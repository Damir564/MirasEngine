#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

enum class GizmoMode {
    None = 0,
    Translate = 1,
    Rotate = 2,
    Scale = 3
};

enum class GizmoAxis {
    None = 0,
    X = 1,
    Y = 2,
    Z = 3
};

struct Gizmo {
    int selectedInstance = -1;
    GizmoMode mode = GizmoMode::None;
    GizmoAxis activeAxis = GizmoAxis::None;
    bool isDragging = false;
    glm::vec2 dragStart{ 0.0f };
    glm::vec3 originalPosition{ 0.0f };
    glm::vec3 originalRotation{ 0.0f };
    glm::vec3 originalScale{ 1.0f };

    void select(int instanceIndex) {
        selectedInstance = instanceIndex;
        if (instanceIndex >= 0 && mode == GizmoMode::None) {
            mode = GizmoMode::Translate;
        }
    }

    void deselect() {
        selectedInstance = -1;
        mode = GizmoMode::None;
        activeAxis = GizmoAxis::None;
        isDragging = false;
    }
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

inline Ray screenToWorldRay(float mouseX, float mouseY,
    float screenWidth, float screenHeight,
    const glm::mat4& view, const glm::mat4& proj)
{
    float x = (2.0f * mouseX) / screenWidth - 1.0f;
    float y = 1.0f - (2.0f * mouseY) / screenHeight;

    glm::vec4 clipNear(x, y, 0.0f, 1.0f);
    glm::vec4 clipFar(x, y, 1.0f, 1.0f);

    glm::mat4 invVP = glm::inverse(proj * view);
    glm::vec4 worldNear = invVP * clipNear;
    glm::vec4 worldFar = invVP * clipFar;
    worldNear /= worldNear.w;
    worldFar /= worldFar.w;

    Ray ray;
    ray.origin = glm::vec3(worldNear);
    ray.direction = glm::normalize(glm::vec3(worldFar - worldNear));
    return ray;
}

inline bool rayIntersectsAABB(const Ray& ray,
    const glm::vec3& boundsMin, const glm::vec3& boundsMax,
    float& tOut)
{
    float tmin = -1e30f;
    float tmax = 1e30f;

    for (int i = 0; i < 3; ++i) {
        if (std::abs(ray.direction[i]) < 1e-8f) {
            if (ray.origin[i] < boundsMin[i] || ray.origin[i] > boundsMax[i])
                return false;
        }
        else {
            float invD = 1.0f / ray.direction[i];
            float t1 = (boundsMin[i] - ray.origin[i]) * invD;
            float t2 = (boundsMax[i] - ray.origin[i]) * invD;

            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);

            if (tmin > tmax)
                return false;
        }
    }

    tOut = tmin > 0.0f ? tmin : tmax;
    return tOut > 0.0f;
}

inline bool rayIntersectsTransformedAABB(const Ray& ray,
    const glm::vec3& localMin, const glm::vec3& localMax,
    const glm::mat4& transform,
    float& tOut)
{
    glm::mat4 invTransform = glm::inverse(transform);
    glm::vec3 localOrigin = glm::vec3(invTransform * glm::vec4(ray.origin, 1.0f));
    glm::vec3 localDir = glm::normalize(glm::vec3(invTransform * glm::vec4(ray.direction, 0.0f)));

    Ray localRay;
    localRay.origin = localOrigin;
    localRay.direction = localDir;

    return rayIntersectsAABB(localRay, localMin, localMax, tOut);
}

// Project a 3D point to screen pixel coordinates
inline glm::vec2 worldToScreen(const glm::vec3& worldPos,
    const glm::mat4& vp,
    float screenWidth, float screenHeight)
{
    glm::vec4 clip = vp * glm::vec4(worldPos, 1.0f);
    if (clip.w <= 0.0001f) return glm::vec2(-10000.0f); // Behind camera
    glm::vec3 ndc = glm::vec3(clip) / clip.w;
    return glm::vec2(
        (ndc.x * 0.5f + 0.5f) * screenWidth,
        (1.0f - (ndc.y * 0.5f + 0.5f)) * screenHeight
    );
}

// Distance from a 2D point to a 2D line segment, returns parametric t along segment
inline float pointToSegment2D(const glm::vec2& point,
    const glm::vec2& segA, const glm::vec2& segB,
    float& segT)
{
    glm::vec2 ab = segB - segA;
    float abLen2 = glm::dot(ab, ab);

    if (abLen2 < 0.0001f) {
        segT = 0.0f;
        return glm::length(point - segA);
    }

    segT = glm::clamp(glm::dot(point - segA, ab) / abLen2, 0.0f, 1.0f);
    glm::vec2 closest = segA + ab * segT;
    return glm::length(point - closest);
}

// Pick gizmo axis entirely in screen space - much more reliable
inline GizmoAxis pickGizmoAxis(const Ray& ray,
    const glm::vec3& gizmoCenter,
    float axisLength,
    float pickRadiusPixels,
    const glm::mat4& view,
    const glm::mat4& proj,
    float screenWidth,
    float screenHeight)
{
    struct AxisCandidate {
        GizmoAxis axis;
        glm::vec3 dir;
    };

    AxisCandidate axes[] = {
        { GizmoAxis::X, glm::vec3(1, 0, 0) },
        { GizmoAxis::Y, glm::vec3(0, 1, 0) },
        { GizmoAxis::Z, glm::vec3(0, 0, 1) }
    };

    glm::mat4 vp = proj * view;

    // Check if gizmo center is behind camera
    glm::vec4 centerClip = vp * glm::vec4(gizmoCenter, 1.0f);
    if (centerClip.w <= 0.0f) return GizmoAxis::None;

    // Get mouse position in screen pixels from ray
    // The ray origin projected to screen IS the mouse position
    glm::vec2 mousePixel = worldToScreen(ray.origin + ray.direction * 0.1f, vp, screenWidth, screenHeight);

    // Actually, we should use the click position directly. Since the ray was constructed
    // from the mouse position, we can reconstruct it:
    // But it's more reliable to project a near-plane point.
    // The ray origin in perspective projection IS the camera position,
    // so we need to use NDC coordinates instead.

    // Reconstruct mouse pixel from ray direction
    glm::vec4 rayPointClip = vp * glm::vec4(ray.origin + ray.direction, 1.0f);
    if (rayPointClip.w <= 0.0f) return GizmoAxis::None;
    glm::vec3 rayPointNDC = glm::vec3(rayPointClip) / rayPointClip.w;
    mousePixel = glm::vec2(
        (rayPointNDC.x * 0.5f + 0.5f) * screenWidth,
        (1.0f - (rayPointNDC.y * 0.5f + 0.5f)) * screenHeight
    );

    float bestDist = pickRadiusPixels;
    GizmoAxis bestAxis = GizmoAxis::None;

    for (auto& a : axes) {
        glm::vec3 axisEnd = gizmoCenter + a.dir * axisLength;

        // Project both endpoints to screen
        glm::vec2 screenStart = worldToScreen(gizmoCenter, vp, screenWidth, screenHeight);
        glm::vec2 screenEnd = worldToScreen(axisEnd, vp, screenWidth, screenHeight);

        // Skip if either point is behind camera (indicated by large negative coords)
        if (screenStart.x < -5000.0f || screenEnd.x < -5000.0f) continue;

        // Skip if the axis has zero screen length (pointing directly at/away from camera)
        float screenLen = glm::length(screenEnd - screenStart);
        if (screenLen < 2.0f) continue; // Less than 2 pixels - can't pick this

        // Find pixel distance from mouse to the screen-space line segment
        float segT;
        float pixelDist = pointToSegment2D(mousePixel, screenStart, screenEnd, segT);

        if (pixelDist < bestDist) {
            bestDist = pixelDist;
            bestAxis = a.axis;
        }
    }

    return bestAxis;
}

inline float getGizmoScale(const glm::vec3& gizmoPos, const glm::vec3& cameraPos,
    float desiredScreenSize = 0.15f,
    const glm::mat4& proj = glm::mat4(1.0f))
{
    float dist = glm::length(gizmoPos - cameraPos);
    float tanHalfFov = 1.0f / proj[1][1];
    return dist * tanHalfFov * desiredScreenSize;
}

struct GizmoVertex {
    glm::vec3 position;
    glm::vec3 color;
};

inline std::vector<GizmoVertex> generateTranslateGizmoLines(float length = 2.0f) {
    return {
        // X axis - Red
        {{ 0, 0, 0 }, { 1, 0, 0 }},
        {{ length, 0, 0 }, { 1, 0, 0 }},
        {{ length, 0, 0 }, { 1, 0, 0 }},
        {{ length - 0.2f, 0.1f, 0 }, { 1, 0, 0 }},
        {{ length, 0, 0 }, { 1, 0, 0 }},
        {{ length - 0.2f, -0.1f, 0 }, { 1, 0, 0 }},

        // Y axis - Green
        {{ 0, 0, 0 }, { 0, 1, 0 }},
        {{ 0, length, 0 }, { 0, 1, 0 }},
        {{ 0, length, 0 }, { 0, 1, 0 }},
        {{ 0.1f, length - 0.2f, 0 }, { 0, 1, 0 }},
        {{ 0, length, 0 }, { 0, 1, 0 }},
        {{ -0.1f, length - 0.2f, 0 }, { 0, 1, 0 }},

        // Z axis - Blue
        {{ 0, 0, 0 }, { 0, 0, 1 }},
        {{ 0, 0, length }, { 0, 0, 1 }},
        {{ 0, 0, length }, { 0, 0, 1 }},
        {{ 0, 0.1f, length - 0.2f }, { 0, 0, 1 }},
        {{ 0, 0, length }, { 0, 0, 1 }},
        {{ 0, -0.1f, length - 0.2f }, { 0, 0, 1 }},
    };
}

inline std::vector<GizmoVertex> generateRotateGizmoLines(int segments = 32, float radius = 1.5f) {
    std::vector<GizmoVertex> lines;
    float step = 2.0f * 3.14159265f / segments;

    for (int i = 0; i < segments; ++i) {
        float a0 = i * step, a1 = (i + 1) * step;
        lines.push_back({ { 0, cosf(a0) * radius, sinf(a0) * radius }, { 1, 0, 0 } });
        lines.push_back({ { 0, cosf(a1) * radius, sinf(a1) * radius }, { 1, 0, 0 } });
    }
    for (int i = 0; i < segments; ++i) {
        float a0 = i * step, a1 = (i + 1) * step;
        lines.push_back({ { cosf(a0) * radius, 0, sinf(a0) * radius }, { 0, 1, 0 } });
        lines.push_back({ { cosf(a1) * radius, 0, sinf(a1) * radius }, { 0, 1, 0 } });
    }
    for (int i = 0; i < segments; ++i) {
        float a0 = i * step, a1 = (i + 1) * step;
        lines.push_back({ { cosf(a0) * radius, sinf(a0) * radius, 0 }, { 0, 0, 1 } });
        lines.push_back({ { cosf(a1) * radius, sinf(a1) * radius, 0 }, { 0, 0, 1 } });
    }
    return lines;
}

inline std::vector<GizmoVertex> generateScaleGizmoLines(float length = 2.0f) {
    float box = 0.15f;
    return {
        {{ 0, 0, 0 }, { 1, 0, 0 }},
        {{ length, 0, 0 }, { 1, 0, 0 }},
        {{ length - box, -box, -box }, { 1, 0, 0 }},
        {{ length + box, -box, -box }, { 1, 0, 0 }},
        {{ length + box, -box, -box }, { 1, 0, 0 }},
        {{ length + box, box, -box }, { 1, 0, 0 }},
        {{ length + box, box, -box }, { 1, 0, 0 }},
        {{ length - box, box, -box }, { 1, 0, 0 }},
        {{ length - box, box, -box }, { 1, 0, 0 }},
        {{ length - box, -box, -box }, { 1, 0, 0 }},

        {{ 0, 0, 0 }, { 0, 1, 0 }},
        {{ 0, length, 0 }, { 0, 1, 0 }},
        {{ -box, length - box, -box }, { 0, 1, 0 }},
        {{ box, length - box, -box }, { 0, 1, 0 }},
        {{ box, length - box, -box }, { 0, 1, 0 }},
        {{ box, length + box, -box }, { 0, 1, 0 }},
        {{ box, length + box, -box }, { 0, 1, 0 }},
        {{ -box, length + box, -box }, { 0, 1, 0 }},
        {{ -box, length + box, -box }, { 0, 1, 0 }},
        {{ -box, length - box, -box }, { 0, 1, 0 }},

        {{ 0, 0, 0 }, { 0, 0, 1 }},
        {{ 0, 0, length }, { 0, 0, 1 }},
        {{ -box, -box, length - box }, { 0, 0, 1 }},
        {{ box, -box, length - box }, { 0, 0, 1 }},
        {{ box, -box, length - box }, { 0, 0, 1 }},
        {{ box, box, length - box }, { 0, 0, 1 }},
        {{ box, box, length - box }, { 0, 0, 1 }},
        {{ -box, box, length - box }, { 0, 0, 1 }},
        {{ -box, box, length - box }, { 0, 0, 1 }},
        {{ -box, -box, length - box }, { 0, 0, 1 }},
    };
}