#pragma once

// Constants and helpers shared by EditorLevel*.cpp; private to them (anonymous namespace).
#include <array>
#include <cfloat>
#include <numeric>
#include "Editor.h"

namespace {
constexpr const char* kTexturesRoot = "textures";
constexpr ImU32 kFaceOutline = IM_COL32(255, 160, 40, 255);
constexpr ImU32 kFaceGroupOutline = IM_COL32(255, 200, 120, 190);
constexpr ImU32 kFaceEdges = IM_COL32(255, 255, 255, 70);
constexpr ImU32 kVertexDot = IM_COL32(255, 255, 255, 200);
constexpr ImU32 kClipOutline = IM_COL32(80, 200, 255, 255);
constexpr ImU32 kMirrorOutline = IM_COL32(230, 110, 255, 255);
constexpr ImU32 kDrawOutline = IM_COL32(110, 255, 140, 255);
constexpr float kVertexPickRadius = 10.0f;

int dominantAxis(const glm::vec3& n)
{
    const glm::vec3 a = glm::abs(n);
    return a.x >= a.y && a.x >= a.z ? 0 : (a.y >= a.z ? 1 : 2);
}

// Recomputes coordinate `axis` of p so that p lies on the plane through `point` with normal n.
glm::vec3 ontoPlaneAlong(glm::vec3 p, int axis, const glm::vec3& point, const glm::vec3& n)
{
    p[axis] = 0.0f;
    p[axis] = (glm::dot(n, point) - glm::dot(n, p)) / n[axis];
    return p;
}

bool rayHitsPlane(const Ray& ray, const glm::vec3& point, const glm::vec3& normal, glm::vec3& hit)
{
    const float denom = glm::dot(ray.direction, normal);
    if (std::abs(denom) < 1e-6f)
        return false;
    const float t = glm::dot(point - ray.origin, normal) / denom;
    if (t <= 0.0f)
        return false;
    hit = ray.origin + ray.direction * t;
    return true;
}

// Smallest and largest dot(normal, p) over the mesh's vertices.
glm::vec2 meshExtent(const PolyMesh& mesh, const glm::vec3& normal)
{
    glm::vec2 range(FLT_MAX, -FLT_MAX);
    for (const glm::vec3& p : mesh.positions) {
        const float d = glm::dot(normal, p);
        range = { std::min(range.x, d), std::max(range.y, d) };
    }
    return mesh.positions.empty() ? glm::vec2(0.0f) : range;
}

// Edge edits move along directions that are often diagonal (the edge between two box faces), so the
// offset is snapped per axis: ends that started on the grid stay on it.
glm::vec3 edgeOffset(const glm::vec3& direction, float distance, float grid, bool snap)
{
    const glm::vec3 offset = direction * distance;
    return snap ? glm::round(offset / grid) * grid : offset;
}

// Moves edge a-b by offset, or extrudes it; returns the edge to select afterwards (the new one).
std::array<uint32_t, 2> editEdge(PolyMesh& mesh, uint32_t a, uint32_t b, const glm::vec3& offset, bool extrude)
{
    if (extrude) {
        uint32_t newA, newB;
        const size_t oldFaces = mesh.faces.size();
        if (!mesh.extrudeEdge(a, b, offset, newA, newB))
            return { a, b };
        // Continuing a flat surface keeps it one face: the new quad joins the face it extends.
        std::vector<int64_t> keys(mesh.faces.size(), -1);
        std::iota(keys.begin(), keys.begin() + oldFaces, int64_t(0));
        std::vector<uint32_t> remap;
        mesh.mergeCoplanarFaces(keys, &remap);
        return { remap[newA], remap[newB] };
    }
    mesh.positions[a] += offset;
    mesh.positions[b] += offset;
    return { a, b };
}
}
