#pragma once

// Helpers shared by PolyMesh*.cpp; private to them (anonymous namespace).
#include <array>
#include <cmath>
#include <vector>
#include "PolyMesh.h"

namespace {

// Right-handed tangent frame (t, b, n) on the face plane; t stays horizontal on walls so textures
// line up across neighbouring faces.
void faceBasis(const glm::vec3& n, glm::vec3& t, glm::vec3& b)
{
    if (std::abs(n.y) > 0.9f)
        t = glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f) - n * n.x);
    else
        t = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), n));
    b = glm::cross(n, t);
}

float cross2(const glm::vec2& a, const glm::vec2& b) { return a.x * b.y - a.y * b.x; }

bool pointInTriangle(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c)
{
    return cross2(b - a, p - a) >= 0.0f && cross2(c - b, p - b) >= 0.0f && cross2(a - c, p - c) >= 0.0f;
}

// Ear clipping on the face projected to its plane. Outputs triangles as positions within face.verts.
void triangulateFace(const PolyMesh& mesh, const PolyFace& face, const glm::vec3& n,
    std::vector<std::array<uint32_t, 3>>& out)
{
    const uint32_t count = static_cast<uint32_t>(face.verts.size());
    if (count < 3)
        return;
    if (count == 3) {
        out.push_back({ 0, 1, 2 });
        return;
    }

    glm::vec3 t, b;
    faceBasis(n, t, b);
    std::vector<glm::vec2> pts(count);
    for (uint32_t i = 0; i < count; ++i) {
        const glm::vec3& p = mesh.positions[face.verts[i]];
        pts[i] = { glm::dot(p, t), glm::dot(p, b) };
    }

    std::vector<uint32_t> remaining(count);
    for (uint32_t i = 0; i < count; ++i)
        remaining[i] = i;

    while (remaining.size() > 3) {
        const size_t size = remaining.size();
        bool clipped = false;
        for (size_t i = 0; i < size; ++i) {
            const uint32_t ia = remaining[(i + size - 1) % size];
            const uint32_t ib = remaining[i];
            const uint32_t ic = remaining[(i + 1) % size];
            const glm::vec2 &a = pts[ia], &bb = pts[ib], &c = pts[ic];
            if (cross2(bb - a, c - bb) <= 1e-9f)
                continue; // reflex or degenerate corner
            bool containsOther = false;
            for (uint32_t j : remaining) {
                if (j != ia && j != ib && j != ic && pointInTriangle(pts[j], a, bb, c)) {
                    containsOther = true;
                    break;
                }
            }
            if (containsOther)
                continue;
            out.push_back({ ia, ib, ic });
            remaining.erase(remaining.begin() + i);
            clipped = true;
            break;
        }
        // Self-intersecting or degenerate polygon: fall back to a fan so the face still renders.
        if (!clipped) {
            for (size_t i = 1; i + 1 < remaining.size(); ++i)
                out.push_back({ remaining[0], remaining[i], remaining[i + 1] });
            return;
        }
    }
    out.push_back({ remaining[0], remaining[1], remaining[2] });
}

// The face's planar projection: uv = rotate(dot(p, t), -dot(p, b)) / scale + offset.
struct FaceUVMapping {
    glm::vec3 t{ 0.0f }, b{ 0.0f };
    float cr = 1.0f, sr = 0.0f;
    glm::vec2 scale{ 1.0f };
    glm::vec2 offset{ 0.0f };

    FaceUVMapping(const PolyMesh& mesh, const PolyFace& face, float texelSize)
    {
        faceBasis(mesh.faceNormal(face), t, b);
        const float r = glm::radians(face.uvRotation);
        cr = std::cos(r);
        sr = std::sin(r);
        scale = glm::max(glm::abs(face.uvScale) * texelSize, glm::vec2(1e-4f));
        offset = face.uvOffset;
    }
    // Before rotation, scale and offset.
    glm::vec2 project(const glm::vec3& p) const { return glm::vec2(glm::dot(p, t), -glm::dot(p, b)); }
    glm::vec2 rotate(const glm::vec2& uv) const { return glm::vec2(cr * uv.x - sr * uv.y, sr * uv.x + cr * uv.y); }
    glm::vec2 operator()(const glm::vec3& p) const { return rotate(project(p)) / scale + offset; }
};

} // namespace
