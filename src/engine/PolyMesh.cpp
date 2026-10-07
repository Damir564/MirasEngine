#include "PolyMesh.h"
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <istream>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <glm/gtc/constants.hpp>
#include "MaterialLibrary.h"
#include "Vertex.h"

glm::vec3 PolyMesh::faceNormal(const PolyFace& face) const
{
    // Newell's method: robust for concave and slightly non-planar polygons.
    glm::vec3 n(0.0f);
    const size_t count = face.verts.size();
    for (size_t i = 0; i < count; ++i) {
        const glm::vec3& a = positions[face.verts[i]];
        const glm::vec3& b = positions[face.verts[(i + 1) % count]];
        n.x += (a.y - b.y) * (a.z + b.z);
        n.y += (a.z - b.z) * (a.x + b.x);
        n.z += (a.x - b.x) * (a.y + b.y);
    }
    const float len = glm::length(n);
    return len > 1e-12f ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
}

glm::vec3 PolyMesh::faceCenter(const PolyFace& face) const
{
    glm::vec3 c(0.0f);
    for (uint32_t v : face.verts)
        c += positions[v];
    return face.verts.empty() ? c : c / float(face.verts.size());
}

void PolyMesh::addFace(std::vector<uint32_t> verts, const glm::vec3& expectedNormal)
{
    PolyFace face;
    face.verts = std::move(verts);
    if (glm::dot(faceNormal(face), expectedNormal) < 0.0f)
        std::reverse(face.verts.begin(), face.verts.end());
    faces.push_back(std::move(face));
}

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
}

void PolyMesh::triangulate(const PolyFace& face, std::vector<std::array<uint32_t, 3>>& out) const
{
    triangulateFace(*this, face, faceNormal(face), out);
}

void PolyMesh::moveFace(uint32_t faceIndex, float distance)
{
    if (faceIndex >= faces.size())
        return;
    const glm::vec3 offset = faceNormal(faces[faceIndex]) * distance;
    for (uint32_t v : faces[faceIndex].verts)
        positions[v] += offset;
}

void PolyMesh::extrudeFace(uint32_t faceIndex, float distance)
{
    if (faceIndex >= faces.size() || faces[faceIndex].verts.size() < 3)
        return;
    const glm::vec3 n = faceNormal(faces[faceIndex]);
    const glm::vec3 offset = n * distance;
    const std::vector<uint32_t> oldVerts = faces[faceIndex].verts;
    const size_t count = oldVerts.size();

    std::vector<uint32_t> newVerts(count);
    for (size_t i = 0; i < count; ++i) {
        newVerts[i] = static_cast<uint32_t>(positions.size());
        positions.push_back(positions[oldVerts[i]] + offset);
    }
    faces[faceIndex].verts = newVerts;

    const uint32_t material = faces[faceIndex].material;
    // Pulling inwards turns the sides inside out, so their outward direction flips with the sign.
    const float side = distance < 0.0f ? -1.0f : 1.0f;
    for (size_t i = 0; i < count; ++i) {
        const size_t j = (i + 1) % count;
        const glm::vec3 edge = positions[oldVerts[j]] - positions[oldVerts[i]];
        addFace({ oldVerts[i], oldVerts[j], newVerts[j], newVerts[i] }, glm::cross(edge, n) * side);
        faces.back().material = material;
    }
}

bool PolyMesh::insetFace(uint32_t faceIndex, float distance)
{
    if (faceIndex >= faces.size() || faces[faceIndex].verts.size() < 3 || distance <= 0.0f)
        return false;
    const glm::vec3 n = faceNormal(faces[faceIndex]);
    const std::vector<uint32_t> oldVerts = faces[faceIndex].verts;
    const size_t count = oldVerts.size();

    // With counter-clockwise winding, cross(n, edge) points into the face. Each corner moves so that
    // both of its edges shift inward by exactly `distance` (a miter), concave corners included.
    std::vector<glm::vec3> inner(count);
    for (size_t i = 0; i < count; ++i) {
        const glm::vec3& prev = positions[oldVerts[(i + count - 1) % count]];
        const glm::vec3& cur = positions[oldVerts[i]];
        const glm::vec3& next = positions[oldVerts[(i + 1) % count]];
        const glm::vec3 inPrev = glm::normalize(glm::cross(n, cur - prev));
        const glm::vec3 inNext = glm::normalize(glm::cross(n, next - cur));
        const float denom = 1.0f + glm::dot(inPrev, inNext);
        if (!(denom >= 1e-3f)) // the outline doubles back on itself here (NaN: zero-length edge)
            return false;
        inner[i] = cur + (inPrev + inNext) * (distance / denom);
    }
    // An inset wider than the face flips edges around; reject that instead of making inverted faces.
    for (size_t i = 0; i < count; ++i) {
        const size_t j = (i + 1) % count;
        const glm::vec3 oldEdge = positions[oldVerts[j]] - positions[oldVerts[i]];
        const glm::vec3 newEdge = inner[j] - inner[i];
        if (glm::dot(oldEdge, newEdge) <= 1e-6f * glm::dot(oldEdge, oldEdge))
            return false;
    }

    std::vector<uint32_t> newVerts(count);
    for (size_t i = 0; i < count; ++i) {
        newVerts[i] = static_cast<uint32_t>(positions.size());
        positions.push_back(inner[i]);
    }
    // Border quads copy the face, so the texture runs on across them unchanged.
    PolyFace border = faces[faceIndex];
    faces[faceIndex].verts = newVerts;
    for (size_t i = 0; i < count; ++i) {
        const size_t j = (i + 1) % count;
        border.verts = { oldVerts[i], oldVerts[j], newVerts[j], newVerts[i] };
        faces.push_back(border);
    }
    return true;
}

void PolyMesh::flipFace(uint32_t faceIndex)
{
    if (faceIndex < faces.size())
        std::reverse(faces[faceIndex].verts.begin(), faces[faceIndex].verts.end());
}

void PolyMesh::deleteFace(uint32_t faceIndex)
{
    if (faceIndex >= faces.size())
        return;
    faces.erase(faces.begin() + faceIndex);
    removeUnusedVertices();
}

void PolyMesh::deleteVertices(const std::vector<uint32_t>& verts)
{
    std::vector<bool> doomed(positions.size(), false);
    for (uint32_t v : verts) {
        if (v < doomed.size())
            doomed[v] = true;
    }
    std::erase_if(faces, [&doomed](const PolyFace& face) {
        return std::any_of(face.verts.begin(), face.verts.end(), [&doomed](uint32_t v) { return doomed[v]; });
    });
    removeUnusedVertices();
}

uint32_t PolyMesh::mergeVertices(const std::vector<uint32_t>& verts)
{
    std::vector<bool> merged(positions.size(), false);
    glm::vec3 center(0.0f);
    uint32_t target = UINT32_MAX;
    size_t count = 0;
    for (uint32_t v : verts) {
        if (v >= positions.size() || merged[v])
            continue;
        merged[v] = true;
        center += positions[v];
        target = std::min(target, v);
        ++count;
    }
    if (count == 0)
        return UINT32_MAX;
    positions[target] = center / static_cast<float>(count);

    for (PolyFace& face : faces) {
        std::vector<uint32_t> remapped;
        remapped.reserve(face.verts.size());
        for (uint32_t v : face.verts) {
            const uint32_t mapped = merged[v] ? target : v;
            // Neighbours that both collapsed into the target leave a zero-length edge; keep one.
            if (remapped.empty() || remapped.back() != mapped)
                remapped.push_back(mapped);
        }
        while (remapped.size() > 1 && remapped.front() == remapped.back())
            remapped.pop_back();
        face.verts = std::move(remapped);
    }
    std::erase_if(faces, [](const PolyFace& face) { return face.verts.size() < 3; });
    return removeUnusedVertices()[target];
}

std::vector<uint32_t> PolyMesh::removeUnusedVertices()
{
    std::vector<uint32_t> remap(positions.size(), UINT32_MAX);
    for (const PolyFace& face : faces) {
        for (uint32_t v : face.verts)
            remap[v] = 0;
    }
    uint32_t next = 0;
    for (size_t i = 0; i < positions.size(); ++i) {
        if (remap[i] == UINT32_MAX)
            continue;
        remap[i] = next;
        positions[next++] = positions[i];
    }
    positions.resize(next);
    for (PolyFace& face : faces) {
        for (uint32_t& v : face.verts)
            v = remap[v];
    }
    return remap;
}

namespace {

// Position of v within face.verts, or face.verts.size() if the face doesn't use it.
size_t cornerOf(const PolyFace& face, uint32_t v)
{
    return static_cast<size_t>(std::find(face.verts.begin(), face.verts.end(), v) - face.verts.begin());
}

// Segments a-b and c-d cross at a point inside both.
bool segmentsCross(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, const glm::vec2& d)
{
    return (cross2(b - a, c - a) > 0.0f) != (cross2(b - a, d - a) > 0.0f)
        && (cross2(d - c, a - c) > 0.0f) != (cross2(d - c, b - c) > 0.0f);
}
}

bool PolyMesh::isEdge(uint32_t a, uint32_t b) const
{
    for (const PolyFace& face : faces) {
        const size_t count = face.verts.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t u = face.verts[i], w = face.verts[(i + 1) % count];
            if ((u == a && w == b) || (u == b && w == a))
                return true;
        }
    }
    return false;
}

namespace {
// Position i in face.verts where the face runs along edge a-b (either way), or -1.
int edgeInFace(const PolyFace& face, uint32_t a, uint32_t b)
{
    const size_t count = face.verts.size();
    for (size_t i = 0; i < count; ++i) {
        const uint32_t u = face.verts[i], w = face.verts[(i + 1) % count];
        if ((u == a && w == b) || (u == b && w == a))
            return static_cast<int>(i);
    }
    return -1;
}
}

glm::vec3 PolyMesh::edgeNormal(uint32_t a, uint32_t b) const
{
    glm::vec3 sum(0.0f);
    const PolyFace* first = nullptr;
    for (const PolyFace& face : faces) {
        if (edgeInFace(face, a, b) < 0)
            continue;
        if (!first)
            first = &face;
        sum += faceNormal(face);
    }
    if (!first)
        return glm::vec3(0.0f);
    // Opposite faces (the rim of a thin sheet) cancel out; use one of them.
    const float length = glm::length(sum);
    return length > 1e-4f ? sum / length : faceNormal(*first);
}

glm::vec3 PolyMesh::edgeExtrudeDirection(uint32_t a, uint32_t b) const
{
    const PolyFace* border = nullptr;
    int position = -1;
    int users = 0;
    for (const PolyFace& face : faces) {
        const int i = edgeInFace(face, a, b);
        if (i < 0)
            continue;
        border = &face;
        position = i;
        ++users;
    }
    if (users != 1)
        return edgeNormal(a, b);
    // Faces wind counter-clockwise around their normal, so the outside is to the right of each edge.
    const size_t count = border->verts.size();
    const glm::vec3 along = positions[border->verts[(position + 1) % count]] - positions[border->verts[position]];
    const glm::vec3 out = glm::cross(along, faceNormal(*border));
    const float length = glm::length(out);
    return length > 1e-6f ? out / length : edgeNormal(a, b);
}

bool PolyMesh::extrudeEdge(uint32_t a, uint32_t b, const glm::vec3& offset, uint32_t& newA, uint32_t& newB)
{
    if (a == b || a >= positions.size() || b >= positions.size())
        return false;
    size_t first = 0;
    int position = -1;
    int users = 0;
    for (size_t f = 0; f < faces.size(); ++f) {
        const int i = edgeInFace(faces[f], a, b);
        if (i >= 0 && users++ == 0) {
            first = f;
            position = i;
        }
    }
    if (users == 0)
        return false;
    // The face runs u -> w; a face continuing it across the edge must run w -> u.
    const PolyFace& face = faces[first];
    const uint32_t u = face.verts[position];
    const uint32_t w = face.verts[(position + 1) % face.verts.size()];
    const uint32_t material = face.material;

    newA = static_cast<uint32_t>(positions.size());
    positions.push_back(positions[a] + offset);
    newB = static_cast<uint32_t>(positions.size());
    positions.push_back(positions[b] + offset);
    const uint32_t newU = u == a ? newA : newB;
    const uint32_t newW = u == a ? newB : newA;

    PolyFace quad;
    quad.material = material;
    // A border edge continues the face's surface, texture included.
    if (users == 1) {
        quad.uvScale = face.uvScale;
        quad.uvOffset = face.uvOffset;
        quad.uvRotation = face.uvRotation;
    }
    quad.verts = { w, u, newU, newW };
    faces.push_back(quad);
    if (users > 1) {
        quad.verts = { newW, newU, u, w };
        faces.push_back(quad);
    }
    return true;
}

uint32_t PolyMesh::splitEdge(uint32_t a, uint32_t b)
{
    if (a >= positions.size() || b >= positions.size())
        return UINT32_MAX;
    return splitEdgeAt(a, b, (positions[a] + positions[b]) * 0.5f);
}

uint32_t PolyMesh::splitEdgeAt(uint32_t a, uint32_t b, const glm::vec3& point)
{
    if (a == b || a >= positions.size() || b >= positions.size())
        return UINT32_MAX;
    uint32_t mid = UINT32_MAX;
    for (PolyFace& face : faces) {
        const size_t count = face.verts.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t u = face.verts[i], w = face.verts[(i + 1) % count];
            if (!((u == a && w == b) || (u == b && w == a)))
                continue;
            if (mid == UINT32_MAX) {
                mid = static_cast<uint32_t>(positions.size());
                positions.push_back(point);
            }
            // At i == count - 1 this appends, which is between the last and first corner.
            face.verts.insert(face.verts.begin() + static_cast<std::ptrdiff_t>(i + 1), mid);
            break;
        }
    }
    return mid;
}

uint32_t PolyMesh::faceToConnect(uint32_t a, uint32_t b) const
{
    if (a == b)
        return UINT32_MAX;
    for (size_t f = 0; f < faces.size(); ++f) {
        const PolyFace& face = faces[f];
        const size_t count = face.verts.size();
        const size_t ia = cornerOf(face, a), ib = cornerOf(face, b);
        if (ia == count || ib == count)
            continue;
        const size_t gap = (ib + count - ia) % count;
        if (gap == 1 || gap == count - 1)
            continue; // already an edge

        // On a concave face the line can leave the face, and the halves would overlap.
        glm::vec3 t, bt;
        faceBasis(faceNormal(face), t, bt);
        auto project = [&](uint32_t v) { return glm::vec2(glm::dot(positions[v], t), glm::dot(positions[v], bt)); };
        const glm::vec2 pa = project(a), pb = project(b);
        const glm::vec2 mid = (pa + pb) * 0.5f;
        bool inside = false;
        bool crosses = false;
        for (size_t i = 0; i < count && !crosses; ++i) {
            const uint32_t u = face.verts[i], w = face.verts[(i + 1) % count];
            const glm::vec2 pu = project(u), pw = project(w);
            if ((pu.y > mid.y) != (pw.y > mid.y) && mid.x < pu.x + (mid.y - pu.y) * (pw.x - pu.x) / (pw.y - pu.y))
                inside = !inside;
            if (u != a && u != b && w != a && w != b && segmentsCross(pa, pb, pu, pw))
                crosses = true;
        }
        if (inside && !crosses)
            return static_cast<uint32_t>(f);
    }
    return UINT32_MAX;
}

uint32_t PolyMesh::connectVertices(uint32_t a, uint32_t b)
{
    const uint32_t f = faceToConnect(a, b);
    if (f == UINT32_MAX)
        return UINT32_MAX;
    PolyFace other = faces[f]; // keeps material and UV params
    const std::vector<uint32_t>& verts = faces[f].verts;
    const size_t count = verts.size();
    const size_t ia = cornerOf(faces[f], a), ib = cornerOf(faces[f], b);
    // Walking the corners in order keeps both halves wound like the original.
    std::vector<uint32_t> first, second;
    for (size_t i = ia;; i = (i + 1) % count) {
        first.push_back(verts[i]);
        if (i == ib)
            break;
    }
    for (size_t i = ib;; i = (i + 1) % count) {
        second.push_back(verts[i]);
        if (i == ia)
            break;
    }
    faces[f].verts = std::move(first);
    other.verts = std::move(second);
    faces.push_back(std::move(other));
    return static_cast<uint32_t>(faces.size() - 1);
}

namespace {

// Drawn points closer than this (object units) to the outline count as on it.
constexpr float kOnBorder = 1e-4f;

// 2D coordinates on a face's plane; counter-clockwise faces stay counter-clockwise.
struct FacePlane {
    glm::vec3 t, b;
    explicit FacePlane(const glm::vec3& n) { faceBasis(n, t, b); }
    glm::vec2 operator()(const glm::vec3& p) const { return { glm::dot(p, t), glm::dot(p, b) }; }
};

std::vector<glm::vec2> projectFace(const PolyMesh& mesh, const PolyFace& face, const FacePlane& plane)
{
    std::vector<glm::vec2> out;
    out.reserve(face.verts.size());
    for (uint32_t v : face.verts)
        out.push_back(plane(mesh.positions[v]));
    return out;
}

float segmentDistance(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b)
{
    const glm::vec2 ab = b - a;
    const float len2 = glm::dot(ab, ab);
    const float s = len2 > 0.0f ? std::clamp(glm::dot(p - a, ab) / len2, 0.0f, 1.0f) : 0.0f;
    return glm::length(p - (a + ab * s));
}

bool insidePolygon(const glm::vec2& p, const std::vector<glm::vec2>& poly)
{
    bool inside = false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
        if ((poly[i].y > p.y) != (poly[j].y > p.y) &&
            p.x < poly[i].x + (p.y - poly[i].y) * (poly[j].x - poly[i].x) / (poly[j].y - poly[i].y))
            inside = !inside;
    return inside;
}

bool onPolygonBorder(const glm::vec2& p, const std::vector<glm::vec2>& poly)
{
    for (size_t i = 0; i < poly.size(); ++i)
        if (segmentDistance(p, poly[i], poly[(i + 1) % poly.size()]) <= kOnBorder)
            return true;
    return false;
}

// Segments a-b and c-d cross at a point clearly inside both; touching at or along them doesn't count.
bool segmentsCrossClearly(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, const glm::vec2& d)
{
    const auto side = [](const glm::vec2& from, const glm::vec2& to, const glm::vec2& p) {
        const float len = glm::length(to - from);
        return len > 0.0f ? cross2(to - from, p - from) / len : 0.0f;
    };
    const auto opposite = [](float u, float w) {
        return (u > kOnBorder && w < -kOnBorder) || (u < -kOnBorder && w > kOnBorder);
    };
    return opposite(side(a, b, c), side(a, b, d)) && opposite(side(c, d, a), side(c, d, b));
}

float signedArea(const std::vector<glm::vec2>& poly)
{
    float area = 0.0f;
    for (size_t i = 0; i < poly.size(); ++i)
        area += cross2(poly[i], poly[(i + 1) % poly.size()]);
    return area * 0.5f;
}
}

bool PolyMesh::onFaceBorder(uint32_t faceIndex, const glm::vec3& point) const
{
    if (faceIndex >= faces.size() || faces[faceIndex].verts.size() < 3)
        return false;
    const FacePlane plane(faceNormal(faces[faceIndex]));
    return onPolygonBorder(plane(point), projectFace(*this, faces[faceIndex], plane));
}

uint32_t PolyMesh::divideFace(uint32_t faceIndex, const std::vector<glm::vec3>& points, bool closed, std::string* error)
{
    const auto fail = [error](const char* message) {
        if (error)
            *error = message;
        return UINT32_MAX;
    };
    if (faceIndex >= faces.size() || faces[faceIndex].verts.size() < 3)
        return fail("No face to divide");
    const glm::vec3 n = faceNormal(faces[faceIndex]);
    const FacePlane plane(n);
    const float planeOffset = glm::dot(n, positions[faces[faceIndex].verts[0]]);

    // Onto the plane; repeats (double clicks, a closing point on the first one) are dropped.
    std::vector<glm::vec3> pts;
    for (const glm::vec3& p : points) {
        const glm::vec3 flat = p - n * (glm::dot(n, p) - planeOffset);
        if (pts.empty() || glm::length(flat - pts.back()) > kOnBorder)
            pts.push_back(flat);
    }
    if (closed && pts.size() > 1 && glm::length(pts.front() - pts.back()) <= kOnBorder)
        pts.pop_back();
    if (pts.size() < (closed ? 3u : 2u))
        return fail(closed ? "A closed shape needs at least three points" : "A cut needs at least two points");

    const std::vector<glm::vec2> outline = projectFace(*this, faces[faceIndex], plane);
    const size_t count = pts.size();
    const size_t segments = closed ? count : count - 1;
    std::vector<glm::vec2> q(count);
    std::vector<bool> onBorder(count);
    for (size_t i = 0; i < count; ++i) {
        q[i] = plane(pts[i]);
        onBorder[i] = onPolygonBorder(q[i], outline);
        if (!onBorder[i] && !insidePolygon(q[i], outline))
            return fail("The shape must stay on the face");
        for (size_t j = 0; j < i; ++j)
            if (glm::length(q[i] - q[j]) <= kOnBorder)
                return fail("The shape passes the same point twice");
    }
    for (size_t s = 0; s < segments; ++s) {
        const glm::vec2 a = q[s], b = q[(s + 1) % count];
        for (size_t e = 0; e < outline.size(); ++e)
            if (segmentsCrossClearly(a, b, outline[e], outline[(e + 1) % outline.size()]))
                return fail("The shape must stay on the face");
        // Catches segments leaving a concave face through one of its corners.
        for (float f : { 0.25f, 0.5f, 0.75f }) {
            const glm::vec2 p = glm::mix(a, b, f);
            if (!onPolygonBorder(p, outline) && !insidePolygon(p, outline))
                return fail("The shape must stay on the face");
        }
        for (size_t s2 = s + 2; s2 < segments; ++s2)
            if (!(closed && s == 0 && s2 == count - 1) && segmentsCrossClearly(a, b, q[s2], q[(s2 + 1) % count]))
                return fail("The shape crosses itself");
    }

    PolyMesh work = *this;
    // Border points become corners of the face, splitting the edge they are on in every face using it so
    // neighbours get no cracks; the others become new vertices.
    std::vector<uint32_t> ids(count);
    for (size_t i = 0; i < count; ++i) {
        if (!onBorder[i]) {
            ids[i] = static_cast<uint32_t>(work.positions.size());
            work.positions.push_back(pts[i]);
            continue;
        }
        const std::vector<uint32_t> verts = work.faces[faceIndex].verts;
        uint32_t id = UINT32_MAX;
        for (uint32_t v : verts) {
            if (glm::length(plane(work.positions[v]) - q[i]) <= kOnBorder) {
                id = v;
                break;
            }
        }
        for (size_t e = 0; e < verts.size() && id == UINT32_MAX; ++e) {
            const uint32_t u = verts[e], w = verts[(e + 1) % verts.size()];
            const glm::vec2 pu = plane(work.positions[u]), pw = plane(work.positions[w]);
            if (segmentDistance(q[i], pu, pw) > kOnBorder)
                continue;
            // Placed on the edge itself, so the edge stays straight.
            const float s = glm::dot(q[i] - pu, pw - pu) / glm::dot(pw - pu, pw - pu);
            id = work.splitEdgeAt(u, w, glm::mix(work.positions[u], work.positions[w], s));
        }
        if (id == UINT32_MAX)
            return fail("The shape must stay on the face");
        ids[i] = id;
    }

    if (std::none_of(onBorder.begin(), onBorder.end(), [](bool b) { return b; })) {
        if (!closed)
            return fail("A cut must start and end on the face's border");
        // A hole: the shape becomes the face and the ring around it is split in two by two bridges
        // between outline and shape corners (shortest ones that stay inside the ring and don't cross).
        std::vector<uint32_t> inner = ids;
        std::vector<glm::vec2> innerPts = q;
        if (signedArea(innerPts) < 0.0f) {
            std::reverse(inner.begin(), inner.end());
            std::reverse(innerPts.begin(), innerPts.end());
        }
        const std::vector<uint32_t> outer = work.faces[faceIndex].verts;
        struct Bridge {
            size_t o, i;
            float length;
        };
        std::vector<Bridge> bridges;
        for (size_t o = 0; o < outer.size(); ++o) {
            for (size_t i = 0; i < inner.size(); ++i) {
                const glm::vec2 a = outline[o], b = innerPts[i];
                bool valid = true;
                for (size_t e = 0; e < outline.size() && valid; ++e)
                    valid = !segmentsCrossClearly(a, b, outline[e], outline[(e + 1) % outline.size()]);
                for (size_t e = 0; e < innerPts.size() && valid; ++e)
                    valid = !segmentsCrossClearly(a, b, innerPts[e], innerPts[(e + 1) % innerPts.size()]);
                const glm::vec2 mid = (a + b) * 0.5f;
                if (valid && insidePolygon(mid, outline) && !insidePolygon(mid, innerPts))
                    bridges.push_back({ o, i, glm::length(b - a) });
            }
        }
        std::sort(bridges.begin(), bridges.end(), [](const Bridge& x, const Bridge& y) { return x.length < y.length; });
        const Bridge* first = bridges.empty() ? nullptr : &bridges.front();
        const Bridge* second = nullptr;
        for (const Bridge& bridge : bridges) {
            if (first && bridge.o != first->o && bridge.i != first->i &&
                !segmentsCrossClearly(outline[bridge.o], innerPts[bridge.i], outline[first->o], innerPts[first->i])) {
                second = &bridge;
                break;
            }
        }
        if (!second)
            return fail("Can't split the face around this shape");
        // Outline forward from oFrom to oTo, then the shape backward (the ring runs around it clockwise).
        const auto piece = [&](size_t oFrom, size_t oTo, size_t iFrom, size_t iTo) {
            std::vector<uint32_t> verts;
            for (size_t k = oFrom;; k = (k + 1) % outer.size()) {
                verts.push_back(outer[k]);
                if (k == oTo)
                    break;
            }
            for (size_t k = iFrom;; k = (k + inner.size() - 1) % inner.size()) {
                verts.push_back(inner[k]);
                if (k == iTo)
                    break;
            }
            return verts;
        };
        PolyFace ring = work.faces[faceIndex];
        work.faces[faceIndex].verts = inner;
        ring.verts = piece(first->o, second->o, second->i, first->i);
        work.faces.push_back(ring);
        ring.verts = piece(second->o, first->o, first->i, second->i);
        work.faces.push_back(std::move(ring));
        *this = std::move(work);
        return faceIndex;
    }

    if (!closed && (!onBorder.front() || !onBorder.back()))
        return fail("A cut must start and end on the face's border");
    // Walked from a border point; each run between two border points cuts the piece it crosses in two.
    std::vector<uint32_t> walk;
    std::vector<bool> walkBorder;
    const size_t start = closed ? static_cast<size_t>(std::find(onBorder.begin(), onBorder.end(), true) - onBorder.begin()) : 0;
    for (size_t k = 0; k < (closed ? count + 1 : count); ++k) {
        walk.push_back(ids[(start + k) % count]);
        walkBorder.push_back(onBorder[(start + k) % count]);
    }
    std::vector<uint32_t> pieces{ faceIndex };
    bool divided = false;
    size_t runStart = 0;
    for (size_t k = 1; k < walk.size(); ++k) {
        if (!walkBorder[k])
            continue;
        const uint32_t b0 = walk[runStart], b1 = walk[k];
        const std::vector<uint32_t> interior(walk.begin() + static_cast<std::ptrdiff_t>(runStart + 1),
            walk.begin() + static_cast<std::ptrdiff_t>(k));
        runStart = k;
        if (b0 == b1)
            return fail("The shape must touch the border at two different points");
        const glm::vec2 p0 = plane(work.positions[b0]);
        const glm::vec2 probe = (p0 + plane(work.positions[interior.empty() ? b1 : interior.front()])) * 0.5f;
        uint32_t target = UINT32_MAX;
        bool along = false;
        for (uint32_t f : pieces) {
            const PolyFace& face = work.faces[f];
            const size_t c = face.verts.size();
            const size_t ia = cornerOf(face, b0), ib = cornerOf(face, b1);
            if (ia == c || ib == c)
                continue;
            const std::vector<glm::vec2> piecePts = projectFace(work, face, plane);
            // A straight run along the border divides nothing.
            const size_t gap = (ib + c - ia) % c;
            if (interior.empty() && (gap == 1 || gap == c - 1 || onPolygonBorder(probe, piecePts))) {
                along = true;
                break;
            }
            if (insidePolygon(probe, piecePts)) {
                target = f;
                break;
            }
        }
        if (along)
            continue;
        if (target == UINT32_MAX)
            return fail("The shape must stay on the face");

        PolyFace& face = work.faces[target];
        const std::vector<uint32_t> verts = face.verts;
        const size_t c = verts.size();
        const size_t ia = cornerOf(face, b0), ib = cornerOf(face, b1);
        // Both halves keep the face's winding: outline one way round, then back along the run.
        std::vector<uint32_t> firstHalf, secondHalf;
        for (size_t i = ia;; i = (i + 1) % c) {
            firstHalf.push_back(verts[i]);
            if (i == ib)
                break;
        }
        firstHalf.insert(firstHalf.end(), interior.rbegin(), interior.rend());
        for (size_t i = ib;; i = (i + 1) % c) {
            secondHalf.push_back(verts[i]);
            if (i == ia)
                break;
        }
        secondHalf.insert(secondHalf.end(), interior.begin(), interior.end());
        PolyFace other = face;
        face.verts = std::move(firstHalf);
        other.verts = std::move(secondHalf);
        work.faces.push_back(std::move(other));
        pieces.push_back(static_cast<uint32_t>(work.faces.size() - 1));
        divided = true;
    }
    if (!divided)
        return fail("The shape doesn't divide the face");

    uint32_t result = faceIndex;
    if (closed) {
        // Pieces lie wholly inside or outside the shape, so one point of each tells which is which.
        std::vector<std::array<uint32_t, 3>> tris;
        for (uint32_t f : pieces) {
            tris.clear();
            work.triangulate(work.faces[f], tris);
            if (tris.empty())
                continue;
            const std::vector<uint32_t>& verts = work.faces[f].verts;
            const glm::vec2 centroid = (plane(work.positions[verts[tris[0][0]]]) + plane(work.positions[verts[tris[0][1]]]) +
                plane(work.positions[verts[tris[0][2]]])) / 3.0f;
            if (insidePolygon(centroid, q)) {
                result = f;
                break;
            }
        }
    }
    *this = std::move(work);
    return result;
}

size_t PolyMesh::clip(const glm::vec3& normal, float offset)
{
    // Vertices this close count as on the plane, so cuts through corners don't leave slivers.
    constexpr float kOnPlane = 1e-4f;
    std::vector<float> dist(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        const float d = glm::dot(normal, positions[i]) - offset;
        dist[i] = std::abs(d) < kOnPlane ? 0.0f : d;
    }
    // Each crossing edge gets one new vertex, shared by both faces along it.
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> crossings;
    const auto crossing = [&](uint32_t a, uint32_t b) {
        if (b < a)
            std::swap(a, b);
        const auto [it, inserted] = crossings.try_emplace({ a, b }, static_cast<uint32_t>(positions.size()));
        if (inserted) {
            const glm::vec3 p = glm::mix(positions[a], positions[b], dist[a] / (dist[a] - dist[b]));
            positions.push_back(p);
            dist.push_back(0.0f);
        }
        return it->second;
    };

    using Edge = std::pair<uint32_t, uint32_t>;
    const auto collectEdges = [](const std::vector<PolyFace>& list) {
        std::set<Edge> edges;
        for (const PolyFace& face : list) {
            for (size_t i = 0; i < face.verts.size(); ++i)
                edges.insert({ face.verts[i], face.verts[(i + 1) % face.verts.size()] });
        }
        return edges;
    };
    const std::set<Edge> originalEdges = collectEdges(faces);

    std::vector<PolyFace> kept;
    uint32_t capMaterial = kNoPolyMaterial;
    for (PolyFace& face : faces) {
        const bool above = std::any_of(face.verts.begin(), face.verts.end(), [&dist](uint32_t v) { return dist[v] > 0.0f; });
        const bool below = std::any_of(face.verts.begin(), face.verts.end(), [&dist](uint32_t v) { return dist[v] < 0.0f; });
        if (!above) {
            // A face lying in the plane but facing backwards bounds the part being removed.
            if (below || glm::dot(faceNormal(face), normal) > 0.0f)
                kept.push_back(std::move(face));
            continue;
        }
        if (!below)
            continue;
        if (capMaterial == kNoPolyMaterial)
            capMaterial = face.material;

        // Walk the outline keeping the part behind the plane; gapAfter marks where it went in front.
        std::vector<uint32_t> out;
        std::vector<bool> gapAfter;
        bool wrapGap = false;
        const size_t count = face.verts.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t a = face.verts[i], b = face.verts[(i + 1) % count];
            if (dist[a] <= 0.0f) {
                out.push_back(a);
                gapAfter.push_back(false);
            }
            else if (out.empty()) {
                wrapGap = true;
            }
            else {
                gapAfter.back() = true;
            }
            if ((dist[a] < 0.0f && dist[b] > 0.0f) || (dist[a] > 0.0f && dist[b] < 0.0f)) {
                out.push_back(crossing(a, b));
                gapAfter.push_back(false);
            }
        }
        if (wrapGap)
            gapAfter.back() = true;

        // The kept outline falls apart into chains, each running from one gap to the next.
        const size_t m = out.size();
        std::vector<size_t> gaps;
        for (size_t i = 0; i < m; ++i) {
            if (gapAfter[i])
                gaps.push_back(i);
        }
        std::vector<std::vector<uint32_t>> chains(gaps.size());
        for (size_t k = 0; k < gaps.size(); ++k) {
            const size_t last = gaps[(k + 1) % gaps.size()];
            for (size_t i = (gaps[k] + 1) % m;; i = (i + 1) % m) {
                chains[k].push_back(out[i]);
                if (i == last)
                    break;
            }
        }

        // Walk order joins the chains correctly only for convex faces. A concave face (an arch's
        // front) can cross the plane several times: its pieces are joined by pairing the chain
        // ends in order along the cut line, where the outline alternately leaves and re-enters.
        std::vector<size_t> link(chains.size());
        for (size_t k = 0; k < chains.size(); ++k)
            link[k] = (k + 1) % chains.size();
        if (chains.size() > 1) {
            const glm::vec3 along = glm::cross(faceNormal(face), normal);
            struct End {
                float t;
                size_t chain;
                bool isEnd;
            };
            std::vector<End> ends;
            for (size_t k = 0; k < chains.size(); ++k) {
                ends.push_back({ glm::dot(positions[chains[k].front()], along), k, false });
                ends.push_back({ glm::dot(positions[chains[k].back()], along), k, true });
            }
            std::sort(ends.begin(), ends.end(), [](const End& x, const End& y) { return x.t < y.t; });
            std::vector<size_t> paired(chains.size());
            bool valid = true;
            for (size_t i = 0; i + 1 < ends.size() && valid; i += 2) {
                const End& x = ends[i];
                const End& y = ends[i + 1];
                valid = x.isEnd != y.isEnd;
                paired[x.isEnd ? x.chain : y.chain] = x.isEnd ? y.chain : x.chain;
            }
            if (valid)
                link = std::move(paired);
        }

        std::vector<bool> used(chains.size(), false);
        for (size_t k0 = 0; k0 < chains.size(); ++k0) {
            if (used[k0])
                continue;
            PolyFace piece = face; // keeps material and UV params
            piece.verts.clear();
            for (size_t k = k0; !used[k]; k = link[k]) {
                used[k] = true;
                for (uint32_t v : chains[k]) {
                    if (piece.verts.empty() || piece.verts.back() != v)
                        piece.verts.push_back(v);
                }
            }
            while (piece.verts.size() > 1 && piece.verts.front() == piece.verts.back())
                piece.verts.pop_back();
            if (piece.verts.size() >= 3)
                kept.push_back(std::move(piece));
        }
    }
    faces = std::move(kept);

    // Caps close every edge on the plane that the clip opened: new cut edges, and old edges whose
    // neighbour was removed (a cut along existing edges, like stairs at a step height). Edges that
    // were open before stay open.
    const std::set<Edge> keptEdges = collectEdges(faces);
    std::map<uint32_t, uint32_t> capNext; // cap outline; runs against the faces it closes
    for (const auto& [u, w] : keptEdges) {
        const bool wasOpen = originalEdges.contains({ u, w }) && !originalEdges.contains({ w, u });
        if (dist[u] == 0.0f && dist[w] == 0.0f && !keptEdges.contains({ w, u }) && !wasOpen)
            capNext[w] = u;
    }

    size_t caps = 0;
    while (!capNext.empty()) {
        const uint32_t start = capNext.begin()->first;
        PolyFace cap;
        cap.material = capMaterial;
        bool closed = false;
        for (uint32_t v = start;;) {
            const auto it = capNext.find(v);
            if (it == capNext.end())
                break; // open outline (non-manifold input); leave the hole
            cap.verts.push_back(v);
            v = it->second;
            capNext.erase(it);
            if (v == start) {
                closed = true;
                break;
            }
        }
        if (!closed || cap.verts.size() < 3)
            continue;
        if (glm::dot(faceNormal(cap), normal) < 0.0f)
            std::reverse(cap.verts.begin(), cap.verts.end());
        faces.push_back(std::move(cap));
        ++caps;
    }
    removeUnusedVertices();
    return caps;
}

void PolyMesh::mirror(int axis, float pivot)
{
    for (glm::vec3& p : positions)
        p[axis] = 2.0f * pivot - p[axis];
    // A reflection turns every winding inside out.
    for (PolyFace& face : faces)
        std::reverse(face.verts.begin(), face.verts.end());
}

bool PolyMesh::symmetrize(int axis, float pivot, bool fromNegative)
{
    glm::vec3 normal(0.0f);
    normal[axis] = fromNegative ? 1.0f : -1.0f;
    clip(normal, normal[axis] * pivot);

    // Same tolerance as clip(). Faces lying on the plane (the cap clip() just made, or an existing
    // face there) would end up inside once the reflection is joined on.
    constexpr float kOnPlane = 1e-4f;
    const auto onPlane = [&](uint32_t v) { return std::abs(positions[v][axis] - pivot) < kOnPlane; };
    std::erase_if(faces, [&](const PolyFace& face) {
        return std::all_of(face.verts.begin(), face.verts.end(), onPlane);
    });
    removeUnusedVertices();
    if (faces.empty())
        return false;

    // Vertices on the plane are their own reflection, which welds the two halves together.
    const size_t vertexCount = positions.size();
    std::vector<uint32_t> reflected(vertexCount);
    for (size_t v = 0; v < vertexCount; ++v) {
        if (onPlane(static_cast<uint32_t>(v))) {
            positions[v][axis] = pivot;
            reflected[v] = static_cast<uint32_t>(v);
            continue;
        }
        glm::vec3 p = positions[v];
        p[axis] = 2.0f * pivot - p[axis];
        reflected[v] = static_cast<uint32_t>(positions.size());
        positions.push_back(p);
    }
    const size_t faceCount = faces.size();
    faces.reserve(faceCount * 2);
    for (size_t i = 0; i < faceCount; ++i) {
        PolyFace copy = faces[i];
        for (uint32_t& v : copy.verts)
            v = reflected[v];
        std::reverse(copy.verts.begin(), copy.verts.end());
        faces.push_back(std::move(copy));
    }
    return true;
}

namespace {

// Range the face covers in rotated texture space, before scale and offset (see polyMeshToMesh()).
void faceUVRange(const PolyMesh& mesh, const PolyFace& face, glm::vec2& lo, glm::vec2& hi)
{
    glm::vec3 t, b;
    faceBasis(mesh.faceNormal(face), t, b);
    const float r = glm::radians(face.uvRotation);
    const float cr = std::cos(r), sr = std::sin(r);
    lo = glm::vec2(FLT_MAX);
    hi = glm::vec2(-FLT_MAX);
    for (uint32_t v : face.verts) {
        const glm::vec3& p = mesh.positions[v];
        const glm::vec2 uv(glm::dot(p, t), -glm::dot(p, b));
        const glm::vec2 rotated(cr * uv.x - sr * uv.y, sr * uv.x + cr * uv.y);
        lo = glm::min(lo, rotated);
        hi = glm::max(hi, rotated);
    }
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

void PolyMesh::faceUVs(uint32_t faceIndex, float texelSize, std::vector<glm::vec2>& out) const
{
    out.clear();
    const PolyFace& face = faces[faceIndex];
    if (face.verts.size() < 3)
        return;
    const FaceUVMapping mapping(*this, face, texelSize);
    for (uint32_t v : face.verts)
        out.push_back(mapping(positions[v]));
}

bool PolyMesh::wrapFaceUVs(uint32_t from, uint32_t to, float fromTexelSize, float toTexelSize)
{
    const PolyFace& source = faces[from];
    PolyFace& target = faces[to];
    if (source.verts.size() < 3 || target.verts.size() < 3)
        return false;
    // A shared edge: two corners next to each other in both faces.
    const auto adjacentInTarget = [&](uint32_t p, uint32_t q) {
        const size_t n = target.verts.size();
        for (size_t i = 0; i < n; ++i) {
            const uint32_t u = target.verts[i], w = target.verts[(i + 1) % n];
            if ((u == p && w == q) || (u == q && w == p))
                return true;
        }
        return false;
    };
    uint32_t a = UINT32_MAX, b = UINT32_MAX;
    for (size_t i = 0; i < source.verts.size() && a == UINT32_MAX; ++i) {
        const uint32_t p = source.verts[i], q = source.verts[(i + 1) % source.verts.size()];
        if (adjacentInTarget(p, q)) {
            a = p;
            b = q;
        }
    }
    if (a == UINT32_MAX)
        return false;

    const FaceUVMapping sourceMap(*this, source, fromTexelSize);
    const glm::vec2 wantA = sourceMap(positions[a]);
    const glm::vec2 wantB = sourceMap(positions[b]);
    // Turn the target's projected edge to point the same way as the source's in texture space.
    const FaceUVMapping targetMap(*this, target, toTexelSize);
    const glm::vec2 edge = targetMap.project(positions[b]) - targetMap.project(positions[a]);
    const glm::vec2 wanted = (wantB - wantA) * targetMap.scale;
    if (glm::length(edge) < 1e-6f || glm::length(wanted) < 1e-6f)
        return false;
    const float angle = std::atan2(wanted.y, wanted.x) - std::atan2(edge.y, edge.x);
    float degrees = glm::degrees(angle);
    degrees = std::fmod(degrees + 540.0f, 360.0f) - 180.0f;
    target.uvRotation = degrees;
    const FaceUVMapping turned(*this, target, toTexelSize);
    target.uvOffset = wantA - turned.rotate(turned.project(positions[a])) / turned.scale;
    return true;
}

void PolyMesh::fitFaceUVs(uint32_t faceIndex, bool fitU, bool fitV, float texelSize)
{
    PolyFace& face = faces[faceIndex];
    if (face.verts.size() < 3)
        return;
    glm::vec2 lo, hi;
    faceUVRange(*this, face, lo, hi);
    const float texel = std::max(texelSize, 1e-3f);
    for (int i = 0; i < 2; ++i) {
        if (!(i == 0 ? fitU : fitV))
            continue;
        face.uvScale[i] = std::max((hi[i] - lo[i]) / texel, 1e-3f);
        face.uvOffset[i] = -lo[i] / (face.uvScale[i] * texel);
        face.uvOffset[i] -= std::floor(face.uvOffset[i]); // whole tiles make no difference
    }
}

void PolyMesh::alignFaceUVs(uint32_t faceIndex, const glm::vec2& anchor, float texelSize)
{
    PolyFace& face = faces[faceIndex];
    if (face.verts.size() < 3)
        return;
    glm::vec2 lo, hi;
    faceUVRange(*this, face, lo, hi);
    const glm::vec2 scale = glm::max(glm::abs(face.uvScale) * texelSize, glm::vec2(1e-4f));
    for (int i = 0; i < 2; ++i) {
        if (anchor[i] < 0.0f)
            continue;
        // The point `anchor` of the way across the face lands on texture coordinate `anchor`.
        const float point = lo[i] + anchor[i] * (hi[i] - lo[i]);
        face.uvOffset[i] = anchor[i] - point / scale[i];
        face.uvOffset[i] -= std::floor(face.uvOffset[i]);
    }
}

void PolyMesh::hollow(float thickness)
{
    const size_t vertexCount = positions.size();
    const size_t faceCount = faces.size();

    // Each vertex moves so that every face plane through it moves `thickness` inward: least squares
    // over the distinct normals there. The damping keeps it solvable where those normals don't span
    // 3D (flat areas, edges) and barely changes the answer elsewhere.
    std::vector<glm::mat3> normalSums(vertexCount, glm::mat3(1e-3f));
    std::vector<glm::vec3> directionSums(vertexCount, glm::vec3(0.0f));
    std::vector<std::vector<glm::vec3>> seen(vertexCount);
    for (const PolyFace& face : faces) {
        const glm::vec3 n = faceNormal(face);
        for (uint32_t v : face.verts) {
            if (std::any_of(seen[v].begin(), seen[v].end(), [&](const glm::vec3& m) { return glm::dot(m, n) > 0.999f; }))
                continue;
            seen[v].push_back(n);
            normalSums[v] += glm::outerProduct(n, n);
            directionSums[v] += n;
        }
    }
    std::vector<uint32_t> inner(vertexCount, UINT32_MAX);
    positions.reserve(vertexCount * 2);
    for (size_t v = 0; v < vertexCount; ++v) {
        if (seen[v].empty())
            continue;
        inner[v] = static_cast<uint32_t>(positions.size());
        positions.push_back(positions[v] - glm::inverse(normalSums[v]) * directionSums[v] * thickness);
    }

    // Edges used by one face only are open borders; a wall joins them to their inner copy.
    std::set<std::pair<uint32_t, uint32_t>> edges;
    for (const PolyFace& face : faces)
        for (size_t i = 0; i < face.verts.size(); ++i)
            edges.insert({ face.verts[i], face.verts[(i + 1) % face.verts.size()] });

    std::vector<PolyFace> walls;
    faces.reserve(faceCount * 2);
    for (size_t f = 0; f < faceCount; ++f) {
        PolyFace copy = faces[f];
        const size_t count = copy.verts.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t a = copy.verts[i];
            const uint32_t b = copy.verts[(i + 1) % count];
            if (edges.count({ b, a }))
                continue;
            // Runs b -> a against this face's a -> b, and inner a -> inner b against the inner face.
            PolyFace wall = copy;
            wall.verts = { b, a, inner[a], inner[b] };
            walls.push_back(std::move(wall));
        }
        for (uint32_t& v : copy.verts)
            v = inner[v];
        std::reverse(copy.verts.begin(), copy.verts.end());
        faces.push_back(std::move(copy));
    }
    std::move(walls.begin(), walls.end(), std::back_inserter(faces));
}

void PolyMesh::transform(const glm::mat4& matrix)
{
    for (glm::vec3& p : positions)
        p = glm::vec3(matrix * glm::vec4(p, 1.0f));
    if (glm::determinant(glm::mat3(matrix)) < 0.0f)
        for (PolyFace& face : faces)
            std::reverse(face.verts.begin(), face.verts.end());
}

namespace {

// Moves the faces' vertices (or every position) and refits the faces' UVs; see transformKeepingUVs().
void transformRefittingUVs(PolyMesh& mesh, const std::vector<uint32_t>& faceIndices, bool everyPosition,
    const glm::mat4& matrix, const std::vector<float>& slotTexelSizes)
{
    const auto texelSize = [&](const PolyFace& face) {
        return face.material < slotTexelSizes.size() ? slotTexelSizes[face.material] : 1.0f;
    };
    std::vector<std::vector<glm::vec2>> before(faceIndices.size());
    for (size_t i = 0; i < faceIndices.size(); ++i)
        mesh.faceUVs(faceIndices[i], texelSize(mesh.faces[faceIndices[i]]), before[i]);

    std::vector<uint8_t> moves(mesh.positions.size(), everyPosition ? 1 : 0);
    for (uint32_t f : faceIndices)
        for (uint32_t v : mesh.faces[f].verts)
            moves[v] = 1;
    for (size_t v = 0; v < mesh.positions.size(); ++v)
        if (moves[v])
            mesh.positions[v] = glm::vec3(matrix * glm::vec4(mesh.positions[v], 1.0f));
    const bool mirrored = glm::determinant(glm::mat3(matrix)) < 0.0f;

    for (size_t n = 0; n < faceIndices.size(); ++n) {
        PolyFace& face = mesh.faces[faceIndices[n]];
        std::vector<glm::vec2>& uvs = before[n];
        if (mirrored) {
            // Keeps the faces pointing out; the corners' UVs follow them.
            std::reverse(face.verts.begin(), face.verts.end());
            std::reverse(uvs.begin(), uvs.end());
        }
        const size_t count = face.verts.size();
        if (count < 3 || uvs.size() != count)
            continue;
        // The texture is an affine map of the face plane: fit it from three corners spanning the face,
        // in the projection of the new plane (uv = rotate(q) / scale + offset).
        const FaceUVMapping mapping(mesh, face, 1.0f);
        const std::vector<glm::vec3>& positions = mesh.positions;
        std::vector<glm::vec2> q(count);
        for (size_t i = 0; i < count; ++i)
            q[i] = mapping.project(positions[face.verts[i]]);
        size_t i1 = 1, i2 = 2;
        for (size_t i = 1; i < count; ++i)
            if (glm::length(q[i] - q[0]) > glm::length(q[i1] - q[0]))
                i1 = i;
        float bestArea = -1.0f;
        for (size_t i = 1; i < count; ++i) {
            const float area = std::abs(cross2(q[i1] - q[0], q[i] - q[0]));
            if (i != i1 && area > bestArea) {
                bestArea = area;
                i2 = i;
            }
        }
        const glm::mat2 dq(q[i1] - q[0], q[i2] - q[0]);
        if (std::abs(glm::determinant(dq)) < 1e-10f)
            continue;
        const glm::mat2 du(uvs[i1] - uvs[0], uvs[i2] - uvs[0]);
        const glm::mat2 a = du * glm::inverse(dq); // uv = a * q + c
        // Rows of a: (cos r, -sin r) / (scale.x * texel) and (sin r, cos r) / (scale.y * texel).
        const glm::vec2 row0(a[0][0], a[1][0]);
        const glm::vec2 row1(a[0][1], a[1][1]);
        const float texel = std::max(texelSize(face), 1e-4f);
        glm::vec2 scale(1.0f / std::max(glm::length(row0) * texel, 1e-8f), 1.0f / std::max(glm::length(row1) * texel, 1e-8f));
        float rotation = glm::degrees(std::atan2(-row0.y, row0.x));
        // Keep the old numbers when only the offset changed (e.g. a move), so they don't pick up noise.
        const float turn = std::remainder(rotation - face.uvRotation, 360.0f);
        if (std::abs(turn) < 1e-3f)
            rotation = face.uvRotation;
        for (int axis = 0; axis < 2; ++axis)
            if (std::abs(scale[axis] - std::abs(face.uvScale[axis])) < 1e-5f * std::abs(face.uvScale[axis]))
                scale[axis] = std::abs(face.uvScale[axis]);
        face.uvRotation = rotation;
        face.uvScale = scale;
        face.uvOffset = glm::vec2(0.0f);
        const FaceUVMapping fitted(mesh, face, texel);
        // The texture repeats every unit, so only the fraction of the offset matters.
        const glm::vec2 offset = uvs[0] - fitted(positions[face.verts[0]]);
        face.uvOffset = offset - glm::floor(offset);
    }
}

} // namespace

void PolyMesh::transformKeepingUVs(const glm::mat4& matrix, const std::vector<float>& slotTexelSizes)
{
    std::vector<uint32_t> all(faces.size());
    std::iota(all.begin(), all.end(), 0u);
    transformRefittingUVs(*this, all, true, matrix, slotTexelSizes);
}

void PolyMesh::transformFacesKeepingUVs(const std::vector<uint32_t>& faceIndices, const glm::mat4& matrix,
    const std::vector<float>& slotTexelSizes)
{
    transformRefittingUVs(*this, faceIndices, false, matrix, slotTexelSizes);
}

std::vector<uint32_t> PolyMesh::copyFaces(const std::vector<uint32_t>& faceIndices)
{
    std::vector<uint32_t> copyOf(positions.size(), UINT32_MAX);
    std::vector<uint32_t> added;
    for (uint32_t f : faceIndices) {
        PolyFace face = faces[f];
        for (uint32_t& v : face.verts) {
            if (copyOf[v] == UINT32_MAX) {
                copyOf[v] = static_cast<uint32_t>(positions.size());
                const glm::vec3 p = positions[v];
                positions.push_back(p);
            }
            v = copyOf[v];
        }
        added.push_back(static_cast<uint32_t>(faces.size()));
        faces.push_back(std::move(face));
    }
    return added;
}

bool PolyMesh::append(const PolyMesh& other)
{
    const auto sameSlot = [](const PolyMaterial& a, const PolyMaterial& b) {
        if (!a.materialPath.empty() || !b.materialPath.empty())
            return a.materialPath == b.materialPath;
        return a.color == b.color && a.roughness == b.roughness && a.metallic == b.metallic &&
            a.texturePath == b.texturePath;
    };
    std::vector<uint32_t> slotMap(other.materials.size());
    std::vector<PolyMaterial> added;
    for (size_t s = 0; s < other.materials.size(); ++s) {
        const PolyMaterial& material = other.materials[s];
        const auto here = std::find_if(materials.begin(), materials.end(),
            [&](const PolyMaterial& m) { return sameSlot(m, material); });
        if (here != materials.end()) {
            slotMap[s] = static_cast<uint32_t>(here - materials.begin());
            continue;
        }
        const size_t pending = static_cast<size_t>(std::find_if(added.begin(), added.end(),
            [&](const PolyMaterial& m) { return sameSlot(m, material); }) - added.begin());
        if (pending == added.size())
            added.push_back(material);
        slotMap[s] = static_cast<uint32_t>(materials.size() + pending);
    }
    if (materials.size() + added.size() > kMaxPolyMaterialSlots)
        return false;
    materials.insert(materials.end(), added.begin(), added.end());

    const uint32_t base = static_cast<uint32_t>(positions.size());
    positions.insert(positions.end(), other.positions.begin(), other.positions.end());
    for (PolyFace face : other.faces) {
        for (uint32_t& v : face.verts)
            v += base;
        // Past the end renders white; it must not land on one of this mesh's slots.
        face.material = face.material < slotMap.size() ? slotMap[face.material] : kNoPolyMaterial;
        faces.push_back(std::move(face));
    }
    return true;
}

size_t PolyMesh::partIds(std::vector<uint32_t>& partOfFace) const
{
    const PolyMesh& mesh = *this;
    std::vector<uint32_t> parent(mesh.positions.size());
    for (uint32_t i = 0; i < parent.size(); ++i)
        parent[i] = i;
    const auto find = [&](uint32_t v) {
        while (parent[v] != v)
            v = parent[v] = parent[parent[v]];
        return v;
    };
    for (const PolyFace& face : mesh.faces)
        for (size_t i = 1; i < face.verts.size(); ++i)
            parent[find(face.verts[i])] = find(face.verts[0]);

    std::vector<uint32_t> partOfRoot(mesh.positions.size(), UINT32_MAX);
    uint32_t parts = 0;
    partOfFace.assign(mesh.faces.size(), UINT32_MAX);
    for (size_t f = 0; f < mesh.faces.size(); ++f) {
        const PolyFace& face = mesh.faces[f];
        if (face.verts.empty())
            continue;
        uint32_t& part = partOfRoot[find(face.verts[0])];
        if (part == UINT32_MAX)
            part = parts++;
        partOfFace[f] = part;
    }
    return parts;
}

size_t PolyMesh::partCount() const
{
    std::vector<uint32_t> partOfFace;
    return partIds(partOfFace);
}

std::vector<PolyMesh> PolyMesh::splitParts() const
{
    std::vector<uint32_t> partOfFace;
    const size_t partCount = partIds(partOfFace);
    std::vector<PolyMesh> parts(partCount);
    std::vector<std::vector<uint32_t>> vertexMap(partCount);
    std::vector<std::vector<uint32_t>> slotMap(partCount);
    for (size_t p = 0; p < partCount; ++p) {
        parts[p].gridSize = gridSize;
        vertexMap[p].assign(positions.size(), UINT32_MAX);
        slotMap[p].assign(materials.size(), UINT32_MAX);
    }
    for (size_t f = 0; f < faces.size(); ++f) {
        if (partOfFace[f] == UINT32_MAX)
            continue;
        const uint32_t p = partOfFace[f];
        PolyMesh& part = parts[p];
        PolyFace face = faces[f];
        for (uint32_t& v : face.verts) {
            uint32_t& mapped = vertexMap[p][v];
            if (mapped == UINT32_MAX) {
                mapped = static_cast<uint32_t>(part.positions.size());
                part.positions.push_back(positions[v]);
            }
            v = mapped;
        }
        if (face.material < materials.size()) {
            uint32_t& slot = slotMap[p][face.material];
            if (slot == UINT32_MAX) {
                slot = static_cast<uint32_t>(part.materials.size());
                part.materials.push_back(materials[face.material]);
            }
            face.material = slot;
        }
        part.faces.push_back(std::move(face));
    }
    return parts;
}

namespace {

uint64_t edgeKey(uint32_t from, uint32_t to)
{
    return (static_cast<uint64_t>(from) << 32) | to;
}

// Directed edge -> face running along it.
std::unordered_map<uint64_t, uint32_t> faceEdges(const std::vector<PolyFace>& faces)
{
    std::unordered_map<uint64_t, uint32_t> edges;
    for (uint32_t f = 0; f < faces.size(); ++f) {
        const size_t count = faces[f].verts.size();
        for (size_t i = 0; i < count; ++i)
            edges[edgeKey(faces[f].verts[i], faces[f].verts[(i + 1) % count])] = f;
    }
    return edges;
}

// Material indices past the slots all mean plain white.
bool sameLook(const PolyFace& a, const PolyFace& b, size_t slots)
{
    const uint32_t materialA = a.material < slots ? a.material : kNoPolyMaterial;
    const uint32_t materialB = b.material < slots ? b.material : kNoPolyMaterial;
    return materialA == materialB && a.uvScale == b.uvScale && a.uvOffset == b.uvOffset &&
        a.uvRotation == b.uvRotation;
}

// The outline around two faces sharing edges (run opposite ways by them), when it is a single loop: the
// shared edges are one unbroken chain and no corner is passed twice.
bool joinOutlines(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, std::vector<uint32_t>& out)
{
    std::unordered_set<uint64_t> edgesA, edgesB;
    for (size_t i = 0; i < a.size(); ++i)
        edgesA.insert(edgeKey(a[i], a[(i + 1) % a.size()]));
    for (size_t i = 0; i < b.size(); ++i)
        edgesB.insert(edgeKey(b[i], b[(i + 1) % b.size()]));
    std::unordered_map<uint32_t, uint32_t> next;
    const auto keep = [&](const std::vector<uint32_t>& verts, const std::unordered_set<uint64_t>& other) {
        for (size_t i = 0; i < verts.size(); ++i) {
            const uint32_t u = verts[i], w = verts[(i + 1) % verts.size()];
            if (!other.contains(edgeKey(w, u)) && !next.emplace(u, w).second)
                return false;
        }
        return true;
    };
    if (!keep(a, edgesB) || !keep(b, edgesA) || next.size() < 3)
        return false;
    out.clear();
    const uint32_t start = next.begin()->first;
    uint32_t v = start;
    do {
        out.push_back(v);
        const auto it = next.find(v);
        if (it == next.end() || out.size() > next.size())
            return false;
        v = it->second;
    } while (v != start);
    return out.size() == next.size();
}

} // namespace

bool PolyMesh::isClosed() const
{
    std::unordered_map<uint64_t, int> edges;
    for (const PolyFace& face : faces) {
        const size_t count = face.verts.size();
        for (size_t i = 0; i < count; ++i)
            ++edges[edgeKey(face.verts[i], face.verts[(i + 1) % count])];
    }
    if (edges.empty())
        return false;
    for (const auto& [key, uses] : edges) {
        const auto reverse = edges.find((key << 32) | (key >> 32));
        if (uses != 1 || reverse == edges.end() || reverse->second != 1)
            return false;
    }
    return true;
}

std::vector<uint32_t> PolyMesh::coplanarRegion(uint32_t faceIndex) const
{
    if (faceIndex >= faces.size() || faces[faceIndex].verts.size() < 3)
        return {};
    const glm::vec3 n = faceNormal(faces[faceIndex]);
    const float offset = glm::dot(n, positions[faces[faceIndex].verts[0]]);
    const auto edges = faceEdges(faces);
    std::vector<bool> seen(faces.size(), false);
    std::vector<uint32_t> region{ faceIndex };
    seen[faceIndex] = true;
    for (size_t next = 0; next < region.size(); ++next) {
        const PolyFace& face = faces[region[next]];
        const size_t count = face.verts.size();
        for (size_t i = 0; i < count; ++i) {
            const auto it = edges.find(edgeKey(face.verts[(i + 1) % count], face.verts[i]));
            if (it == edges.end() || seen[it->second])
                continue;
            const PolyFace& other = faces[it->second];
            const bool flat = glm::dot(faceNormal(other), n) > 0.999f &&
                std::all_of(other.verts.begin(), other.verts.end(),
                    [&](uint32_t v) { return std::abs(glm::dot(n, positions[v]) - offset) < 1e-3f; });
            if (!flat)
                continue;
            seen[it->second] = true;
            region.push_back(it->second);
        }
    }
    return region;
}

std::vector<uint32_t> PolyMesh::mergeCoplanarFaces(const std::vector<int64_t>& joinKeys,
    std::vector<uint32_t>* vertexRemap)
{
    const uint32_t faceCount = static_cast<uint32_t>(faces.size());
    std::vector<int64_t> keys = joinKeys;
    keys.resize(faceCount, -1);
    std::vector<glm::vec3> normals(faceCount);
    std::vector<float> offsets(faceCount);
    for (uint32_t f = 0; f < faceCount; ++f) {
        normals[f] = faceNormal(faces[f]);
        offsets[f] = faces[f].verts.empty() ? 0.0f : glm::dot(normals[f], positions[faces[f].verts[0]]);
    }
    const auto coplanar = [&](uint32_t f, uint32_t g) {
        if (glm::dot(normals[f], normals[g]) < 0.9999f)
            return false;
        return std::all_of(faces[g].verts.begin(), faces[g].verts.end(),
            [&](uint32_t v) { return std::abs(glm::dot(normals[f], positions[v]) - offsets[f]) < 1e-4f; });
    };

    auto edges = faceEdges(faces);
    std::vector<uint32_t> joinedInto(faceCount);
    std::iota(joinedInto.begin(), joinedInto.end(), 0u);
    std::vector<bool> grown(faceCount, false);
    std::vector<uint32_t> work(joinedInto.rbegin(), joinedInto.rend());
    std::vector<uint32_t> outline;
    const auto setEdges = [&](uint32_t f, bool add) {
        const std::vector<uint32_t>& verts = faces[f].verts;
        for (size_t i = 0; i < verts.size(); ++i) {
            const uint64_t key = edgeKey(verts[i], verts[(i + 1) % verts.size()]);
            if (add)
                edges[key] = f;
            else if (const auto it = edges.find(key); it != edges.end() && it->second == f)
                edges.erase(it);
        }
    };
    while (!work.empty()) {
        const uint32_t f = work.back();
        work.pop_back();
        if (joinedInto[f] != f)
            continue;
        const std::vector<uint32_t>& verts = faces[f].verts;
        for (size_t i = 0; i < verts.size(); ++i) {
            const auto it = edges.find(edgeKey(verts[(i + 1) % verts.size()], verts[i]));
            if (it == edges.end())
                continue;
            const uint32_t g = it->second;
            if (g == f || joinedInto[g] != g || (keys[f] >= 0 && keys[g] >= 0 && keys[f] != keys[g]) ||
                !sameLook(faces[f], faces[g], materials.size()) || !coplanar(f, g) || !joinOutlines(verts, faces[g].verts, outline))
                continue;
            setEdges(f, false);
            setEdges(g, false);
            faces[f].verts = outline;
            setEdges(f, true);
            joinedInto[g] = f;
            if (keys[f] < 0)
                keys[f] = keys[g];
            grown[f] = true;
            work.push_back(f); // look again from the new outline
            break;
        }
    }

    // Corners left in the middle of straight edges go, where every face using them runs straight through.
    std::unordered_map<uint32_t, std::vector<uint32_t>> users;
    for (uint32_t f = 0; f < faceCount; ++f)
        if (joinedInto[f] == f)
            for (uint32_t v : faces[f].verts)
                users[v].push_back(f);
    const auto straightIn = [&](uint32_t f, uint32_t v) {
        const std::vector<uint32_t>& verts = faces[f].verts;
        const size_t count = verts.size();
        const size_t i = static_cast<size_t>(std::find(verts.begin(), verts.end(), v) - verts.begin());
        if (i == count || count <= 3)
            return false;
        const glm::vec3 in = positions[v] - positions[verts[(i + count - 1) % count]];
        const glm::vec3 out = positions[verts[(i + 1) % count]] - positions[v];
        return glm::dot(in, out) > 0.0f &&
            glm::length(glm::cross(in, out)) <= 1e-4f * glm::length(in) * glm::length(out);
    };
    std::unordered_set<uint32_t> removable;
    for (uint32_t f = 0; f < faceCount; ++f) {
        if (!grown[f] || joinedInto[f] != f)
            continue;
        for (uint32_t v : faces[f].verts) {
            const std::vector<uint32_t>& list = users[v];
            if (std::all_of(list.begin(), list.end(), [&](uint32_t g) { return straightIn(g, v); }))
                removable.insert(v);
        }
    }
    if (!removable.empty())
        for (uint32_t f = 0; f < faceCount; ++f)
            if (joinedInto[f] == f && faces[f].verts.size() > 3) {
                std::vector<uint32_t> kept;
                for (uint32_t v : faces[f].verts)
                    if (!removable.contains(v))
                        kept.push_back(v);
                if (kept.size() >= 3)
                    faces[f].verts = std::move(kept);
            }

    std::vector<uint32_t> newIndex(faceCount, UINT32_MAX);
    std::vector<PolyFace> kept;
    for (uint32_t f = 0; f < faceCount; ++f) {
        if (joinedInto[f] != f)
            continue;
        newIndex[f] = static_cast<uint32_t>(kept.size());
        kept.push_back(std::move(faces[f]));
    }
    std::vector<uint32_t> remap(faceCount);
    for (uint32_t f = 0; f < faceCount; ++f) {
        uint32_t root = f;
        while (joinedInto[root] != root)
            root = joinedInto[root];
        remap[f] = newIndex[root];
    }
    faces = std::move(kept);
    std::vector<uint32_t> vertices = removeUnusedVertices();
    if (vertexRemap)
        *vertexRemap = std::move(vertices);
    return remap;
}

namespace {

PolyMesh makeBox(const glm::vec3& size)
{
    const glm::vec3 h(size.x * 0.5f, size.y, size.z * 0.5f);
    PolyMesh m;
    for (int i = 0; i < 8; ++i)
        m.positions.push_back({ (i & 1) ? h.x : -h.x, (i & 2) ? h.y : 0.0f, (i & 4) ? h.z : -h.z });
    // Bit 0 = +X, bit 1 = top, bit 2 = +Z.
    m.addFace({ 2, 3, 7, 6 }, { 0, 1, 0 });
    m.addFace({ 0, 1, 5, 4 }, { 0, -1, 0 });
    m.addFace({ 1, 3, 7, 5 }, { 1, 0, 0 });
    m.addFace({ 0, 2, 6, 4 }, { -1, 0, 0 });
    m.addFace({ 4, 5, 7, 6 }, { 0, 0, 1 });
    m.addFace({ 0, 1, 3, 2 }, { 0, 0, -1 });
    return m;
}

PolyMesh makePlane(const glm::vec3& size, int segments)
{
    const int n = std::clamp(segments, 1, 64);
    PolyMesh m;
    for (int z = 0; z <= n; ++z)
        for (int x = 0; x <= n; ++x)
            m.positions.push_back({ (float(x) / n - 0.5f) * size.x, 0.0f, (float(z) / n - 0.5f) * size.z });
    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            const uint32_t a = uint32_t(z * (n + 1) + x);
            m.addFace({ a, a + 1, a + uint32_t(n) + 2, a + uint32_t(n) + 1 }, { 0, 1, 0 });
        }
    }
    return m;
}

PolyMesh makeCylinder(const glm::vec3& size, int segments)
{
    const uint32_t n = uint32_t(std::clamp(segments, 3, 128));
    PolyMesh m;
    for (int ring = 0; ring < 2; ++ring) {
        for (uint32_t i = 0; i < n; ++i) {
            const float a = glm::two_pi<float>() * float(i) / float(n);
            m.positions.push_back({ std::cos(a) * size.x * 0.5f, ring ? size.y : 0.0f, std::sin(a) * size.z * 0.5f });
        }
    }
    std::vector<uint32_t> bottom(n), top(n);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t j = (i + 1) % n;
        const float mid = glm::two_pi<float>() * (float(i) + 0.5f) / float(n);
        m.addFace({ i, j, n + j, n + i }, { std::cos(mid), 0.0f, std::sin(mid) });
        bottom[i] = i;
        top[i] = n + i;
    }
    m.addFace(std::move(top), { 0, 1, 0 });
    m.addFace(std::move(bottom), { 0, -1, 0 });
    return m;
}

PolyMesh makeWedge(const glm::vec3& size)
{
    const float hx = size.x * 0.5f, hz = size.z * 0.5f, y = size.y;
    PolyMesh m;
    // Bottom rectangle, then the top edge along the back (-Z); the slope faces +Z.
    m.positions = { { -hx, 0, -hz }, { hx, 0, -hz }, { hx, 0, hz }, { -hx, 0, hz }, { -hx, y, -hz }, { hx, y, -hz } };
    m.addFace({ 0, 1, 2, 3 }, { 0, -1, 0 });
    m.addFace({ 0, 1, 5, 4 }, { 0, 0, -1 });
    m.addFace({ 3, 2, 5, 4 }, { 0, size.z, size.y });
    m.addFace({ 0, 3, 4 }, { -1, 0, 0 });
    m.addFace({ 1, 2, 5 }, { 1, 0, 0 });
    return m;
}

// Extrudes a closed 2D profile symmetrically. alongX: profile is (z, y), extruded along X;
// otherwise the profile is (x, y), extruded along Z.
PolyMesh extrudeProfile(const std::vector<glm::vec2>& profile, float depth, bool alongX)
{
    const auto to3 = [alongX](const glm::vec2& p, float w) {
        return alongX ? glm::vec3(w, p.y, p.x) : glm::vec3(p.x, p.y, w);
    };
    const uint32_t n = static_cast<uint32_t>(profile.size());
    PolyMesh m;
    for (const glm::vec2& p : profile)
        m.positions.push_back(to3(p, -depth * 0.5f));
    for (const glm::vec2& p : profile)
        m.positions.push_back(to3(p, depth * 0.5f));

    float area = 0.0f;
    for (uint32_t i = 0; i < n; ++i)
        area += cross2(profile[i], profile[(i + 1) % n]);
    const float winding = area >= 0.0f ? 1.0f : -1.0f;

    std::vector<uint32_t> front(n), back(n);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t j = (i + 1) % n;
        const glm::vec2 e = profile[j] - profile[i];
        const glm::vec2 outward = winding * glm::vec2(e.y, -e.x);
        m.addFace({ i, j, n + j, n + i }, to3(outward, 0.0f));
        front[i] = i;
        back[i] = n + i;
    }
    const glm::vec3 axis = to3(glm::vec2(0.0f), 1.0f);
    m.addFace(std::move(front), -axis);
    m.addFace(std::move(back), axis);
    return m;
}

PolyMesh makeStairs(const glm::vec3& size, int steps)
{
    const int n = std::clamp(steps, 1, 64);
    const float hz = size.z * 0.5f, rise = size.y / n, run = size.z / n;
    // Side profile in (z, y): the stairs climb towards -Z.
    std::vector<glm::vec2> profile{ { hz, 0.0f } };
    for (int k = 1; k <= n; ++k) {
        profile.push_back({ hz - (k - 1) * run, k * rise });
        profile.push_back({ hz - k * run, k * rise });
    }
    profile.push_back({ -hz, 0.0f });
    return extrudeProfile(profile, size.x, true);
}

PolyMesh makeArch(const glm::vec3& size, int segments, float thickness)
{
    const int n = std::clamp(segments, 2, 128);
    const float rx = size.x * 0.5f, ry = size.y;
    const float t = std::clamp(thickness, 0.01f, std::min(rx, ry) - 0.01f);
    // Outer half-ellipse from right to left, then the inner one back.
    std::vector<glm::vec2> profile;
    for (int i = 0; i <= n; ++i) {
        const float a = glm::pi<float>() * float(i) / float(n);
        profile.push_back({ std::cos(a) * rx, std::sin(a) * ry });
    }
    for (int i = n; i >= 0; --i) {
        const float a = glm::pi<float>() * float(i) / float(n);
        profile.push_back({ std::cos(a) * (rx - t), std::sin(a) * (ry - t) });
    }
    return extrudeProfile(profile, size.z, false);
}

} // namespace

PolyMesh makePolyShape(const PolyShapeParams& params)
{
    const glm::vec3 size = glm::max(params.size, glm::vec3(0.01f));
    switch (params.shape) {
    case PolyShape::Plane: return makePlane(size, params.subdivisions);
    case PolyShape::Cylinder: return makeCylinder(size, params.segments);
    case PolyShape::Wedge: return makeWedge(size);
    case PolyShape::Stairs: return makeStairs(size, params.steps);
    case PolyShape::Arch: return makeArch(size, params.segments, params.thickness);
    case PolyShape::Box:
    default: return makeBox(size);
    }
}

PolyShapeParams snapPolyShapeSize(PolyShapeParams params, float grid)
{
    params.size = glm::max(glm::round(params.size / grid), glm::vec3(1.0f)) * grid;
    return params;
}

PolyMesh makePolyShapeOnGrid(const PolyShapeParams& params, float grid, bool snap)
{
    if (!snap) {
        PolyMesh mesh = makePolyShape(params);
        mesh.gridSize = grid;
        return mesh;
    }
    const PolyShapeParams snapped = snapPolyShapeSize(params, grid);
    PolyMesh mesh = makePolyShape(snapped);
    mesh.gridSize = grid;
    // Centered on X/Z, an odd number of cells puts the sides between grid lines; shift them half a cell.
    const glm::vec2 half(snapped.size.x * 0.5f, snapped.size.z * 0.5f);
    const glm::vec2 shift = glm::round(half / grid) * grid - half;
    if (shift != glm::vec2(0.0f))
        for (glm::vec3& p : mesh.positions)
            p += glm::vec3(shift.x, 0.0f, shift.y);
    return mesh;
}

Mesh polyMeshToMesh(const PolyMesh& poly, MaterialLibrary* library)
{
    Mesh mesh;
    // Slot values with links resolved; a missing shared material falls back to the inline values.
    std::vector<MaterialAsset> slots(poly.materials.size());
    for (size_t i = 0; i < poly.materials.size(); ++i) {
        const PolyMaterial& source = poly.materials[i];
        const MaterialAsset* shared = library ? library->find(source.materialPath) : nullptr;
        if (shared) {
            slots[i] = *shared;
            continue;
        }
        slots[i].color = source.color;
        slots[i].roughness = source.roughness;
        slots[i].metallic = source.metallic;
        slots[i].baseColorTexture = source.texturePath;
    }
    const auto toMaterial = [&](uint32_t index) {
        Material material;
        material.metallicFactor = 0.0f;
        material.roughnessFactor = 0.8f;
        return index < slots.size() ? toRenderMaterial(slots[index], mesh) : material;
    };

    std::vector<uint32_t> usedMaterials;
    for (const PolyFace& face : poly.faces)
        usedMaterials.push_back(face.material);
    std::sort(usedMaterials.begin(), usedMaterials.end());
    usedMaterials.erase(std::unique(usedMaterials.begin(), usedMaterials.end()), usedMaterials.end());

    std::vector<std::array<uint32_t, 3>> tris;
    for (uint32_t mat : usedMaterials) {
        SubmeshInfo sub{};
        sub.indexOffset = static_cast<uint32_t>(mesh.indices.size());
        sub.vertexOffset = 0;

        for (const PolyFace& face : poly.faces) {
            if (face.material != mat || face.verts.size() < 3)
                continue;
            const glm::vec3 n = poly.faceNormal(face);
            // A shared material's texel size sets how much of the world one texture repeat covers.
            const float texel = mat < slots.size() ? slots[mat].texelSize : 1.0f;
            const FaceUVMapping mapping(poly, face, texel);
            // The tangent must follow +U after rotation; the bitangent sign (w = 1) matches generateCube().
            const glm::vec3 tangent = glm::normalize(mapping.cr * mapping.t + mapping.sr * mapping.b);

            const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
            for (uint32_t v : face.verts) {
                const glm::vec3& p = poly.positions[v];
                Vertex vert{};
                vert.position = p;
                vert.normal = n;
                vert.texCoord = mapping(p);
                vert.tangent = glm::vec4(tangent, 1.0f);
                mesh.vertices.push_back(vert);
                sub.boundsMin = glm::min(sub.boundsMin, p);
                sub.boundsMax = glm::max(sub.boundsMax, p);
            }

            tris.clear();
            triangulateFace(poly, face, n, tris);
            for (const auto& tri : tris)
                for (uint32_t k : tri)
                    mesh.indices.push_back(base + k);
        }

        sub.indexCount = static_cast<uint32_t>(mesh.indices.size()) - sub.indexOffset;
        if (sub.indexCount > 0) {
            // Only materials in use pull in their texture.
            sub.material = toMaterial(mat);
            mesh.submeshes.push_back(sub);
        }
    }
    return mesh;
}

namespace {
// Caps keep a corrupt file from triggering huge allocations.
constexpr uint32_t kMaxPolyPositions = 1u << 22;
constexpr uint32_t kMaxPolyFaces = 1u << 22;
constexpr uint32_t kMaxFaceVerts = 1u << 16;
constexpr uint32_t kMaxPolyMaterials = 256;
constexpr uint32_t kMaxTexturePath = 4096;

template <typename T>
void writePod(std::ostream& out, const T& value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool readPod(std::istream& in, T& value)
{
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}
}

void writePolyMesh(std::ostream& out, const PolyMesh& mesh)
{
    writePod(out, static_cast<uint32_t>(mesh.positions.size()));
    out.write(reinterpret_cast<const char*>(mesh.positions.data()), mesh.positions.size() * sizeof(glm::vec3));
    writePod(out, static_cast<uint32_t>(mesh.faces.size()));
    for (const PolyFace& face : mesh.faces) {
        writePod(out, face.material);
        writePod(out, face.uvScale);
        writePod(out, face.uvOffset);
        writePod(out, face.uvRotation);
        writePod(out, static_cast<uint32_t>(face.verts.size()));
        out.write(reinterpret_cast<const char*>(face.verts.data()), face.verts.size() * sizeof(uint32_t));
    }
    writePod(out, static_cast<uint32_t>(mesh.materials.size()));
    for (const PolyMaterial& material : mesh.materials) {
        writePod(out, material.color);
        writePod(out, material.roughness);
        writePod(out, material.metallic);
        writePod(out, static_cast<uint32_t>(material.texturePath.size()));
        out.write(material.texturePath.data(), material.texturePath.size());
        writePod(out, static_cast<uint32_t>(material.materialPath.size()));
        out.write(material.materialPath.data(), material.materialPath.size());
    }
    writePod(out, mesh.gridSize);
}

bool readPolyMesh(std::istream& in, PolyMesh& mesh, int format)
{
    mesh = {};
    uint32_t positionCount = 0;
    if (!readPod(in, positionCount) || positionCount > kMaxPolyPositions)
        return false;
    mesh.positions.resize(positionCount);
    if (!in.read(reinterpret_cast<char*>(mesh.positions.data()), positionCount * sizeof(glm::vec3)))
        return false;

    uint32_t faceCount = 0;
    if (!readPod(in, faceCount) || faceCount > kMaxPolyFaces)
        return false;
    mesh.faces.resize(faceCount);
    for (PolyFace& face : mesh.faces) {
        uint32_t vertCount = 0;
        if (!readPod(in, face.material) || !readPod(in, face.uvScale) || !readPod(in, face.uvOffset) ||
            !readPod(in, face.uvRotation) || !readPod(in, vertCount))
            return false;
        if ((face.material >= kMaxPolyMaterials && face.material != kNoPolyMaterial) || vertCount > kMaxFaceVerts)
            return false;
        face.verts.resize(vertCount);
        if (!in.read(reinterpret_cast<char*>(face.verts.data()), vertCount * sizeof(uint32_t)))
            return false;
        if (std::any_of(face.verts.begin(), face.verts.end(), [&](uint32_t v) { return v >= positionCount; }))
            return false;
    }
    if (format < 1)
        return true;
    const auto readString = [&](std::string& out) {
        uint32_t length = 0;
        if (!readPod(in, length) || length > kMaxTexturePath)
            return false;
        out.resize(length);
        return static_cast<bool>(in.read(out.data(), length));
    };

    uint32_t materialCount = 0;
    if (!readPod(in, materialCount) || materialCount > kMaxPolyMaterials)
        return false;
    mesh.materials.resize(materialCount);
    for (PolyMaterial& material : mesh.materials) {
        if (!readPod(in, material.color) || !readPod(in, material.roughness) || !readPod(in, material.metallic) ||
            !readString(material.texturePath))
            return false;
        if (format >= 3 && !readString(material.materialPath))
            return false;
    }
    if (format >= 2) {
        if (!readPod(in, mesh.gridSize))
            return false;
        // Also catches NaN.
        if (!(mesh.gridSize >= kMinPolyGridSize && mesh.gridSize <= kMaxPolyGridSize))
            mesh.gridSize = 1.0f;
    }
    return true;
}
