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
#include "PolyMeshInternal.h"

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
