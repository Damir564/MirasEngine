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
