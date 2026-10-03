#include "PolyMesh.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <map>
#include <utility>

// BSP-tree CSG after Evan Wallace's csg.js. Trees are stored as node arrays and walked with explicit
// stacks: a convex shape builds a chain one node per face, which gets deep after a few cuts.

namespace {

constexpr float kPlaneEpsilon = 1e-4f; // closer than this to a plane counts as on it
constexpr float kWeldDistance = 1e-3f;
constexpr size_t kMaxMaterials = 255; // the editor's limit, below the scene reader's

struct CsgPlane {
    glm::vec3 normal;
    float w;
    void flip()
    {
        normal = -normal;
        w = -w;
    }
};

struct CsgPolygon {
    std::vector<glm::vec3> verts; // convex
    CsgPlane plane;
    uint32_t sourceFace;
    bool fromCutter;
    void flip()
    {
        std::reverse(verts.begin(), verts.end());
        plane.flip();
    }
};

using PolygonList = std::vector<CsgPolygon>;

enum : int { Coplanar = 0, Front = 1, Back = 2, Spanning = 3 };

// Sorts the polygon into the lists by which side of the plane it lies on, cutting it if it spans
// the plane. Coplanar polygons go by whether they face the same way as the plane.
void splitPolygon(const CsgPlane& plane, const CsgPolygon& polygon, PolygonList& coplanarFront,
    PolygonList& coplanarBack, PolygonList& front, PolygonList& back)
{
    const size_t count = polygon.verts.size();
    std::vector<int> types(count);
    int polygonType = Coplanar;
    for (size_t i = 0; i < count; ++i) {
        const float t = glm::dot(plane.normal, polygon.verts[i]) - plane.w;
        types[i] = t < -kPlaneEpsilon ? Back : t > kPlaneEpsilon ? Front : Coplanar;
        polygonType |= types[i];
    }
    switch (polygonType) {
    case Coplanar:
        (glm::dot(plane.normal, polygon.plane.normal) > 0.0f ? coplanarFront : coplanarBack).push_back(polygon);
        break;
    case Front:
        front.push_back(polygon);
        break;
    case Back:
        back.push_back(polygon);
        break;
    default: {
        CsgPolygon f{ {}, polygon.plane, polygon.sourceFace, polygon.fromCutter };
        CsgPolygon b = f;
        for (size_t i = 0; i < count; ++i) {
            const size_t j = (i + 1) % count;
            const glm::vec3& vi = polygon.verts[i];
            const glm::vec3& vj = polygon.verts[j];
            if (types[i] != Back)
                f.verts.push_back(vi);
            if (types[i] != Front)
                b.verts.push_back(vi);
            if ((types[i] | types[j]) == Spanning) {
                const float t = (plane.w - glm::dot(plane.normal, vi)) / glm::dot(plane.normal, vj - vi);
                const glm::vec3 v = glm::mix(vi, vj, t);
                f.verts.push_back(v);
                b.verts.push_back(v);
            }
        }
        if (f.verts.size() >= 3)
            front.push_back(std::move(f));
        if (b.verts.size() >= 3)
            back.push_back(std::move(b));
        break;
    }
    }
}

// Solid space: behind every node's plane along some path is inside.
struct CsgTree {
    struct Node {
        CsgPlane plane;
        int front = -1;
        int back = -1;
        PolygonList polygons;
    };
    std::vector<Node> nodes; // nodes[0] is the root; every node has a plane

    void build(PolygonList list)
    {
        if (list.empty())
            return;
        if (nodes.empty())
            nodes.push_back({ list[0].plane });
        std::vector<std::pair<int, PolygonList>> work;
        work.emplace_back(0, std::move(list));
        while (!work.empty()) {
            auto [index, polygons] = std::move(work.back());
            work.pop_back();
            PolygonList front, back;
            const CsgPlane plane = nodes[index].plane;
            for (const CsgPolygon& polygon : polygons)
                splitPolygon(plane, polygon, nodes[index].polygons, nodes[index].polygons, front, back);
            // Adding children may reallocate nodes, so nothing holds a reference across this.
            for (auto [side, part] : { std::pair{ &Node::front, &front }, std::pair{ &Node::back, &back } }) {
                if (part->empty())
                    continue;
                if (nodes[index].*side < 0) {
                    nodes[index].*side = static_cast<int>(nodes.size());
                    nodes.push_back({ (*part)[0].plane });
                }
                work.emplace_back(nodes[index].*side, std::move(*part));
            }
        }
    }

    // Removes the parts of the polygons that are inside this solid.
    PolygonList clipPolygons(PolygonList list) const
    {
        if (nodes.empty())
            return list;
        PolygonList result;
        std::vector<std::pair<int, PolygonList>> work;
        work.emplace_back(0, std::move(list));
        while (!work.empty()) {
            auto [index, polygons] = std::move(work.back());
            work.pop_back();
            const Node& node = nodes[index];
            PolygonList front, back;
            for (const CsgPolygon& polygon : polygons)
                splitPolygon(node.plane, polygon, front, back, front, back);
            if (node.front >= 0)
                work.emplace_back(node.front, std::move(front));
            else
                std::move(front.begin(), front.end(), std::back_inserter(result));
            if (node.back >= 0)
                work.emplace_back(node.back, std::move(back));
            // With no back child, the back part is inside and dropped.
        }
        return result;
    }

    // Removes the parts of this tree's polygons that are inside the other solid.
    void clipTo(const CsgTree& other)
    {
        for (Node& node : nodes)
            node.polygons = other.clipPolygons(std::move(node.polygons));
    }

    // Swaps inside and outside.
    void invert()
    {
        for (Node& node : nodes) {
            for (CsgPolygon& polygon : node.polygons)
                polygon.flip();
            node.plane.flip();
            std::swap(node.front, node.back);
        }
    }

    PolygonList allPolygons() const
    {
        PolygonList result;
        for (const Node& node : nodes)
            result.insert(result.end(), node.polygons.begin(), node.polygons.end());
        return result;
    }
};

bool isConvex(const PolyMesh& mesh, const PolyFace& face, const glm::vec3& n)
{
    const size_t count = face.verts.size();
    for (size_t i = 0; i < count; ++i) {
        const glm::vec3& a = mesh.positions[face.verts[(i + count - 1) % count]];
        const glm::vec3& b = mesh.positions[face.verts[i]];
        const glm::vec3& c = mesh.positions[face.verts[(i + 1) % count]];
        if (glm::dot(glm::cross(b - a, c - b), n) < -1e-6f)
            return false;
    }
    return true;
}

// The BSP split assumes convex polygons, so concave faces go in as their triangles.
PolygonList toPolygons(const PolyMesh& mesh, bool fromCutter)
{
    PolygonList result;
    std::vector<std::array<uint32_t, 3>> tris;
    for (uint32_t f = 0; f < mesh.faces.size(); ++f) {
        const PolyFace& face = mesh.faces[f];
        if (face.verts.size() < 3)
            continue;
        const glm::vec3 n = mesh.faceNormal(face);
        const CsgPlane plane{ n, glm::dot(n, mesh.faceCenter(face)) };
        if (isConvex(mesh, face, n)) {
            CsgPolygon polygon{ {}, plane, f, fromCutter };
            for (uint32_t v : face.verts)
                polygon.verts.push_back(mesh.positions[v]);
            result.push_back(std::move(polygon));
            continue;
        }
        tris.clear();
        mesh.triangulate(face, tris);
        for (const auto& tri : tris) {
            CsgPolygon polygon{ {}, plane, f, fromCutter };
            for (uint32_t k : tri)
                polygon.verts.push_back(mesh.positions[face.verts[k]]);
            result.push_back(std::move(polygon));
        }
    }
    return result;
}

bool sameMaterial(const PolyMaterial& a, const PolyMaterial& b)
{
    return a.color == b.color && a.roughness == b.roughness && a.metallic == b.metallic &&
        a.texturePath == b.texturePath;
}

// Merges positions closer than kWeldDistance, using a grid of that cell size.
class Welder {
public:
    explicit Welder(std::vector<glm::vec3>& positions) : m_positions(positions) {}

    uint32_t add(const glm::vec3& p)
    {
        const glm::ivec3 cell(glm::floor(p / kWeldDistance));
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    const auto it = m_cells.find({ cell.x + dx, cell.y + dy, cell.z + dz });
                    if (it == m_cells.end())
                        continue;
                    for (uint32_t index : it->second)
                        if (glm::distance(m_positions[index], p) < kWeldDistance)
                            return index;
                }
        const uint32_t index = static_cast<uint32_t>(m_positions.size());
        m_positions.push_back(p);
        m_cells[{ cell.x, cell.y, cell.z }].push_back(index);
        return index;
    }

private:
    std::vector<glm::vec3>& m_positions;
    std::map<std::array<int, 3>, std::vector<uint32_t>> m_cells;
};

// Splitting leaves vertices in the middle of a neighbour's edge (T-junctions), which show as cracks
// and keep the faces from sharing vertices. Inserts every such vertex into the edge.
void fixTJunctions(PolyMesh& mesh)
{
    for (PolyFace& face : mesh.faces) {
        std::vector<uint32_t> verts;
        const size_t count = face.verts.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t a = face.verts[i];
            const uint32_t b = face.verts[(i + 1) % count];
            verts.push_back(a);
            const glm::vec3 pa = mesh.positions[a];
            const glm::vec3 edge = mesh.positions[b] - pa;
            const float length2 = glm::dot(edge, edge);
            if (length2 < kWeldDistance * kWeldDistance)
                continue;
            const glm::vec3 lo = glm::min(pa, mesh.positions[b]) - kWeldDistance;
            const glm::vec3 hi = glm::max(pa, mesh.positions[b]) + kWeldDistance;
            std::vector<std::pair<float, uint32_t>> inside;
            for (uint32_t v = 0; v < mesh.positions.size(); ++v) {
                const glm::vec3& p = mesh.positions[v];
                if (v == a || v == b || glm::any(glm::lessThan(p, lo)) || glm::any(glm::greaterThan(p, hi)))
                    continue;
                const float t = glm::dot(p - pa, edge) / length2;
                if (t > 0.0f && t < 1.0f && glm::distance(pa + edge * t, p) < kWeldDistance)
                    inside.emplace_back(t, v);
            }
            std::sort(inside.begin(), inside.end());
            for (const auto& [t, v] : inside)
                verts.push_back(v);
        }
        face.verts = std::move(verts);
    }
}

} // namespace

PolyMesh polyMeshSubtract(const PolyMesh& target, const PolyMesh& cutter)
{
    CsgTree a, b;
    a.build(toPolygons(target, false));
    b.build(toPolygons(cutter, true));
    a.invert();
    a.clipTo(b);
    b.clipTo(a);
    b.invert();
    b.clipTo(a);
    b.invert();
    a.build(b.allPolygons());
    a.invert();

    PolyMesh result;
    result.materials = target.materials;
    std::vector<uint32_t> cutterMaterials(cutter.materials.size(), UINT32_MAX);
    const auto materialFor = [&](const CsgPolygon& polygon) {
        const uint32_t material = (polygon.fromCutter ? cutter : target).faces[polygon.sourceFace].material;
        if (!polygon.fromCutter || material >= cutter.materials.size())
            return polygon.fromCutter ? kNoPolyMaterial : material;
        uint32_t& mapped = cutterMaterials[material];
        if (mapped == UINT32_MAX) {
            const PolyMaterial& source = cutter.materials[material];
            const auto it = std::find_if(result.materials.begin(), result.materials.end(),
                [&](const PolyMaterial& m) { return sameMaterial(m, source); });
            if (it != result.materials.end()) {
                mapped = static_cast<uint32_t>(it - result.materials.begin());
            }
            else if (result.materials.size() < kMaxMaterials) {
                mapped = static_cast<uint32_t>(result.materials.size());
                result.materials.push_back(source);
            }
            else {
                mapped = kNoPolyMaterial;
            }
        }
        return mapped;
    };

    Welder welder(result.positions);
    for (const CsgPolygon& polygon : a.allPolygons()) {
        const PolyFace& source = (polygon.fromCutter ? cutter : target).faces[polygon.sourceFace];
        PolyFace face;
        face.material = materialFor(polygon);
        face.uvScale = source.uvScale;
        face.uvOffset = source.uvOffset;
        face.uvRotation = source.uvRotation;
        for (const glm::vec3& p : polygon.verts) {
            const uint32_t v = welder.add(p);
            if (face.verts.empty() || face.verts.back() != v)
                face.verts.push_back(v);
        }
        while (face.verts.size() > 1 && face.verts.front() == face.verts.back())
            face.verts.pop_back();
        if (face.verts.size() < 3)
            continue;
        // Slivers thinner than the weld distance can survive welding with almost no area.
        glm::vec3 area(0.0f);
        for (size_t i = 0; i < face.verts.size(); ++i)
            area += glm::cross(result.positions[face.verts[i]], result.positions[face.verts[(i + 1) % face.verts.size()]]);
        if (glm::length(area) * 0.5f < 1e-7f)
            continue;
        result.faces.push_back(std::move(face));
    }
    fixTJunctions(result);
    result.removeUnusedVertices();
    return result;
}
