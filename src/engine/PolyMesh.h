#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "ModelTypes.h"

// One polygon of a PolyMesh. Vertices wind counter-clockwise seen from outside; faces may be concave
// but must be (close to) planar.
struct PolyFace {
    std::vector<uint32_t> verts;
    uint32_t material = 0;
    // UVs are projected onto the face plane in object-space units, then rotated, scaled and offset.
    glm::vec2 uvScale{ 1.0f };
    glm::vec2 uvOffset{ 0.0f };
    float uvRotation = 0.0f; // degrees
};

// Surface of level geometry; faces refer to it by index.
struct PolyMaterial {
    glm::vec4 color{ 1.0f };
    float roughness = 0.8f;
    float metallic = 0.0f;
    std::string texturePath; // base color; empty for none, kCheckerTexturePath for the built-in checker
};

// PolyFace::material value for "no material": the face renders plain white.
inline constexpr uint32_t kNoPolyMaterial = UINT32_MAX;

// Generated in code, so UVs can be checked without any texture files.
inline constexpr const char* kCheckerTexturePath = "builtin:checker";

// Editable polygon mesh used by the level tool. Faces share positions, so moving a vertex moves
// every face touching it; toMesh() splits them again so each face gets flat normals.
struct PolyMesh {
    std::vector<glm::vec3> positions;
    std::vector<PolyFace> faces;
    // Faces whose material index is past the end (e.g. kNoPolyMaterial) render plain white.
    std::vector<PolyMaterial> materials;

    glm::vec3 faceNormal(const PolyFace& face) const;
    glm::vec3 faceCenter(const PolyFace& face) const;
    // Adds a face, flipping its winding when its normal points away from expectedNormal.
    void addFace(std::vector<uint32_t> verts, const glm::vec3& expectedNormal);
    // Triangles covering the face, as positions within face.verts (handles concave faces).
    void triangulate(const PolyFace& face, std::vector<std::array<uint32_t, 3>>& out) const;
    // Moves the face's vertices along its normal; faces sharing them stretch along.
    void moveFace(uint32_t faceIndex, float distance);
    // Copies the face's vertices out along its normal and joins old and new outlines with one quad
    // per edge. The face keeps its index and now caps the extrusion; side faces take its material.
    void extrudeFace(uint32_t faceIndex, float distance);
    // Reverses the winding, so the face points the other way.
    void flipFace(uint32_t faceIndex);
    // Removes the face and any vertices left unused.
    void deleteFace(uint32_t faceIndex);
    // Removes every face touching one of the vertices, then the vertices left unused.
    void deleteVertices(const std::vector<uint32_t>& verts);
    // Collapses the vertices into one at their average position. Faces that shrink below three
    // vertices are removed. Returns the merged vertex's new index, or UINT32_MAX if no face uses it.
    uint32_t mergeVertices(const std::vector<uint32_t>& verts);
    // Drops positions no face refers to. Returns old -> new index, UINT32_MAX for removed ones.
    std::vector<uint32_t> removeUnusedVertices();
    // True when some face has a and b as neighbouring corners.
    bool isEdge(uint32_t a, uint32_t b) const;
    // Adds a vertex halfway along edge a-b, inserted into every face using that edge.
    // Returns its index, or UINT32_MAX if a-b is not an edge.
    uint32_t splitEdge(uint32_t a, uint32_t b);
    // A face holding a and b as non-neighbouring corners, where the line between them stays inside
    // the face. UINT32_MAX if there is none.
    uint32_t faceToConnect(uint32_t a, uint32_t b) const;
    // Cuts that face in two along a-b; both halves keep its material and UVs.
    // Returns the new half's index, or UINT32_MAX if no face can be cut.
    uint32_t connectVertices(uint32_t a, uint32_t b);
    // Cuts away everything in front of the plane dot(normal, p) == offset and closes each opening
    // with a cap face. Caps are appended last; returns how many there are.
    size_t clip(const glm::vec3& normal, float offset);
    // Reflects the mesh across the plane where coordinate `axis` (0-2) equals pivot.
    void mirror(int axis, float pivot);
    // Keeps the part above pivot along the axis (below if fromNegative) and replaces the rest with its
    // reflection, welded along the plane. Returns false, with the mesh emptied, if nothing was kept.
    bool symmetrize(int axis, float pivot, bool fromNegative);
    // Scales and offsets the face's texture so it covers the face exactly once along U and/or V.
    void fitFaceUVs(uint32_t faceIndex, bool fitU, bool fitV);
    // Offsets the texture so the face's edge lines up with a texture edge, keeping scale and rotation.
    // anchor per axis: 0 = left/top, 0.5 = center, 1 = right/bottom; negative leaves that axis alone.
    void alignFaceUVs(uint32_t faceIndex, const glm::vec2& anchor);
    // Turns the shape into a shell `thickness` thick: adds an inner copy of the surface, moved inward
    // and facing in, and closes open borders (so a plane becomes a slab). Thickness larger than the
    // shape leaves the inner surface poking through the outer one.
    void hollow(float thickness);
    // Applies the transform to every position; faces are reversed when it mirrors.
    void transform(const glm::mat4& matrix);
};

// Constructive solid geometry (PolyMeshCsg.cpp). Both meshes must be closed and in the same space.
// The result keeps `target`'s materials and appends those of `cutter`'s faces that end up in it.
// Faces come out cut into convex pieces; positions closer than a millimetre are welded.
PolyMesh polyMeshSubtract(const PolyMesh& target, const PolyMesh& cutter);

enum class PolyShape {
    Box,
    Plane,
    Cylinder,
    Wedge,
    Stairs,
    Arch,
};

inline constexpr const char* kPolyShapeNames[] = { "Box", "Plane", "Cylinder", "Wedge", "Stairs", "Arch" };

struct PolyShapeParams {
    PolyShape shape = PolyShape::Box;
    glm::vec3 size{ 2.0f };
    int segments = 16;     // plane grid cells per side, cylinder sides, arch segments
    int steps = 6;         // stairs
    float thickness = 0.5f; // arch
};

// Every shape is centered on X/Z and rests on y = 0.
PolyMesh makePolyShape(const PolyShapeParams& params);

// Triangulates the mesh with flat per-face normals, one submesh per material. Textures are listed in
// mesh.textureData by path only (one entry per distinct path, in first-use order); the caller decodes them.
Mesh polyMeshToMesh(const PolyMesh& mesh);

// Models built from a PolyMesh use "level:#<n>" as their source path; scene files store the mesh itself.
inline constexpr const char* kLevelModelPathPrefix = "level:#";
inline bool isLevelModelPath(const std::string& path) { return path.rfind(kLevelModelPathPrefix, 0) == 0; }

// Binary form used inside scene files; materials are stored from scene version 4 on.
// readPolyMesh() rejects out-of-range indices and returns false on truncated or corrupt data.
void writePolyMesh(std::ostream& out, const PolyMesh& mesh);
bool readPolyMesh(std::istream& in, PolyMesh& mesh, bool withMaterials);
