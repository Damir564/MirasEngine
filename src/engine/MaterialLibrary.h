#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "ModelTypes.h"

// A surface shared between objects, stored as JSON in its own .mat file. Level shape faces and
// imported model material slots refer to it by path.
struct MaterialAsset {
    std::string path; // as stored in scenes, e.g. "materials/brick.mat"
    std::string name; // file stem
    glm::vec4 color{ 1.0f };
    float roughness = 0.8f;
    float metallic = 0.0f;
    std::string baseColorTexture; // empty for none, kCheckerTexturePath for the built-in checker
    std::string normalTexture;
    // World units one texture repeat covers on level shapes, so it looks the same size on every face.
    float texelSize = 1.0f;
};

// Index of the texture with this path and color space in mesh.textureData, appended undecoded (path
// only) when new; -1 for an empty path.
int addPathTexture(Mesh& mesh, const std::string& path, bool linear);
// Opaque render material for the asset; its textures are added to mesh with addPathTexture().
Material toRenderMaterial(const MaterialAsset& asset, Mesh& mesh);

// The .mat files under one folder (relative to the working directory), loaded on demand and kept
// until they change. Main thread only.
class MaterialLibrary {
public:
    explicit MaterialLibrary(std::string root = "materials");

    const std::string& root() const { return m_root; }
    // Reloads every .mat file in the folder.
    void scan();
    // Sorted by path.
    const std::map<std::string, MaterialAsset>& materials() const { return m_materials; }
    // Loads the file on first use; null when it is missing or unreadable. Pointers stay valid until the
    // material is removed or the library rescanned.
    const MaterialAsset* find(const std::string& path);
    // Writes the material to its path and updates the library.
    bool save(const MaterialAsset& material);
    // Path a new material named `name` would get, made unique with a number suffix.
    std::string uniquePath(const std::string& name) const;
    // Creates a .mat file from `initial` (path and name are replaced) and returns its path.
    std::optional<std::string> create(const std::string& name, const MaterialAsset& initial = {});
    bool remove(const std::string& path);
    // Renames the file; returns the new path. Objects using the old path keep it, so callers relink them.
    std::optional<std::string> rename(const std::string& path, const std::string& newName);
    // Increases whenever a material is created, saved, renamed or removed.
    uint64_t revision() const { return m_revision; }

    static std::optional<MaterialAsset> loadFile(const std::string& path);
    static bool saveFile(const MaterialAsset& material);

private:
    std::string m_root;
    std::map<std::string, MaterialAsset> m_materials;
    // Paths find() failed on, so it does not hit the disk for them every frame; cleared by scan().
    std::vector<std::string> m_missing;
    uint64_t m_revision = 1;
};
