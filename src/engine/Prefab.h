#pragma once
#include <optional>
#include <string>
#include "PolyMesh.h"

// A prefab is a level object saved to its own .prefab file: name, geometry and materials, but no
// transform. Instances of one prefab in a scene share a single model, so editing it changes them all.
struct Prefab {
    std::string name;
    PolyMesh mesh;
};

inline constexpr const char* kPrefabExtension = ".prefab";

bool savePrefab(const std::string& path, const Prefab& prefab);
// Empty when the file is missing, not a prefab, or corrupt.
std::optional<Prefab> loadPrefab(const std::string& path);
