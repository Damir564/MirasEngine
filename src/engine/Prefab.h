#pragma once
#include <optional>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "PolyMesh.h"

// A prefab is a level object saved to its own .prefab file: name, geometry and materials, but no
// transform. Instances of one prefab in a scene share a single model, so editing it changes them all.
struct Prefab {
    std::string name;
    PolyMesh mesh;
};

// A prefab of scene objects: any models (by file path), level shapes (their geometry, or the shape prefab
// they are linked to) and game entities, with transforms relative to the prefab's origin. Placing one adds
// copies of the objects as a group. Same .prefab extension; isObjectPrefab() tells the kinds apart.
struct ObjectPrefab {
    struct Object {
        std::string name;
        std::string modelPath;          // file model; empty for level geometry
        std::string modelName;
        std::optional<PolyMesh> mesh;   // level geometry not linked to a shape prefab
        std::string prefabPath;         // level geometry linked to a shape prefab
        glm::vec3 position{ 0.0f };
        glm::vec3 rotation{ 0.0f };
        glm::vec3 scale{ 1.0f };
        glm::vec3 color{ 1.0f };
        bool visible = true;
        bool locked = false;
        std::string entity;
        std::string entityParams;
    };
    std::string name;
    std::vector<Object> objects;
};

inline constexpr const char* kPrefabExtension = ".prefab";

bool savePrefab(const std::string& path, const Prefab& prefab);
// Empty when the file is missing, not a prefab, or corrupt.
std::optional<Prefab> loadPrefab(const std::string& path);
bool saveObjectPrefab(const std::string& path, const ObjectPrefab& prefab);
std::optional<ObjectPrefab> loadObjectPrefab(const std::string& path);
// The file holds an ObjectPrefab (not a single shape).
bool isObjectPrefab(const std::string& path);
