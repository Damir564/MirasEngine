#pragma once

#include <string>
#include "Animation.h"
#include "ModelTypes.h"
#include "Vertex.h"

// Paths with this prefix name procedurally generated models instead of files.
inline constexpr const char* kBuiltinCubePath = "builtin:cube";
bool isBuiltinModelPath(const std::string& path);

// Loads a glTF/GLB model from disk via fastgltf (other formats are rejected), going through the
// .cache file next to the source when it is up to date.
// Builtin paths return generated geometry and never touch the disk or the cache.
// Throws std::runtime_error when the source cannot be imported.
Mesh loadModelSmart(const std::string& path);

// Loads a glTF/GLB file with its node hierarchy and animation clips (no cache). Throws like loadModelSmart().
AnimatedModelData loadAnimatedModel(const std::string& path);
