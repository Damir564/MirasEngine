#pragma once

#include <string>
#include "ModelTypes.h"

// Binary cache of an imported Mesh with textures stored already decoded (RGBA8), so reloading a
// model costs a disk read instead of a full import + image decode. IFC metadata is not cached.
namespace ModelCache {

// True when the cache exists and is newer than its source file.
bool isValid(const std::string& sourcePath, const std::string& cachePath);
bool save(const std::string& cachePath, const Mesh& mesh);
bool load(const std::string& cachePath, Mesh& outMesh);

} // namespace ModelCache
