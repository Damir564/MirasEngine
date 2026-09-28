#pragma once

#include <string>
#include "ModelTypes.h"
#include "Vertex.h"

// Paths with this prefix name procedurally generated models instead of files.
inline constexpr const char* kBuiltinCubePath = "builtin:cube";
bool isBuiltinModelPath(const std::string& path);

// "ifcdirect:<file.ifc>" loads the IFC in-process with web-ifc instead of IfcConvert, cached (with
// its metadata) in "<file.ifc>.direct.cache".
// The prefix is part of the model path so scenes remember which importer was used.
inline constexpr const char* kIfcDirectPrefix = "ifcdirect:";
bool isIfcDirectPath(const std::string& path);
// The file on disk behind a model path (strips "ifcdirect:"; builtin paths are returned as is).
std::string modelSourceFile(const std::string& path);

// Loads a model from disk (glTF/GLB via fastgltf, IFC via IfcConvert, everything else via assimp),
// going through the .cache file next to the source when it is up to date.
// Builtin paths return generated geometry and never touch the disk or the cache.
// Throws std::runtime_error when the source cannot be imported.
Mesh loadModelSmart(const std::string& path);
