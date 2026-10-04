#include "Prefab.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include "Log.h"

namespace {
constexpr char kPrefabMagic[4] = { 'P', 'F', 'A', 'B' };
// Version 1 files have no grid size, version 2 files no shared material links.
constexpr uint32_t kPrefabVersion = kPolyMeshFormat;
constexpr uint32_t kMaxPrefabNameLength = 4096;
}

bool savePrefab(const std::string& path, const Prefab& prefab)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("[PREFAB] Failed to open file for writing: " << path << "\n");
        return false;
    }
    const uint32_t nameLength = static_cast<uint32_t>(prefab.name.size());
    file.write(kPrefabMagic, sizeof(kPrefabMagic));
    file.write(reinterpret_cast<const char*>(&kPrefabVersion), sizeof(kPrefabVersion));
    file.write(reinterpret_cast<const char*>(&nameLength), sizeof(nameLength));
    file.write(prefab.name.data(), nameLength);
    writePolyMesh(file, prefab.mesh);
    if (!file) {
        LOG_ERROR("[PREFAB] Failed to write " << path << "\n");
        return false;
    }
    LOG_INFO("[PREFAB] Saved '" << prefab.name << "' to " << path << "\n");
    return true;
}

std::optional<Prefab> loadPrefab(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("[PREFAB] Failed to open file for reading: " << path << "\n");
        return std::nullopt;
    }
    char magic[4] = {};
    uint32_t version = 0;
    uint32_t nameLength = 0;
    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
    if (!file || std::memcmp(magic, kPrefabMagic, sizeof(magic)) != 0 || version < 1 || version > kPrefabVersion ||
        nameLength > kMaxPrefabNameLength) {
        LOG_ERROR("[PREFAB] Not a supported prefab file: " << path << "\n");
        return std::nullopt;
    }
    Prefab prefab;
    prefab.name.resize(nameLength);
    file.read(prefab.name.data(), nameLength);
    // Prefab versions match the PolyMesh formats they hold.
    if (!file || !readPolyMesh(file, prefab.mesh, static_cast<int>(version))) {
        LOG_ERROR("[PREFAB] Corrupt prefab file: " << path << "\n");
        return std::nullopt;
    }
    return prefab;
}
