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

// Object prefabs: magic, version, name, object count, then per object its strings (name, model path,
// model name, prefab path, entity, entity params), uint8 hasMesh (+ writePolyMesh()), position, rotation,
// scale and color as 3 floats each, uint8 visible, uint8 locked. Meshes are in the version's PolyMesh format.
constexpr char kObjectPrefabMagic[4] = { 'P', 'O', 'B', 'J' };
constexpr uint32_t kObjectPrefabVersion = 1;
static_assert(kPolyMeshFormat == 3, "object prefab version 1 stores PolyMesh format 3: bump the version and map it");
constexpr uint32_t kMaxObjectPrefabObjects = 100000;

void writeText(std::ostream& out, const std::string& text)
{
    const uint32_t length = static_cast<uint32_t>(text.size());
    out.write(reinterpret_cast<const char*>(&length), sizeof(length));
    out.write(text.data(), length);
}

bool readText(std::istream& in, std::string& text)
{
    uint32_t length = 0;
    in.read(reinterpret_cast<char*>(&length), sizeof(length));
    if (!in || length > kMaxPrefabNameLength)
        return false;
    text.resize(length);
    in.read(text.data(), length);
    return static_cast<bool>(in);
}

void writeVec3(std::ostream& out, const glm::vec3& v)
{
    out.write(reinterpret_cast<const char*>(&v.x), sizeof(float) * 3);
}

bool readVec3(std::istream& in, glm::vec3& v)
{
    in.read(reinterpret_cast<char*>(&v.x), sizeof(float) * 3);
    return static_cast<bool>(in);
}

bool readMagic(std::istream& in, const char (&magic)[4])
{
    char found[4] = {};
    in.read(found, sizeof(found));
    return in && std::memcmp(found, magic, sizeof(found)) == 0;
}
}

bool isObjectPrefab(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    return file.is_open() && readMagic(file, kObjectPrefabMagic);
}

bool saveObjectPrefab(const std::string& path, const ObjectPrefab& prefab)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("[PREFAB] Failed to open file for writing: " << path << "\n");
        return false;
    }
    file.write(kObjectPrefabMagic, sizeof(kObjectPrefabMagic));
    file.write(reinterpret_cast<const char*>(&kObjectPrefabVersion), sizeof(kObjectPrefabVersion));
    writeText(file, prefab.name);
    const uint32_t count = static_cast<uint32_t>(prefab.objects.size());
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const ObjectPrefab::Object& object : prefab.objects) {
        for (const std::string* text : { &object.name, &object.modelPath, &object.modelName, &object.prefabPath,
                 &object.entity, &object.entityParams })
            writeText(file, *text);
        const uint8_t hasMesh = object.mesh ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&hasMesh), sizeof(hasMesh));
        if (object.mesh)
            writePolyMesh(file, *object.mesh);
        for (const glm::vec3* v : { &object.position, &object.rotation, &object.scale, &object.color })
            writeVec3(file, *v);
        const uint8_t flags[2] = { uint8_t(object.visible ? 1 : 0), uint8_t(object.locked ? 1 : 0) };
        file.write(reinterpret_cast<const char*>(flags), sizeof(flags));
    }
    if (!file) {
        LOG_ERROR("[PREFAB] Failed to write " << path << "\n");
        return false;
    }
    LOG_INFO("[PREFAB] Saved '" << prefab.name << "' (" << count << " objects) to " << path << "\n");
    return true;
}

std::optional<ObjectPrefab> loadObjectPrefab(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("[PREFAB] Failed to open file for reading: " << path << "\n");
        return std::nullopt;
    }
    uint32_t version = 0;
    ObjectPrefab prefab;
    uint32_t count = 0;
    const bool header = readMagic(file, kObjectPrefabMagic) &&
        file.read(reinterpret_cast<char*>(&version), sizeof(version)) && version >= 1 &&
        version <= kObjectPrefabVersion && readText(file, prefab.name) &&
        file.read(reinterpret_cast<char*>(&count), sizeof(count)) && count <= kMaxObjectPrefabObjects;
    if (!header) {
        LOG_ERROR("[PREFAB] Not a supported object prefab: " << path << "\n");
        return std::nullopt;
    }
    prefab.objects.resize(count);
    for (ObjectPrefab::Object& object : prefab.objects) {
        bool ok = true;
        for (std::string* text : { &object.name, &object.modelPath, &object.modelName, &object.prefabPath,
                 &object.entity, &object.entityParams })
            ok = ok && readText(file, *text);
        uint8_t hasMesh = 0;
        ok = ok && file.read(reinterpret_cast<char*>(&hasMesh), sizeof(hasMesh));
        if (ok && hasMesh)
            ok = readPolyMesh(file, object.mesh.emplace(), kPolyMeshFormat);
        for (glm::vec3* v : { &object.position, &object.rotation, &object.scale, &object.color })
            ok = ok && readVec3(file, *v);
        uint8_t flags[2] = {};
        ok = ok && file.read(reinterpret_cast<char*>(flags), sizeof(flags));
        if (!ok) {
            LOG_ERROR("[PREFAB] Corrupt object prefab: " << path << "\n");
            return std::nullopt;
        }
        object.visible = flags[0] != 0;
        object.locked = flags[1] != 0;
    }
    return prefab;
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
