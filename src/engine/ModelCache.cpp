#include "ModelCache.h"
#include "Vertex.h"
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

constexpr uint32_t kCacheMagic = 0x564B4D44; // "VKMD"
constexpr uint32_t kCacheVersion = 3;

// Written verbatim (including the implicit padding after version), so its layout is part of the file format.
struct ModelCacheHeader {
    uint32_t magic = kCacheMagic;
    uint32_t version = kCacheVersion;
    uint64_t vertexCount;
    uint64_t indexCount;
    uint64_t submeshCount;
    uint64_t textureCount;
};
static_assert(sizeof(ModelCacheHeader) == 40);

// Textures are always decoded to 4 channels regardless of the stored channel count.
size_t rgbaSize(const TextureData& tex) {
    return tex.width * tex.height * 4;
}

} // namespace

namespace ModelCache {

bool isValid(const std::string& sourcePath, const std::string& cachePath) {
    namespace fs = std::filesystem;
    if (!fs::exists(cachePath)) return false;
    if (!fs::exists(sourcePath)) return false;
    return fs::last_write_time(cachePath) > fs::last_write_time(sourcePath);
}

bool save(const std::string& cachePath, const Mesh& mesh) {
    std::ofstream file(cachePath, std::ios::binary);
    if (!file.is_open()) return false;

    ModelCacheHeader header{};
    header.vertexCount = mesh.vertices.size();
    header.indexCount = mesh.indices.size();
    header.submeshCount = mesh.submeshes.size();
    header.textureCount = mesh.textureData.size();

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    if (header.vertexCount > 0)
        file.write(reinterpret_cast<const char*>(mesh.vertices.data()), header.vertexCount * sizeof(Vertex));
    if (header.indexCount > 0)
        file.write(reinterpret_cast<const char*>(mesh.indices.data()), header.indexCount * sizeof(uint32_t));
    if (header.submeshCount > 0)
        file.write(reinterpret_cast<const char*>(mesh.submeshes.data()), header.submeshCount * sizeof(SubmeshInfo));

    for (const auto& tex : mesh.textureData) {
        file.write(reinterpret_cast<const char*>(&tex.width), sizeof(int));
        file.write(reinterpret_cast<const char*>(&tex.height), sizeof(int));
        file.write(reinterpret_cast<const char*>(&tex.channels), sizeof(int));

        bool linear = tex.isLinear;
        file.write(reinterpret_cast<const char*>(&linear), sizeof(bool));

        size_t dataSize = rgbaSize(tex);
        if (tex.pixels) {
            file.write(reinterpret_cast<const char*>(tex.pixels), dataSize);
        }
        else {
            std::vector<unsigned char> white(dataSize, 255);
            file.write(reinterpret_cast<const char*>(white.data()), dataSize);
        }
    }

    return true;
}

bool load(const std::string& cachePath, Mesh& outMesh) {
    std::ifstream file(cachePath, std::ios::binary);
    if (!file.is_open()) return false;

    ModelCacheHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.magic != kCacheMagic || header.version != kCacheVersion) return false;

    outMesh.vertices.resize(header.vertexCount);
    outMesh.indices.resize(header.indexCount);
    outMesh.submeshes.resize(header.submeshCount);
    outMesh.textureData.resize(header.textureCount);

    if (header.vertexCount > 0)
        file.read(reinterpret_cast<char*>(outMesh.vertices.data()), header.vertexCount * sizeof(Vertex));
    if (header.indexCount > 0)
        file.read(reinterpret_cast<char*>(outMesh.indices.data()), header.indexCount * sizeof(uint32_t));
    if (header.submeshCount > 0)
        file.read(reinterpret_cast<char*>(outMesh.submeshes.data()), header.submeshCount * sizeof(SubmeshInfo));

    for (TextureData& tex : outMesh.textureData) {
        file.read(reinterpret_cast<char*>(&tex.width), sizeof(int));
        file.read(reinterpret_cast<char*>(&tex.height), sizeof(int));
        file.read(reinterpret_cast<char*>(&tex.channels), sizeof(int));

        bool linear;
        file.read(reinterpret_cast<char*>(&linear), sizeof(bool));
        tex.isLinear = linear;

        size_t dataSize = rgbaSize(tex);
        // malloc (not stbi) so TextureData::free knows to release it with ::free via fromCache.
        tex.pixels = static_cast<unsigned char*>(malloc(dataSize));
        tex.fromCache = true;
        file.read(reinterpret_cast<char*>(tex.pixels), dataSize);
    }

    return true;
}

} // namespace ModelCache
