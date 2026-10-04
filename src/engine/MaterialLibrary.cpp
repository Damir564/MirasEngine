#include "MaterialLibrary.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string_view>
#include "Log.h"

namespace {

constexpr const char* kMaterialExtension = ".mat";

template <typename T>
void readField(const nlohmann::json& json, const char* key, T& out)
{
    const auto it = json.find(key);
    if (it == json.end())
        return;
    try {
        out = it->get<T>();
    }
    catch (const nlohmann::json::exception&) {
        LOG_ERROR("[MATERIAL] Ignoring invalid value for '" << key << "'\n");
    }
}

// Same form scenes use for stored paths: relative to the working directory when possible, '/' separated.
std::string normalizePath(const std::filesystem::path& path)
{
    std::error_code ec;
    std::filesystem::path relative = std::filesystem::relative(path, std::filesystem::current_path(), ec);
    if (ec || relative.empty())
        relative = path;
    return relative.lexically_normal().generic_string();
}

} // namespace

int addPathTexture(Mesh& mesh, const std::string& path, bool linear)
{
    if (path.empty())
        return -1;
    const auto it = std::find_if(mesh.textureData.begin(), mesh.textureData.end(),
        [&](const TextureData& tex) { return tex.path == path && tex.isLinear == linear; });
    if (it != mesh.textureData.end())
        return static_cast<int>(it - mesh.textureData.begin());
    TextureData tex;
    tex.path = path;
    tex.isLinear = linear;
    mesh.textureData.push_back(std::move(tex));
    return static_cast<int>(mesh.textureData.size() - 1);
}

Material toRenderMaterial(const MaterialAsset& asset, Mesh& mesh)
{
    Material material;
    material.baseColorFactor = asset.color;
    material.metallicFactor = asset.metallic;
    material.roughnessFactor = asset.roughness;
    material.baseColorTextureIndex = addPathTexture(mesh, asset.baseColorTexture, false);
    material.normalTextureIndex = addPathTexture(mesh, asset.normalTexture, true);
    return material;
}

MaterialLibrary::MaterialLibrary(std::string root)
    : m_root(std::move(root))
{
}

void MaterialLibrary::scan()
{
    m_materials.clear();
    m_missing.clear();
    ++m_revision;
    std::error_code ec;
    if (!std::filesystem::is_directory(m_root, ec))
        return;
    for (const auto& entry : std::filesystem::directory_iterator(m_root, ec)) {
        if (!entry.is_regular_file() || entry.path().extension() != kMaterialExtension)
            continue;
        if (auto material = loadFile(normalizePath(entry.path())))
            m_materials[material->path] = std::move(*material);
    }
}

const MaterialAsset* MaterialLibrary::find(const std::string& path)
{
    if (path.empty())
        return nullptr;
    if (const auto it = m_materials.find(path); it != m_materials.end())
        return &it->second;
    if (std::find(m_missing.begin(), m_missing.end(), path) != m_missing.end())
        return nullptr;
    auto material = loadFile(path);
    if (!material) {
        m_missing.push_back(path);
        return nullptr;
    }
    // Keyed by the path asked for, so callers holding that string find it again.
    material->path = path;
    return &(m_materials[path] = std::move(*material));
}

bool MaterialLibrary::save(const MaterialAsset& material)
{
    if (!saveFile(material))
        return false;
    m_materials[material.path] = material;
    std::erase(m_missing, material.path);
    ++m_revision;
    return true;
}

std::string MaterialLibrary::uniquePath(const std::string& name) const
{
    std::string base = name.empty() ? std::string("Material") : name;
    // Keep the name usable as a file name.
    for (char& c : base)
        if (std::string_view("<>:\"/\\|?*").find(c) != std::string_view::npos)
            c = '_';
    const auto pathFor = [&](const std::string& stem) {
        return normalizePath(std::filesystem::path(m_root) / (stem + kMaterialExtension));
    };
    std::string path = pathFor(base);
    for (int i = 2; m_materials.count(path) > 0 || std::filesystem::exists(path); ++i)
        path = pathFor(base + " " + std::to_string(i));
    return path;
}

std::optional<std::string> MaterialLibrary::create(const std::string& name, const MaterialAsset& initial)
{
    std::error_code ec;
    std::filesystem::create_directories(m_root, ec);
    MaterialAsset material = initial;
    material.path = uniquePath(name);
    material.name = std::filesystem::path(material.path).stem().string();
    if (!save(material))
        return std::nullopt;
    return material.path;
}

bool MaterialLibrary::remove(const std::string& path)
{
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
        LOG_ERROR("[MATERIAL] Failed to delete " << path << ": " << ec.message() << "\n");
        return false;
    }
    m_materials.erase(path);
    ++m_revision;
    return true;
}

std::optional<std::string> MaterialLibrary::rename(const std::string& path, const std::string& newName)
{
    const auto it = m_materials.find(path);
    if (it == m_materials.end())
        return std::nullopt;
    const std::string newPath = uniquePath(newName);
    std::error_code ec;
    std::filesystem::rename(path, newPath, ec);
    if (ec) {
        LOG_ERROR("[MATERIAL] Failed to rename " << path << ": " << ec.message() << "\n");
        return std::nullopt;
    }
    MaterialAsset material = std::move(it->second);
    m_materials.erase(it);
    material.path = newPath;
    material.name = std::filesystem::path(newPath).stem().string();
    m_materials[newPath] = std::move(material);
    ++m_revision;
    return newPath;
}

std::optional<MaterialAsset> MaterialLibrary::loadFile(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        return std::nullopt;
    const nlohmann::json json = nlohmann::json::parse(file, nullptr, false);
    if (json.is_discarded() || !json.is_object()) {
        LOG_ERROR("[MATERIAL] Not a valid material file: " << path << "\n");
        return std::nullopt;
    }
    MaterialAsset material;
    material.path = path;
    material.name = std::filesystem::path(path).stem().string();
    std::array<float, 4> color{ 1.0f, 1.0f, 1.0f, 1.0f };
    readField(json, "color", color);
    material.color = glm::clamp(glm::vec4(color[0], color[1], color[2], color[3]), glm::vec4(0.0f), glm::vec4(1.0f));
    readField(json, "roughness", material.roughness);
    readField(json, "metallic", material.metallic);
    readField(json, "baseColorTexture", material.baseColorTexture);
    readField(json, "normalTexture", material.normalTexture);
    readField(json, "texelSize", material.texelSize);
    material.roughness = std::clamp(material.roughness, 0.0f, 1.0f);
    material.metallic = std::clamp(material.metallic, 0.0f, 1.0f);
    if (!std::isfinite(material.texelSize) || material.texelSize < 1e-3f)
        material.texelSize = 1.0f;
    return material;
}

bool MaterialLibrary::saveFile(const MaterialAsset& material)
{
    nlohmann::json json;
    json["color"] = { material.color.r, material.color.g, material.color.b, material.color.a };
    json["roughness"] = material.roughness;
    json["metallic"] = material.metallic;
    json["baseColorTexture"] = material.baseColorTexture;
    json["normalTexture"] = material.normalTexture;
    json["texelSize"] = material.texelSize;
    std::ofstream file(material.path);
    if (!file.is_open()) {
        LOG_ERROR("[MATERIAL] Failed to write " << material.path << "\n");
        return false;
    }
    file << json.dump(4) << '\n';
    return static_cast<bool>(file);
}
