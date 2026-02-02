#pragma once
#include "PCH.h"
#include "Types.h"
#include "VulkanResources.h"

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SubmeshInfo> submeshes;
    std::vector<TextureData> textureData;
};

class Model {
public:
    void loadFromFile(const std::string& path);
    void uploadToGPU(VulkanContext& ctx);

    // Resources
    std::unique_ptr<VulkanBuffer> vertexBuffer;
    std::unique_ptr<VulkanBuffer> indexBuffer;
    std::vector<std::unique_ptr<VulkanTexture>> textures;

    // Data
    MeshData meshData;

private:
    void loadGLTF(const std::string& path);
    void loadOBJ(const std::string& path);
};