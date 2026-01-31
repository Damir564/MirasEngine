#pragma once
#include "PCH.h"

struct Camera {
    glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 10.0f;
    float sensitivity = 0.1f;

    glm::mat4 getView() const;
    glm::mat4 getProjection(float width, float height) const;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec4 tangent;

    static vk::VertexInputBindingDescription2EXT getBindingDescription(uint32_t binding = 0);
    static std::array<vk::VertexInputAttributeDescription2EXT, 5> getAttributeDescriptions(uint32_t locationOffset = 0);
};

struct MeshPushConstants {
    glm::mat4 view;
    glm::mat4 proj;
    alignas(16) glm::vec4 cameraPos;
    alignas(16) glm::vec4 baseColor{ 1.0f, 1.0f, 1.0f, 1.0f };
    float metallic{ 0.0f };
    float roughness{ 0.5f };
    float time;
};

struct InstanceData {
    glm::vec3 offset;
};

struct Material {
    glm::vec4 baseColorFactor{ 1.0f };
    float metallicFactor{ 1.0f };
    float roughnessFactor{ 1.0f };
    int baseColorTextureIndex = -1;
    int normalTextureIndex = -1;
    int metallicRoughnessTextureIndex = -1;
};

struct SubmeshInfo {
    uint32_t indexOffset;
    uint32_t indexCount;
    uint32_t vertexOffset;
    Material material;
};

// Raw CPU texture data holder
struct TextureData {
    int width, height, channels;
    unsigned char* pixels = nullptr;
    std::string path;
    const unsigned char* encodedData = nullptr;
    size_t encodedSize = 0;
    bool isLinear = false;

    void free(); // Helper to free STB memory
};