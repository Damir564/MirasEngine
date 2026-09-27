#pragma once
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <cstdint>

struct ShadowMapResources {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    vk::ImageView view;
    vk::Sampler sampler;
    uint32_t size = 0;
};

struct DirectionalLight {
    glm::vec3 direction{ -0.5f, -1.0f, -0.3f };
    glm::vec3 color{ 1.0f, 1.0f, 1.0f };
    float intensity{ 1.0f };
};

// Orthographic light matrix covering a sphere. The sphere center is snapped to whole shadow-map
// texels in light space so the map does not shimmer while the camera moves.
glm::mat4 calculateLightSpaceMatrix(const DirectionalLight& light, const glm::vec3& center, float radius, uint32_t mapSize);
// Throws std::runtime_error on failure.
ShadowMapResources createShadowMap(VmaAllocator allocator, vk::Device device, uint32_t size);
void destroyShadowMap(ShadowMapResources& shadow, VmaAllocator allocator, vk::Device device);
