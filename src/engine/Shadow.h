#pragma once
#include "ModelInstance.h"
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <vector>
#include "Buffers.h" 

struct ShadowMapResources {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	vk::ImageView view;
	vk::Sampler sampler;
};

struct DirectionalLight {
	glm::vec3 direction{ -0.5f, -1.0f, -0.3f };
	glm::vec3 color{ 1.0f, 1.0f, 1.0f };
	float intensity{ 1.0f };
};

struct ShadowPushConstants {
	float alphaCutoff;      // 4 bytes
	int alphaMode;          // 4 bytes
	float padding[2];       // 8 bytes for alignment
};

struct SceneBounds {
	glm::vec3 center{ 0.0f };
	float radius{ 100.0f };
};

constexpr uint32_t SHADOW_MAP_SIZE = 2048;

glm::mat4 calculateLightSpaceMatrix(const DirectionalLight& light, const glm::vec3& sceneCenter, float sceneRadius);
SceneBounds calculateSceneBounds(const std::vector<Vertex>& vertices, const std::vector<InstanceData>& instances);
ShadowMapResources createShadowMap(VmaAllocator allocator, vk::Device device);
void destroyShadowMap(ShadowMapResources& shadow, VmaAllocator allocator, vk::Device device);
