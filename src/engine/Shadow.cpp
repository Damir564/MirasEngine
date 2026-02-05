#include "Shadow.h"
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <iostream>

glm::mat4 calculateLightSpaceMatrix(const DirectionalLight& light, const glm::vec3& sceneCenter, float sceneRadius) {
	glm::vec3 lightDir = glm::normalize(light.direction);

	// Create stable up vector
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	if (glm::abs(glm::dot(lightDir, up)) > 0.99f) {
		up = glm::vec3(0.0f, 0.0f, 1.0f);
	}

	// Position light very far from scene
	float lightDistance = sceneRadius * 5.0f;
	glm::vec3 lightPos = sceneCenter - lightDir * lightDistance;

	glm::mat4 lightView = glm::lookAt(lightPos, sceneCenter, up);

	// Make ortho bounds much larger to capture everything
	float orthoSize = sceneRadius * 3.0f;

	float nearPlane = 0.1f;
	float farPlane = lightDistance + sceneRadius * 2.0f;

	glm::mat4 lightProj = glm::ortho(
		-orthoSize, orthoSize,
		-orthoSize, orthoSize,
		nearPlane, farPlane
	);

	// Vulkan Y-flip
	lightProj[1][1] *= -1;

	return lightProj * lightView;
}

SceneBounds calculateSceneBounds(const std::vector<Vertex>& vertices,
	const std::vector<InstanceData>& instances) {
	if (vertices.empty()) {
		return { glm::vec3(0.0f), 100.0f };
	}

	glm::vec3 minBounds{ FLT_MAX };
	glm::vec3 maxBounds{ -FLT_MAX };

	if (instances.empty()) {
		// No instances - just use raw vertex positions
		for (const auto& v : vertices) {
			minBounds = glm::min(minBounds, v.position);
			maxBounds = glm::max(maxBounds, v.position);
		}
	}
	else {
		// With instances - compute bounds for all instanced positions
		for (const auto& inst : instances) {
			for (const auto& v : vertices) {
				glm::vec3 worldPos = v.position + inst.offset;
				minBounds = glm::min(minBounds, worldPos);
				maxBounds = glm::max(maxBounds, worldPos);
			}
		}
	}

	SceneBounds bounds;
	bounds.center = (minBounds + maxBounds) * 0.5f;
	bounds.radius = glm::length(maxBounds - minBounds) * 0.5f;

	// Add generous padding
	bounds.radius *= 1.5f;

	std::cout << "Scene bounds: center=(" << bounds.center.x << ", "
		<< bounds.center.y << ", " << bounds.center.z
		<< ") radius=" << bounds.radius << std::endl;
	std::cout << "  Min: (" << minBounds.x << ", " << minBounds.y << ", " << minBounds.z << ")" << std::endl;
	std::cout << "  Max: (" << maxBounds.x << ", " << maxBounds.y << ", " << maxBounds.z << ")" << std::endl;

	return bounds;
}

ShadowMapResources createShadowMap(VmaAllocator allocator, vk::Device device) {
	ShadowMapResources shadow{};

	// 1. Create Depth Image
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_D32_SFLOAT;
	imageInfo.extent = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 };
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	if (vmaCreateImage(allocator, &imageInfo, &allocInfo,
		&shadow.image, &shadow.allocation, nullptr) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create shadow map image");
	}

	// 2. Create Image View
	vk::ImageViewCreateInfo viewInfo{};
	viewInfo.image = vk::Image(shadow.image);
	viewInfo.viewType = vk::ImageViewType::e2D;
	viewInfo.format = vk::Format::eD32Sfloat;
	viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	shadow.view = device.createImageView(viewInfo).value;

	// 3. Create Shadow Sampler (with depth comparison)
	vk::SamplerCreateInfo samplerInfo{};
	samplerInfo.magFilter = vk::Filter::eLinear;
	samplerInfo.minFilter = vk::Filter::eLinear;
	samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToBorder;
	samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToBorder;
	samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToBorder;
	samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
	samplerInfo.compareEnable = VK_TRUE;
	samplerInfo.compareOp = vk::CompareOp::eLessOrEqual;
	samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;

	shadow.sampler = device.createSampler(samplerInfo).value;

	return shadow;
}

void destroyShadowMap(ShadowMapResources& shadow, VmaAllocator allocator, vk::Device device) {
	if (shadow.sampler) device.destroySampler(shadow.sampler);
	if (shadow.view) device.destroyImageView(shadow.view);
	if (shadow.image) vmaDestroyImage(allocator, shadow.image, shadow.allocation);
}

