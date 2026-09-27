#include "Shadow.h"
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <cmath>
#include <stdexcept>

glm::mat4 calculateLightSpaceMatrix(const DirectionalLight& light, const glm::vec3& center, float radius, uint32_t mapSize) {
    const glm::vec3 lightDir = glm::normalize(light.direction);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    if (glm::abs(glm::dot(lightDir, up)) > 0.99f)
        up = glm::vec3(0.0f, 0.0f, 1.0f);

    // Rotation only, so snapping in this space is independent of where the camera is.
    const glm::mat4 lightView = glm::lookAt(glm::vec3(0.0f), lightDir, up);
    glm::vec3 centerLS = glm::vec3(lightView * glm::vec4(center, 1.0f));
    const float texel = (2.0f * radius) / static_cast<float>(mapSize);
    centerLS.x = std::floor(centerLS.x / texel) * texel;
    centerLS.y = std::floor(centerLS.y / texel) * texel;

    // View space looks down -Z. Casters up to several radii towards the sun still land in the map.
    const float depth = -centerLS.z;
    const float nearPlane = depth - radius * 6.0f;
    const float farPlane = depth + radius;
    glm::mat4 lightProj = glm::orthoRH_ZO(centerLS.x - radius, centerLS.x + radius,
        centerLS.y - radius, centerLS.y + radius, nearPlane, farPlane);
    lightProj[1][1] *= -1; // Vulkan clip space has Y pointing down
    return lightProj * lightView;
}

ShadowMapResources createShadowMap(VmaAllocator allocator, vk::Device device, uint32_t size) {
    ShadowMapResources shadow{};
    shadow.size = size;

    // 1. Create Depth Image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_D32_SFLOAT;
    imageInfo.extent = { size, size, 1 };
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

    auto view = device.createImageView(viewInfo);
    if (view.result != vk::Result::eSuccess) {
        vmaDestroyImage(allocator, shadow.image, shadow.allocation);
        throw std::runtime_error("Failed to create shadow map view");
    }
    shadow.view = view.value;

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

    auto sampler = device.createSampler(samplerInfo);
    if (sampler.result != vk::Result::eSuccess) {
        device.destroyImageView(shadow.view);
        vmaDestroyImage(allocator, shadow.image, shadow.allocation);
        throw std::runtime_error("Failed to create shadow map sampler");
    }
    shadow.sampler = sampler.value;

    return shadow;
}

void destroyShadowMap(ShadowMapResources& shadow, VmaAllocator allocator, vk::Device device) {
    if (shadow.sampler) device.destroySampler(shadow.sampler);
    if (shadow.view) device.destroyImageView(shadow.view);
    if (shadow.image) vmaDestroyImage(allocator, shadow.image, shadow.allocation);
    shadow = {};
}

