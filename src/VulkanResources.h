#pragma once
#include "PCH.h"
#include "VulkanContext.h"
#include "Types.h"

// Generic Buffer Wrapper
class VulkanBuffer {
public:
    VulkanBuffer(VulkanContext& ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    ~VulkanBuffer();

    // Disable copy, allow move
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    VulkanBuffer(VulkanBuffer&& other) noexcept;

    void upload(const void* data, size_t size);
    vk::Buffer getBuffer() const { return buffer; }
    uint32_t getCount() const { return count; }
    void setCount(uint32_t c) { count = c; }

private:
    VulkanContext* ctx;
    vk::Buffer buffer;
    VmaAllocation allocation;
    uint32_t count = 0;
};

// Texture Wrapper
class VulkanTexture {
public:
    VulkanTexture(VulkanContext& ctx, const TextureData& data, vk::Format format);
    ~VulkanTexture();

    VulkanTexture(const VulkanTexture&) = delete;
    VulkanTexture& operator=(const VulkanTexture&) = delete;
    VulkanTexture(VulkanTexture&& other) noexcept;

    vk::ImageView getView() const { return view; }

private:
    VulkanContext* ctx;
    vk::Image image;
    VmaAllocation allocation;
    vk::ImageView view;
};