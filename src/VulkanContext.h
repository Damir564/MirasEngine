#pragma once
#include "PCH.h"

class VulkanContext {
public:
    void init(SDL_Window* window, const char* appName);
    void cleanup();

    // Getters
    vk::Device getDevice() const { return device; }
    vk::PhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    VmaAllocator getAllocator() const { return allocator; }
    vk::Queue getGraphicsQueue() const { return graphicsQueue; }
    vk::Queue getPresentQueue() const { return presentQueue; }
    vkb::Swapchain& getSwapchain() { return swapchain; }

    uint32_t getGraphicsQueueFamily() const { return graphicsQueueFamily; }

    // Immediate submit helper (for asset uploading)
    void immediateSubmit(std::function<void(vk::CommandBuffer cmd)>&& function);

private:
    vkb::Instance vkbInstance;
    vk::Instance instance;
    vk::SurfaceKHR surface;

    vkb::Device vkbDevice;
    vk::Device device;
    vk::PhysicalDevice physicalDevice;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    uint32_t graphicsQueueFamily;

    vkb::Swapchain swapchain;
    VmaAllocator allocator;

    // For immediate submit
    vk::UniqueCommandPool immediateCmdPool;
    vk::UniqueFence immediateFence;
};