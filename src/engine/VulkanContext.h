#pragma once
#include <vulkan/vulkan.hpp>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <cstdint>

struct SDL_Window;

// Owns the Vulkan instance, surface, device, queues and VMA allocator.
class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // Prints the reason and releases anything partially created on failure.
    bool init(SDL_Window* window, bool enableValidation);
    void shutdown();

    const vkb::Instance& vkbInstance() const { return m_vkbInstance; }
    vk::Instance instance() const { return m_instance; }
    vk::SurfaceKHR surface() const { return m_surface; }
    const vkb::PhysicalDevice& vkbPhysicalDevice() const { return m_vkbPhysicalDevice; }
    vk::PhysicalDevice physicalDevice() const { return m_physicalDevice; }
    const vkb::Device& vkbDevice() const { return m_vkbDevice; }
    vk::Device device() const { return m_device; }
    VmaAllocator allocator() const { return m_allocator; }
    vk::Queue graphicsQueue() const { return m_graphicsQueue; }
    vk::Queue presentQueue() const { return m_presentQueue; }
    uint32_t graphicsQueueFamily() const { return m_graphicsQueueFamily; }

private:
    bool createInstance(bool enableValidation);
    bool createSurface(SDL_Window* window);
    bool createDevice();
    bool createAllocator();

    vkb::Instance m_vkbInstance;
    vk::Instance m_instance;
    vk::SurfaceKHR m_surface;
    vkb::PhysicalDevice m_vkbPhysicalDevice;
    vk::PhysicalDevice m_physicalDevice;
    vkb::Device m_vkbDevice;
    vk::Device m_device;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    vk::Queue m_graphicsQueue;
    vk::Queue m_presentQueue;
    uint32_t m_graphicsQueueFamily = 0;
};
