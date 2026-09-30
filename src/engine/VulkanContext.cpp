#define VKB_DISABLE_DEBUG_BREAK
#include "VulkanContext.h"
#include <volk.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vector>
#include "Log.h"

namespace {
constexpr uint32_t kApiMajor = 1;
constexpr uint32_t kApiMinor = 3;
}

VulkanContext::~VulkanContext()
{
    shutdown();
}

bool VulkanContext::init(SDL_Window* window, bool enableValidation)
{
    if (volkInitialize() != VK_SUCCESS) {
        LOG_ERROR("volkInitialize failed\n");
        return false;
    }
    vk::detail::defaultDispatchLoaderDynamic.init(vkGetInstanceProcAddr);

    if (!createInstance(enableValidation) || !createSurface(window) || !createDevice() || !createAllocator()) {
        shutdown();
        return false;
    }

    LOG_INFO("SDL3 + Vulkan instance, device, and VMA initialized successfully!\n");
    return true;
}

bool VulkanContext::createInstance(bool enableValidation)
{
    Uint32 extensionCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
    if (sdlExtensions == nullptr) {
        LOG_ERROR("SDL_Vulkan_GetInstanceExtensions failed\n");
        return false;
    }

    std::vector<const char*> extensions(sdlExtensions, sdlExtensions + extensionCount);

    vkb::InstanceBuilder builder;
    builder
        .set_app_name("SDL3 Vulkan App")
        .require_api_version(kApiMajor, kApiMinor, 0)
        .set_minimum_instance_version(kApiMajor, kApiMinor)
        .enable_extensions(extensions);
    if (enableValidation)
        builder.request_validation_layers(true).use_default_debug_messenger();
    auto instRet = builder.build();
    if (!instRet) {
        LOG_ERROR("Failed to create instance: " << instRet.error().message() << "\n");
        return false;
    }

    m_vkbInstance = instRet.value();
    volkLoadInstance(m_vkbInstance.instance);
    m_instance = vk::Instance(m_vkbInstance.instance);
    vk::detail::defaultDispatchLoaderDynamic.init(m_instance);
    return true;
}

bool VulkanContext::createSurface(SDL_Window* window)
{
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window, m_instance, nullptr, &surface)) {
        LOG_ERROR("Failed to create Vulkan surface\n");
        return false;
    }
    m_surface = vk::SurfaceKHR(surface);
    return true;
}

bool VulkanContext::createDevice()
{
    VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
    shaderObjectFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
    shaderObjectFeatures.shaderObject = VK_TRUE;

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;
    features13.pNext = &shaderObjectFeatures;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.timelineSemaphore = VK_TRUE;
    features12.vulkanMemoryModel = VK_TRUE;
    features12.vulkanMemoryModelDeviceScope = VK_TRUE;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.scalarBlockLayout = VK_TRUE;
    features12.storageBuffer8BitAccess = VK_TRUE;
    features12.separateDepthStencilLayouts = VK_TRUE;

    VkPhysicalDeviceFeatures coreFeatures{};
    coreFeatures.fragmentStoresAndAtomics = VK_TRUE;
    coreFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;
    coreFeatures.shaderInt64 = VK_TRUE;
    coreFeatures.wideLines = VK_TRUE;
    coreFeatures.multiDrawIndirect = VK_TRUE;
    coreFeatures.drawIndirectFirstInstance = VK_TRUE;
    coreFeatures.samplerAnisotropy = VK_TRUE;
    // Shadow casters in front of a cascade are flattened onto its near plane instead of being clipped.
    coreFeatures.depthClamp = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{ m_vkbInstance };
    auto physRet = selector
        .set_minimum_version(kApiMajor, kApiMinor)
        .add_required_extension(vk::EXTShaderObjectExtensionName)
        .add_required_extension_features(shaderObjectFeatures)
        .set_required_features(coreFeatures)
        .set_required_features_12(features12)
        .set_required_features_13(features13)
        .set_surface(m_surface)
        .select();
    if (!physRet) {
        LOG_ERROR("Failed to select physical device: " << physRet.error().message() << "\n");
        return false;
    }

    m_vkbPhysicalDevice = physRet.value();
    if (!m_vkbPhysicalDevice.is_extension_present(VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) {
        LOG_ERROR("Shader Object not supported\n");
        return false;
    }
    m_physicalDevice = vk::PhysicalDevice(m_vkbPhysicalDevice.physical_device);

    auto deviceRet = vkb::DeviceBuilder{ m_vkbPhysicalDevice }.build();
    if (!deviceRet) {
        LOG_ERROR("Failed to create device: " << deviceRet.error().message() << "\n");
        return false;
    }

    m_vkbDevice = deviceRet.value();
    volkLoadDevice(m_vkbDevice.device);
    m_device = vk::Device(m_vkbDevice.device);
    vk::detail::defaultDispatchLoaderDynamic.init(m_device);

    auto graphicsQueue = m_vkbDevice.get_queue(vkb::QueueType::graphics);
    auto presentQueue = m_vkbDevice.get_queue(vkb::QueueType::present);
    auto graphicsFamily = m_vkbDevice.get_queue_index(vkb::QueueType::graphics);
    if (!graphicsQueue || !presentQueue || !graphicsFamily) {
        LOG_ERROR("Failed to get device queues\n");
        return false;
    }
    m_graphicsQueue = vk::Queue(graphicsQueue.value());
    m_presentQueue = vk::Queue(presentQueue.value());
    m_graphicsQueueFamily = graphicsFamily.value();
    return true;
}

bool VulkanContext::createAllocator()
{
    VmaVulkanFunctions vulkanFunctions{};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = m_physicalDevice;
    allocatorInfo.device = m_device;
    allocatorInfo.instance = m_instance;
    allocatorInfo.pVulkanFunctions = &vulkanFunctions;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

    if (vmaCreateAllocator(&allocatorInfo, &m_allocator) != VK_SUCCESS) {
        LOG_ERROR("Failed to create VMA allocator\n");
        m_allocator = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

void VulkanContext::shutdown()
{
    if (m_allocator) {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }
    if (m_device) {
        vkb::destroy_device(m_vkbDevice);
        m_vkbDevice = {};
        m_device = nullptr;
    }
    if (m_surface) {
        vkb::destroy_surface(m_vkbInstance, m_surface);
        m_surface = nullptr;
    }
    if (m_instance) {
        vkb::destroy_instance(m_vkbInstance);
        m_vkbInstance = {};
        m_instance = nullptr;
    }
    m_graphicsQueue = nullptr;
    m_presentQueue = nullptr;
}
