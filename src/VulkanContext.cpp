#include "VulkanContext.h"

void VulkanContext::init(SDL_Window* window, const char* appName) {
    // 1. Initialize volk
    if (volkInitialize() != VK_SUCCESS) throw std::runtime_error("volkInitialize failed");
    vk::detail::defaultDispatchLoaderDynamic.init(vkGetInstanceProcAddr);

    // 2. Get SDL Extensions
    Uint32 count;
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&count);
    std::vector<const char*> extensionList(extensions, extensions + count);

    // 3. Create Instance
    vkb::InstanceBuilder builder;
    auto instRet = builder.set_app_name(appName)
        .require_api_version(1, 4, 0)
        .set_minimum_instance_version(1, 4)
        .enable_extensions(extensionList)
        .build();
    if (!instRet) throw std::runtime_error("Instance creation failed: " + instRet.error().message());

    vkbInstance = instRet.value();
    volkLoadInstance(vkbInstance.instance);
    instance = vk::Instance(vkbInstance.instance);
    vk::detail::defaultDispatchLoaderDynamic.init(instance);
    // 4. Create Surface
    VkSurfaceKHR surfaceVk;
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surfaceVk)) throw std::runtime_error("Surface creation failed");
    surface = vk::SurfaceKHR(surfaceVk);

    // 5. Select Physical Device (Shader Object reqs)
    VkPhysicalDeviceShaderObjectFeaturesEXT shaderFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT };
    shaderFeatures.shaderObject = VK_TRUE;

    VkPhysicalDeviceVulkan13Features features13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;
    features13.pNext = &shaderFeatures;

    VkPhysicalDeviceVulkan12Features features12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{ vkbInstance };
    auto physRet = selector
        .set_minimum_version(1, 4)
        .add_required_extension(vk::EXTShaderObjectExtensionName)
        .add_required_extension_features(shaderFeatures)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_surface(surface)
        .select();
    if (!physRet) throw std::runtime_error("No suitable device found: " + physRet.error().message());

    // 6. Create Device
    vkb::DeviceBuilder deviceBuilder{ physRet.value() };
    auto deviceRet = deviceBuilder.build();
    if (!deviceRet) throw std::runtime_error("Device creation failed");

    vkbDevice = deviceRet.value();
    volkLoadDevice(vkbDevice.device);
    device = vk::Device(vkbDevice.device);
    vk::detail::defaultDispatchLoaderDynamic.init(device);
    physicalDevice = vk::PhysicalDevice(vkbDevice.physical_device);

    graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    presentQueue = vkbDevice.get_queue(vkb::QueueType::present).value();
    graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // 7. VMA
    VmaVulkanFunctions vulkanFunctions{};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    allocatorInfo.pVulkanFunctions = &vulkanFunctions;
    allocatorInfo.vulkanApiVersion = vk::ApiVersion14;
    if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) throw std::runtime_error("VMA failed");

    // 8. Swapchain
    vkb::SwapchainBuilder swapchainBuilder{ vkbDevice };
    auto swapRet = swapchainBuilder
        .set_desired_extent(1280, 720)
        .set_desired_format({ VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .build();
    if (!swapRet) throw std::runtime_error("Swapchain failed");
    swapchain = swapRet.value();

    // 9. Immediate Submit Init
    vk::CommandPoolCreateInfo poolInfo{ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsQueueFamily };
    immediateCmdPool = device.createCommandPoolUnique(poolInfo).value;
    immediateFence = device.createFenceUnique({}).value;
}

void VulkanContext::immediateSubmit(std::function<void(vk::CommandBuffer cmd)>&& function) {
    vk::CommandBufferAllocateInfo allocInfo(immediateCmdPool.get(), vk::CommandBufferLevel::ePrimary, 1);
    vk::UniqueCommandBuffer cmd = std::move(device.allocateCommandBuffersUnique(allocInfo).value[0]);

    cmd->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    function(cmd.get());
    cmd->end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd.get();

    graphicsQueue.submit(submitInfo, immediateFence.get());
    auto result = device.waitForFences(immediateFence.get(), VK_TRUE, 9999999999);
    device.resetFences(immediateFence.get());
}

void VulkanContext::cleanup() {
    // Wait idle
    if (device) device.waitIdle();

    // Destroy VKB/Vulkan wrappers in reverse order
    immediateFence.reset();
    immediateCmdPool.reset();

    vkb::destroy_swapchain(swapchain);
    vmaDestroyAllocator(allocator);
    vkb::destroy_device(vkbDevice);
    vkb::destroy_surface(vkbInstance, surface);
    vkb::destroy_instance(vkbInstance);
}