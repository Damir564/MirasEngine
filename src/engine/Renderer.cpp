#include "Renderer.h"
#include "Atmosphere.h"
#include "ModelManager.h"
#include "RenderUtils.h"
#include "Vertex.h"
#include "VulkanContext.h"
#include <volk.h>
#include <SDL3/SDL.h>
#include <glm/ext/matrix_transform.hpp>
#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <execution>
#include <numeric>
#include "Log.h"

#include "RendererInternal.h"

void Renderer::FrameBatches::clear()
{
    for (size_t index : mainModels)
        main[index].instances.clear();
    for (size_t index : shadowModels)
        shadow[index].instances.clear();
    mainModels.clear();
    shadowModels.clear();
}

Renderer::RenderBatch& Renderer::FrameBatches::add(std::vector<RenderBatch>& batches, std::vector<size_t>& used,
    size_t modelIndex, GPUModel* model)
{
    if (modelIndex >= batches.size())
        batches.resize(modelIndex + 1);
    RenderBatch& batch = batches[modelIndex];
    if (batch.instances.empty())
        used.push_back(modelIndex);
    batch.model = model;
    return batch;
}

Renderer::~Renderer()
{
    shutdown();
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

bool Renderer::init(VulkanContext& context, SDL_Window* window, const GraphicsSettings& settings)
{
    m_context = &context;
    m_window = window;
    m_device = context.device();
    m_allocator = context.allocator();
    m_meshBinding = Vertex::getBindingDescription(0);
    m_meshAttributes = Vertex::getVertexOnlyAttributes(0);
    m_settings = sanitizeGraphicsSettings(settings);
    updateSun();
    queryCapabilities();

    try {
        int pixelWidth = 0, pixelHeight = 0;
        SDL_GetWindowSizeInPixels(window, &pixelWidth, &pixelHeight);
        if (!m_swapchain.create(context, static_cast<uint32_t>(pixelWidth), static_cast<uint32_t>(pixelHeight), m_settings.vsync)) {
            shutdown();
            return false;
        }
        // Fixed at startup; later swapchain rebuilds may change the image count but not this.
        m_framesInFlight = m_swapchain.imageCount();

        if (!createRenderTargets() || !createSkyResources() || !createFrameResources() || !createDescriptors() ||
            !createShaders() || !initImGuiBackend()) {
            shutdown();
            return false;
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Renderer initialization failed: " << e.what() << "\n");
        shutdown();
        return false;
    }
    return true;
}

void Renderer::queryCapabilities()
{
    const vk::PhysicalDeviceLimits limits = m_context->physicalDevice().getProperties().limits;
    const vk::SampleCountFlags counts = limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;
    m_capabilities.maxMsaaSamples = 1;
    for (int samples : { 2, 4, 8 })
        if (counts & toSampleCount(samples))
            m_capabilities.maxMsaaSamples = samples;
    const OptionalDeviceFeatures& features = m_context->features();
    m_capabilities.maxAnisotropy = features.samplerAnisotropy ? limits.maxSamplerAnisotropy : 1.0f;
    m_maxDrawIndirectCount = features.multiDrawIndirect ? limits.maxDrawIndirectCount : 1;
}

vk::SampleCountFlagBits Renderer::effectiveSampleCount() const
{
    if (classic())
        return vk::SampleCountFlagBits::e1;
    return toSampleCount(std::min(m_settings.msaaSamples, m_capabilities.maxMsaaSamples));
}

bool Renderer::createRenderImage(vk::Format format, vk::ImageUsageFlags usage, vk::SampleCountFlagBits samples,
    vk::ImageAspectFlags aspect, RenderImage& out, vk::Extent2D extent, uint32_t mipLevels)
{
    if (extent.width == 0 || extent.height == 0)
        extent = m_swapchain.extent();

    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent = vk::Extent3D{ extent.width, extent.height, 1 };
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = usage;
    imageInfo.samples = samples;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (usage & vk::ImageUsageFlagBits::eTransientAttachment)
        allocInfo.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;

    if (vmaCreateImage(m_allocator, reinterpret_cast<const VkImageCreateInfo*>(&imageInfo), &allocInfo,
        &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
        LOG_ERROR("Failed to create render target image\n");
        out = {};
        return false;
    }

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = vk::Image(out.image);
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = { aspect, 0, mipLevels, 0, 1 };
    if (!takeResult(m_device.createImageView(viewInfo), out.view, "render target view")) {
        destroyRenderImage(out);
        return false;
    }
    return true;
}

bool Renderer::createMipView(const RenderImage& image, vk::Format format, uint32_t mip, vk::ImageView& out)
{
    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = vk::Image(image.image);
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1 };
    return takeResult(m_device.createImageView(viewInfo), out, "mip view");
}

void Renderer::destroyRenderImage(RenderImage& image)
{
    if (image.view) m_device.destroyImageView(image.view);
    if (image.image) vmaDestroyImage(m_allocator, image.image, image.allocation);
    image = {};
}

bool Renderer::createRenderTargets()
{
    m_samples = effectiveSampleCount();
    const vk::Extent2D extent = m_swapchain.extent();
    m_halfExtent.setWidth((extent.width + 1) / 2);
    m_halfExtent.setHeight((extent.height + 1) / 2);
    const auto depthAspect = vk::ImageAspectFlagBits::eDepth;
    const auto colorAspect = vk::ImageAspectFlagBits::eColor;
    const auto target = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
    const auto single = vk::SampleCountFlagBits::e1;

    if (!createRenderImage(kDepthFormat,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            single, depthAspect, m_sceneDepth) ||
        !createRenderImage(kSelectionMaskFormat, target, single, colorAspect, m_selectionMask) ||
        !createRenderImage(kHdrFormat, target, single, colorAspect, m_hdrColor) ||
        !createRenderImage(kLdrFormat, target, single, colorAspect, m_ldrColor) ||
        !createRenderImage(kAoDepthFormat, target, single, colorAspect, m_aoDepth, m_halfExtent) ||
        !createRenderImage(kAoFormat, target, single, colorAspect, m_aoRaw, m_halfExtent) ||
        !createRenderImage(kAoFormat, target, single, colorAspect, m_aoTemp, m_halfExtent))
        return false;

    // Bloom mips down to roughly 8 texels on the short side.
    m_bloomMips = 1;
    while (m_bloomMips < kMaxBloomMips && (std::min(m_halfExtent.width, m_halfExtent.height) >> m_bloomMips) >= 8)
        ++m_bloomMips;
    if (!createRenderImage(kHdrFormat, target, single, colorAspect, m_bloom, m_halfExtent, m_bloomMips))
        return false;
    for (uint32_t mip = 0; mip < m_bloomMips; ++mip)
        if (!createMipView(m_bloom, kHdrFormat, mip, m_bloomMipViews[mip]))
            return false;

    if (m_samples != single) {
        // The depth is not transient: the depth prepass result is loaded again by the scene pass.
        if (!createRenderImage(kHdrFormat,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransientAttachment,
                m_samples, colorAspect, m_msaaColor) ||
            !createRenderImage(kDepthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment,
                m_samples, depthAspect, m_msaaDepth))
            return false;
    }

    writeImageDescriptors();
    return true;
}

void Renderer::destroyRenderTargets()
{
    for (vk::ImageView& view : m_bloomMipViews) {
        if (view) m_device.destroyImageView(view);
        view = nullptr;
    }
    m_bloomMips = 0;
    for (RenderImage* image : { &m_sceneDepth, &m_selectionMask, &m_msaaColor, &m_msaaDepth, &m_hdrColor,
             &m_ldrColor, &m_aoDepth, &m_aoRaw, &m_aoTemp, &m_bloom })
        destroyRenderImage(*image);
}

bool Renderer::createSkyResources()
{
    const auto colorAspect = vk::ImageAspectFlagBits::eColor;
    // Mips come from blits and give rough reflections their blur.
    const auto lutUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    m_skyValid = false;
    return createRenderImage(kHdrFormat, lutUsage, vk::SampleCountFlagBits::e1, colorAspect, m_skyLut,
               kSkyLutExtent, kSkyLutMips) &&
        createMipView(m_skyLut, kHdrFormat, 0, m_skyLutTargetView) &&
        createRenderImage(kHdrFormat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::SampleCountFlagBits::e1, colorAspect, m_skyIrradiance, kSkyIrradianceExtent);
}

void Renderer::destroySkyResources()
{
    if (m_skyLutTargetView) m_device.destroyImageView(m_skyLutTargetView);
    m_skyLutTargetView = nullptr;
    destroyRenderImage(m_skyLut);
    destroyRenderImage(m_skyIrradiance);
}

std::pair<uint32_t, uint32_t> Renderer::wantedShadowMap() const
{
    if (!m_settings.shadows || classic())
        return { kStubShadowMapSize, 1u };
    return { static_cast<uint32_t>(m_settings.shadowMapSize), static_cast<uint32_t>(m_settings.shadowCascades) };
}

bool Renderer::createShadowMapResources()
{
    const auto [size, layers] = wantedShadowMap();
    m_shadowMap = createShadowMap(m_allocator, m_device, size, layers);
    m_shadowMapValid = false;
    if (!m_lightShadowMap.image)
        m_lightShadowMap = createShadowMap(m_allocator, m_device, kStubShadowMapSize, 1);
    writeImageDescriptors();
    return true;
}

bool Renderer::lightShadowsAllowed() const
{
    return m_settings.lightShadows && !classic();
}

void Renderer::resizeLightShadowMap(bool full)
{
    const uint32_t size = full ? kLightShadowMapSize : kStubShadowMapSize;
    if (m_lightShadowMap.size == size)
        return;
    // The other frame in flight may still sample the old map through the lighting set.
    (void)m_device.waitIdle();
    destroyShadowMap(m_lightShadowMap, m_allocator, m_device);
    try {
        m_lightShadowMap = createShadowMap(m_allocator, m_device, size, full ? kMaxLightShadowLayers : 1);
    }
    catch (const std::exception& e) {
        LOG_ERROR(e.what() << "; light shadows are off\n");
        m_lightShadowMap = createShadowMap(m_allocator, m_device, kStubShadowMapSize, 1);
    }
    writeImageDescriptors();
}

bool Renderer::createFrameResources()
{
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = m_context->graphicsQueueFamily();
    if (!takeResult(m_device.createCommandPool(poolInfo), m_commandPool, "command pool"))
        return false;

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = m_framesInFlight;
    if (!takeResult(m_device.allocateCommandBuffers(allocInfo), m_commandBuffers, "command buffers"))
        return false;

    m_frameUBOs.resize(m_framesInFlight);
    for (UBOBuffer& ubo : m_frameUBOs) {
        VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = sizeof(FrameUBO);
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo uboAllocInfo{};
        uboAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        uboAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo allocationInfo{};
        if (vmaCreateBuffer(m_allocator, &bufferInfo, &uboAllocInfo, &ubo.buffer, &ubo.allocation, &allocationInfo) != VK_SUCCESS) {
            LOG_ERROR("Failed to create UBO buffer\n");
            ubo = {};
            return false;
        }
        ubo.mapped = allocationInfo.pMappedData;
    }

    vk::FenceCreateInfo fenceInfo{ vk::FenceCreateFlagBits::eSignaled };
    m_imageAvailableSemaphores.resize(m_framesInFlight);
    m_renderFinishedSemaphores.resize(m_framesInFlight);
    m_inFlightFences.resize(m_framesInFlight);
    for (uint32_t i = 0; i < m_framesInFlight; ++i) {
        if (!takeResult(m_device.createSemaphore({}), m_imageAvailableSemaphores[i], "semaphore") ||
            !takeResult(m_device.createSemaphore({}), m_renderFinishedSemaphores[i], "semaphore") ||
            !takeResult(m_device.createFence(fenceInfo), m_inFlightFences[i], "fence"))
            return false;
    }
    return true;
}

bool Renderer::createDescriptors()
{
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.anisotropyEnable = m_capabilities.maxAnisotropy > 1.0f ? VK_TRUE : VK_FALSE;
    samplerInfo.maxAnisotropy = std::min(16.0f, m_capabilities.maxAnisotropy);
    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    if (!takeResult(m_device.createSampler(samplerInfo), m_textureSampler, "texture sampler"))
        return false;

    vk::SamplerCreateInfo nearestInfo{};
    nearestInfo.magFilter = vk::Filter::eNearest;
    nearestInfo.minFilter = vk::Filter::eNearest;
    nearestInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    nearestInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    nearestInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    nearestInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    if (!takeResult(m_device.createSampler(nearestInfo), m_nearestSampler, "nearest sampler"))
        return false;

    vk::SamplerCreateInfo linearInfo = nearestInfo;
    linearInfo.magFilter = vk::Filter::eLinear;
    linearInfo.minFilter = vk::Filter::eLinear;
    if (!takeResult(m_device.createSampler(linearInfo), m_linearClampSampler, "linear sampler"))
        return false;

    vk::SamplerCreateInfo skyInfo = linearInfo;
    skyInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    skyInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    skyInfo.maxLod = VK_LOD_CLAMP_NONE;
    if (!takeResult(m_device.createSampler(skyInfo), m_skySampler, "sky sampler"))
        return false;

    const vk::DescriptorSetLayoutBinding textureBinding{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment };
    if (!takeResult(m_device.createDescriptorSetLayout({ {}, 1, &textureBinding }), m_textureSetLayout, "texture set layout"))
        return false;

    // Binding 0: FrameUBO, 1: per-draw data (GpuDrawData[]), 2: per-instance transforms (GpuTransform[]),
    // 3: point and spot lights with their shadow matrices (GpuLightHeader + GpuLight[]).
    const auto vertexAndFragment = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    const vk::DescriptorSetLayoutBinding frameBindings[4] = {
        { 0, vk::DescriptorType::eUniformBuffer, 1, vertexAndFragment },
        { 1, vk::DescriptorType::eStorageBuffer, 1, vertexAndFragment },
        { 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        { 3, vk::DescriptorType::eStorageBuffer, 1, vertexAndFragment },
    };
    if (!takeResult(m_device.createDescriptorSetLayout({ {}, 4, frameBindings }), m_frameSetLayout, "frame set layout"))
        return false;

    // Set 4 of triangle.frag: shadow map (comparison and raw depth), AO, AO depth, sky LUT, sky irradiance,
    // light shadow map.
    std::array<vk::DescriptorSetLayoutBinding, 7> lightingBindings;
    for (uint32_t i = 0; i < lightingBindings.size(); ++i)
        lightingBindings[i] = { i, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment };
    vk::DescriptorSetLayoutCreateInfo lightingInfo{};
    lightingInfo.setBindings(lightingBindings);
    if (!takeResult(m_device.createDescriptorSetLayout(lightingInfo), m_lightingSetLayout, "lighting set layout"))
        return false;

    // Shared with ModelManager, which allocates one set per texture from it.
    const vk::DescriptorPoolSize poolSizes[] = {
        { vk::DescriptorType::eCombinedImageSampler, 1100 },
        { vk::DescriptorType::eUniformBuffer, m_framesInFlight + 10 },
        { vk::DescriptorType::eStorageBuffer, 3 * m_framesInFlight + 10 },
    };
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = 1164;
    poolInfo.setPoolSizes(poolSizes);
    if (!takeResult(m_device.createDescriptorPool(poolInfo), m_descriptorPool, "descriptor pool"))
        return false;

    std::vector<vk::DescriptorSet> lightingSets;
    if (!takeResult(m_device.allocateDescriptorSets({ m_descriptorPool, 1, &m_lightingSetLayout }), lightingSets, "lighting set"))
        return false;
    m_lightingSet = lightingSets[0];

    vk::DescriptorSet* imageSets[] = { &m_sceneDepthSet, &m_selectionMaskSet, &m_hdrSet, &m_ldrSet, &m_aoDepthSet,
        &m_aoRawSet, &m_aoTempSet, &m_skyLutSet };
    const uint32_t imageSetCount = static_cast<uint32_t>(std::size(imageSets)) + kMaxBloomMips;
    const std::vector<vk::DescriptorSetLayout> imageLayouts(imageSetCount, m_textureSetLayout);
    std::vector<vk::DescriptorSet> allocated;
    if (!takeResult(m_device.allocateDescriptorSets({ m_descriptorPool, imageLayouts }), allocated, "image descriptor sets"))
        return false;
    for (size_t i = 0; i < std::size(imageSets); ++i)
        *imageSets[i] = allocated[i];
    for (uint32_t i = 0; i < kMaxBloomMips; ++i)
        m_bloomSets[i] = allocated[std::size(imageSets) + i];

    // Also writes every image descriptor now that the sets exist.
    if (!createShadowMapResources())
        return false;

    const std::vector<vk::DescriptorSetLayout> frameLayouts(m_framesInFlight, m_frameSetLayout);
    vk::DescriptorSetAllocateInfo frameAllocInfo{ m_descriptorPool, frameLayouts };
    if (!takeResult(m_device.allocateDescriptorSets(frameAllocInfo), m_frameSets, "frame descriptor sets"))
        return false;

    for (uint32_t i = 0; i < m_framesInFlight; ++i) {
        const vk::DescriptorBufferInfo bufferInfo{ vk::Buffer(m_frameUBOs[i].buffer), 0, sizeof(FrameUBO) };
        const vk::WriteDescriptorSet write{ m_frameSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo };
        m_device.updateDescriptorSets(1, &write, 0, nullptr);
    }

    for (uint32_t i = 0; i < m_framesInFlight; ++i) {
        m_frameDrawBuffers.emplace_back(new FrameDrawBuffers{
            HostBuffer(m_allocator, vk::BufferUsageFlagBits::eStorageBuffer, sizeof(GpuDrawData) * 4096),
            HostBuffer(m_allocator, vk::BufferUsageFlagBits::eStorageBuffer, sizeof(GpuTransform) * 256),
            HostBuffer(m_allocator, vk::BufferUsageFlagBits::eIndirectBuffer, sizeof(vk::DrawIndexedIndirectCommand) * 4096),
        });
        writeDrawDescriptors(i);

        // Fixed size: the light count is capped at kMaxLights.
        auto& lightBuffer = m_lightBuffers.emplace_back(
            std::make_unique<HostBuffer>(m_allocator, vk::BufferUsageFlagBits::eStorageBuffer, kLightBufferSize));
        std::memset(lightBuffer->mapped(), 0, sizeof(GpuLightHeader));
        const vk::DescriptorBufferInfo lightsInfo(lightBuffer->getBuffer(), 0, kLightBufferSize);
        const vk::WriteDescriptorSet write(m_frameSets[i], 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &lightsInfo);
        m_device.updateDescriptorSets(1, &write, 0, nullptr);
    }
    return true;
}

void Renderer::writeDrawDescriptors(uint32_t frame)
{
    const FrameDrawBuffers& buffers = *m_frameDrawBuffers[frame];
    const vk::DescriptorBufferInfo drawsInfo(buffers.draws.getBuffer(), 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo transformsInfo(buffers.transforms.getBuffer(), 0, VK_WHOLE_SIZE);
    const vk::WriteDescriptorSet writes[2] = {
        vk::WriteDescriptorSet(m_frameSets[frame], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &drawsInfo),
        vk::WriteDescriptorSet(m_frameSets[frame], 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &transformsInfo),
    };
    m_device.updateDescriptorSets(2, writes, 0, nullptr);
}

void Renderer::writeImageDescriptors()
{
    // Before createDescriptors() ran there is nothing to write; it calls this itself once the sets exist.
    if (!m_lightingSet || !m_shadowMap.view || !m_lightShadowMap.view)
        return;

    constexpr size_t kMaxWrites = 16 + kMaxBloomMips;
    std::array<vk::DescriptorImageInfo, kMaxWrites> infos;
    std::array<vk::WriteDescriptorSet, kMaxWrites> writes;
    uint32_t count = 0;
    auto add = [&](vk::DescriptorSet set, uint32_t binding, vk::Sampler sampler, vk::ImageView view, vk::ImageLayout layout) {
        infos[count] = { sampler, view, layout };
        writes[count] = { set, binding, 0, 1, vk::DescriptorType::eCombinedImageSampler, &infos[count] };
        ++count;
    };
    const auto sampled = vk::ImageLayout::eShaderReadOnlyOptimal;
    const auto shadowLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    add(m_lightingSet, 0, m_shadowMap.sampler, m_shadowMap.view, shadowLayout);
    add(m_lightingSet, 1, m_shadowMap.depthSampler, m_shadowMap.view, shadowLayout);
    add(m_lightingSet, 2, m_nearestSampler, m_aoRaw.view, sampled);
    add(m_lightingSet, 3, m_nearestSampler, m_aoDepth.view, sampled);
    add(m_lightingSet, 4, m_skySampler, m_skyLut.view, sampled);
    add(m_lightingSet, 5, m_skySampler, m_skyIrradiance.view, sampled);
    add(m_lightingSet, 6, m_lightShadowMap.sampler, m_lightShadowMap.view, shadowLayout);
    add(m_sceneDepthSet, 0, m_nearestSampler, m_sceneDepth.view, vk::ImageLayout::eDepthReadOnlyOptimal);
    add(m_selectionMaskSet, 0, m_nearestSampler, m_selectionMask.view, sampled);
    add(m_hdrSet, 0, m_linearClampSampler, m_hdrColor.view, sampled);
    add(m_ldrSet, 0, m_linearClampSampler, m_ldrColor.view, sampled);
    add(m_aoDepthSet, 0, m_nearestSampler, m_aoDepth.view, sampled);
    add(m_aoRawSet, 0, m_nearestSampler, m_aoRaw.view, sampled);
    add(m_aoTempSet, 0, m_nearestSampler, m_aoTemp.view, sampled);
    add(m_skyLutSet, 0, m_skySampler, m_skyLut.view, sampled);
    // Sets past the current chain length are never used; they just need a valid view.
    for (uint32_t i = 0; i < kMaxBloomMips; ++i)
        add(m_bloomSets[i], 0, m_linearClampSampler, m_bloomMipViews[std::min(i, m_bloomMips - 1)], sampled);
    m_device.updateDescriptorSets(count, writes.data(), 0, nullptr);
}

bool Renderer::createShaders()
{
    // Main: set 0 = frame data, sets 1-3 = base color / normal / metallic-roughness, set 4 = lighting.
    const vk::DescriptorSetLayout meshLayouts[] = {
        m_frameSetLayout, m_textureSetLayout, m_textureSetLayout, m_textureSetLayout, m_lightingSetLayout,
    };
    // Depth-only (shadow, prepass): set 0 = frame data, set 1 = base color for alpha testing.
    const vk::DescriptorSetLayout depthLayouts[] = { m_frameSetLayout, m_textureSetLayout };
    const vk::DescriptorSetLayout frameOnlyLayouts[] = { m_frameSetLayout };
    const vk::DescriptorSetLayout fxLayouts[] = { m_frameSetLayout, m_textureSetLayout, m_textureSetLayout, m_textureSetLayout };

    const vk::PushConstantRange shadowPushRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(ShadowPushConstants) };
    const vk::PushConstantRange gizmoPushRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4) };
    const vk::PushConstantRange fxPushRange{ vk::ShaderStageFlagBits::eFragment, 0, kFxPushConstantSize };

    try {
        m_meshShaders = createShaderPair(m_device, "shaders/triangle.vert.spv", "shaders/triangle.frag.spv", meshLayouts);
        // Same layout as the main mesh shaders, so the scene pass binds descriptor sets the same way.
        m_classicShaders = createShaderPair(m_device, "shaders/classic.vert.spv", "shaders/classic.frag.spv", meshLayouts);
        m_prepassShaders = createShaderPair(m_device, "shaders/prepass.vert.spv", "shaders/alpha_test.frag.spv", depthLayouts);
        m_shadowShaders = createShaderPair(m_device, "shaders/shadow.vert.spv", "shaders/alpha_test.frag.spv",
            depthLayouts, { &shadowPushRange, 1 });
        m_gizmoShaders = createShaderPair(m_device, "shaders/gizmo.vert.spv", "shaders/gizmo.frag.spv",
            frameOnlyLayouts, { &gizmoPushRange, 1 });
        m_maskShaders = createShaderPair(m_device, "shaders/mask.vert.spv", "shaders/mask.frag.spv",
            fxLayouts, { &fxPushRange, 1 });
        m_levelGridShaders = createShaderPair(m_device, "shaders/level_grid.vert.spv", "shaders/level_grid.frag.spv",
            fxLayouts, { &fxPushRange, 1 });

        const std::pair<ShaderPair*, const char*> fullscreen[] = {
            { &m_skyShaders, "shaders/sky.frag.spv" },
            { &m_gridShaders, "shaders/grid.frag.spv" },
            { &m_outlineShaders, "shaders/outline.frag.spv" },
            { &m_aoDepthShaders, "shaders/ao_depth.frag.spv" },
            { &m_aoShaders, "shaders/ao.frag.spv" },
            { &m_aoBlurShaders, "shaders/ao_blur.frag.spv" },
            { &m_bloomDownShaders, "shaders/bloom_down.frag.spv" },
            { &m_bloomUpShaders, "shaders/bloom_up.frag.spv" },
            { &m_compositeShaders, "shaders/composite.frag.spv" },
            { &m_fxaaShaders, "shaders/fxaa.frag.spv" },
            { &m_skyLutShaders, "shaders/sky_lut.frag.spv" },
            { &m_skyIrradianceShaders, "shaders/sky_irradiance.frag.spv" },
            { &m_classicSkyShaders, "shaders/classic_sky.frag.spv" },
            { &m_classicPresentShaders, "shaders/classic_present.frag.spv" },
        };
        for (const auto& [pair, fragment] : fullscreen)
            *pair = createShaderPair(m_device, "shaders/fullscreen.vert.spv", fragment, fxLayouts, { &fxPushRange, 1 });
    }
    catch (const std::exception& e) {
        LOG_ERROR("Failed to load shaders: " << e.what() << "\n");
        return false;
    }

    vk::PipelineLayoutCreateInfo meshLayoutInfo{};
    meshLayoutInfo.setSetLayouts(meshLayouts);
    vk::PipelineLayoutCreateInfo prepassLayoutInfo{};
    prepassLayoutInfo.setSetLayouts(depthLayouts);
    vk::PipelineLayoutCreateInfo shadowLayoutInfo{};
    shadowLayoutInfo.setSetLayouts(depthLayouts).setPushConstantRanges(shadowPushRange);
    vk::PipelineLayoutCreateInfo gizmoLayoutInfo{};
    gizmoLayoutInfo.setSetLayouts(frameOnlyLayouts).setPushConstantRanges(gizmoPushRange);
    vk::PipelineLayoutCreateInfo fxLayoutInfo{};
    fxLayoutInfo.setSetLayouts(fxLayouts).setPushConstantRanges(fxPushRange);

    return takeResult(m_device.createPipelineLayout(meshLayoutInfo), m_meshLayout, "mesh pipeline layout") &&
        takeResult(m_device.createPipelineLayout(prepassLayoutInfo), m_prepassLayout, "prepass pipeline layout") &&
        takeResult(m_device.createPipelineLayout(shadowLayoutInfo), m_shadowLayout, "shadow pipeline layout") &&
        takeResult(m_device.createPipelineLayout(gizmoLayoutInfo), m_gizmoLayout, "gizmo pipeline layout") &&
        takeResult(m_device.createPipelineLayout(fxLayoutInfo), m_fxLayout, "effects pipeline layout");
}

bool Renderer::createLineBuffer(const std::vector<GizmoVertex>& vertices, LineBuffer& out)
{
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = sizeof(GizmoVertex) * vertices.size();
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    LineBuffer created;
    if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &created.buffer, &created.allocation, nullptr) != VK_SUCCESS) {
        LOG_ERROR("Failed to create line vertex buffer\n");
        return false;
    }
    void* mapped = nullptr;
    if (vmaMapMemory(m_allocator, created.allocation, &mapped) != VK_SUCCESS) {
        LOG_ERROR("Failed to map line vertex buffer\n");
        vmaDestroyBuffer(m_allocator, created.buffer, created.allocation);
        return false;
    }
    memcpy(mapped, vertices.data(), bufferInfo.size);
    vmaUnmapMemory(m_allocator, created.allocation);

    created.vertexCount = static_cast<uint32_t>(vertices.size());
    out = created;
    return true;
}

void Renderer::destroyLineBuffer(LineBuffer& buffer)
{
    if (buffer.buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(m_allocator, buffer.buffer, buffer.allocation);
    buffer = {};
}

bool Renderer::initImGuiBackend()
{
    // ImGui recreates its font atlas texture when the atlas grows and frees the old set a few frames later,
    // so several sets can be live at once. A failed allocation leaves ImGui writing to a garbage handle.
    constexpr uint32_t kImGuiMaxSets = 16;
    const vk::DescriptorPoolSize imguiPoolSize{ vk::DescriptorType::eCombinedImageSampler, kImGuiMaxSets };
    vk::DescriptorPoolCreateInfo imguiPoolInfo{ vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, kImGuiMaxSets, 1, &imguiPoolSize };
    if (!takeResult(m_device.createDescriptorPool(imguiPoolInfo), m_imguiDescriptorPool, "ImGui descriptor pool"))
        return false;

    // ImGui draws in the single-sample final pass, which has only the swapchain color attachment.
    const VkFormat colorFormat = static_cast<VkFormat>(m_swapchain.format());
    VkPipelineRenderingCreateInfoKHR renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

    const VkInstance instance = m_context->instance();
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.ApiVersion = m_context->apiVersion();
    initInfo.Instance = instance;
    initInfo.PhysicalDevice = m_context->physicalDevice();
    initInfo.Device = m_device;
    initInfo.QueueFamily = m_context->graphicsQueueFamily();
    initInfo.Queue = m_context->graphicsQueue();
    initInfo.DescriptorPool = m_imguiDescriptorPool;
    initInfo.MinImageCount = m_swapchain.imageCount();
    initInfo.ImageCount = m_swapchain.imageCount();
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.RenderPass = VK_NULL_HANDLE;
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingInfo;
    initInfo.CheckVkResultFn = [](VkResult result) {
        if (result != VK_SUCCESS)
            LOG_ERROR("ImGui Vulkan backend call failed: " << static_cast<int>(result) << "\n");
    };
    // Built with IMGUI_IMPL_VULKAN_NO_PROTOTYPES, so the backend loads its entry points through volk's loader.
    // ImGui tries the core dynamic rendering commands before the KHR ones. On a 1.2 device the instance-level
    // vkCmdBeginRendering exists but leads nowhere; the device-level lookup fails instead, so ImGui falls back.
    const bool loaded = ImGui_ImplVulkan_LoadFunctions(m_context->apiVersion(), [](const char* functionName, void* userData) {
        const auto* context = static_cast<const VulkanContext*>(userData);
        if (std::strcmp(functionName, "vkCmdBeginRendering") == 0 || std::strcmp(functionName, "vkCmdEndRendering") == 0)
            return vkGetDeviceProcAddr(context->device(), functionName);
        return vkGetInstanceProcAddr(context->instance(), functionName);
    }, m_context);
    if (!loaded || !ImGui_ImplVulkan_Init(&initInfo)) {
        LOG_ERROR("Failed to initialize the ImGui Vulkan backend\n");
        return false;
    }
    m_imguiInitialized = true;
    return true;
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

void Renderer::shutdown()
{
    if (!m_device)
        return;
    (void)m_device.waitIdle();

    if (m_imguiInitialized) {
        ImGui_ImplVulkan_Shutdown();
        m_imguiInitialized = false;
    }
    if (m_imguiDescriptorPool) {
        m_device.destroyDescriptorPool(m_imguiDescriptorPool);
        m_imguiDescriptorPool = nullptr;
    }

    destroyLineBuffer(m_pathLines);

    for (ShaderPair* pair : { &m_meshShaders, &m_prepassShaders, &m_shadowShaders, &m_gizmoShaders, &m_skyShaders,
             &m_gridShaders, &m_maskShaders, &m_levelGridShaders, &m_outlineShaders, &m_aoDepthShaders, &m_aoShaders, &m_aoBlurShaders,
             &m_bloomDownShaders, &m_bloomUpShaders, &m_compositeShaders, &m_fxaaShaders, &m_skyLutShaders,
             &m_skyIrradianceShaders, &m_classicShaders, &m_classicSkyShaders, &m_classicPresentShaders })
        destroyShaderPair(m_device, *pair);
    for (vk::PipelineLayout* layout : { &m_meshLayout, &m_prepassLayout, &m_shadowLayout, &m_gizmoLayout, &m_fxLayout }) {
        if (*layout) m_device.destroyPipelineLayout(*layout);
        *layout = nullptr;
    }

    m_frameDrawBuffers.clear();
    m_lightBuffers.clear();
    for (UBOBuffer& ubo : m_frameUBOs)
        if (ubo.buffer) vmaDestroyBuffer(m_allocator, ubo.buffer, ubo.allocation);
    m_frameUBOs.clear();

    for (vk::Semaphore semaphore : m_imageAvailableSemaphores)
        if (semaphore) m_device.destroySemaphore(semaphore);
    for (vk::Semaphore semaphore : m_renderFinishedSemaphores)
        if (semaphore) m_device.destroySemaphore(semaphore);
    for (vk::Fence fence : m_inFlightFences)
        if (fence) m_device.destroyFence(fence);
    m_imageAvailableSemaphores.clear();
    m_renderFinishedSemaphores.clear();
    m_inFlightFences.clear();

    // Destroying the pool frees every set allocated from it, including ModelManager's.
    if (m_descriptorPool) m_device.destroyDescriptorPool(m_descriptorPool);
    m_descriptorPool = nullptr;
    m_frameSets.clear();
    for (vk::DescriptorSet* set : { &m_lightingSet, &m_sceneDepthSet, &m_selectionMaskSet, &m_hdrSet, &m_ldrSet,
             &m_aoDepthSet, &m_aoRawSet, &m_aoTempSet, &m_skyLutSet })
        *set = nullptr;
    m_bloomSets = {};
    for (vk::DescriptorSetLayout* layout : { &m_frameSetLayout, &m_textureSetLayout, &m_lightingSetLayout }) {
        if (*layout) m_device.destroyDescriptorSetLayout(*layout);
        *layout = nullptr;
    }
    for (vk::Sampler* sampler : { &m_textureSampler, &m_nearestSampler, &m_linearClampSampler, &m_skySampler }) {
        if (*sampler) m_device.destroySampler(*sampler);
        *sampler = nullptr;
    }

    if (m_commandPool) m_device.destroyCommandPool(m_commandPool);
    m_commandPool = nullptr;
    m_commandBuffers.clear();

    destroyRenderTargets();
    destroySkyResources();
    destroyShadowMap(m_shadowMap, m_allocator, m_device);
    destroyShadowMap(m_lightShadowMap, m_allocator, m_device);
    m_swapchain.destroy();

    m_device = nullptr;
    m_allocator = VK_NULL_HANDLE;
    m_context = nullptr;
    m_window = nullptr;
}

// ---------------------------------------------------------------------------
// Settings / swapchain / path lines
// ---------------------------------------------------------------------------

void Renderer::applySettings(const GraphicsSettings& requested)
{
    const GraphicsSettings settings = sanitizeGraphicsSettings(requested);
    if (settings == m_settings || !m_device)
        return;
    const GraphicsSettings old = m_settings;
    m_settings = settings;
    updateSun();

    if (old.vsync != settings.vsync)
        m_swapchainDirty = true;

    if (effectiveSampleCount() != m_samples) {
        (void)m_device.waitIdle();
        destroyRenderTargets();
        if (!createRenderTargets())
            LOG_ERROR("Failed to recreate render targets for MSAA " << settings.msaaSamples << "x\n");
    }

    if (wantedShadowMap() != std::pair(m_shadowMap.size, m_shadowMap.layers)) {
        (void)m_device.waitIdle();
        destroyShadowMap(m_shadowMap, m_allocator, m_device);
        try {
            createShadowMapResources();
        }
        catch (const std::exception& e) {
            LOG_ERROR(e.what() << "; falling back to 1024\n");
            m_settings.shadowMapSize = 1024;
            createShadowMapResources();
        }
    }
    // Sun direction, shadow distance and caster changes are caught by the per-cascade hashes.
    if (old.shadows != settings.shadows)
        m_shadowMapValid = false;
    // Grows again when a light asks for shadows (gatherLights()).
    if (!lightShadowsAllowed())
        resizeLightShadowMap(false);
}

void Renderer::updateSun()
{
    m_sun.direction = sunDirectionFromAngles(m_settings.sunAzimuth, m_settings.sunElevation);
    m_sun.topIrradiance = srgbToLinear(m_settings.sunColor) * (m_settings.sunIntensity * kSunIrradiance);
    m_sun.groundIrradiance = m_settings.sun
        ? m_sun.topIrradiance * sunTransmittance(-m_sun.direction, m_settings.haze)
        : glm::vec3(0.0f);
}

bool Renderer::shadowsActive() const
{
    // Nothing to shadow once the sun is off, set or has no intensity.
    return m_settings.shadows && !classic() && glm::dot(m_sun.groundIrradiance, glm::vec3(1.0f)) > 1e-4f;
}

bool Renderer::contactShadowsActive() const
{
    return m_settings.contactShadows && !classic() && glm::dot(m_sun.groundIrradiance, glm::vec3(1.0f)) > 1e-4f;
}

bool Renderer::depthPrepassEnabled() const
{
    return (m_settings.ambientOcclusion > 0 && !classic()) || contactShadowsActive();
}

uint64_t Renderer::skyHash() const
{
    Hasher hash;
    const float values[] = { m_sun.direction.x, m_sun.direction.y, m_sun.direction.z, m_sun.topIrradiance.r,
        m_sun.topIrradiance.g, m_sun.topIrradiance.b, m_settings.haze, kGroundAlbedo };
    hash.bytes(values, sizeof(values));
    return hash.value();
}

bool Renderer::prepareSwapchain()
{
    if (!m_swapchainDirty)
        return true;
    if (!recreateSwapchain())
        return false;
    m_swapchainDirty = false;
    return true;
}

bool Renderer::recreateSwapchain()
{
    int pixelWidth = 0, pixelHeight = 0;
    SDL_GetWindowSizeInPixels(m_window, &pixelWidth, &pixelHeight);
    if (pixelWidth <= 0 || pixelHeight <= 0)
        return false;

    // Waits for the device to go idle, so the old render targets below are no longer in use.
    if (!m_swapchain.recreate(static_cast<uint32_t>(pixelWidth), static_cast<uint32_t>(pixelHeight), m_settings.vsync))
        return false;

    while (m_renderFinishedSemaphores.size() < m_swapchain.imageCount()) {
        vk::Semaphore semaphore;
        if (!takeResult(m_device.createSemaphore({}), semaphore, "semaphore"))
            return false;
        m_renderFinishedSemaphores.push_back(semaphore);
    }

    destroyRenderTargets();
    return createRenderTargets();
}

void Renderer::setPathLines(const std::vector<GizmoVertex>& vertices)
{
    if (m_pathLines.buffer != VK_NULL_HANDLE) {
        // Frames still in flight may be reading the old buffer.
        (void)m_device.waitIdle();
        destroyLineBuffer(m_pathLines);
    }
    if (!vertices.empty())
        createLineBuffer(vertices, m_pathLines);
}
