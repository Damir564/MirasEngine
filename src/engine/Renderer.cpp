#include "Renderer.h"
#include "ModelManager.h"
#include "Vertex.h"
#include "VulkanContext.h"
#include <volk.h>
#include <SDL3/SDL.h>
#include <glm/ext/matrix_transform.hpp>
#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <execution>
#include <iostream>
#include <numeric>

namespace {

constexpr uint8_t kVisibleMain = 1;
constexpr uint8_t kVisibleShadow = 2;
constexpr size_t kCullChunkSize = 1024;
constexpr float kShadowBias = 0.0015f;
// Unity's selection orange; occluded parts of the outline are drawn fainter.
constexpr glm::vec3 kOutlineColor{ 1.0f, 0.4f, 0.0f };
constexpr float kOutlineOccludedAlpha = 0.4f;
constexpr float kOutlineWidthPixels = 2.0f;

const vk::VertexInputBindingDescription2EXT kLineBinding{ 0, sizeof(GizmoVertex), vk::VertexInputRate::eVertex, 1 };
const std::array<vk::VertexInputAttributeDescription2EXT, 2> kLineAttributes = { {
    { 0, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(GizmoVertex, position)) },
    { 1, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(GizmoVertex, color)) },
} };

template <typename T>
bool takeResult(vk::ResultValue<T>&& created, T& out, const char* what)
{
    if (created.result != vk::Result::eSuccess) {
        std::cerr << "Failed to create " << what << ": " << vk::to_string(created.result) << "\n";
        return false;
    }
    out = std::move(created.value);
    return true;
}

// FNV-1a, used to detect frames where the shadow map would come out identical.
class ShadowHasher {
public:
    void bytes(const void* data, size_t size) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
            byte(p[i]);
    }
    void byte(uint8_t value) {
        m_hash ^= value;
        m_hash *= 1099511628211ull;
    }
    uint64_t value() const { return m_hash; }

private:
    uint64_t m_hash = 1469598103934665603ull;
};

vk::ImageMemoryBarrier2 imageBarrier(vk::Image image, vk::ImageAspectFlags aspect,
    vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
    vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess,
    vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    vk::ImageMemoryBarrier2 barrier{};
    barrier.setSrcStageMask(srcStage)
        .setSrcAccessMask(srcAccess)
        .setDstStageMask(dstStage)
        .setDstAccessMask(dstAccess)
        .setOldLayout(oldLayout)
        .setNewLayout(newLayout)
        .setImage(image)
        .setSubresourceRange({ aspect, 0, 1, 0, 1 });
    return barrier;
}

void pipelineBarriers(vk::CommandBuffer cmd, std::span<const vk::ImageMemoryBarrier2> barriers)
{
    vk::DependencyInfo info{};
    info.setImageMemoryBarrierCount(static_cast<uint32_t>(barriers.size()))
        .setPImageMemoryBarriers(barriers.data());
    cmd.pipelineBarrier2(info);
}

vk::SampleCountFlagBits toSampleCount(int samples)
{
    switch (samples) {
    case 8: return vk::SampleCountFlagBits::e8;
    case 4: return vk::SampleCountFlagBits::e4;
    case 2: return vk::SampleCountFlagBits::e2;
    default: return vk::SampleCountFlagBits::e1;
    }
}

// Perspective near/far recovered from a right-handed zero-to-one projection matrix.
glm::vec2 projectionDepthRange(const glm::mat4& proj)
{
    const float a = proj[2][2];
    const float b = proj[3][2];
    if (std::abs(a) < 1e-12f || std::abs(a + 1.0f) < 1e-12f)
        return { 0.1f, 1000.0f };
    return { b / a, b / (a + 1.0f) };
}

} // namespace

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

        if (!createRenderTargets() || !createFrameResources() || !createDescriptors() ||
            !createShaders() || !initImGuiBackend()) {
            shutdown();
            return false;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Renderer initialization failed: " << e.what() << "\n";
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
    m_capabilities.maxAnisotropy = limits.maxSamplerAnisotropy;
    m_maxDrawIndirectCount = limits.maxDrawIndirectCount;
}

vk::SampleCountFlagBits Renderer::effectiveSampleCount() const
{
    return toSampleCount(std::min(m_settings.msaaSamples, m_capabilities.maxMsaaSamples));
}

bool Renderer::createRenderImage(vk::Format format, vk::ImageUsageFlags usage, vk::SampleCountFlagBits samples,
    vk::ImageAspectFlags aspect, RenderImage& out)
{
    const vk::Extent2D extent = m_swapchain.extent();

    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent = vk::Extent3D{ extent.width, extent.height, 1 };
    imageInfo.mipLevels = 1;
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
        std::cerr << "Failed to create render target image\n";
        out = {};
        return false;
    }

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = vk::Image(out.image);
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = { aspect, 0, 1, 0, 1 };
    if (!takeResult(m_device.createImageView(viewInfo), out.view, "render target view")) {
        destroyRenderImage(out);
        return false;
    }
    return true;
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
    const auto depthAspect = vk::ImageAspectFlagBits::eDepth;
    const auto colorAspect = vk::ImageAspectFlagBits::eColor;

    if (!createRenderImage(kDepthFormat,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::SampleCountFlagBits::e1, depthAspect, m_sceneDepth) ||
        !createRenderImage(kSelectionMaskFormat,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::SampleCountFlagBits::e1, colorAspect, m_selectionMask))
        return false;

    if (m_samples != vk::SampleCountFlagBits::e1) {
        const auto transient = vk::ImageUsageFlagBits::eTransientAttachment;
        if (!createRenderImage(m_swapchain.format(), vk::ImageUsageFlagBits::eColorAttachment | transient,
                m_samples, colorAspect, m_msaaColor) ||
            !createRenderImage(kDepthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment | transient,
                m_samples, depthAspect, m_msaaDepth))
            return false;
    }

    // The sets exist once createDescriptors() ran; on later rebuilds point them at the new images.
    if (m_sceneDepthSet && m_selectionMaskSet) {
        const vk::DescriptorImageInfo depthInfo{ m_nearestSampler, m_sceneDepth.view, vk::ImageLayout::eShaderReadOnlyOptimal };
        const vk::DescriptorImageInfo maskInfo{ m_nearestSampler, m_selectionMask.view, vk::ImageLayout::eShaderReadOnlyOptimal };
        const vk::WriteDescriptorSet writes[2] = {
            { m_sceneDepthSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &depthInfo },
            { m_selectionMaskSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &maskInfo },
        };
        m_device.updateDescriptorSets(2, writes, 0, nullptr);
    }
    return true;
}

void Renderer::destroyRenderTargets()
{
    destroyRenderImage(m_sceneDepth);
    destroyRenderImage(m_selectionMask);
    destroyRenderImage(m_msaaColor);
    destroyRenderImage(m_msaaDepth);
}

bool Renderer::createShadowMapResources()
{
    m_shadowMap = createShadowMap(m_allocator, m_device, static_cast<uint32_t>(m_settings.shadowMapSize));
    m_shadowMapValid = false;
    if (m_shadowMapSet) {
        const vk::DescriptorImageInfo shadowImageInfo{ m_shadowMap.sampler, m_shadowMap.view, vk::ImageLayout::eDepthStencilReadOnlyOptimal };
        const vk::WriteDescriptorSet shadowWrite{ m_shadowMapSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &shadowImageInfo };
        m_device.updateDescriptorSets(1, &shadowWrite, 0, nullptr);
    }
    return true;
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
            std::cerr << "Failed to create UBO buffer\n";
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
    samplerInfo.anisotropyEnable = VK_TRUE;
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

    const vk::DescriptorSetLayoutBinding textureBinding{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment };
    if (!takeResult(m_device.createDescriptorSetLayout({ {}, 1, &textureBinding }), m_textureSetLayout, "texture set layout"))
        return false;

    // Binding 0: FrameUBO, 1: per-draw data (GpuDrawData[]), 2: per-instance transforms (GpuTransform[]).
    const vk::DescriptorSetLayoutBinding frameBindings[3] = {
        { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
        { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
        { 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex },
    };
    if (!takeResult(m_device.createDescriptorSetLayout({ {}, 3, frameBindings }), m_frameSetLayout, "frame set layout"))
        return false;

    // Shared with ModelManager, which allocates one set per texture from it.
    const vk::DescriptorPoolSize poolSizes[] = {
        { vk::DescriptorType::eCombinedImageSampler, 1000 },
        { vk::DescriptorType::eUniformBuffer, m_framesInFlight + 10 },
        { vk::DescriptorType::eStorageBuffer, 2 * m_framesInFlight + 10 },
    };
    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = 1100;
    poolInfo.setPoolSizes(poolSizes);
    if (!takeResult(m_device.createDescriptorPool(poolInfo), m_descriptorPool, "descriptor pool"))
        return false;

    const vk::DescriptorSetLayout imageLayouts[3] = { m_textureSetLayout, m_textureSetLayout, m_textureSetLayout };
    std::vector<vk::DescriptorSet> imageSets;
    if (!takeResult(m_device.allocateDescriptorSets({ m_descriptorPool, 3, imageLayouts }), imageSets, "image descriptor sets"))
        return false;
    m_shadowMapSet = imageSets[0];
    m_sceneDepthSet = imageSets[1];
    m_selectionMaskSet = imageSets[2];

    // Writes the shadow map set; the render target sets are written by rebuilding the targets' views.
    if (!createShadowMapResources())
        return false;
    const vk::DescriptorImageInfo depthInfo{ m_nearestSampler, m_sceneDepth.view, vk::ImageLayout::eShaderReadOnlyOptimal };
    const vk::DescriptorImageInfo maskInfo{ m_nearestSampler, m_selectionMask.view, vk::ImageLayout::eShaderReadOnlyOptimal };
    const vk::WriteDescriptorSet targetWrites[2] = {
        { m_sceneDepthSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &depthInfo },
        { m_selectionMaskSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &maskInfo },
    };
    m_device.updateDescriptorSets(2, targetWrites, 0, nullptr);

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

bool Renderer::createShaders()
{
    // Main: set 0 = frame data, sets 1-3 = base color / normal / metallic-roughness, set 4 = shadow map.
    const vk::DescriptorSetLayout meshLayouts[] = {
        m_frameSetLayout, m_textureSetLayout, m_textureSetLayout, m_textureSetLayout, m_textureSetLayout,
    };
    // Shadow: set 0 = frame data (lightSpaceMatrix), set 1 = base color for alpha testing.
    const vk::DescriptorSetLayout shadowLayouts[] = { m_frameSetLayout, m_textureSetLayout };
    const vk::DescriptorSetLayout frameOnlyLayouts[] = { m_frameSetLayout };
    const vk::DescriptorSetLayout fxLayouts[] = { m_frameSetLayout, m_textureSetLayout };

    const vk::PushConstantRange gizmoPushRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4) };
    const vk::PushConstantRange fxPushRange{ vk::ShaderStageFlagBits::eFragment, 0, sizeof(OutlinePushConstants) };

    try {
        m_meshShaders = createShaderPair(m_device, "shaders/triangle.vert.spv", "shaders/triangle.frag.spv", meshLayouts);
        m_shadowShaders = createShaderPair(m_device, "shaders/shadow.vert.spv", "shaders/shadow.frag.spv", shadowLayouts);
        m_gizmoShaders = createShaderPair(m_device, "shaders/gizmo.vert.spv", "shaders/gizmo.frag.spv",
            frameOnlyLayouts, { &gizmoPushRange, 1 });
        m_skyShaders = createShaderPair(m_device, "shaders/fullscreen.vert.spv", "shaders/sky.frag.spv",
            fxLayouts, { &fxPushRange, 1 });
        m_gridShaders = createShaderPair(m_device, "shaders/fullscreen.vert.spv", "shaders/grid.frag.spv",
            fxLayouts, { &fxPushRange, 1 });
        m_maskShaders = createShaderPair(m_device, "shaders/mask.vert.spv", "shaders/mask.frag.spv",
            fxLayouts, { &fxPushRange, 1 });
        m_outlineShaders = createShaderPair(m_device, "shaders/fullscreen.vert.spv", "shaders/outline.frag.spv",
            fxLayouts, { &fxPushRange, 1 });
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to load shaders: " << e.what() << "\n";
        return false;
    }

    vk::PipelineLayoutCreateInfo meshLayoutInfo{};
    meshLayoutInfo.setSetLayouts(meshLayouts);
    vk::PipelineLayoutCreateInfo shadowLayoutInfo{};
    shadowLayoutInfo.setSetLayouts(shadowLayouts);
    vk::PipelineLayoutCreateInfo gizmoLayoutInfo{};
    gizmoLayoutInfo.setSetLayouts(frameOnlyLayouts).setPushConstantRanges(gizmoPushRange);
    vk::PipelineLayoutCreateInfo fxLayoutInfo{};
    fxLayoutInfo.setSetLayouts(fxLayouts).setPushConstantRanges(fxPushRange);

    return takeResult(m_device.createPipelineLayout(meshLayoutInfo), m_meshLayout, "mesh pipeline layout") &&
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
        std::cerr << "Failed to create line vertex buffer\n";
        return false;
    }
    void* mapped = nullptr;
    if (vmaMapMemory(m_allocator, created.allocation, &mapped) != VK_SUCCESS) {
        std::cerr << "Failed to map line vertex buffer\n";
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
    const vk::DescriptorPoolSize imguiPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1 };
    vk::DescriptorPoolCreateInfo imguiPoolInfo{ vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, 1, &imguiPoolSize };
    if (!takeResult(m_device.createDescriptorPool(imguiPoolInfo), m_imguiDescriptorPool, "ImGui descriptor pool"))
        return false;

    // ImGui draws in the single-sample overlay pass, which has only the swapchain color attachment.
    const VkFormat colorFormat = static_cast<VkFormat>(m_swapchain.format());
    VkPipelineRenderingCreateInfoKHR renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

    const VkInstance instance = m_context->instance();
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
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

    // Built with IMGUI_IMPL_VULKAN_NO_PROTOTYPES, so the backend loads its entry points through volk's loader.
    const bool loaded = ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_3, [](const char* functionName, void* userData) {
        return vkGetInstanceProcAddr(static_cast<VkInstance>(userData), functionName);
    }, instance);
    if (!loaded || !ImGui_ImplVulkan_Init(&initInfo)) {
        std::cerr << "Failed to initialize the ImGui Vulkan backend\n";
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

    for (ShaderPair* pair : { &m_meshShaders, &m_shadowShaders, &m_gizmoShaders, &m_skyShaders,
             &m_gridShaders, &m_maskShaders, &m_outlineShaders })
        destroyShaderPair(m_device, *pair);
    for (vk::PipelineLayout* layout : { &m_meshLayout, &m_shadowLayout, &m_gizmoLayout, &m_fxLayout }) {
        if (*layout) m_device.destroyPipelineLayout(*layout);
        *layout = nullptr;
    }

    m_frameDrawBuffers.clear();
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
    m_shadowMapSet = nullptr;
    m_sceneDepthSet = nullptr;
    m_selectionMaskSet = nullptr;
    if (m_frameSetLayout) m_device.destroyDescriptorSetLayout(m_frameSetLayout);
    if (m_textureSetLayout) m_device.destroyDescriptorSetLayout(m_textureSetLayout);
    if (m_textureSampler) m_device.destroySampler(m_textureSampler);
    if (m_nearestSampler) m_device.destroySampler(m_nearestSampler);
    m_frameSetLayout = nullptr;
    m_textureSetLayout = nullptr;
    m_textureSampler = nullptr;
    m_nearestSampler = nullptr;

    if (m_commandPool) m_device.destroyCommandPool(m_commandPool);
    m_commandPool = nullptr;
    m_commandBuffers.clear();

    destroyRenderTargets();
    destroyShadowMap(m_shadowMap, m_allocator, m_device);
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

    if (old.vsync != settings.vsync)
        m_swapchainDirty = true;

    if (effectiveSampleCount() != m_samples) {
        (void)m_device.waitIdle();
        destroyRenderTargets();
        if (!createRenderTargets())
            std::cerr << "Failed to recreate render targets for MSAA " << settings.msaaSamples << "x\n";
    }

    if (old.shadowMapSize != settings.shadowMapSize) {
        (void)m_device.waitIdle();
        destroyShadowMap(m_shadowMap, m_allocator, m_device);
        try {
            createShadowMapResources();
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "; falling back to 1024\n";
            m_settings.shadowMapSize = 1024;
            createShadowMapResources();
        }
    }
    if (old.shadows != settings.shadows || old.shadowDistance != settings.shadowDistance)
        m_shadowMapValid = false;
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

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

Renderer::FrameStatus Renderer::renderFrame(const FrameInput& input)
{
    const vk::Fence fence = m_inFlightFences[m_currentFrame];
    (void)m_device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    // Pointer overload: returns the raw result instead of asserting on eErrorOutOfDateKHR.
    const vk::Result result = m_device.acquireNextImageKHR(m_swapchain.handle(), UINT64_MAX,
        m_imageAvailableSemaphores[m_currentFrame], {}, &imageIndex);
    if (result == vk::Result::eErrorOutOfDateKHR) {
        m_swapchainDirty = true;
        return FrameStatus::Skipped;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        std::cerr << "Failed to acquireNextImageKHR\n";
        return FrameStatus::Failed;
    }
    if (result == vk::Result::eSuboptimalKHR)
        m_swapchainDirty = true;

    // Reset only once we know this frame will be submitted; otherwise the next wait would hang.
    (void)m_device.resetFences(fence);

    const FrameUBO frameData = buildFrameUBO(input);
    memcpy(m_frameUBOs[m_currentFrame].mapped, &frameData, sizeof(FrameUBO));

    FrameBatches batches;
    const uint64_t shadowHash = cullAndBatch(input, frameData.proj * frameData.view, frameData.lightSpaceMatrix, batches);
    const bool renderShadowMap = !m_shadowMapValid || shadowHash != m_lastShadowHash;
    m_lastShadowHash = shadowHash;
    m_shadowMapValid = true;

    buildDrawStreams(input, batches, renderShadowMap);
    uploadDrawStreams();

    const vk::CommandBuffer cmd = m_commandBuffers[m_currentFrame];
    (void)cmd.reset();
    (void)cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    // When skipped, the map keeps its previous contents in DepthStencilReadOnlyOptimal. With shadows
    // off it is still cleared once so the (unused) binding refers to an initialized image.
    if (renderShadowMap)
        recordShadowPass(cmd, *input.models);

    recordScenePass(cmd, imageIndex, input);
    const bool drawOutline = !m_highlightRuns.empty();
    if (drawOutline)
        recordSelectionMask(cmd, input);
    recordOverlayPass(cmd, imageIndex, input, drawOutline);

    (void)cmd.end();
    return submitAndPresent(cmd, imageIndex);
}

FrameUBO Renderer::buildFrameUBO(const FrameInput& input) const
{
    FrameUBO frameData{};
    frameData.view = input.view;
    frameData.proj = input.proj;
    frameData.invViewProj = glm::inverse(input.proj * input.view);

    // Fit the shadow map to a sphere in front of the camera, so it covers what is actually on screen
    // up to the shadow distance instead of the whole scene.
    const glm::mat4 invView = glm::inverse(input.view);
    const glm::vec3 forward = -glm::normalize(glm::vec3(invView[2]));
    const float radius = m_settings.shadowDistance * 0.5f;
    const glm::vec3 center = input.cameraPosition + forward * radius;
    frameData.lightSpaceMatrix = calculateLightSpaceMatrix(input.sun, center, radius, m_shadowMap.size);

    frameData.cameraPos = glm::vec4(input.cameraPosition, 1.0f);
    frameData.lightDir = glm::vec4(glm::normalize(input.sun.direction), 0.0f);
    frameData.sunColor = glm::vec4(input.sun.color, input.sun.intensity * 3.0f);
    frameData.skyZenith = glm::vec4(0.16f, 0.34f, 0.72f, 0.0f);
    frameData.skyHorizon = glm::vec4(0.62f, 0.74f, 0.88f, 0.0f);
    frameData.groundColor = glm::vec4(0.28f, 0.26f, 0.23f, 0.0f);

    const glm::vec2 depthRange = projectionDepthRange(input.proj);
    frameData.fogParams = glm::vec4(m_settings.fog ? 1.0f : 0.0f, depthRange.y * 0.6f, 0.0f, 0.0f);
    frameData.shadowParams = glm::vec4(kShadowBias, m_settings.shadows ? 1.0f : 0.0f, m_settings.shadowDistance,
        1.0f / static_cast<float>(std::max(m_shadowMap.size, 1u)));
    frameData.time = input.time;
    frameData.nearPlane = depthRange.x;
    frameData.farPlane = depthRange.y;
    return frameData;
}

uint64_t Renderer::cullAndBatch(const FrameInput& input, const glm::mat4& viewProj, const glm::mat4& lightSpaceMatrix,
    FrameBatches& batches)
{
    ModelManager& models = *input.models;
    const glm::vec3 cameraPos = input.cameraPosition;
    const bool shadows = m_settings.shadows;
    const float shadowDistance = m_settings.shadowDistance;
    const auto& instances = models.getInstances();
    m_cullResults.resize(instances.size());
    m_cullIndices.resize(instances.size());
    std::iota(m_cullIndices.begin(), m_cullIndices.end(), 0);

    // 1. Per-instance phase: transform + whole-model culling.
    std::for_each(std::execution::par, m_cullIndices.begin(), m_cullIndices.end(), [&](size_t i) {
        const auto& inst = instances[i];
        auto& res = m_cullResults[i];

        res.instance = &inst;
        res.modelIndex = inst.modelIndex;
        res.gpuModel = nullptr;
        res.visibleMain = false;
        res.visibleShadow = false;

        if (!inst.visible) return;

        GPUModel* gpuModel = models.getModel(inst.modelIndex);
        if (!gpuModel || !gpuModel->isValid()) return;

        res.gpuModel = gpuModel;
        res.transform = inst.getTransformMatrix();

        res.mainPlanes = extractFrustumPlanes(viewProj * res.transform);
        res.visibleMain = isAABBInFrustum(res.mainPlanes, gpuModel->boundsMin, gpuModel->boundsMax);

        if (shadows) {
            glm::vec3 worldCenter = glm::vec3(res.transform * glm::vec4(gpuModel->boundsCenter, 1.0f));
            float maxScale = std::max({ inst.scale.x, inst.scale.y, inst.scale.z });
            float distToCamera = glm::distance(worldCenter, cameraPos) - (gpuModel->boundsRadius * maxScale);
            if (distToCamera < shadowDistance) {
                res.shadowPlanes = extractFrustumPlanes(lightSpaceMatrix * res.transform);
                res.visibleShadow = isAABBInFrustum(res.shadowPlanes, gpuModel->boundsMin, gpuModel->boundsMax);
            }
        }

        res.submeshFlags.assign(gpuModel->submeshes.size(), 0);
    });

    // 2. Per-submesh phase, split into chunks so a single huge model still uses all cores.
    m_cullChunks.clear();
    for (size_t i = 0; i < m_cullResults.size(); ++i) {
        const auto& res = m_cullResults[i];
        if (!res.gpuModel || !(res.visibleMain || res.visibleShadow)) continue;
        const size_t n = res.gpuModel->submeshes.size();
        for (size_t b = 0; b < n; b += kCullChunkSize)
            m_cullChunks.push_back({ i, b, std::min(b + kCullChunkSize, n) });
    }

    std::for_each(std::execution::par, m_cullChunks.begin(), m_cullChunks.end(), [&](const CullChunk& chunk) {
        auto& res = m_cullResults[chunk.instanceIdx];
        const auto& submeshes = res.gpuModel->submeshes;
        const IfcScene* ifc = res.instance->ifcScene ? &*res.instance->ifcScene : nullptr;

        for (size_t s = chunk.begin; s < chunk.end; ++s) {
            if (ifc && !ifc->isSubmeshVisible(s)) continue;

            const SubmeshInfo& sub = submeshes[s];
            const bool validBounds = sub.boundsMin.x <= sub.boundsMax.x;
            uint8_t flags = 0;
            if (res.visibleMain &&
                (!validBounds || isAABBInFrustum(res.mainPlanes, sub.boundsMin, sub.boundsMax)))
                flags |= kVisibleMain;
            if (res.visibleShadow && sub.material.alphaMode != AlphaMode::BLEND &&
                (!validBounds || isAABBInFrustum(res.shadowPlanes, sub.boundsMin, sub.boundsMax)))
                flags |= kVisibleShadow;
            res.submeshFlags[s] = flags;
        }
    });

    // 3. Sequential aggregation, hashing everything that affects the shadow map on the way.
    ShadowHasher shadowHash;
    shadowHash.byte(static_cast<uint8_t>(shadows ? 1 : 0));
    shadowHash.bytes(&m_shadowMap.size, sizeof(m_shadowMap.size));
    if (shadows)
        shadowHash.bytes(&lightSpaceMatrix, sizeof(glm::mat4));

    m_frameTransforms.clear();
    for (const auto& res : m_cullResults) {
        if (!res.gpuModel || !(res.visibleMain || res.visibleShadow)) continue;

        const uint32_t transformIndex = pushTransform(res.transform);
        InstanceRenderData renderData{ res.instance, res.transform, res.submeshFlags.data(), transformIndex };

        if (res.visibleMain) {
            auto& batch = batches.main[res.modelIndex];
            batch.model = res.gpuModel;
            batch.instances.push_back(renderData);
        }

        if (res.visibleShadow) {
            auto& batch = batches.shadow[res.modelIndex];
            batch.model = res.gpuModel;
            batch.instances.push_back(renderData);

            shadowHash.bytes(&res.gpuModel, sizeof(res.gpuModel));
            shadowHash.bytes(&res.transform, sizeof(glm::mat4));
            for (uint8_t f : res.submeshFlags)
                shadowHash.byte(f & kVisibleShadow);
        }
    }
    return shadowHash.value();
}

void Renderer::materialSets(const ModelManager& models, const GPUModel* model, const Material& material,
    vk::DescriptorSet out[3]) const
{
    auto pick = [&](int texIndex, vk::DescriptorSet fallback) {
        return (texIndex >= 0 && texIndex < static_cast<int>(model->textureDescriptorSets.size()))
            ? model->textureDescriptorSets[texIndex] : fallback;
    };
    out[0] = pick(material.baseColorTextureIndex, models.getDefaultBaseColorSet());
    out[1] = pick(material.normalTextureIndex, models.getDefaultNormalSet());
    out[2] = pick(material.metallicRoughnessTextureIndex, models.getDefaultMRSet());
}

uint32_t Renderer::pushTransform(const glm::mat4& transform)
{
    m_frameTransforms.push_back({ transform, glm::mat4(glm::transpose(glm::inverse(glm::mat3(transform)))) });
    return static_cast<uint32_t>(m_frameTransforms.size() - 1);
}

uint32_t Renderer::pushDrawData(const SubmeshInfo& sub, uint32_t transformIndex, const glm::vec3& tint)
{
    GpuDrawData d{};
    d.baseColor = sub.material.baseColorFactor * glm::vec4(tint, 1.0f);
    d.transformIndex = transformIndex;
    d.alphaMode = static_cast<int32_t>(sub.material.alphaMode);
    d.metallic = sub.material.metallicFactor;
    d.roughness = sub.material.roughnessFactor;
    d.alphaCutoff = sub.material.alphaCutoff;
    m_frameDraws.push_back(d);
    return static_cast<uint32_t>(m_frameDraws.size() - 1);
}

void Renderer::appendDraw(std::vector<DrawRun>& runs, GPUModel* model, const vk::DescriptorSet sets[3], bool blend,
    const SubmeshInfo& sub, uint32_t transformIndex, const glm::vec3& tint)
{
    const uint32_t commandIndex = static_cast<uint32_t>(m_frameCommands.size());
    const uint32_t drawIndex = pushDrawData(sub, transformIndex, tint);
    m_frameCommands.push_back(vk::DrawIndexedIndirectCommand(
        sub.indexCount, 1, sub.indexOffset, static_cast<int32_t>(sub.vertexOffset), drawIndex));

    if (!runs.empty()) {
        DrawRun& run = runs.back();
        bool compatible = run.model == model && run.blend == blend &&
            run.firstCommand + run.commandCount == commandIndex;
        for (int k = 0; k < 3 && compatible; ++k)
            compatible = !sets[k] || !run.sets[k] || sets[k] == run.sets[k];
        if (compatible) {
            for (int k = 0; k < 3; ++k)
                if (!run.sets[k]) run.sets[k] = sets[k];
            ++run.commandCount;
            return;
        }
    }
    runs.push_back({ model, { sets[0], sets[1], sets[2] }, blend, commandIndex, 1 });
}

// Every visible (instance, submesh) pair becomes one GpuDrawData entry plus one indirect command whose
// firstInstance points at that entry; compatible neighbours are merged into DrawRuns.
void Renderer::buildDrawStreams(const FrameInput& input, FrameBatches& batches, bool renderShadowMap)
{
    ModelManager& models = *input.models;
    const glm::vec3 cameraPos = input.cameraPosition;

    m_frameDraws.clear();
    m_frameCommands.clear();
    m_shadowRuns.clear();
    m_opaqueRuns.clear();
    m_blendRuns.clear();

    if (renderShadowMap) {
        for (auto& [modelIdx, batch] : batches.shadow) {
            GPUModel* model = batch.model;
            for (uint32_t si : model->drawOrder) {
                const SubmeshInfo& sub = model->submeshes[si];
                // Only alpha-masked submeshes need a specific texture in the shadow pass.
                vk::DescriptorSet sets[3] = {};
                if (sub.material.alphaMode == AlphaMode::MASK) {
                    vk::DescriptorSet all[3];
                    materialSets(models, model, sub.material, all);
                    sets[0] = all[0];
                }
                for (const auto& rd : batch.instances)
                    if (rd.submeshFlags[si] & kVisibleShadow)
                        appendDraw(m_shadowRuns, model, sets, false, sub, rd.transformIndex, glm::vec3(1.0f));
            }
        }
    }

    for (auto& [modelIdx, batch] : batches.main) {
        GPUModel* model = batch.model;
        for (uint32_t si : model->drawOrder) {
            const SubmeshInfo& sub = model->submeshes[si];
            if (sub.material.alphaMode == AlphaMode::BLEND) continue;
            vk::DescriptorSet sets[3];
            materialSets(models, model, sub.material, sets);
            for (const auto& rd : batch.instances)
                if (rd.submeshFlags[si] & kVisibleMain)
                    appendDraw(m_opaqueRuns, model, sets, false, sub, rd.transformIndex, rd.instance->color);
        }
    }

    std::vector<InstanceRenderData> sortedInstances;
    for (auto& [modelIdx, batch] : batches.main) {
        GPUModel* model = batch.model;
        bool sorted = false;
        for (std::size_t si = 0; si < model->submeshes.size(); ++si) {
            const SubmeshInfo& sub = model->submeshes[si];
            if (sub.material.alphaMode != AlphaMode::BLEND) continue;

            if (!sorted) {
                sortedInstances = batch.instances;
                std::sort(sortedInstances.begin(), sortedInstances.end(), [&](const InstanceRenderData& a, const InstanceRenderData& b) {
                    return glm::distance(cameraPos, a.instance->position) > glm::distance(cameraPos, b.instance->position);
                });
                sorted = true;
            }

            vk::DescriptorSet sets[3];
            materialSets(models, model, sub.material, sets);
            for (const auto& rd : sortedInstances)
                if (rd.submeshFlags[si] & kVisibleMain)
                    appendDraw(m_blendRuns, model, sets, true, sub, rd.transformIndex, rd.instance->color);
        }
    }

    buildHighlightStream(input);
}

// The selection mask draws the highlighted submeshes whether or not they passed culling, so the
// outline of partly off-screen objects stays correct.
void Renderer::buildHighlightStream(const FrameInput& input)
{
    m_highlightRuns.clear();
    const SelectionHighlight& highlight = input.highlight;
    ModelManager& models = *input.models;
    const auto& instances = models.getInstances();
    if (highlight.instance < 0 || highlight.instance >= static_cast<int>(instances.size()))
        return;
    const ModelInstance& inst = instances[highlight.instance];
    GPUModel* model = models.getModel(inst.modelIndex);
    if (!inst.visible || !model || !model->isValid())
        return;
    if (!highlight.wholeInstance && highlight.submeshes.empty())
        return;

    const IfcScene* ifc = inst.ifcScene ? &*inst.ifcScene : nullptr;
    const uint32_t transformIndex = pushTransform(inst.getTransformMatrix());
    const vk::DescriptorSet noSets[3] = {};
    auto add = [&](size_t si) {
        if (si >= model->submeshes.size() || (ifc && !ifc->isSubmeshVisible(si)))
            return;
        appendDraw(m_highlightRuns, model, noSets, false, model->submeshes[si], transformIndex, glm::vec3(1.0f));
    };
    if (highlight.wholeInstance) {
        for (size_t si = 0; si < model->submeshes.size(); ++si)
            add(si);
    }
    else {
        for (uint32_t si : highlight.submeshes)
            add(si);
    }
}

void Renderer::uploadDrawStreams()
{
    // Safe to grow/rewrite: this frame's fence was waited on, so the GPU no longer reads these
    // buffers or this frame's descriptor set.
    FrameDrawBuffers& buffers = *m_frameDrawBuffers[m_currentFrame];
    const vk::DeviceSize drawBytes = sizeof(GpuDrawData) * m_frameDraws.size();
    const vk::DeviceSize transformBytes = sizeof(GpuTransform) * m_frameTransforms.size();
    const vk::DeviceSize commandBytes = sizeof(vk::DrawIndexedIndirectCommand) * m_frameCommands.size();
    bool descriptorsStale = buffers.draws.reserve(drawBytes);
    descriptorsStale = buffers.transforms.reserve(transformBytes) || descriptorsStale;
    buffers.indirect.reserve(commandBytes);
    if (descriptorsStale) writeDrawDescriptors(m_currentFrame);

    if (drawBytes) memcpy(buffers.draws.mapped(), m_frameDraws.data(), drawBytes);
    if (transformBytes) memcpy(buffers.transforms.mapped(), m_frameTransforms.data(), transformBytes);
    if (commandBytes) memcpy(buffers.indirect.mapped(), m_frameCommands.data(), commandBytes);
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

void Renderer::setDefaultDrawState(vk::CommandBuffer cmd, vk::SampleCountFlagBits samples,
    const vk::Viewport& viewport, const vk::Rect2D& scissor) const
{
    cmd.setViewportWithCount(1, &viewport);
    cmd.setScissorWithCount(1, &scissor);
    cmd.setRasterizerDiscardEnable(VK_FALSE);
    cmd.setCullMode(vk::CullModeFlagBits::eNone);
    cmd.setFrontFace(vk::FrontFace::eCounterClockwise);
    cmd.setDepthTestEnable(VK_FALSE);
    cmd.setDepthWriteEnable(VK_FALSE);
    cmd.setDepthCompareOp(vk::CompareOp::eLessOrEqual);
    cmd.setDepthBiasEnable(VK_FALSE);
    cmd.setStencilTestEnable(VK_FALSE);
    cmd.setPolygonModeEXT(vk::PolygonMode::eFill);
    cmd.setRasterizationSamplesEXT(samples);
    const vk::SampleMask sampleMask = ~0u;
    cmd.setSampleMaskEXT(samples, &sampleMask);
    cmd.setAlphaToCoverageEnableEXT(VK_FALSE);
    cmd.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
    cmd.setPrimitiveRestartEnable(VK_FALSE);
    cmd.setColorWriteMaskEXT(0, vk::ColorComponentFlags(0xF));
    setAlphaBlending(cmd, false);
}

void Renderer::setAlphaBlending(vk::CommandBuffer cmd, bool enabled) const
{
    cmd.setColorBlendEnableEXT(0, enabled ? VK_TRUE : VK_FALSE);
    vk::ColorBlendEquationEXT blendEquation{};
    blendEquation.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    blendEquation.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    blendEquation.colorBlendOp = vk::BlendOp::eAdd;
    blendEquation.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    blendEquation.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    blendEquation.alphaBlendOp = vk::BlendOp::eAdd;
    cmd.setColorBlendEquationEXT(0, 1, &blendEquation);
}

// The editor's scene area in framebuffer pixels (window coordinates are scaled on high-DPI displays).
vk::Rect2D Renderer::sceneRect(const FrameInput& input) const
{
    const vk::Extent2D extent = m_swapchain.extent();
    const int32_t fbWidth = static_cast<int32_t>(extent.width);
    const int32_t fbHeight = static_cast<int32_t>(extent.height);
    const ViewRect& view = input.viewport;
    const float pixelScaleX = float(extent.width) / input.windowWidth;
    const float pixelScaleY = float(extent.height) / input.windowHeight;
    const int32_t x0 = std::clamp(static_cast<int32_t>(std::lround(view.x * pixelScaleX)), 0, fbWidth - 1);
    const int32_t y0 = std::clamp(static_cast<int32_t>(std::lround(view.y * pixelScaleY)), 0, fbHeight - 1);
    const int32_t x1 = std::clamp(static_cast<int32_t>(std::lround((view.x + view.width) * pixelScaleX)), x0 + 1, fbWidth);
    const int32_t y1 = std::clamp(static_cast<int32_t>(std::lround((view.y + view.height) * pixelScaleY)), y0 + 1, fbHeight);
    return { { x0, y0 }, { uint32_t(x1 - x0), uint32_t(y1 - y0) } };
}

void Renderer::bindModelBuffers(vk::CommandBuffer cmd, const GPUModel* model) const
{
    const vk::Buffer buffers[1] = { model->vertexBuffer->getBuffer() };
    const vk::DeviceSize offsets[1] = { 0 };
    const vk::DeviceSize sizes[1] = { sizeof(Vertex) * model->vertexCount };
    const vk::DeviceSize strides[1] = { sizeof(Vertex) };
    cmd.bindVertexBuffers2(0, 1, buffers, offsets, sizes, strides);
    cmd.bindIndexBuffer(model->indexBuffer->getBuffer(), 0, vk::IndexType::eUint32);
}

void Renderer::recordRun(vk::CommandBuffer cmd, const DrawRun& run) const
{
    constexpr uint32_t stride = sizeof(vk::DrawIndexedIndirectCommand);
    const vk::Buffer indirect = m_frameDrawBuffers[m_currentFrame]->indirect.getBuffer();
    for (uint32_t done = 0; done < run.commandCount;) {
        const uint32_t count = std::min(run.commandCount - done, m_maxDrawIndirectCount);
        cmd.drawIndexedIndirect(indirect, vk::DeviceSize(run.firstCommand + done) * stride, count, stride);
        done += count;
    }
}

void Renderer::recordShadowPass(vk::CommandBuffer cmd, const ModelManager& models)
{
    const uint32_t size = m_shadowMap.size;
    const vk::Image shadowImage(m_shadowMap.image);
    const auto depthAspect = vk::ImageAspectFlagBits::eDepth;

    const vk::ImageMemoryBarrier2 toAttachment = imageBarrier(shadowImage, depthAspect,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal);
    pipelineBarriers(cmd, { &toAttachment, 1 });

    vk::RenderingAttachmentInfo shadowDepthAttachment{};
    shadowDepthAttachment.setImageView(m_shadowMap.view)
        .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue(vk::ClearDepthStencilValue{ 1.0f, 0 }));

    vk::RenderingInfo shadowRenderInfo{};
    shadowRenderInfo.setRenderArea({ {0, 0}, {size, size} })
        .setLayerCount(1)
        .setColorAttachmentCount(0)
        .setPDepthAttachment(&shadowDepthAttachment);

    cmd.beginRendering(shadowRenderInfo);
    if (!m_shadowRuns.empty()) {
        bindShaderPair(cmd, m_shadowShaders);
        const vk::Viewport shadowViewport{ 0, 0, float(size), float(size), 0.f, 1.f };
        const vk::Rect2D shadowRect{ {0, 0}, {size, size} };
        setDefaultDrawState(cmd, vk::SampleCountFlagBits::e1, shadowViewport, shadowRect);
        cmd.setCullMode(vk::CullModeFlagBits::eFront);
        cmd.setDepthTestEnable(VK_TRUE);
        cmd.setDepthWriteEnable(VK_TRUE);
        cmd.setDepthBiasEnable(VK_TRUE);
        cmd.setDepthBias(1.25f, 0.0f, 1.75f);
        cmd.setVertexInputEXT(1, &m_meshBinding, static_cast<uint32_t>(m_meshAttributes.size()), m_meshAttributes.data());

        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_shadowLayout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);
        // The shadow fragment shader statically uses set 1, so it must be bound even
        // when no submesh is alpha-masked.
        vk::DescriptorSet boundShadowSet = models.getDefaultBaseColorSet();
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_shadowLayout, 1, 1, &boundShadowSet, 0, nullptr);

        const GPUModel* boundModel = nullptr;
        for (const DrawRun& run : m_shadowRuns) {
            if (run.model != boundModel) {
                bindModelBuffers(cmd, run.model);
                boundModel = run.model;
            }
            if (run.sets[0] && run.sets[0] != boundShadowSet) {
                boundShadowSet = run.sets[0];
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_shadowLayout, 1, 1, &boundShadowSet, 0, nullptr);
            }
            recordRun(cmd, run);
        }
        cmd.setDepthBiasEnable(VK_FALSE);
    }
    cmd.endRendering();

    const vk::ImageMemoryBarrier2 toSampled = imageBarrier(shadowImage, depthAspect,
        vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
        vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthStencilReadOnlyOptimal);
    pipelineBarriers(cmd, { &toSampled, 1 });
}

void Renderer::recordScenePass(vk::CommandBuffer cmd, uint32_t imageIndex, const FrameInput& input)
{
    const vk::Extent2D extent = m_swapchain.extent();
    const bool msaa = m_samples != vk::SampleCountFlagBits::e1;
    const vk::Image swapImage(m_swapchain.images()[imageIndex]);
    const auto colorAspect = vk::ImageAspectFlagBits::eColor;
    const auto depthAspect = vk::ImageAspectFlagBits::eDepth;
    const auto depthStages = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
    const auto depthAccess = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;

    // Previous contents are discarded. The scene depth was last sampled by the selection mask pass
    // of an earlier frame, hence the fragment shader source stage.
    std::vector<vk::ImageMemoryBarrier2> barriers = {
        imageBarrier(swapImage, colorAspect,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal),
        imageBarrier(vk::Image(m_sceneDepth.image), depthAspect,
            vk::PipelineStageFlagBits2::eFragmentShader | depthStages | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            // Depth resolves write in the color output stage with color attachment access.
            depthStages | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            depthAccess | vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal),
    };
    if (msaa) {
        barriers.push_back(imageBarrier(vk::Image(m_msaaColor.image), colorAspect,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal));
        barriers.push_back(imageBarrier(vk::Image(m_msaaDepth.image), depthAspect,
            depthStages, vk::AccessFlagBits2::eDepthStencilAttachmentWrite, depthStages, depthAccess,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal));
    }
    pipelineBarriers(cmd, barriers);

    // Outside the scene viewport the UI covers everything, so a plain clear is enough there.
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setClearValue(vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{ 0.1f, 0.1f, 0.1f, 1.0f })));
    vk::RenderingAttachmentInfo depthAttachment{};
    depthAttachment.setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setClearValue(vk::ClearValue(vk::ClearDepthStencilValue{ 1.0f, 0 }));
    if (msaa) {
        colorAttachment.setImageView(m_msaaColor.view)
            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setResolveMode(vk::ResolveModeFlagBits::eAverage)
            .setResolveImageView(m_swapchain.imageViews()[imageIndex])
            .setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
        depthAttachment.setImageView(m_msaaDepth.view)
            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setResolveMode(vk::ResolveModeFlagBits::eSampleZero)
            .setResolveImageView(m_sceneDepth.view)
            .setResolveImageLayout(vk::ImageLayout::eDepthAttachmentOptimal);
    }
    else {
        colorAttachment.setImageView(m_swapchain.imageViews()[imageIndex])
            .setStoreOp(vk::AttachmentStoreOp::eStore);
        depthAttachment.setImageView(m_sceneDepth.view)
            .setStoreOp(vk::AttachmentStoreOp::eStore);
    }

    vk::RenderingInfo renderInfo{};
    renderInfo.setRenderArea({ {0, 0}, extent })
        .setLayerCount(1)
        .setColorAttachments(colorAttachment)
        .setPDepthAttachment(&depthAttachment);
    cmd.beginRendering(renderInfo);

    const vk::Rect2D rect = sceneRect(input);
    const vk::Viewport viewport{ float(rect.offset.x), float(rect.offset.y),
        float(rect.extent.width), float(rect.extent.height), 0.f, 1.f };
    setDefaultDrawState(cmd, m_samples, viewport, rect);

    // Sky first, behind everything (no depth).
    recordFullscreen(cmd, m_skyShaders);

    cmd.setDepthTestEnable(VK_TRUE);
    cmd.setDepthWriteEnable(VK_TRUE);
    cmd.setDepthCompareOp(vk::CompareOp::eLess);
    bindShaderPair(cmd, m_meshShaders);
    cmd.setVertexInputEXT(1, &m_meshBinding, static_cast<uint32_t>(m_meshAttributes.size()), m_meshAttributes.data());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_meshLayout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_meshLayout, 4, 1, &m_shadowMapSet, 0, nullptr);
    recordMeshRuns(cmd, m_opaqueRuns);

    // Transparent-ish layers: depth-tested against the opaque scene, no depth writes.
    cmd.setDepthWriteEnable(VK_FALSE);
    cmd.setDepthCompareOp(vk::CompareOp::eLessOrEqual);
    setAlphaBlending(cmd, true);

    if (input.showGrid)
        recordFullscreen(cmd, m_gridShaders);

    if (!m_blendRuns.empty()) {
        bindShaderPair(cmd, m_meshShaders);
        cmd.setVertexInputEXT(1, &m_meshBinding, static_cast<uint32_t>(m_meshAttributes.size()), m_meshAttributes.data());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_meshLayout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_meshLayout, 4, 1, &m_shadowMapSet, 0, nullptr);
        recordMeshRuns(cmd, m_blendRuns);
    }

    setAlphaBlending(cmd, false);
    recordPathLines(cmd, input);
    cmd.endRendering();

    // The resolved depth is sampled by the selection mask; the swapchain image is drawn on again
    // by the overlay pass. Resolves count as attachment writes in the color output stage.
    const vk::ImageMemoryBarrier2 after[2] = {
        imageBarrier(vk::Image(m_sceneDepth.image), depthAspect,
            vk::PipelineStageFlagBits2::eLateFragmentTests | vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
            vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal),
        imageBarrier(swapImage, colorAspect,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal),
    };
    pipelineBarriers(cmd, after);
}

void Renderer::recordMeshRuns(vk::CommandBuffer cmd, const std::vector<DrawRun>& runs)
{
    const GPUModel* boundModel = nullptr;
    vk::DescriptorSet boundSets[3] = { nullptr, nullptr, nullptr };
    for (const DrawRun& run : runs) {
        if (run.model != boundModel) {
            bindModelBuffers(cmd, run.model);
            boundModel = run.model;
        }
        if (run.sets[0] != boundSets[0] || run.sets[1] != boundSets[1] || run.sets[2] != boundSets[2]) {
            for (int k = 0; k < 3; ++k) boundSets[k] = run.sets[k];
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_meshLayout, 1, 3, boundSets, 0, nullptr);
        }
        recordRun(cmd, run);
    }
}

void Renderer::recordFullscreen(vk::CommandBuffer cmd, const ShaderPair& shaders)
{
    bindShaderPair(cmd, shaders);
    cmd.setVertexInputEXT(0, nullptr, 0, nullptr);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_fxLayout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);
    cmd.draw(3, 1, 0, 0);
}

void Renderer::recordPathLines(vk::CommandBuffer cmd, const FrameInput& input)
{
    if (!input.showPath || m_pathLines.buffer == VK_NULL_HANDLE || m_pathLines.vertexCount == 0)
        return;

    bindShaderPair(cmd, m_gizmoShaders);
    cmd.setPrimitiveTopology(vk::PrimitiveTopology::eLineList);
    cmd.setLineWidth(2.0f);
    cmd.setDepthTestEnable(VK_TRUE); // the path is hidden behind objects
    cmd.setDepthWriteEnable(VK_FALSE);
    cmd.setVertexInputEXT(1, &kLineBinding, static_cast<uint32_t>(kLineAttributes.size()), kLineAttributes.data());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_gizmoLayout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);

    const glm::mat4 identity(1.0f);
    cmd.pushConstants(m_gizmoLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4), &identity);
    const vk::Buffer buffer(m_pathLines.buffer);
    const vk::DeviceSize offset = 0;
    const vk::DeviceSize size = sizeof(GizmoVertex) * m_pathLines.vertexCount;
    const vk::DeviceSize stride = sizeof(GizmoVertex);
    cmd.bindVertexBuffers2(0, 1, &buffer, &offset, &size, &stride);
    cmd.draw(m_pathLines.vertexCount, 1, 0, 0);
    cmd.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
}

// R = silhouette of the highlighted geometry regardless of depth, G = where it is visible in the scene.
void Renderer::recordSelectionMask(vk::CommandBuffer cmd, const FrameInput& input)
{
    const vk::Image maskImage(m_selectionMask.image);
    const auto colorAspect = vk::ImageAspectFlagBits::eColor;

    const vk::ImageMemoryBarrier2 toAttachment = imageBarrier(maskImage, colorAspect,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
    pipelineBarriers(cmd, { &toAttachment, 1 });

    vk::RenderingAttachmentInfo maskAttachment{};
    maskAttachment.setImageView(m_selectionMask.view)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f })));
    vk::RenderingInfo renderInfo{};
    renderInfo.setRenderArea({ {0, 0}, m_swapchain.extent() })
        .setLayerCount(1)
        .setColorAttachments(maskAttachment);
    cmd.beginRendering(renderInfo);

    const vk::Rect2D rect = sceneRect(input);
    const vk::Viewport viewport{ float(rect.offset.x), float(rect.offset.y),
        float(rect.extent.width), float(rect.extent.height), 0.f, 1.f };
    setDefaultDrawState(cmd, vk::SampleCountFlagBits::e1, viewport, rect);
    cmd.setColorWriteMaskEXT(0, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG);
    cmd.setColorBlendEnableEXT(0, VK_TRUE);
    vk::ColorBlendEquationEXT maxEquation{};
    maxEquation.srcColorBlendFactor = vk::BlendFactor::eOne;
    maxEquation.dstColorBlendFactor = vk::BlendFactor::eOne;
    maxEquation.colorBlendOp = vk::BlendOp::eMax;
    maxEquation.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    maxEquation.dstAlphaBlendFactor = vk::BlendFactor::eOne;
    maxEquation.alphaBlendOp = vk::BlendOp::eMax;
    cmd.setColorBlendEquationEXT(0, 1, &maxEquation);

    bindShaderPair(cmd, m_maskShaders);
    cmd.setVertexInputEXT(1, &m_meshBinding, static_cast<uint32_t>(m_meshAttributes.size()), m_meshAttributes.data());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_fxLayout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_fxLayout, 1, 1, &m_sceneDepthSet, 0, nullptr);

    const GPUModel* boundModel = nullptr;
    for (const DrawRun& run : m_highlightRuns) {
        if (run.model != boundModel) {
            bindModelBuffers(cmd, run.model);
            boundModel = run.model;
        }
        recordRun(cmd, run);
    }
    cmd.endRendering();

    const vk::ImageMemoryBarrier2 toSampled = imageBarrier(maskImage, colorAspect,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    pipelineBarriers(cmd, { &toSampled, 1 });
}

// Single-sample pass straight on the swapchain image: selection outline, then ImGui.
void Renderer::recordOverlayPass(vk::CommandBuffer cmd, uint32_t imageIndex, const FrameInput& input, bool drawOutline)
{
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.setImageView(m_swapchain.imageViews()[imageIndex])
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eLoad)
        .setStoreOp(vk::AttachmentStoreOp::eStore);
    vk::RenderingInfo renderInfo{};
    renderInfo.setRenderArea({ {0, 0}, m_swapchain.extent() })
        .setLayerCount(1)
        .setColorAttachments(colorAttachment);
    cmd.beginRendering(renderInfo);

    const vk::Rect2D rect = sceneRect(input);
    const vk::Viewport viewport{ float(rect.offset.x), float(rect.offset.y),
        float(rect.extent.width), float(rect.extent.height), 0.f, 1.f };
    setDefaultDrawState(cmd, vk::SampleCountFlagBits::e1, viewport, rect);

    if (drawOutline) {
        setAlphaBlending(cmd, true);
        bindShaderPair(cmd, m_outlineShaders);
        cmd.setVertexInputEXT(0, nullptr, 0, nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_fxLayout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_fxLayout, 1, 1, &m_selectionMaskSet, 0, nullptr);
        OutlinePushConstants push{};
        push.color = glm::vec4(kOutlineColor, 1.0f);
        push.occludedAlpha = kOutlineOccludedAlpha;
        push.widthPixels = kOutlineWidthPixels;
        cmd.pushConstants(m_fxLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(push), &push);
        cmd.draw(3, 1, 0, 0);
        setAlphaBlending(cmd, false);
    }

    if (input.imgui && input.imgui->TotalVtxCount > 0)
        ImGui_ImplVulkan_RenderDrawData(input.imgui, cmd);
    cmd.endRendering();

    const vk::ImageMemoryBarrier2 toPresent = imageBarrier(vk::Image(m_swapchain.images()[imageIndex]),
        vk::ImageAspectFlagBits::eColor,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eBottomOfPipe, vk::AccessFlagBits2::eNone,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);
    pipelineBarriers(cmd, { &toPresent, 1 });
}

Renderer::FrameStatus Renderer::submitAndPresent(vk::CommandBuffer cmd, uint32_t imageIndex)
{
    const vk::CommandBufferSubmitInfo cmdInfo{ cmd };
    const vk::SemaphoreSubmitInfo waitInfo{
        m_imageAvailableSemaphores[m_currentFrame], 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput };
    const vk::SemaphoreSubmitInfo signalInfo{
        m_renderFinishedSemaphores[imageIndex], 0, vk::PipelineStageFlagBits2::eAllCommands };

    vk::SubmitInfo2 submit{};
    submit.setCommandBufferInfos(cmdInfo)
        .setWaitSemaphoreInfos(waitInfo)
        .setSignalSemaphoreInfos(signalInfo);
    (void)m_context->graphicsQueue().submit2(submit, m_inFlightFences[m_currentFrame]);

    const vk::SwapchainKHR swapchain = m_swapchain.handle();
    vk::PresentInfoKHR present{};
    present.setWaitSemaphores(m_renderFinishedSemaphores[imageIndex])
        .setSwapchains(swapchain)
        .setImageIndices(imageIndex);

    // Pointer overload: returns the raw result instead of asserting on eErrorOutOfDateKHR.
    const vk::Result presentResult = m_context->presentQueue().presentKHR(&present);
    if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR)
        m_swapchainDirty = true;
    else if (presentResult != vk::Result::eSuccess)
        std::cerr << "Failed to present\n";

    m_currentFrame = (m_currentFrame + 1) % m_framesInFlight;
    return FrameStatus::Rendered;
}
