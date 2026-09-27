#pragma once
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>
#include "Buffers.h"
#include "Camera.h"
#include "Gizmo.h"
#include "GraphicsSettings.h"
#include "RenderTypes.h"
#include "ShaderUtils.h"
#include "Shadow.h"
#include "Swapchain.h"

struct SDL_Window;
struct ImDrawData;
class VulkanContext;
class ModelManager;
struct GPUModel;
struct ModelInstance;
struct SubmeshInfo;
struct Material;

// Rectangle in window coordinates (not pixels).
struct ViewRect {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

// Geometry drawn with the selection outline.
struct SelectionHighlight {
    int instance = -1;
    bool wholeInstance = true;
    // Submesh indices of `instance`; used only when wholeInstance is false.
    std::vector<uint32_t> submeshes;
};

struct FrameInput {
    ModelManager* models = nullptr;
    // Computed by the mode; the projection must come from getProjection() so picking matches.
    glm::mat4 view{ 1.0f };
    glm::mat4 proj{ 1.0f };
    glm::vec3 cameraPosition{ 0.0f };
    // The scene is drawn only into this part of the window; UI covers the rest.
    ViewRect viewport;
    float windowWidth = 1.0f;
    float windowHeight = 1.0f;
    DirectionalLight sun;
    SelectionHighlight highlight;
    bool showPath = false;
    bool showGrid = false;
    ImDrawData* imgui = nullptr;
    float time = 0.0f;
};

// Owns every GPU resource used to draw a frame and records/submits/presents it.
// Requires an ImGui context to exist before init() (it initializes the ImGui Vulkan backend).
class Renderer {
public:
    enum class FrameStatus {
        Rendered,
        Skipped, // swapchain out of date; it is rebuilt on the next prepareSwapchain()
        Failed,  // unrecoverable acquire error
    };

    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool init(VulkanContext& context, SDL_Window* window, const GraphicsSettings& settings);
    void shutdown();

    // Applies changed settings: VSync rebuilds the swapchain on the next prepareSwapchain(); MSAA and
    // shadow map size recreate their resources right away (waits for the GPU to go idle).
    void applySettings(const GraphicsSettings& settings);
    const GraphicsSettings& settings() const { return m_settings; }
    const RenderCapabilities& capabilities() const { return m_capabilities; }

    // Resources ModelManager needs to upload textures and allocate their descriptor sets.
    vk::CommandPool commandPool() const { return m_commandPool; }
    vk::DescriptorPool descriptorPool() const { return m_descriptorPool; }
    vk::DescriptorSetLayout textureSetLayout() const { return m_textureSetLayout; }
    vk::Sampler textureSampler() const { return m_textureSampler; }

    // Replaces the camera-path line list (pairs of vertices). An empty list hides the path.
    void setPathLines(const std::vector<GizmoVertex>& vertices);

    void requestSwapchainRebuild() { m_swapchainDirty = true; }
    // Rebuilds the swapchain if a rebuild was requested. False while there is nothing to render into
    // (window minimized / zero-sized) or recreation failed; skip the frame then.
    bool prepareSwapchain();

    uint32_t framebufferWidth() const { return m_swapchain.extent().width; }
    uint32_t framebufferHeight() const { return m_swapchain.extent().height; }

    FrameStatus renderFrame(const FrameInput& input);

private:
    struct FrameDrawBuffers {
        HostBuffer draws;
        HostBuffer transforms;
        HostBuffer indirect;
    };

    struct LineBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        uint32_t vertexCount = 0;
    };

    struct InstanceRenderData {
        const ModelInstance* instance;
        glm::mat4 transform;
        const uint8_t* submeshFlags;
        uint32_t transformIndex;
    };

    struct RenderBatch {
        GPUModel* model = nullptr;
        std::vector<InstanceRenderData> instances;
    };

    // Visible models this frame, keyed by model index.
    struct FrameBatches {
        std::unordered_map<size_t, RenderBatch> main;
        std::unordered_map<size_t, RenderBatch> shadow;
    };

    struct CullResult {
        bool visibleMain = false;
        bool visibleShadow = false;
        glm::mat4 transform = glm::mat4(1.0f);
        const ModelInstance* instance = nullptr;
        size_t modelIndex = 0;
        GPUModel* gpuModel = nullptr;
        FrustumPlanes mainPlanes{};
        FrustumPlanes shadowPlanes{};
        std::vector<uint8_t> submeshFlags;
    };

    struct RenderImage {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        vk::ImageView view;
    };

    struct CullChunk {
        size_t instanceIdx;
        size_t begin;
        size_t end;
    };

    // Consecutive indirect commands that share model buffers, texture sets and blend state.
    struct DrawRun {
        GPUModel* model;
        vk::DescriptorSet sets[3]; // null = "any set works" (only used by the shadow pass)
        bool blend;
        uint32_t firstCommand;
        uint32_t commandCount;
    };

    void queryCapabilities();
    bool createFrameResources();
    bool createDescriptors();
    bool createShaders();
    bool initImGuiBackend();
    bool createShadowMapResources();
    bool createRenderTargets();
    void destroyRenderTargets();
    bool createRenderImage(vk::Format format, vk::ImageUsageFlags usage, vk::SampleCountFlagBits samples,
        vk::ImageAspectFlags aspect, RenderImage& out);
    void destroyRenderImage(RenderImage& image);
    bool recreateSwapchain();
    void writeDrawDescriptors(uint32_t frame);
    vk::SampleCountFlagBits effectiveSampleCount() const;

    bool createLineBuffer(const std::vector<GizmoVertex>& vertices, LineBuffer& out);
    void destroyLineBuffer(LineBuffer& buffer);

    FrameUBO buildFrameUBO(const FrameInput& input) const;
    // Returns the hash of everything that affects the shadow map.
    uint64_t cullAndBatch(const FrameInput& input, const glm::mat4& viewProj, const glm::mat4& lightSpaceMatrix,
        FrameBatches& batches);
    void buildDrawStreams(const FrameInput& input, FrameBatches& batches, bool renderShadowMap);
    void buildHighlightStream(const FrameInput& input);
    void appendDraw(std::vector<DrawRun>& runs, GPUModel* model, const vk::DescriptorSet sets[3], bool blend,
        const SubmeshInfo& sub, uint32_t transformIndex, const glm::vec3& tint);
    uint32_t pushDrawData(const SubmeshInfo& sub, uint32_t transformIndex, const glm::vec3& tint);
    uint32_t pushTransform(const glm::mat4& transform);
    void materialSets(const ModelManager& models, const GPUModel* model, const Material& material,
        vk::DescriptorSet out[3]) const;
    void uploadDrawStreams();

    // Sets every piece of dynamic state the shader-object draws rely on to a known default.
    void setDefaultDrawState(vk::CommandBuffer cmd, vk::SampleCountFlagBits samples,
        const vk::Viewport& viewport, const vk::Rect2D& scissor) const;
    void setAlphaBlending(vk::CommandBuffer cmd, bool enabled) const;
    vk::Rect2D sceneRect(const FrameInput& input) const;
    void bindModelBuffers(vk::CommandBuffer cmd, const GPUModel* model) const;
    void recordRun(vk::CommandBuffer cmd, const DrawRun& run) const;
    void recordShadowPass(vk::CommandBuffer cmd, const ModelManager& models);
    void recordScenePass(vk::CommandBuffer cmd, uint32_t imageIndex, const FrameInput& input);
    void recordMeshRuns(vk::CommandBuffer cmd, const std::vector<DrawRun>& runs);
    void recordFullscreen(vk::CommandBuffer cmd, const ShaderPair& shaders);
    void recordPathLines(vk::CommandBuffer cmd, const FrameInput& input);
    void recordSelectionMask(vk::CommandBuffer cmd, const FrameInput& input);
    void recordOverlayPass(vk::CommandBuffer cmd, uint32_t imageIndex, const FrameInput& input, bool drawOutline);
    FrameStatus submitAndPresent(vk::CommandBuffer cmd, uint32_t imageIndex);

    VulkanContext* m_context = nullptr;
    SDL_Window* m_window = nullptr;
    vk::Device m_device;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    Swapchain m_swapchain;
    bool m_swapchainDirty = false;
    uint32_t m_framesInFlight = 0;
    uint32_t m_currentFrame = 0;

    GraphicsSettings m_settings;
    RenderCapabilities m_capabilities;

    static constexpr vk::Format kDepthFormat = vk::Format::eD32Sfloat;
    static constexpr vk::Format kSelectionMaskFormat = vk::Format::eR8G8Unorm;
    // Swapchain-sized targets, rebuilt on resize and MSAA changes. With MSAA off the scene renders
    // straight into the swapchain image and sceneDepth; otherwise the multisampled targets resolve into them.
    vk::SampleCountFlagBits m_samples = vk::SampleCountFlagBits::e1;
    RenderImage m_sceneDepth;
    RenderImage m_msaaColor;
    RenderImage m_msaaDepth;
    RenderImage m_selectionMask;

    vk::CommandPool m_commandPool;
    std::vector<vk::CommandBuffer> m_commandBuffers;
    std::vector<vk::Semaphore> m_imageAvailableSemaphores;
    // Indexed by swapchain image, not by frame in flight.
    std::vector<vk::Semaphore> m_renderFinishedSemaphores;
    std::vector<vk::Fence> m_inFlightFences;
    std::vector<UBOBuffer> m_frameUBOs;

    vk::Sampler m_textureSampler;
    vk::Sampler m_nearestSampler;
    vk::DescriptorSetLayout m_textureSetLayout;
    vk::DescriptorSetLayout m_frameSetLayout;
    vk::DescriptorPool m_descriptorPool;
    std::vector<vk::DescriptorSet> m_frameSets;
    std::vector<std::unique_ptr<FrameDrawBuffers>> m_frameDrawBuffers;
    uint32_t m_maxDrawIndirectCount = 0;

    ShadowMapResources m_shadowMap;
    vk::DescriptorSet m_shadowMapSet;
    vk::DescriptorSet m_sceneDepthSet;
    vk::DescriptorSet m_selectionMaskSet;

    ShaderPair m_meshShaders;
    ShaderPair m_shadowShaders;
    ShaderPair m_gizmoShaders;
    ShaderPair m_skyShaders;
    ShaderPair m_gridShaders;
    ShaderPair m_maskShaders;
    ShaderPair m_outlineShaders;
    vk::PipelineLayout m_meshLayout;
    vk::PipelineLayout m_shadowLayout;
    vk::PipelineLayout m_gizmoLayout;
    // Sky, grid, selection mask and outline: set 0 = frame data, set 1 = one sampled image.
    vk::PipelineLayout m_fxLayout;
    vk::VertexInputBindingDescription2EXT m_meshBinding;
    std::array<vk::VertexInputAttributeDescription2EXT, 4> m_meshAttributes;

    LineBuffer m_pathLines;

    vk::DescriptorPool m_imguiDescriptorPool;
    bool m_imguiInitialized = false;

    // Per-frame scratch kept as members so their capacity survives between frames.
    std::vector<CullResult> m_cullResults;
    std::vector<size_t> m_cullIndices;
    std::vector<CullChunk> m_cullChunks;
    std::vector<GpuTransform> m_frameTransforms;
    std::vector<GpuDrawData> m_frameDraws;
    std::vector<vk::DrawIndexedIndirectCommand> m_frameCommands;
    std::vector<DrawRun> m_shadowRuns;
    std::vector<DrawRun> m_opaqueRuns;
    std::vector<DrawRun> m_blendRuns;
    std::vector<DrawRun> m_highlightRuns;

    uint64_t m_lastShadowHash = 0;
    bool m_shadowMapValid = false;
};
