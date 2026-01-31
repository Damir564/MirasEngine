#pragma once
#include "PCH.h"
#include "VulkanContext.h"
#include "ModelLoader.h"
#include "Types.h"

class Renderer {
public:
    Renderer(VulkanContext& ctx, SDL_Window* window);
    ~Renderer();

    void loadModel(const std::string& path);
    void render(Camera& camera, float time);

private:
    void initSwapchainResources();
    void initShaders();
    void initDescriptors();
    void initSyncObjects();

    VulkanContext& ctx;
    SDL_Window* window;

    // Assets
    Model model;
    std::unique_ptr<VulkanBuffer> instanceBuffer; // Simple single instance for now

    // Rendering Resources
    vk::UniqueDescriptorSetLayout descriptorLayout;
    vk::UniqueDescriptorPool descriptorPool;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueSampler textureSampler;

    // Shader Objects
    vk::ShaderEXT vertShader, fragShader;

    // Depth
    vk::Image depthImage;
    VmaAllocation depthAlloc;
    vk::ImageView depthImageView;
    std::vector<vk::ImageView> swapchainImageViews;

    // Sync
    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::UniqueFence> inFlightFences;
    uint32_t currentFrame = 0;
    const int MAX_FRAMES_IN_FLIGHT = 2;

    // Default textures & sets
    std::vector<vk::DescriptorSet> textureSets;
    std::unique_ptr<VulkanTexture> whiteTex, normalTex, mrTex;

    std::vector<vk::UniqueCommandPool> frameCommandPools;
};