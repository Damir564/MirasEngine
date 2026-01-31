#include "Renderer.h"

Renderer::Renderer(VulkanContext& c, SDL_Window* w) : ctx(c), window(w) {
    initSwapchainResources();
    initSyncObjects();
    initShaders();

    // Default Instance Buffer
    std::vector<InstanceData> instances = { {{0.0f, 0.0f, 0.0f}} };
    instanceBuffer = std::make_unique<VulkanBuffer>(ctx, sizeof(InstanceData), vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
    instanceBuffer->upload(instances.data(), sizeof(InstanceData));
    instanceBuffer->setCount(1);
}

Renderer::~Renderer() {
    if (depthImageView) ctx.getDevice().destroyImageView(depthImageView);
    if (depthImage) vmaDestroyImage(ctx.getAllocator(), VkImage(depthImage), depthAlloc);
    ctx.getDevice().destroyShaderEXT(vertShader);
    ctx.getDevice().destroyShaderEXT(fragShader);
    // Unique handles auto-destroy layouts/pools/fences
}

void Renderer::initSwapchainResources() {
    auto& swapchain = ctx.getSwapchain();
    auto rawViews = swapchain.get_image_views().value();
    swapchainImageViews.clear();
    for (const auto& v : rawViews) {
        swapchainImageViews.push_back(vk::ImageView(v));
    }

    // Depth Image
    VkImageCreateInfo dInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    dInfo.imageType = VK_IMAGE_TYPE_2D;
    dInfo.extent = { 1280, 720, 1 };
    dInfo.mipLevels = 1; dInfo.arrayLayers = 1;
    dInfo.format = VK_FORMAT_D32_SFLOAT;
    dInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    dInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    dInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    VkImage rawD;
    vmaCreateImage(ctx.getAllocator(), &dInfo, &allocInfo, &rawD, &depthAlloc, nullptr);
    depthImage = vk::Image(rawD);

    vk::ImageViewCreateInfo vInfo{};
    vInfo.image = depthImage;
    vInfo.viewType = vk::ImageViewType::e2D;
    vInfo.format = vk::Format::eD32Sfloat;
    vInfo.subresourceRange = { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 };
    depthImageView = ctx.getDevice().createImageView(vInfo).value;
}

void Renderer::initShaders() {
    auto vertCode = loadSpirv("shaders/triangle.vert.spv");
    auto fragCode = loadSpirv("shaders/triangle.frag.spv");

    vk::DescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorCount = 1;
    binding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    binding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    descriptorLayout = ctx.getDevice().createDescriptorSetLayoutUnique(layoutInfo).value;

    vk::PushConstantRange pcRange{ vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(MeshPushConstants) };
    vk::DescriptorSetLayout layouts[] = { descriptorLayout.get(), descriptorLayout.get(), descriptorLayout.get() };

    vk::PipelineLayoutCreateInfo plInfo{};
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    plInfo.setLayoutCount = 3;
    plInfo.setPSetLayouts(layouts);
    pipelineLayout = ctx.getDevice().createPipelineLayoutUnique(plInfo).value;

    // Shader Objects creation
    vk::ShaderCreateInfoEXT vInfo{};
    vInfo.setStage(vk::ShaderStageFlagBits::eVertex).setCodeType(vk::ShaderCodeTypeEXT::eSpirv).setCodeSize(vertCode.size() * 4).setPCode(vertCode.data()).setPName("main").setSetLayoutCount(3).setPSetLayouts(layouts).setPushConstantRangeCount(1).setPPushConstantRanges(&pcRange);
    vertShader = ctx.getDevice().createShaderEXT(vInfo).value;

    vk::ShaderCreateInfoEXT fInfo{};
    fInfo.setStage(vk::ShaderStageFlagBits::eFragment).setCodeType(vk::ShaderCodeTypeEXT::eSpirv).setCodeSize(fragCode.size() * 4).setPCode(fragCode.data()).setPName("main").setSetLayoutCount(3).setPSetLayouts(layouts).setPushConstantRangeCount(1).setPPushConstantRanges(&pcRange);
    fragShader = ctx.getDevice().createShaderEXT(fInfo).value;
}

void Renderer::loadModel(const std::string& path) {
    model.loadFromFile(path);
    model.uploadToGPU(ctx);
    initDescriptors(); // Re-init descriptors based on new texture count
}

void Renderer::initDescriptors() {
    // 1. Create Sampler
    vk::SamplerCreateInfo sInfo{};
    sInfo.magFilter = vk::Filter::eLinear; sInfo.minFilter = vk::Filter::eLinear;
    sInfo.addressModeU = vk::SamplerAddressMode::eRepeat; sInfo.addressModeV = vk::SamplerAddressMode::eRepeat; sInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    sInfo.maxAnisotropy = 1.0f;
    textureSampler = ctx.getDevice().createSamplerUnique(sInfo).value;

    // 2. Create Default Textures (White/Normal/MR)
    TextureData whiteData{ 1, 1, 4 }; unsigned char wP[] = { 255,255,255,255 }; whiteData.pixels = wP;
    whiteTex = std::make_unique<VulkanTexture>(ctx, whiteData, vk::Format::eR8G8B8A8Srgb);

    TextureData nData{ 1, 1, 4 }; unsigned char nP[] = { 128,128,255,255 }; nData.pixels = nP;
    normalTex = std::make_unique<VulkanTexture>(ctx, nData, vk::Format::eR8G8B8A8Unorm);

    TextureData mData{ 1, 1, 4 }; unsigned char mP[] = { 0,128,0,255 }; mData.pixels = mP;
    mrTex = std::make_unique<VulkanTexture>(ctx, mData, vk::Format::eR8G8B8A8Unorm);

    // 3. Pool & Sets
    uint32_t count = 3 + (uint32_t)model.textures.size();
    vk::DescriptorPoolSize size{ vk::DescriptorType::eCombinedImageSampler, count };
    vk::DescriptorPoolCreateInfo poolInfo{ {}, count, 1, &size };
    descriptorPool = ctx.getDevice().createDescriptorPoolUnique(poolInfo).value;

    std::vector<vk::DescriptorSetLayout> layouts(count, descriptorLayout.get());
    vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool.get(), count, layouts.data() };
    textureSets = ctx.getDevice().allocateDescriptorSets(allocInfo).value;

    auto updateSet = [&](int index, vk::ImageView view) {
        vk::DescriptorImageInfo info{ textureSampler.get(), view, vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::WriteDescriptorSet write{ textureSets[index], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &info };
        ctx.getDevice().updateDescriptorSets(1, &write, 0, nullptr);
        };

    updateSet(0, whiteTex->getView());
    updateSet(1, normalTex->getView());
    updateSet(2, mrTex->getView());

    for (size_t i = 0; i < model.textures.size(); ++i) {
        updateSet(3 + i, model.textures[i]->getView());
    }
}

void Renderer::initSyncObjects() {
    vk::FenceCreateInfo fInfo{ vk::FenceCreateFlagBits::eSignaled };
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = ctx.getGraphicsQueueFamily();
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        imageAvailableSemaphores.push_back(ctx.getDevice().createSemaphoreUnique({}).value);
        renderFinishedSemaphores.push_back(ctx.getDevice().createSemaphoreUnique({}).value);
        inFlightFences.push_back(ctx.getDevice().createFenceUnique(fInfo).value);
        frameCommandPools.push_back(ctx.getDevice().createCommandPoolUnique(poolInfo).value);
    }
}

void Renderer::render(Camera& cam, float time) {
    if (ctx.getDevice().waitForFences(inFlightFences[currentFrame].get(), VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) return;

    uint32_t imgIndex;
    auto acqResult = ctx.getDevice().acquireNextImageKHR(ctx.getSwapchain().swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame].get(), {}, &imgIndex);
    ctx.getDevice().resetFences(inFlightFences[currentFrame].get());
    if (acqResult != vk::Result::eSuccess && acqResult != vk::Result::eSuboptimalKHR) return;

    vk::CommandPool& currentPool = frameCommandPools[currentFrame].get();
    ctx.getDevice().resetCommandPool(currentPool, {});
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = currentPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;
    vk::CommandBuffer cmd = ctx.getDevice().allocateCommandBuffers(allocInfo).value[0];

    cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    // Transitions
    vk::ImageMemoryBarrier2 colorBarrier{};
    colorBarrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    colorBarrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    colorBarrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    colorBarrier.oldLayout = vk::ImageLayout::eUndefined;
    colorBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorBarrier.image = ctx.getSwapchain().get_images().value()[imgIndex];
    colorBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

    vk::ImageMemoryBarrier2 depthBarrier{};
    depthBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    depthBarrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests;
    depthBarrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    depthBarrier.oldLayout = vk::ImageLayout::eUndefined;
    depthBarrier.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    depthBarrier.image = depthImage;
    depthBarrier.subresourceRange = { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 };

    vk::DependencyInfo depInfo{};
    vk::ImageMemoryBarrier2 barriers[] = { colorBarrier, depthBarrier };
    depInfo.setImageMemoryBarrierCount(2);
    depInfo.setImageMemoryBarriers(barriers);
    cmd.pipelineBarrier2(depInfo);

    // Rendering
    vk::RenderingAttachmentInfo colorAtt{};
    colorAtt.imageView = swapchainImageViews[imgIndex];
    colorAtt.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAtt.loadOp = vk::AttachmentLoadOp::eClear;
    colorAtt.storeOp = vk::AttachmentStoreOp::eStore;
    colorAtt.clearValue = vk::ClearValue{ std::array<float,4>{0.0f,0.2f,1.0f,1.0f} };

    vk::RenderingAttachmentInfo depthAtt{};
    depthAtt.imageView = depthImageView;
    depthAtt.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    depthAtt.loadOp = vk::AttachmentLoadOp::eClear;
    depthAtt.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAtt.clearValue = vk::ClearDepthStencilValue{ 1.0f, 0 };

    vk::RenderingInfo rInfo{};
    rInfo.renderArea = vk::Rect2D{ {0,0}, {1280,720} };
    rInfo.layerCount = 1;
    rInfo.colorAttachmentCount = 1;
    rInfo.pColorAttachments = &colorAtt;
    rInfo.pDepthAttachment = &depthAtt;

    cmd.beginRendering(rInfo);

    // Shader Objects Bindings
    vk::ShaderStageFlagBits stages[] = { vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment };
    vk::ShaderEXT shaders[] = { vertShader, fragShader };
    cmd.bindShadersEXT(2, stages, shaders);

    // Dynamic State
    vk::Viewport vp{ 0,0,1280,720,0,1 };
    vk::Rect2D sc{ {0,0},{1280,720} };
    cmd.setViewportWithCount(1, &vp);
    cmd.setScissorWithCount(1, &sc);
    cmd.setCullMode(vk::CullModeFlagBits::eBack);
    cmd.setFrontFace(vk::FrontFace::eCounterClockwise);
    cmd.setDepthTestEnable(VK_TRUE);
    cmd.setDepthWriteEnable(VK_TRUE);
    cmd.setDepthCompareOp(vk::CompareOp::eLess);
    cmd.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
    // Needed for dynamic rendering + shader objects:
    cmd.setRasterizerDiscardEnable(VK_FALSE);
    cmd.setStencilTestEnable(VK_FALSE);
    cmd.setPolygonModeEXT(vk::PolygonMode::eFill);
    cmd.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
    vk::SampleMask mask = ~0u;
    cmd.setSampleMaskEXT(vk::SampleCountFlagBits::e1, &mask);
    cmd.setAlphaToCoverageEnableEXT(VK_FALSE);
    cmd.setColorBlendEnableEXT(0, VK_FALSE);
    cmd.setColorWriteMaskEXT(0, vk::ColorComponentFlags(0xF));

    // Input Bindings
    auto attr = Vertex::getAttributeDescriptions();
    auto bind = Vertex::getBindingDescription();
    vk::VertexInputBindingDescription2EXT instanceBind{ 1, sizeof(InstanceData), vk::VertexInputRate::eInstance, 1 };
    vk::VertexInputBindingDescription2EXT bindings[] = { bind, instanceBind };
    cmd.setVertexInputEXT(2, bindings, (uint32_t)attr.size(), attr.data());

    vk::Buffer buffs[] = { model.vertexBuffer->getBuffer(), instanceBuffer->getBuffer() };
    vk::DeviceSize offsets[] = { 0,0 };
    vk::DeviceSize sizes[] = { sizeof(Vertex) * model.meshData.vertices.size(), sizeof(InstanceData) };
    vk::DeviceSize strides[] = { sizeof(Vertex), sizeof(InstanceData) };
    cmd.bindVertexBuffers2(0, 2, buffs, offsets, sizes, strides);
    cmd.bindIndexBuffer(model.indexBuffer->getBuffer(), 0, vk::IndexType::eUint32);

    MeshPushConstants pc{};
    pc.view = cam.getView();
    pc.proj = cam.getProjection(1280, 720);
    pc.time = time;
    pc.cameraPos = glm::vec4(cam.position, 0.0f);

    for (const auto& sub : model.meshData.submeshes) {
        pc.baseColor = sub.material.baseColorFactor;
        pc.metallic = sub.material.metallicFactor;
        pc.roughness = sub.material.roughnessFactor;
        cmd.pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);

        // Bind Textures
        // 0=White, 1=Normal, 2=MR. Real textures start at 3.
        int baseIdx = sub.material.baseColorTextureIndex >= 0 ? (sub.material.baseColorTextureIndex + 3) : 0;
        int normIdx = sub.material.normalTextureIndex >= 0 ? (sub.material.normalTextureIndex + 3) : 1;
        int mrIdx = sub.material.metallicRoughnessTextureIndex >= 0 ? (sub.material.metallicRoughnessTextureIndex + 3) : 2;

        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 0, 1, &textureSets[baseIdx], 0, nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 1, 1, &textureSets[normIdx], 0, nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 2, 1, &textureSets[mrIdx], 0, nullptr);

        cmd.drawIndexed(sub.indexCount, 1, sub.indexOffset, sub.vertexOffset, 0);
    }
    cmd.endRendering();

    // Transition back for Present
    vk::ImageMemoryBarrier2 presentBarrier{};
    presentBarrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    presentBarrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    presentBarrier.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
    presentBarrier.dstAccessMask = vk::AccessFlagBits2::eNone;
    presentBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
    presentBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
    presentBarrier.image = ctx.getSwapchain().get_images().value()[imgIndex];
    presentBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

    vk::DependencyInfo presDep{}; presDep.setImageMemoryBarriers(presentBarrier);
    cmd.pipelineBarrier2(presDep);

    cmd.end();

    // Submit
    vk::CommandBufferSubmitInfo csi{}; csi.commandBuffer = cmd;
    vk::SemaphoreSubmitInfo waitS{ imageAvailableSemaphores[currentFrame].get(), 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput };
    vk::SemaphoreSubmitInfo sigS{ renderFinishedSemaphores[currentFrame].get(), 0, vk::PipelineStageFlagBits2::eAllCommands };
    vk::SubmitInfo2 submit{};
    submit.setCommandBufferInfos(csi).setWaitSemaphoreInfos(waitS).setSignalSemaphoreInfos(sigS);

    ctx.getGraphicsQueue().submit2(submit, inFlightFences[currentFrame].get());

    // Present
    vk::PresentInfoKHR present{};
    present.setWaitSemaphores(renderFinishedSemaphores[currentFrame].get());
    vk::SwapchainKHR swapchainHPP(ctx.getSwapchain().swapchain);
    present.setSwapchains(swapchainHPP);
    present.setImageIndices(imgIndex);
    ctx.getPresentQueue().presentKHR(present);

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}