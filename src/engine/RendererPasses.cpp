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
    cmd.setDepthClampEnableEXT(VK_FALSE);
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

void Renderer::setAdditiveBlending(vk::CommandBuffer cmd) const
{
    cmd.setColorBlendEnableEXT(0, VK_TRUE);
    vk::ColorBlendEquationEXT blendEquation{};
    blendEquation.srcColorBlendFactor = vk::BlendFactor::eOne;
    blendEquation.dstColorBlendFactor = vk::BlendFactor::eOne;
    blendEquation.colorBlendOp = vk::BlendOp::eAdd;
    blendEquation.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    blendEquation.dstAlphaBlendFactor = vk::BlendFactor::eOne;
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

vk::Rect2D Renderer::renderRect(const FrameInput& input) const
{
    vk::Rect2D rect = sceneRect(input);
    if (!classic() || m_settings.renderScale >= 1.0f)
        return rect;
    const auto scaled = [&](uint32_t size) {
        return std::max(1u, static_cast<uint32_t>(std::lround(static_cast<float>(size) * m_settings.renderScale)));
    };
    rect.extent = vk::Extent2D{ scaled(rect.extent.width), scaled(rect.extent.height) };
    return rect;
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

// Depth-only runs: only the base color texture (set 1, for alpha testing) is bound.
void Renderer::recordDepthRuns(vk::CommandBuffer cmd, vk::PipelineLayout layout, const std::vector<DrawRun>& runs,
    const ModelManager& models) const
{
    cmd.setVertexInputEXT(1, &m_meshBinding, static_cast<uint32_t>(m_meshAttributes.size()), m_meshAttributes.data());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);
    // alpha_test.frag statically uses set 1, so it must be bound even when nothing is alpha-masked.
    vk::DescriptorSet boundSet = models.getDefaultBaseColorSet();
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 1, 1, &boundSet, 0, nullptr);

    const GPUModel* boundModel = nullptr;
    for (const DrawRun& run : runs) {
        if (run.model != boundModel) {
            bindModelBuffers(cmd, run.model);
            boundModel = run.model;
        }
        if (run.sets[0] && run.sets[0] != boundSet) {
            boundSet = run.sets[0];
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 1, 1, &boundSet, 0, nullptr);
        }
        recordRun(cmd, run);
    }
}

void Renderer::recordShadowPasses(vk::CommandBuffer cmd, const ModelManager& models)
{
    const uint32_t size = m_shadowMap.size;
    const vk::Image shadowImage(m_shadowMap.image);
    const auto depthAspect = vk::ImageAspectFlagBits::eDepth;
    const auto depthStages = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
    const vk::Rect2D area{ { 0, 0 }, { size, size } };

    for (uint32_t c = 0; c < m_shadowMap.layers; ++c) {
        if (!m_renderCascade[c])
            continue;
        // Unchanged cascades keep their contents; this one is redrawn from scratch.
        const vk::ImageMemoryBarrier2 toAttachment = imageBarrier(shadowImage, depthAspect,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eNone,
            depthStages, vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal, 0, 1, c, 1);
        pipelineBarriers(cmd, { &toAttachment, 1 });

        vk::RenderingAttachmentInfo depthAttachment{};
        depthAttachment.setImageView(m_shadowMap.layerViews[c])
            .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(vk::ClearValue(vk::ClearDepthStencilValue{ 1.0f, 0 }));
        vk::RenderingInfo renderInfo{};
        renderInfo.setRenderArea(area).setLayerCount(1).setPDepthAttachment(&depthAttachment);

        beginRendering(cmd, renderInfo);
        if (!m_shadowRuns[c].empty()) {
            bindShaderPair(cmd, m_shadowShaders);
            setDefaultDrawState(cmd, vk::SampleCountFlagBits::e1, viewportFor(area), area);
            // Both faces cast: single-sided geometry facing the sun must still block it. The normal offset
            // and slope bias keep lit faces from shadowing themselves.
            cmd.setDepthTestEnable(VK_TRUE);
            cmd.setDepthWriteEnable(VK_TRUE);
            cmd.setDepthBiasEnable(VK_TRUE);
            cmd.setDepthBias(1.25f, 0.0f, 1.75f);
            // Casters in front of the cascade are flattened onto its near plane instead of being clipped.
            cmd.setDepthClampEnableEXT(m_context->features().depthClamp ? VK_TRUE : VK_FALSE);
            const ShadowPushConstants push{ c, {} };
            cmd.pushConstants(m_shadowLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(push), &push);
            recordDepthRuns(cmd, m_shadowLayout, m_shadowRuns[c], models);
        }
        cmd.endRendering();

        const vk::ImageMemoryBarrier2 toSampled = imageBarrier(shadowImage, depthAspect,
            vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
            vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthStencilReadOnlyOptimal, 0, 1, c, 1);
        pipelineBarriers(cmd, { &toSampled, 1 });
    }
}

// ---------------------------------------------------------------------------
// Point and spot lights
// ---------------------------------------------------------------------------

void Renderer::gatherLights(const FrameInput& input)
{
    m_sceneLights.clear();
    m_lightShadowViews.clear();
    if (input.models)
        collectSceneLights(input.models->getInstances(), m_sceneLights);

    // Nearest first, measured to the edge of each light's reach: past kMaxLights the distant ones drop out,
    // and shadows go to the near ones.
    const glm::vec3 camera = input.cameraPosition;
    const auto reach = [&](const SceneLight& light) { return glm::distance(light.position, camera) - light.range; };
    std::sort(m_sceneLights.begin(), m_sceneLights.end(),
        [&](const SceneLight& a, const SceneLight& b) { return reach(a) < reach(b); });
    if (m_sceneLights.size() > kMaxLights)
        m_sceneLights.resize(kMaxLights);

    const bool shadowsAllowed = lightShadowsAllowed();
    if (shadowsAllowed && std::any_of(m_sceneLights.begin(), m_sceneLights.end(),
            [](const SceneLight& light) { return light.shadows; }))
        resizeLightShadowMap(true);
    const bool shadowMapReady = shadowsAllowed && m_lightShadowMap.size == kLightShadowMapSize;
    const float shadowSize = static_cast<float>(kLightShadowMapSize);

    std::byte* mapped = static_cast<std::byte*>(m_lightBuffers[m_currentFrame]->mapped());
    GpuLight* gpuLights = reinterpret_cast<GpuLight*>(mapped + sizeof(GpuLightHeader));
    for (size_t i = 0; i < m_sceneLights.size(); ++i) {
        const SceneLight& light = m_sceneLights[i];
        const float outer = glm::radians(light.coneDegrees);
        const float inner = outer * (1.0f - light.softness);
        GpuLight& gpu = gpuLights[i];
        gpu.positionRange = glm::vec4(light.position, light.range);
        gpu.color = glm::vec4(light.color * light.intensity, light.spot ? 1.0f : 0.0f);
        gpu.direction = glm::vec4(light.direction, std::cos(outer));
        gpu.params = glm::vec4(std::cos(inner), -1.0f, 0.0f, 0.0f);

        const size_t layersNeeded = light.spot ? 1 : 6;
        if (!light.shadows || !shadowMapReady || m_lightShadowViews.size() + layersNeeded > kMaxLightShadowLayers)
            continue;
        // A cube face sees a little more than 90 degrees, so filtering near its edge stays on the face.
        const float fov = light.spot ? std::min(2.0f * outer + glm::radians(4.0f), glm::radians(170.0f))
                                     : 2.0f * std::atan(1.0f + 4.0f / shadowSize);
        gpu.params.y = static_cast<float>(m_lightShadowViews.size());
        gpu.params.z = 2.0f * std::tan(0.5f * fov) / shadowSize;
        const glm::mat4 proj = glm::perspectiveRH_ZO(fov, 1.0f, kLightShadowNear, light.range);
        const auto addView = [&](const glm::vec3& forward, const glm::vec3& up) {
            const glm::mat4 view = glm::lookAtRH(light.position, light.position + forward, up);
            m_lightShadowViews.push_back({ proj * view, light.position, light.range });
        };
        if (light.spot) {
            addView(light.direction, std::abs(light.direction.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f));
        }
        else {
            // In the order lightShadowLayer() in lights.glsl picks them.
            addView({ 1.0f, 0.0f, 0.0f }, { 0.0f, -1.0f, 0.0f });
            addView({ -1.0f, 0.0f, 0.0f }, { 0.0f, -1.0f, 0.0f });
            addView({ 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f });
            addView({ 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, -1.0f });
            addView({ 0.0f, 0.0f, 1.0f }, { 0.0f, -1.0f, 0.0f });
            addView({ 0.0f, 0.0f, -1.0f }, { 0.0f, -1.0f, 0.0f });
        }
    }

    GpuLightHeader header{};
    header.info.x = static_cast<uint32_t>(m_sceneLights.size());
    for (size_t layer = 0; layer < m_lightShadowViews.size(); ++layer)
        header.shadowMatrices[layer] = m_lightShadowViews[layer].matrix;
    std::memcpy(mapped, &header, sizeof(header));
}

// Every caster within a shadow view's light range and frustum draws into that view. Entity markers (a lamp's
// own model around the light) cast none.
void Renderer::buildLightShadowStreams(const FrameInput& input)
{
    for (auto& runs : m_lightShadowRuns)
        runs.clear();
    if (m_lightShadowViews.empty())
        return;
    ModelManager& models = *input.models;
    const size_t sceneCount = models.getInstances().size();
    m_lightTransformIndices.assign(m_cullResults.size(), std::numeric_limits<uint32_t>::max());

    for (size_t layer = 0; layer < m_lightShadowViews.size(); ++layer) {
        const LightShadowView& view = m_lightShadowViews[layer];
        for (size_t i = 0; i < m_cullResults.size(); ++i) {
            const CullResult& res = m_cullResults[i];
            GPUModel* model = res.gpuModel;
            if (!model)
                continue;
            const bool dynamic = i >= sceneCount;
            if (dynamic ? !input.dynamicInstances[i - sceneCount].castShadow : !res.instance->entity.empty())
                continue;
            const glm::vec3 center = glm::vec3(res.transform * glm::vec4(model->boundsCenter, 1.0f));
            if (glm::distance(center, view.lightPosition) > view.range + model->boundsRadius * res.maxScale)
                continue;
            if (!isAABBInFrustum(extractFrustumPlanes(view.matrix * res.transform), model->boundsMin, model->boundsMax))
                continue;
            uint32_t& transformIndex = m_lightTransformIndices[i];
            if (transformIndex == std::numeric_limits<uint32_t>::max())
                transformIndex = pushTransform(res.transform);
            for (uint32_t si : model->drawOrder) {
                const SubmeshInfo& sub = model->submeshes[si];
                if (sub.material.alphaMode == AlphaMode::BLEND)
                    continue;
                vk::DescriptorSet sets[3] = {};
                if (sub.material.alphaMode == AlphaMode::MASK) {
                    vk::DescriptorSet all[3];
                    materialSets(models, model, sub.material, all);
                    sets[0] = all[0];
                }
                appendDraw(m_lightShadowRuns[layer], model, sets, false, sub, transformIndex, glm::vec3(1.0f));
            }
        }
    }
}

void Renderer::recordLightShadowPasses(vk::CommandBuffer cmd, const ModelManager& models)
{
    using Stage = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;
    const vk::Image image(m_lightShadowMap.image);
    const auto depthAspect = vk::ImageAspectFlagBits::eDepth;
    const auto depthStages = Stage::eEarlyFragmentTests | Stage::eLateFragmentTests;
    const uint32_t layers = m_lightShadowMap.layers;

    // Every layer starts over each frame; the ones no light uses this frame only need the layout the
    // lighting set promises.
    const vk::ImageMemoryBarrier2 toAttachment = imageBarrier(image, depthAspect,
        Stage::eFragmentShader, Access::eNone,
        depthStages, Access::eDepthStencilAttachmentRead | Access::eDepthStencilAttachmentWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal, 0, 1, 0, layers);
    pipelineBarriers(cmd, { &toAttachment, 1 });

    const uint32_t size = m_lightShadowMap.size;
    const vk::Rect2D area{ { 0, 0 }, { size, size } };
    const uint32_t used = std::min(static_cast<uint32_t>(m_lightShadowViews.size()), layers);
    for (uint32_t layer = 0; layer < used; ++layer) {
        vk::RenderingAttachmentInfo depthAttachment{};
        depthAttachment.setImageView(m_lightShadowMap.layerViews[layer])
            .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setClearValue(vk::ClearValue(vk::ClearDepthStencilValue{ 1.0f, 0 }));
        vk::RenderingInfo renderInfo{};
        renderInfo.setRenderArea(area).setLayerCount(1).setPDepthAttachment(&depthAttachment);

        beginRendering(cmd, renderInfo);
        if (!m_lightShadowRuns[layer].empty()) {
            bindShaderPair(cmd, m_shadowShaders);
            setDefaultDrawState(cmd, vk::SampleCountFlagBits::e1, viewportFor(area), area);
            // As the sun's: both faces cast, slope bias against acne (the shader adds a normal offset).
            cmd.setDepthTestEnable(VK_TRUE);
            cmd.setDepthWriteEnable(VK_TRUE);
            cmd.setDepthBiasEnable(VK_TRUE);
            cmd.setDepthBias(1.25f, 0.0f, 1.75f);
            cmd.setDepthClampEnableEXT(VK_FALSE);
            const ShadowPushConstants push{ kMaxShadowCascades + layer, {} };
            cmd.pushConstants(m_shadowLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(push), &push);
            recordDepthRuns(cmd, m_shadowLayout, m_lightShadowRuns[layer], models);
        }
        cmd.endRendering();
    }

    const vk::ImageMemoryBarrier2 toSampled = imageBarrier(image, depthAspect,
        depthStages, Access::eDepthStencilAttachmentWrite,
        Stage::eFragmentShader, Access::eShaderSampledRead,
        vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthStencilReadOnlyOptimal, 0, 1, 0, layers);
    pipelineBarriers(cmd, { &toSampled, 1 });
}

void Renderer::transitionFrameTargets(vk::CommandBuffer cmd, bool depthPrepass)
{
    using Stage = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;
    const auto colorAspect = vk::ImageAspectFlagBits::eColor;
    const auto depthAspect = vk::ImageAspectFlagBits::eDepth;
    const auto depthStages = Stage::eEarlyFragmentTests | Stage::eLateFragmentTests;
    const auto depthAccess = Access::eDepthStencilAttachmentRead | Access::eDepthStencilAttachmentWrite;
    // Earlier frames rendered to these images or sampled them; their contents are discarded.
    const auto previousStages = Stage::eFragmentShader | Stage::eColorAttachmentOutput | depthStages;
    const auto previousWrites = Access::eColorAttachmentWrite | Access::eDepthStencilAttachmentWrite;

    std::vector<vk::ImageMemoryBarrier2> barriers;
    auto toColorTarget = [&](const RenderImage& image, uint32_t mips) {
        barriers.push_back(imageBarrier(vk::Image(image.image), colorAspect, previousStages, previousWrites,
            Stage::eColorAttachmentOutput, Access::eColorAttachmentRead | Access::eColorAttachmentWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, 0, mips));
    };
    // Inputs of passes skipped this frame must still be in the layout their descriptors promise.
    auto toUnusedInput = [&](const RenderImage& image) {
        barriers.push_back(imageBarrier(vk::Image(image.image), colorAspect, previousStages, previousWrites,
            Stage::eFragmentShader, Access::eShaderSampledRead,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal));
    };

    // Depth resolves write in the color output stage with color attachment access.
    barriers.push_back(imageBarrier(vk::Image(m_sceneDepth.image), depthAspect, previousStages, previousWrites,
        depthStages | Stage::eColorAttachmentOutput, depthAccess | Access::eColorAttachmentWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal));
    toColorTarget(sceneColorTarget(), 1);
    if (m_samples != vk::SampleCountFlagBits::e1) {
        toColorTarget(m_msaaColor, 1);
        barriers.push_back(imageBarrier(vk::Image(m_msaaDepth.image), depthAspect, previousStages, previousWrites,
            depthStages, depthAccess, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal));
    }
    if (depthPrepass) {
        toColorTarget(m_aoDepth, 1);
        toColorTarget(m_aoRaw, 1);
        toColorTarget(m_aoTemp, 1);
    }
    else {
        toUnusedInput(m_aoDepth);
        toUnusedInput(m_aoRaw);
    }
    if (bloomActive())
        toColorTarget(m_bloom, m_bloomMips);
    else
        toUnusedInput(m_bloom);
    if (fxaaActive())
        toColorTarget(m_ldrColor, 1);
    pipelineBarriers(cmd, barriers);
}

// Depth of the opaque scene before shading: the AO passes read it, and the scene pass then shades each
// pixel once (depth test against these exact depths, no writes).
void Renderer::recordDepthPrepass(vk::CommandBuffer cmd, const FrameInput& input)
{
    const bool msaa = m_samples != vk::SampleCountFlagBits::e1;
    const vk::Rect2D rect = sceneRect(input);

    vk::RenderingAttachmentInfo depthAttachment{};
    depthAttachment.setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue(vk::ClearDepthStencilValue{ 1.0f, 0 }));
    if (msaa) {
        depthAttachment.setImageView(m_msaaDepth.view)
            .setResolveMode(vk::ResolveModeFlagBits::eSampleZero)
            .setResolveImageView(m_sceneDepth.view)
            .setResolveImageLayout(vk::ImageLayout::eDepthAttachmentOptimal);
    }
    else {
        depthAttachment.setImageView(m_sceneDepth.view);
    }
    vk::RenderingInfo renderInfo{};
    renderInfo.setRenderArea(rect).setLayerCount(1).setPDepthAttachment(&depthAttachment);

    beginRendering(cmd, renderInfo);
    if (!m_opaqueRuns.empty()) {
        bindShaderPair(cmd, m_prepassShaders);
        setDefaultDrawState(cmd, m_samples, viewportFor(rect), rect);
        cmd.setDepthTestEnable(VK_TRUE);
        cmd.setDepthWriteEnable(VK_TRUE);
        cmd.setDepthCompareOp(vk::CompareOp::eLess);
        recordDepthRuns(cmd, m_prepassLayout, m_opaqueRuns, *input.models);
    }
    cmd.endRendering();

    using Stage = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;
    const auto depthAspect = vk::ImageAspectFlagBits::eDepth;
    const auto depthStages = Stage::eEarlyFragmentTests | Stage::eLateFragmentTests;
    // Read-only from here on: sampled by the AO passes and the selection mask, depth-tested by the scene pass.
    std::vector<vk::ImageMemoryBarrier2> barriers = {
        imageBarrier(vk::Image(m_sceneDepth.image), depthAspect,
            depthStages | Stage::eColorAttachmentOutput,
            Access::eDepthStencilAttachmentWrite | Access::eColorAttachmentWrite,
            Stage::eFragmentShader | depthStages, Access::eShaderSampledRead | Access::eDepthStencilAttachmentRead,
            vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthReadOnlyOptimal),
    };
    if (msaa) {
        barriers.push_back(imageBarrier(vk::Image(m_msaaDepth.image), depthAspect,
            depthStages | Stage::eColorAttachmentOutput, Access::eDepthStencilAttachmentWrite,
            depthStages, Access::eDepthStencilAttachmentRead | Access::eDepthStencilAttachmentWrite,
            vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthAttachmentOptimal));
    }
    pipelineBarriers(cmd, barriers);
}

void Renderer::recordScenePass(vk::CommandBuffer cmd, const FrameInput& input, bool depthPrepass)
{
    const bool msaa = m_samples != vk::SampleCountFlagBits::e1;
    const vk::Rect2D rect = renderRect(input);
    const bool classicFrame = classic();
    const ShaderPair& meshShaders = classicFrame ? m_classicShaders : m_meshShaders;

    // Cleared to zero coverage: in the solid-background mode, composite.frag fills uncovered pixels.
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setClearValue(vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f })));
    if (msaa) {
        colorAttachment.setImageView(m_msaaColor.view)
            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setResolveMode(vk::ResolveModeFlagBits::eAverage)
            .setResolveImageView(m_hdrColor.view)
            .setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
    }
    else {
        colorAttachment.setImageView(sceneColorTarget().view).setStoreOp(vk::AttachmentStoreOp::eStore);
    }

    vk::RenderingAttachmentInfo depthAttachment{};
    if (depthPrepass) {
        // Depth is final already; this pass only tests against it.
        depthAttachment.setLoadOp(vk::AttachmentLoadOp::eLoad).setStoreOp(vk::AttachmentStoreOp::eNone);
        if (msaa)
            depthAttachment.setImageView(m_msaaDepth.view).setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal);
        else
            depthAttachment.setImageView(m_sceneDepth.view).setImageLayout(vk::ImageLayout::eDepthReadOnlyOptimal);
    }
    else {
        depthAttachment.setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setClearValue(vk::ClearValue(vk::ClearDepthStencilValue{ 1.0f, 0 }));
        if (msaa) {
            depthAttachment.setImageView(m_msaaDepth.view)
                .setStoreOp(vk::AttachmentStoreOp::eDontCare)
                .setResolveMode(vk::ResolveModeFlagBits::eSampleZero)
                .setResolveImageView(m_sceneDepth.view)
                .setResolveImageLayout(vk::ImageLayout::eDepthAttachmentOptimal);
        }
        else {
            depthAttachment.setImageView(m_sceneDepth.view).setStoreOp(vk::AttachmentStoreOp::eStore);
        }
    }

    vk::RenderingInfo renderInfo{};
    renderInfo.setRenderArea(rect)
        .setLayerCount(1)
        .setColorAttachments(colorAttachment)
        .setPDepthAttachment(&depthAttachment);
    beginRendering(cmd, renderInfo);

    setDefaultDrawState(cmd, m_samples, viewportFor(rect), rect);
    cmd.setDepthTestEnable(VK_TRUE);
    cmd.setDepthWriteEnable(depthPrepass ? VK_FALSE : VK_TRUE);
    cmd.setDepthCompareOp(depthPrepass ? vk::CompareOp::eLessOrEqual : vk::CompareOp::eLess);
    // The Classic shaders do not read the lighting set (shadow map, AO, sky textures).
    const auto bindMeshShaders = [&] {
        bindShaderPair(cmd, meshShaders);
        cmd.setVertexInputEXT(1, &m_meshBinding, static_cast<uint32_t>(m_meshAttributes.size()), m_meshAttributes.data());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_meshLayout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);
        if (!classicFrame)
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_meshLayout, 4, 1, &m_lightingSet, 0, nullptr);
    };
    bindMeshShaders();
    recordMeshRuns(cmd, m_opaqueRuns);

    // Sky after the opaque geometry, so it is only shaded where nothing covers it.
    cmd.setDepthWriteEnable(VK_FALSE);
    cmd.setDepthCompareOp(vk::CompareOp::eLessOrEqual);
    if (classicFrame)
        drawFx(cmd, m_classicSkyShaders, {}); // also fills the solid background, so there is no coverage alpha
    else if (m_settings.background == BackgroundMode::Realistic)
        drawFx(cmd, m_skyShaders, { m_skyLutSet });

    // Transparent-ish layers: depth-tested against the opaque scene, no depth writes.
    setAlphaBlending(cmd, true);
    const GridPushConstants gridPush{ std::max(input.gridCellSize, 1e-3f), 8.0f, 0.0f, 0.0f };
    if (input.showGrid)
        drawFx(cmd, m_gridShaders, {}, &gridPush, sizeof(gridPush));
    if (input.levelGrid && !m_highlightRuns.empty()) {
        // Same invariant position as the mesh pass, so LessOrEqual puts the lines exactly on the surface.
        bindShaderPair(cmd, m_levelGridShaders);
        cmd.setVertexInputEXT(1, &m_meshBinding, static_cast<uint32_t>(m_meshAttributes.size()), m_meshAttributes.data());
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_fxLayout, 0, 1, &m_frameSets[m_currentFrame], 0, nullptr);
        std::array<std::byte, kFxPushConstantSize> push{};
        std::memcpy(push.data(), &gridPush, sizeof(gridPush));
        cmd.pushConstants(m_fxLayout, vk::ShaderStageFlagBits::eFragment, 0, kFxPushConstantSize, push.data());
        const GPUModel* boundModel = nullptr;
        for (const DrawRun& run : m_highlightRuns) {
            if (run.model != boundModel) {
                bindModelBuffers(cmd, run.model);
                boundModel = run.model;
            }
            recordRun(cmd, run);
        }
    }

    if (!m_blendRuns.empty()) {
        bindMeshShaders();
        recordMeshRuns(cmd, m_blendRuns);
    }

    setAlphaBlending(cmd, false);
    recordPathLines(cmd, input);
    cmd.endRendering();

    // The HDR image feeds bloom and the composite (the LDR one the Classic upscale); without a prepass the
    // depth was only written just now and the selection mask samples it next. Resolves count as color
    // attachment writes.
    using Stage = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;
    std::vector<vk::ImageMemoryBarrier2> barriers = {
        imageBarrier(vk::Image(sceneColorTarget().image), vk::ImageAspectFlagBits::eColor,
            Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite,
            Stage::eFragmentShader, Access::eShaderSampledRead,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal),
    };
    if (!depthPrepass) {
        barriers.push_back(imageBarrier(vk::Image(m_sceneDepth.image), vk::ImageAspectFlagBits::eDepth,
            Stage::eLateFragmentTests | Stage::eColorAttachmentOutput,
            Access::eDepthStencilAttachmentWrite | Access::eColorAttachmentWrite,
            Stage::eFragmentShader, Access::eShaderSampledRead,
            vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthReadOnlyOptimal));
    }
    pipelineBarriers(cmd, barriers);
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

void Renderer::recordPathLines(vk::CommandBuffer cmd, const FrameInput& input)
{
    if (!input.showPath || m_pathLines.buffer == VK_NULL_HANDLE || m_pathLines.vertexCount == 0)
        return;

    bindShaderPair(cmd, m_gizmoShaders);
    cmd.setPrimitiveTopology(vk::PrimitiveTopology::eLineList);
    cmd.setLineWidth(m_context->features().wideLines ? 2.0f : 1.0f);
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
    beginRendering(cmd, renderInfo);

    const vk::Rect2D rect = sceneRect(input);
    setDefaultDrawState(cmd, vk::SampleCountFlagBits::e1, viewportFor(rect), rect);
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
    // The mask is full resolution; the scene depth may be rendered smaller (Classic render scale).
    const vk::Rect2D depthRect = renderRect(input);
    const glm::vec2 depthScale(float(depthRect.extent.width) / float(rect.extent.width),
        float(depthRect.extent.height) / float(rect.extent.height));
    const glm::vec2 rectOffset(float(rect.offset.x), float(rect.offset.y));
    const glm::vec2 depthOffset = glm::vec2(depthRect.offset.x, depthRect.offset.y) - rectOffset * depthScale;
    std::array<std::byte, kFxPushConstantSize> push{};
    const MaskPushConstants maskPush{ glm::vec4(depthScale, depthOffset) };
    std::memcpy(push.data(), &maskPush, sizeof(maskPush));
    cmd.pushConstants(m_fxLayout, vk::ShaderStageFlagBits::eFragment, 0, kFxPushConstantSize, push.data());

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

// Single-sample pass on the swapchain image: the tone-mapped scene (or its FXAA'd copy), the selection
// outline, then ImGui.
void Renderer::recordFinalPass(vk::CommandBuffer cmd, uint32_t imageIndex, const FrameInput& input, bool drawOutline)
{
    const vk::Rect2D rect = sceneRect(input);
    const vk::Extent2D extent = m_swapchain.extent();
    const CompositePushConstants composite = compositeConstants();
    if (fxaaActive()) {
        beginFxPass(cmd, m_ldrColor.view, rect, vk::AttachmentLoadOp::eDontCare);
        drawFx(cmd, m_compositeShaders, { m_hdrSet, m_bloomSets[0] }, &composite, sizeof(composite));
        cmd.endRendering();
        colorTargetToSampled(cmd, m_ldrColor);
    }

    const vk::Image swapImage(m_swapchain.images()[imageIndex]);
    const vk::ImageMemoryBarrier2 toAttachment = imageBarrier(swapImage, vk::ImageAspectFlagBits::eColor,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
    pipelineBarriers(cmd, { &toAttachment, 1 });

    // Outside the scene viewport the UI covers everything, so a plain clear is enough there.
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.setImageView(m_swapchain.imageViews()[imageIndex])
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{ 0.1f, 0.1f, 0.1f, 1.0f })));
    vk::RenderingInfo renderInfo{};
    renderInfo.setRenderArea({ {0, 0}, extent })
        .setLayerCount(1)
        .setColorAttachments(colorAttachment);
    beginRendering(cmd, renderInfo);
    setDefaultDrawState(cmd, vk::SampleCountFlagBits::e1, viewportFor(rect), rect);

    if (classic()) {
        // Maps each viewport pixel to the matching point of the (possibly smaller) rendered scene.
        const vk::Rect2D source = renderRect(input);
        const glm::vec2 size(float(extent.width), float(extent.height));
        const glm::vec2 scale = glm::vec2(source.extent.width, source.extent.height) /
            glm::vec2(rect.extent.width, rect.extent.height);
        const glm::vec2 offset = glm::vec2(source.offset.x, source.offset.y) -
            glm::vec2(rect.offset.x, rect.offset.y) * scale;
        ClassicPresentPushConstants push{};
        push.transform = glm::vec4(scale / size, offset / size);
        push.uvClamp = glm::vec4((source.offset.x + 0.5f) / size.x, (source.offset.y + 0.5f) / size.y,
            (source.offset.x + source.extent.width - 0.5f) / size.x,
            (source.offset.y + source.extent.height - 0.5f) / size.y);
        drawFx(cmd, m_classicPresentShaders, { m_ldrSet }, &push, sizeof(push));
    }
    else if (fxaaActive()) {
        FxaaPushConstants push{};
        push.texel = glm::vec4(1.0f / extent.width, 1.0f / extent.height, 0.0f, 0.0f);
        push.uvClamp = glm::vec4((rect.offset.x + 0.5f) / extent.width, (rect.offset.y + 0.5f) / extent.height,
            (rect.offset.x + rect.extent.width - 0.5f) / extent.width,
            (rect.offset.y + rect.extent.height - 0.5f) / extent.height);
        drawFx(cmd, m_fxaaShaders, { m_ldrSet }, &push, sizeof(push));
    }
    else {
        drawFx(cmd, m_compositeShaders, { m_hdrSet, m_bloomSets[0] }, &composite, sizeof(composite));
    }

    if (drawOutline) {
        setAlphaBlending(cmd, true);
        OutlinePushConstants push{};
        push.color = glm::vec4(kOutlineColor, 1.0f);
        push.occludedAlpha = kOutlineOccludedAlpha;
        push.widthPixels = kOutlineWidthPixels;
        drawFx(cmd, m_outlineShaders, { m_selectionMaskSet }, &push, sizeof(push));
        setAlphaBlending(cmd, false);
    }

    if (input.imgui && input.imgui->TotalVtxCount > 0)
        ImGui_ImplVulkan_RenderDrawData(input.imgui, cmd);
    cmd.endRendering();

    const vk::ImageMemoryBarrier2 toPresent = imageBarrier(swapImage, vk::ImageAspectFlagBits::eColor,
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
        LOG_ERROR("Failed to present\n");

    m_currentFrame = (m_currentFrame + 1) % m_framesInFlight;
    return FrameStatus::Rendered;
}
