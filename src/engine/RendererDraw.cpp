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
        LOG_ERROR("Failed to acquireNextImageKHR\n");
        return FrameStatus::Failed;
    }
    if (result == vk::Result::eSuboptimalKHR)
        m_swapchainDirty = true;

    // Reset only once we know this frame will be submitted; otherwise the next wait would hang.
    (void)m_device.resetFences(fence);

    const glm::vec2 depthRange = projectionDepthRange(input.proj);
    computeShadowCascades(input.view, input.proj, depthRange.x, m_settings.shadowDistance, m_sun.direction,
        m_shadowMap.size, std::span(m_cascades.data(), m_shadowMap.layers));

    const FrameUBO frameData = buildFrameUBO(input);
    memcpy(m_frameUBOs[m_currentFrame].mapped, &frameData, sizeof(FrameUBO));
    gatherLights(input);

    std::array<uint64_t, kMaxShadowCascades> shadowHashes{};
    cullAndBatch(input, frameData.proj * frameData.view, m_batches, shadowHashes);
    // A cascade whose casters and matrix did not change keeps last frame's contents.
    for (uint32_t c = 0; c < m_shadowMap.layers; ++c) {
        m_renderCascade[c] = !m_shadowMapValid || shadowHashes[c] != m_cascadeHashes[c];
        m_cascadeHashes[c] = shadowHashes[c];
    }
    // Classic uses neither the shadow map nor the sky textures; they are rebuilt once Standard is back.
    const bool classicFrame = classic();
    m_shadowMapValid = !classicFrame;

    const uint64_t sky = skyHash();
    const bool renderSky = !classicFrame && (!m_skyValid || sky != m_skyHash);
    m_skyHash = sky;
    m_skyValid = !classicFrame;

    buildDrawStreams(input, m_batches);
    uploadDrawStreams();

    const vk::CommandBuffer cmd = m_commandBuffers[m_currentFrame];
    (void)cmd.reset();
    (void)cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    if (renderSky)
        recordSkyPasses(cmd);
    if (!classicFrame) {
        recordShadowPasses(cmd, *input.models);
        recordLightShadowPasses(cmd, *input.models);
    }

    const bool depthPrepass = depthPrepassEnabled();
    transitionFrameTargets(cmd, depthPrepass);
    if (depthPrepass) {
        recordDepthPrepass(cmd, input);
        recordAoPasses(cmd, input);
    }
    recordScenePass(cmd, input, depthPrepass);
    const bool drawOutline = !m_highlightRuns.empty();
    if (drawOutline)
        recordSelectionMask(cmd, input);
    if (bloomActive())
        recordBloom(cmd, input);
    recordFinalPass(cmd, imageIndex, input, drawOutline);

    (void)cmd.end();
    ++m_frameIndex;
    return submitAndPresent(cmd, imageIndex);
}

FrameUBO Renderer::buildFrameUBO(const FrameInput& input) const
{
    FrameUBO frameData{};
    frameData.view = input.view;
    frameData.proj = input.proj;
    frameData.invViewProj = glm::inverse(input.proj * input.view);

    for (uint32_t c = 0; c < kMaxShadowCascades; ++c) {
        const ShadowCascade& cascade = m_cascades[c];
        frameData.cascadeMatrices[c] = cascade.matrix;
        frameData.cascadeSplits[c] = cascade.splitEnd;
        frameData.cascadeParams[c] = glm::vec4(cascade.texelWorld, cascade.depthRange,
            1.0f / (2.0f * std::max(cascade.radius, 1e-3f)), 0.0f);
    }

    frameData.cameraPos = glm::vec4(input.cameraPosition, 1.0f);
    frameData.lightDir = glm::vec4(m_sun.direction, kSunAngularRadius);
    frameData.sunColor = glm::vec4(m_sun.groundIrradiance, 0.0f);
    frameData.sunTopColor = glm::vec4(m_sun.topIrradiance, 0.0f);
    const bool realistic = m_settings.background == BackgroundMode::Realistic;
    frameData.backgroundColor = glm::vec4(srgbToLinear(m_settings.backgroundColor), realistic ? 1.0f : 0.0f);
    frameData.atmosphereParams = glm::vec4(m_settings.haze, kGroundAlbedo, 0.0f, 0.0f);

    const glm::vec2 depthRange = projectionDepthRange(input.proj);
    frameData.fogParams = glm::vec4(m_settings.fog ? 1.0f : 0.0f, kFogDensity * m_settings.fogDensity,
        kFogHeightFalloff, depthRange.y * 0.8f);
    frameData.shadowParams = glm::vec4(shadowsActive() ? 1.0f : 0.0f, static_cast<float>(m_shadowMap.layers),
        m_settings.shadowDistance, m_settings.softShadows ? 1.0f : 0.0f);
    frameData.shadowParams2 = glm::vec4(1.0f / static_cast<float>(std::max(m_shadowMap.size, 1u)),
        kPenumbraPerMeter, 0.0f, 0.0f);
    frameData.aoParams = glm::vec4(m_settings.ambientOcclusion > 0 ? 1.0f : 0.0f, m_settings.aoRadius,
        m_settings.aoIntensity, static_cast<float>(kAoSamples[m_settings.ambientOcclusion]));
    frameData.aoParams2 = glm::vec4(contactShadowsActive() ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    frameData.projParams = glm::vec4(1.0f / input.proj[0][0], 1.0f / input.proj[1][1], 0.0f, 0.0f);

    const vk::Rect2D rect = renderRect(input);
    frameData.viewport = glm::vec4(rect.offset.x, rect.offset.y, rect.extent.width, rect.extent.height);
    const vk::Extent2D extent = m_swapchain.extent();
    frameData.renderSize = glm::vec4(extent.width, extent.height, 1.0f / extent.width, 1.0f / extent.height);

    frameData.time = input.time;
    frameData.nearPlane = depthRange.x;
    frameData.farPlane = depthRange.y;
    frameData.frameIndex = m_frameIndex;
    return frameData;
}

void Renderer::cullAndBatch(const FrameInput& input, const glm::mat4& viewProj, FrameBatches& batches,
    std::array<uint64_t, kMaxShadowCascades>& shadowHashes)
{
    ModelManager& models = *input.models;
    const bool shadows = shadowsActive();
    const uint32_t cascadeCount = shadows ? m_shadowMap.layers : 0;
    const auto& instances = models.getInstances();
    // Scene instances first, then the frame's dynamic ones.
    const size_t sceneCount = instances.size();
    const size_t totalCount = sceneCount + input.dynamicInstances.size();
    m_cullResults.resize(totalCount);
    m_cullIndices.resize(totalCount);
    std::iota(m_cullIndices.begin(), m_cullIndices.end(), 0);

    // Detail culling: a sphere of radius r at distance d covers about 2 r * pixelsPerUnit / d pixels.
    const float detailPixels = m_settings.detailCulling;
    const float pixelsPerUnit = 0.5f * std::abs(input.proj[1][1]) * static_cast<float>(sceneRect(input).extent.height);
    const glm::vec3 cameraPos = input.cameraPosition;
    const auto largeEnough = [&](const glm::mat4& transform, const glm::vec3& center, float worldRadius) {
        if (detailPixels <= 0.0f)
            return true;
        const float distance = glm::length(glm::vec3(transform * glm::vec4(center, 1.0f)) - cameraPos);
        return distance <= worldRadius || 2.0f * worldRadius * pixelsPerUnit >= detailPixels * distance;
    };

    // 1. Per-instance phase: transform + whole-model culling.
    forEachMaybeParallel(totalCount >= kParallelCullThreshold, m_cullIndices.begin(), m_cullIndices.end(), [&](size_t i) {
        const bool dynamic = i >= sceneCount;
        const auto& inst = dynamic ? input.dynamicInstances[i - sceneCount].instance : instances[i];
        const bool castShadow = !dynamic || input.dynamicInstances[i - sceneCount].castShadow;
        auto& res = m_cullResults[i];

        res.instance = &inst;
        res.modelIndex = inst.modelIndex;
        res.gpuModel = nullptr;
        res.visibleMain = false;
        res.shadowMask = 0;

        if (!inst.visible || (input.hideEntities && !dynamic && !inst.entity.empty())) return;

        GPUModel* gpuModel = models.getModel(inst.modelIndex);
        if (!gpuModel || !gpuModel->isValid()) return;

        res.gpuModel = gpuModel;
        if (dynamic) {
            res.transform = input.dynamicInstances[i - sceneCount].transform;
            res.maxScale = std::max({ glm::length(glm::vec3(res.transform[0])), glm::length(glm::vec3(res.transform[1])),
                glm::length(glm::vec3(res.transform[2])) });
        }
        else {
            res.transform = inst.getTransformMatrix();
            res.maxScale = std::max({ std::abs(inst.scale.x), std::abs(inst.scale.y), std::abs(inst.scale.z) });
        }

        res.mainPlanes = extractFrustumPlanes(viewProj * res.transform);
        res.visibleMain = isAABBInFrustum(res.mainPlanes, gpuModel->boundsMin, gpuModel->boundsMax) &&
            largeEnough(res.transform, gpuModel->boundsCenter, gpuModel->boundsRadius * res.maxScale);

        for (uint32_t c = 0; castShadow && c < cascadeCount; ++c) {
            FrustumPlanes planes = extractFrustumPlanes(m_cascades[c].matrix * res.transform);
            // Casters between the sun and the cascade still throw shadows into it (depth clamp flattens them).
            planes.planes[4] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            if (isAABBInFrustum(planes, gpuModel->boundsMin, gpuModel->boundsMax)) {
                res.shadowPlanes[c] = planes;
                res.shadowMask |= static_cast<uint8_t>(1u << c);
            }
        }

        res.submeshFlags.assign(gpuModel->submeshes.size(), 0);
    });

    // 2. Per-submesh phase, split into chunks so a single huge model still uses all cores.
    m_cullChunks.clear();
    for (size_t i = 0; i < m_cullResults.size(); ++i) {
        const auto& res = m_cullResults[i];
        if (!res.gpuModel || !(res.visibleMain || res.shadowMask)) continue;
        const size_t n = res.gpuModel->submeshes.size();
        for (size_t b = 0; b < n; b += kCullChunkSize)
            m_cullChunks.push_back({ i, b, std::min(b + kCullChunkSize, n) });
    }

    forEachMaybeParallel(m_cullChunks.size() > 1, m_cullChunks.begin(), m_cullChunks.end(), [&](const CullChunk& chunk) {
        auto& res = m_cullResults[chunk.instanceIdx];
        const auto& submeshes = res.gpuModel->submeshes;

        for (size_t s = chunk.begin; s < chunk.end; ++s) {
            const SubmeshInfo& sub = submeshes[s];
            const bool validBounds = sub.boundsMin.x <= sub.boundsMax.x;
            uint8_t flags = 0;
            if (res.visibleMain && (!validBounds ||
                    (isAABBInFrustum(res.mainPlanes, sub.boundsMin, sub.boundsMax) &&
                     largeEnough(res.transform, 0.5f * (sub.boundsMin + sub.boundsMax),
                         0.5f * glm::length(sub.boundsMax - sub.boundsMin) * res.maxScale))))
                flags |= kVisibleMain;
            if (res.shadowMask && sub.material.alphaMode != AlphaMode::BLEND) {
                const float radius = validBounds
                    ? 0.5f * glm::length(sub.boundsMax - sub.boundsMin) * res.maxScale
                    : std::numeric_limits<float>::max();
                for (uint32_t c = 0; c < cascadeCount; ++c) {
                    // Casters smaller than a texel would only add aliasing noise to a cascade.
                    if (!(res.shadowMask & (1u << c)) || radius < m_cascades[c].texelWorld)
                        continue;
                    if (!validBounds || isAABBInFrustum(res.shadowPlanes[c], sub.boundsMin, sub.boundsMax))
                        flags |= static_cast<uint8_t>(kVisibleShadow << c);
                }
            }
            res.submeshFlags[s] = flags;
        }
    });

    // 3. Sequential aggregation, hashing everything that affects each cascade on the way.
    std::array<Hasher, kMaxShadowCascades> hashers;
    for (uint32_t c = 0; c < m_shadowMap.layers; ++c) {
        hashers[c].byte(static_cast<uint8_t>(shadows ? 1 : 0));
        hashers[c].bytes(&m_shadowMap.size, sizeof(m_shadowMap.size));
        if (shadows)
            hashers[c].bytes(&m_cascades[c].matrix, sizeof(glm::mat4));
    }

    m_frameTransforms.clear();
    batches.clear();
    for (const auto& res : m_cullResults) {
        if (!res.gpuModel || !(res.visibleMain || res.shadowMask)) continue;

        const uint32_t transformIndex = pushTransform(res.transform);
        InstanceRenderData renderData{ res.instance, res.transform, res.submeshFlags.data(), transformIndex };

        if (res.visibleMain)
            batches.add(batches.main, batches.mainModels, res.modelIndex, res.gpuModel).instances.push_back(renderData);

        if (res.shadowMask) {
            batches.add(batches.shadow, batches.shadowModels, res.modelIndex, res.gpuModel).instances.push_back(renderData);

            for (uint32_t c = 0; c < cascadeCount; ++c) {
                if (!(res.shadowMask & (1u << c))) continue;
                Hasher& hash = hashers[c];
                hash.bytes(&res.gpuModel, sizeof(res.gpuModel));
                hash.bytes(&res.transform, sizeof(glm::mat4));
                const uint8_t bit = static_cast<uint8_t>(kVisibleShadow << c);
                for (uint8_t f : res.submeshFlags)
                    hash.byte(f & bit);
            }
        }
    }
    for (uint32_t c = 0; c < m_shadowMap.layers; ++c)
        shadowHashes[c] = hashers[c].value();
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
void Renderer::buildDrawStreams(const FrameInput& input, FrameBatches& batches)
{
    ModelManager& models = *input.models;
    const glm::vec3 cameraPos = input.cameraPosition;

    m_frameDraws.clear();
    m_frameCommands.clear();
    for (auto& runs : m_shadowRuns)
        runs.clear();
    m_opaqueRuns.clear();
    m_blendRuns.clear();

    // Only cascades that are re-rendered this frame need draws.
    for (uint32_t c = 0; c < m_shadowMap.layers; ++c) {
        if (!m_renderCascade[c]) continue;
        const uint8_t bit = static_cast<uint8_t>(kVisibleShadow << c);
        for (size_t modelIdx : batches.shadowModels) {
            const RenderBatch& batch = batches.shadow[modelIdx];
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
                    if (rd.submeshFlags[si] & bit)
                        appendDraw(m_shadowRuns[c], model, sets, false, sub, rd.transformIndex, glm::vec3(1.0f));
            }
        }
    }
    buildLightShadowStreams(input);

    for (size_t modelIdx : batches.mainModels) {
        const RenderBatch& batch = batches.main[modelIdx];
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

    for (size_t modelIdx : batches.mainModels) {
        const RenderBatch& batch = batches.main[modelIdx];
        GPUModel* model = batch.model;
        bool sorted = false;
        for (std::size_t si = 0; si < model->submeshes.size(); ++si) {
            const SubmeshInfo& sub = model->submeshes[si];
            if (sub.material.alphaMode != AlphaMode::BLEND) continue;

            if (!sorted) {
                m_sortedInstances.assign(batch.instances.begin(), batch.instances.end());
                std::sort(m_sortedInstances.begin(), m_sortedInstances.end(), [&](const InstanceRenderData& a, const InstanceRenderData& b) {
                    return glm::distance(cameraPos, a.instance->position) > glm::distance(cameraPos, b.instance->position);
                });
                sorted = true;
            }

            vk::DescriptorSet sets[3];
            materialSets(models, model, sub.material, sets);
            for (const auto& rd : m_sortedInstances)
                if (rd.submeshFlags[si] & kVisibleMain)
                    appendDraw(m_blendRuns, model, sets, true, sub, rd.transformIndex, rd.instance->color);
        }
    }

    buildHighlightStream(input);
}

// The selection mask draws the highlighted instances whether or not their submeshes passed culling, so
// the outline of partly off-screen objects stays correct.
void Renderer::buildHighlightStream(const FrameInput& input)
{
    m_highlightRuns.clear();
    ModelManager& models = *input.models;
    const auto& instances = models.getInstances();
    const vk::DescriptorSet noSets[3] = {};
    for (const int index : input.highlight.instances) {
        if (index < 0 || index >= static_cast<int>(instances.size()))
            continue;
        const ModelInstance& inst = instances[index];
        GPUModel* model = models.getModel(inst.modelIndex);
        if (!inst.visible || !model || !model->isValid())
            continue;
        const uint32_t transformIndex = pushTransform(inst.getTransformMatrix());
        for (const SubmeshInfo& sub : model->submeshes)
            appendDraw(m_highlightRuns, model, noSets, false, sub, transformIndex, glm::vec3(1.0f));
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
