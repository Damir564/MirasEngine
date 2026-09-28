#pragma once
#include <glm/glm.hpp>
#include <cstdint>

// CPU mirrors of GLSL blocks. Layouts must match the shaders exactly, including explicit padding.

// Mirrors FrameUBO (set 0, binding 0) in shaders/frame_ubo.glsl (std140).
struct FrameUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 invViewProj;
    glm::mat4 lightSpaceMatrix;
    glm::vec4 cameraPos;     // xyz; w unused
    glm::vec4 lightDir;      // xyz = direction the sunlight travels; w unused
    glm::vec4 sunColor;      // rgb, a = intensity
    glm::vec4 skyZenith;     // rgb; w unused
    glm::vec4 skyHorizon;    // rgb; w unused
    glm::vec4 groundColor;   // rgb; w unused
    glm::vec4 fogParams;     // x = enabled, y = distance where fog is ~63%, zw unused
    glm::vec4 shadowParams;  // x = bias, y = enabled, z = fade distance, w = 1 / map size
    float time;
    float nearPlane;
    float farPlane;
    float _pad0;
};
static_assert(sizeof(FrameUBO) == 400);

// Mirrors the push_constant block in shaders/outline.frag.
struct OutlinePushConstants {
    glm::vec4 color;          // rgb, a = alpha of the visible outline
    float occludedAlpha;
    float widthPixels;
    float _pad0;
    float _pad1;
};
static_assert(sizeof(OutlinePushConstants) == 32);

// Mirrors DrawData (set 0, binding 1) in shaders/draw_data.glsl (std430), indexed by gl_InstanceIndex.
struct GpuDrawData {
    glm::vec4 baseColor{ 1.0f };
    uint32_t transformIndex = 0;
    int32_t alphaMode = 0;
    float metallic = 0.0f;
    float roughness = 0.5f;
    float alphaCutoff = 0.5f;
    float _pad[3]{};
};
static_assert(sizeof(GpuDrawData) == 48);

// Mirrors TransformData (set 0, binding 2) in shaders/draw_data.glsl (std430). The normal
// matrix is precomputed so shaders don't invert per vertex.
struct GpuTransform {
    glm::mat4 model{ 1.0f };
    glm::mat4 normal{ 1.0f };
};
static_assert(sizeof(GpuTransform) == 128);

// Mirrors CullBounds in shaders/occlusion_cull.comp (std430); one per indirect command.
struct GpuCullBounds {
    glm::vec3 boundsMin{ 0.0f };
    uint32_t transformIndex = 0;
    glm::vec3 boundsMax{ 0.0f };
    uint32_t cullable = 0; // 0 = always draw (no bounds, or not a main-pass draw)
};
static_assert(sizeof(GpuCullBounds) == 32);

// Mirrors the push_constant block in shaders/occlusion_cull.comp.
struct CullPushConstants {
    glm::mat4 prevViewProj;
    glm::vec4 viewRect;
    glm::vec2 pyramidSize;
    uint32_t firstCommand;
    uint32_t commandCount;
};
static_assert(sizeof(CullPushConstants) == 96);

// Mirrors the push_constant block in shaders/depth_reduce.comp.
struct PyramidPushConstants {
    uint32_t dstWidth;
    uint32_t dstHeight;
};
static_assert(sizeof(PyramidPushConstants) == 8);
