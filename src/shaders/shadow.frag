#version 460
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) flat in uint fragDrawIndex;

// Set 1: Base Color texture (for alpha testing)
layout(set = 1, binding = 0) uniform sampler2D baseColorSampler;

// Must match GpuDrawData in main.cpp
struct DrawData {
    vec4 baseColor;
    uint transformIndex;
    int alphaMode;
    float metallic;
    float roughness;
    float alphaCutoff;
    float _pad0;
    float _pad1;
    float _pad2;
};

layout(std430, set = 0, binding = 1) readonly buffer DrawBuffer { DrawData draws[]; };

void main() {
    DrawData d = draws[fragDrawIndex];
    // Only do alpha test for MASK mode (1). BLEND submeshes are never drawn into the shadow map.
    if (d.alphaMode == 1) {
        float alpha = texture(baseColorSampler, fragTexCoord).a;
        if (alpha < d.alphaCutoff) {
            discard;
        }
    }
}
