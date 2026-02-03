#version 460
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec2 fragTexCoord;

// Set 1: Base Color texture (for alpha testing)
layout(set = 1, binding = 0) uniform sampler2D baseColorSampler;

layout(push_constant) uniform ShadowData {
    float alphaCutoff;
    int alphaMode;
} pc;

void main() {
    // Only do alpha test for MASK mode (1)
    if (pc.alphaMode == 1) {
        float alpha = texture(baseColorSampler, fragTexCoord).a;
        if (alpha < pc.alphaCutoff) {
            discard;
        }
    }
    // For BLEND mode (2), skip shadows entirely (handled in CPU code)
}