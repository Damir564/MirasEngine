#version 460
#extension GL_GOOGLE_include_directive : require

#include "frame_ubo.glsl"

// Classic pipeline: upscales the scene, rendered into a scaled-down corner of the scene target, to the
// viewport. No tone mapping; the scene is already in display range.
layout(set = 1, binding = 0) uniform sampler2D sceneColor;

layout(push_constant) uniform ClassicPresentPush {
    vec4 transform; // source UV = gl_FragCoord.xy * xy + zw
    vec4 uvClamp;   // source UVs are clamped to [xy, zw] so filtering does not pick up texels outside it
} pc;

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = clamp(gl_FragCoord.xy * pc.transform.xy + pc.transform.zw, pc.uvClamp.xy, pc.uvClamp.zw);
    vec3 color = textureLod(sceneColor, uv, 0.0).rgb;

    // +-1 LSB of noise in sRGB space hides banding in the sky gradient and fog.
    float noise = interleavedGradientNoise(gl_FragCoord.xy) - 0.5;
    vec3 srgb = mix(color * 12.92, 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), color));
    srgb = clamp(clamp(srgb, 0.0, 1.0) + noise / 255.0, 0.0, 1.0);
    color = mix(srgb / 12.92, pow((srgb + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), srgb));
    outColor = vec4(color, 1.0);
}
