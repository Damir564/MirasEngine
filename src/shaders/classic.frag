#version 460
#extension GL_GOOGLE_include_directive : require

#include "frame_ubo.glsl"
#include "draw_data.glsl"
#include "classic.glsl"

layout(location = 0) in vec3 fragLight;
layout(location = 1) in vec4 fragFog;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) flat in uint fragDrawIndex;

layout(set = 1, binding = 0) uniform sampler2D baseColorSampler;

layout(location = 0) out vec4 outColor;

const int ALPHA_MODE_OPAQUE = 0;
const int ALPHA_MODE_MASK = 1;
const int ALPHA_MODE_BLEND = 2;

void main() {
    DrawData d = draws[fragDrawIndex];
    vec4 texColor = texture(baseColorSampler, fragTexCoord);
    float alpha = texColor.a * d.baseColor.a;

    if (d.alphaMode == ALPHA_MODE_MASK) {
        if (alpha < d.alphaCutoff)
            discard;
        alpha = 1.0;
    } else if (d.alphaMode == ALPHA_MODE_OPAQUE) {
        alpha = 1.0;
    }
    if (alpha < 0.001)
        discard;

    vec3 color = d.baseColor.rgb * texColor.rgb * fragLight;
    outColor = vec4(classicDither(mix(color, fragFog.rgb, fragFog.a), gl_FragCoord.xy), alpha);
}
