#version 460
#extension GL_GOOGLE_include_directive : require

#include "draw_data.glsl"

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) flat in uint fragDrawIndex;

// Base color texture, for alpha testing.
layout(set = 1, binding = 0) uniform sampler2D baseColorSampler;

void main() {
    DrawData d = draws[fragDrawIndex];
    // Only MASK (1) needs the test; BLEND submeshes are never drawn into the shadow map.
    if (d.alphaMode == 1 && texture(baseColorSampler, fragTexCoord).a < d.alphaCutoff)
        discard;
}
