#version 460
#extension GL_GOOGLE_include_directive : require

#include "frame_ubo.glsl"
#include "classic.glsl"

// Classic pipeline background, drawn with fullscreen.vert after the opaque geometry.
layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 world = ubo.invViewProj * vec4(inNdc, 1.0, 1.0);
    vec3 dir = normalize(world.xyz / world.w - ubo.cameraPos.xyz);
    outColor = vec4(classicDither(classicSkyColor(dir), gl_FragCoord.xy), 1.0);
}
