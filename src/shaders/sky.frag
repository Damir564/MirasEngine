#version 460
#extension GL_GOOGLE_include_directive : require

#include "frame_ubo.glsl"

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 world = ubo.invViewProj * vec4(inNdc, 0.5, 1.0);
    vec3 dir = normalize(world.xyz / world.w - ubo.cameraPos.xyz);

    vec3 color = skyGradient(dir);
    float sunAmount = dot(dir, -normalize(ubo.lightDir.xyz));
    float disk = smoothstep(0.99955, 0.99985, sunAmount);
    color += ubo.sunColor.rgb * (disk * 4.0 + pow(max(sunAmount, 0.0), 256.0) * 0.6);
    outColor = vec4(min(color, vec3(1.0)), 1.0);
}
