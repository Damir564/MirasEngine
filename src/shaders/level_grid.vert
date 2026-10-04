#version 460
#extension GL_GOOGLE_include_directive : require

#include "frame_ubo.glsl"
#include "draw_data.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 objectPos;
layout(location = 1) out vec3 objectNormal;
layout(location = 2) out vec3 worldPos;
layout(location = 3) out vec3 worldNormal;

// Same expression and qualifier as triangle.vert so depths compare exactly.
invariant gl_Position;

void main() {
    uint drawIndex = gl_InstanceIndex;
    TransformData t = transforms[draws[drawIndex].transformIndex];
    vec4 worldPosition = t.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPosition;
    objectPos = inPosition;
    objectNormal = inNormal;
    worldPos = worldPosition.xyz;
    worldNormal = mat3(t.normal) * inNormal;
}
