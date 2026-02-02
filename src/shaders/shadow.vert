#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;
layout(location = 4) in vec3 inInstanceOffset;

layout(push_constant) uniform PushConstants {
    mat4 lightSpaceMatrix;
} pc;

void main() {
    vec3 worldPos = inPosition + inInstanceOffset;
    gl_Position = pc.lightSpaceMatrix * vec4(worldPos, 1.0);
}
