#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    float time;
    float shadowBias;
} ubo;

layout(push_constant) uniform GizmoPush {
    mat4 modelMatrix;
} pc;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * pc.modelMatrix * vec4(inPosition, 1.0);
    fragColor = inColor;
}
