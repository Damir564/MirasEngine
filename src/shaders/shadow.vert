#version 460
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

// Set 0: Frame UBO (for lightSpaceMatrix)
layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    float time;
    float shadowBias;
} ubo;

layout(push_constant) uniform ShadowData {
    mat4 modelMatrix;
    float alphaCutoff;
    int alphaMode;
} pc;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    vec4 worldPos = pc.modelMatrix * vec4(inPosition, 1.0);
    gl_Position = ubo.lightSpaceMatrix * worldPos;
    fragTexCoord = inTexCoord;
}