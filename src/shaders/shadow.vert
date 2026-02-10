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

layout(location = 0) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.lightSpaceMatrix * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
}