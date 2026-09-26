#version 460
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

// Set 0: Frame UBO
layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    float time;
    float shadowBias;
} ubo;

// Must match GpuDrawData / GpuTransform in main.cpp
struct DrawData {
    vec4 baseColor;
    uint transformIndex;
    int alphaMode;
    float metallic;
    float roughness;
    float alphaCutoff;
    float _pad0;
    float _pad1;
    float _pad2;
};

struct TransformData {
    mat4 model;
    mat4 normal;
};

layout(std430, set = 0, binding = 1) readonly buffer DrawBuffer { DrawData draws[]; };
layout(std430, set = 0, binding = 2) readonly buffer TransformBuffer { TransformData transforms[]; };

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out mat3 TBN;
layout(location = 6) out vec4 fragPosLightSpace;
layout(location = 7) flat out uint fragDrawIndex;

void main() {
    // firstInstance of each indirect command is the index of its DrawData entry
    uint drawIndex = gl_InstanceIndex;
    TransformData t = transforms[draws[drawIndex].transformIndex];

    vec4 worldPosition = t.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPosition;

    mat3 normalMatrix = mat3(t.normal);

    vec3 worldNormal = normalize(normalMatrix * inNormal);
    vec3 worldTangent = normalize(normalMatrix * inTangent.xyz);
    vec3 worldBitangent = cross(worldNormal, worldTangent) * inTangent.w;

    fragWorldPos = vec3(worldPosition);
    fragNormal = worldNormal;
    fragTexCoord = inTexCoord;
    TBN = mat3(worldTangent, worldBitangent, worldNormal);
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPosition;
    fragDrawIndex = drawIndex;
}
