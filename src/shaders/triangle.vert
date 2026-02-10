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

// Push constants - per-material data only
layout(push_constant) uniform MeshData {
    mat4 modelMatrix;
    vec4 baseColor;
    float metallic;
    float roughness;
    float alphaCutoff;
    int alphaMode;
} pc;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out mat3 TBN;
layout(location = 6) out vec4 fragPosLightSpace;

void main() {
    vec4 worldPosition = pc.modelMatrix * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPosition;

    mat3 normalMatrix = transpose(inverse(mat3(pc.modelMatrix)));

    vec3 worldNormal = normalize(normalMatrix * inNormal);
    vec3 worldTangent = normalize(normalMatrix * inTangent.xyz);
    vec3 worldBitangent = cross(worldNormal, worldTangent) * inTangent.w;

    fragWorldPos = vec3(worldPosition);
    fragNormal = worldNormal;
    fragTexCoord = inTexCoord;
    TBN = mat3(worldTangent, worldBitangent, worldNormal);
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPosition;
}