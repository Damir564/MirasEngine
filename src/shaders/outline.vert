#version 460

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    float time;
    float shadowBias;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    float outlineWidth;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

void main()
{
    // Transform normal to world space
    mat3 normalMatrix = transpose(inverse(mat3(pc.modelMatrix)));
    vec3 worldNormal = normalize(normalMatrix * inNormal);
    
    // Expand vertex along world-space normal
    vec4 worldPos = pc.modelMatrix * vec4(inPosition, 1.0);
    worldPos.xyz += worldNormal * pc.outlineWidth;
    
    gl_Position = ubo.proj * ubo.view * worldPos;
}