#version 460
#extension GL_KHR_vulkan_glsl : enable

// Vertex attributes from VertexBuffer
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;    // still there
layout(location = 2) in vec2 inTexCoord; // [cite: 30]
layout(location = 3) in vec3 inOffset;   // [cite: 31]

layout(push_constant) uniform MeshData {
    mat4 view;
    mat4 proj;
    vec4 baseColor;
    float metallic;
    float roughness;
} pc;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 N;
layout(location = 2) out vec3 L;
layout(location = 3) out vec2 fragTexCoord;

void main()
{
    // add instance offset
    vec3 pos = inPosition * 1.0 + inOffset; // [cite: 34]
    gl_Position = pc.proj * pc.view * vec4(pos, 1.0);
    N = normalize(inNormal);
    fragColor = N * 0.5 + 0.5;
    // L = normalize(vec3(1.0, 1.0, 0.0));
    L = normalize(vec3(-0.4, 0.8, 0.6)); // [cite: 35]
    // Pass UVs to fragment shader
    fragTexCoord = inTexCoord;
}