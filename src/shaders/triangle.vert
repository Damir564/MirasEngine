#version 460
#extension GL_KHR_vulkan_glsl : enable

// Vertex attributes from VertexBuffer
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;    // still there
layout(location = 2) in vec2 inTexCoord;  // still there

// Instance data from InstanceBuffer
layout(location = 3) in vec3 inOffset;    // location 3 now

layout(push_constant) uniform Camera {
    mat4 view;
    mat4 proj;
} pc;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 N;
layout(location = 2) out vec3 L;

void main()
{
    // add instance offset
    vec3 pos = inPosition * 1.0 + inOffset;
    gl_Position = pc.proj * pc.view * vec4(pos, 1.0);
    N = normalize(inNormal);
    fragColor = N * 0.5 + 0.5;
    // L = normalize(vec3(1.0, 1.0, 0.0));
    L = normalize(vec3(-0.5, 1.0, -0.3));
}