#version 460
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inOffset;

layout(push_constant) uniform Camera {
    mat4 view;
    mat4 proj;
} pc;

void main()
{
    vec3 pos = inPosition + vec3(inOffset, 0.0);
    gl_Position = pc.proj * pc.view * vec4(pos, 1.0);
}
