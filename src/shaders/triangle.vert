#version 460
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3 inPos;

layout(push_constant) uniform Camera {
    mat4 view;
    mat4 proj;
} camera;

void main()
{
    gl_Position = camera.proj * camera.view * vec4(inPos, 1.0f);
}
