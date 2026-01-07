#version 460
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec2 inPosition;

layout(push_constant) uniform PushConstants {
    vec2 offset;
} pc;

vec2 positions[3] = vec2[](
    vec2( 0.0, -0.5),
    vec2( 0.5,  0.5),
    vec2(-0.5,  0.5)
);

void main()
{
    gl_Position = vec4(inPosition + pc.offset, 0.0, 1.0);
}
