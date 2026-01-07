#version 460

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pc;

void main()
{
    outColor = vec4(1.0, 0.0, 0.0, 1.0);
}