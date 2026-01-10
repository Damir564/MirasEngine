#version 460
#extension GL_NV_uniform_buffer_std430_layout : enable

layout(std430, push_constant) uniform MaterialPush {
    layout(offset = 128) vec3 diffuseColor;
} material;

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(material.diffuseColor, 1.0);
    // outColor = vec4(1.0, 0.0, 0.0, 1.0);
}