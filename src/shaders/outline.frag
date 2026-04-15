#version 460

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    float outlineWidth;
    float _pad0;
    float _pad1;
    float _pad2;
    vec4 outlineColor;
} pc;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = pc.outlineColor;
}