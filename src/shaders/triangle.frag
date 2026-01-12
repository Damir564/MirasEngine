#version 460
#extension GL_NV_uniform_buffer_std430_layout : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 N;
layout(location = 2) in vec3 L;
layout(location = 0) out vec4 outColor;

void main()
{
    vec4 normalColored = vec4(fragColor, 1.0);
    vec4 lambertianColored = vec4(vec3(max(dot(N, L), 0.01)), 1.0);
    // vec4 lambertianColored = vec4(vec3(max(dot(N, L), 0.0)), 1.0);
    // outColor = mix(lambertianColored, vec4(0.0, 1.0, 0.4, 0.0), 0.1);
    outColor = lambertianColored;
}