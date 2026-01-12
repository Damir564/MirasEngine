#version 460
#extension GL_NV_uniform_buffer_std430_layout : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 N;
layout(location = 2) in vec3 L;

layout(push_constant) uniform MeshData {
    mat4 view;
    mat4 proj;
    vec4 baseColor;
    float metallic;
    float roughness;
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// 1. Normal Distribution Function (Trowbridge-Reitz GGX)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / denom;
}

// 2. Geometry Function (Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0), roughness) * GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

// 3. Fresnel Equation (Schlick)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    float lightIntensity = 1.5;
    vec3 normal = normalize(N);
    vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0)); // Simplification: assuming view is forward
    vec3 lightDir = normalize(L);
    vec3 halfwayDir = normalize(viewDir + lightDir);

    // Surface reflection at zero incidence
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, pc.baseColor.rgb, pc.metallic);

    // Cook-Torrance BRDF components
    float NDF = DistributionGGX(normal, halfwayDir, pc.roughness);
    float G = GeometrySmith(normal, viewDir, lightDir, pc.roughness);
    vec3 F = fresnelSchlick(max(dot(halfwayDir, viewDir), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - pc.metallic;

    float nDotL = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = kD * pc.baseColor.rgb / PI;

    vec3 color = (diffuse + specular) * nDotL;

    // Ambient + Tone mapping (very basic)
    color = color / (color + vec3(1.0));
    
    outColor = vec4(color, 1.0);
}