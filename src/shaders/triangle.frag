#version 460
#extension GL_NV_uniform_buffer_std430_layout : enable
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in mat3 TBN;

// Set 0: Base Color
layout(set = 0, binding = 0) uniform sampler2D baseColorSampler;
// Set 1: Normal Map 
layout(set = 1, binding = 0) uniform sampler2D normalMapSampler;
// Set 2: Metallic-Roughness Map
layout(set = 2, binding = 0) uniform sampler2D mrSampler;

layout(push_constant) uniform MeshData {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 baseColor;
    float metallic;
    float roughness;
    float time;
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
    float ggx2 = GeometrySchlickGGX(max(dot(N, V), 0.0), roughness);
    float ggx1 = GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
    return ggx1 * ggx2;
}

// 3. Fresnel Equation (Schlick)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    // 1. Sample the texture
    vec4 texColor = texture(baseColorSampler, fragTexCoord);
    vec3 albedo = pc.baseColor.rgb * texColor.rgb;
    
    // 2. Normal Mapping
    vec3 normalMapValue = texture(normalMapSampler, fragTexCoord).rgb;
    normalMapValue = normalMapValue * 2.0 - 1.0; 
    vec3 N = normalize(TBN * normalMapValue); 

    // 3. Metallic & Roughness
    // glTF standard: G = Roughness, B = Metallic
    vec4 mrSample = texture(mrSampler, fragTexCoord);
    // Combine factor (push constant) with texture value
    float metallic = pc.metallic * mrSample.b;
    float roughness = pc.roughness * mrSample.g;

    // DEBUG: Visualize metallic (red) and roughness (green)
//    outColor = vec4(metallic, roughness, 0.0, 1.0);
//    return;  // Skip normal rendering

    // 3. Lighting Setup
    vec3 L = normalize(vec3(-0.4, 0.8, -0.6)); // Light Direction
    // Approximate view pos (assuming camera at 0,0,100 or similar, pass actual cam pos for better accuracy)
    vec3 V = normalize(pc.cameraPos.xyz - fragWorldPos);
    vec3 H = normalize(V + L);

    // 4. PBR Calculation
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = kD * albedo / PI;

    vec3 ambient = vec3(0.05) * albedo;
    float intensity = 2.0;
    vec3 color = ambient + (diffuse + specular) * NdotL * intensity; // 2.0 = light intensity

    // Tone Mapping
    color = color / (color + vec3(1.0));
    
    outColor = vec4(color, texColor.a * pc.baseColor.a);
}