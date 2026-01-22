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

// ============================================
// PBR Functions
// ============================================

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

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ============================================
// Calculate PBR lighting for a single light
// ============================================
vec3 calcLight(vec3 N, vec3 V, vec3 L, vec3 lightColor, float lightIntensity, 
               vec3 albedo, float metallic, float roughness, vec3 F0) {
    vec3 H = normalize(V + L);
    
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = kD * albedo / PI;

    return (diffuse + specular) * lightColor * lightIntensity * NdotL;
}

// ============================================
// Point light attenuation
// ============================================
vec3 calcPointLight(vec3 N, vec3 V, vec3 lightPos, vec3 lightColor, float lightIntensity,
                    vec3 worldPos, vec3 albedo, float metallic, float roughness, vec3 F0) {
    vec3 L = lightPos - worldPos;
    float distance = length(L);
    L = normalize(L);
    
    // Attenuation (inverse square with minimum)
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    
    return calcLight(N, V, L, lightColor, lightIntensity * attenuation, albedo, metallic, roughness, F0);
}

void main() {
    // ========================================
    // 1. Sample Textures
    // ========================================
    vec4 texColor = texture(baseColorSampler, fragTexCoord);
    vec3 albedo = pc.baseColor.rgb * texColor.rgb;
    
    // Normal Mapping
    vec3 normalMapValue = texture(normalMapSampler, fragTexCoord).rgb;
    normalMapValue = normalMapValue * 2.0 - 1.0; 
    vec3 N = normalize(TBN * normalMapValue); 

    // Metallic & Roughness (glTF: G = Roughness, B = Metallic)
    vec4 mrSample = texture(mrSampler, fragTexCoord);
    float metallic = pc.metallic * mrSample.b;
    float roughness = pc.roughness * mrSample.g;
    
    // Clamp roughness to avoid divide-by-zero in specular
    roughness = max(roughness, 0.05);

    // ========================================
    // 2. View Direction & F0
    // ========================================
    vec3 V = normalize(pc.cameraPos.xyz - fragWorldPos);
    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    // ========================================
    // 3. Lighting Accumulation
    // ========================================
    vec3 Lo = vec3(0.0);

    // --- Key Light (Main Sun/Directional) ---
    {
        vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
        vec3 lightColor = vec3(1.0, 0.98, 0.95);  // Slightly warm white
        float intensity = 3.0;  // ⬆️ Increased from 2.0
        Lo += calcLight(N, V, lightDir, lightColor, intensity, albedo, metallic, roughness, F0);
    }

    // --- Fill Light (Softer, opposite side) ---
    {
        vec3 lightDir = normalize(vec3(-0.7, 0.5, -0.5));
        vec3 lightColor = vec3(0.6, 0.7, 1.0);  // Cool blue tint
        float intensity = 1.5;
        Lo += calcLight(N, V, lightDir, lightColor, intensity, albedo, metallic, roughness, F0);
    }

    // --- Rim/Back Light (Edge highlighting) ---
    {
        vec3 lightDir = normalize(vec3(0.0, 0.3, -1.0));
        vec3 lightColor = vec3(1.0, 0.9, 0.8);  // Warm back light
        float intensity = 1.0;
        Lo += calcLight(N, V, lightDir, lightColor, intensity, albedo, metallic, roughness, F0);
    }

    // --- Top Light (Sky simulation) ---
    {
        vec3 lightDir = normalize(vec3(0.0, 1.0, 0.0));
        vec3 lightColor = vec3(0.8, 0.9, 1.0);  // Sky blue
        float intensity = 0.8;
        Lo += calcLight(N, V, lightDir, lightColor, intensity, albedo, metallic, roughness, F0);
    }

    // --- Optional: Point Light following camera ---
    // Uncomment for a "headlamp" effect
    /*
    {
        vec3 lightPos = pc.cameraPos + vec3(0.0, 0.5, 0.0);
        vec3 lightColor = vec3(1.0, 1.0, 0.9);
        float intensity = 50.0;
        Lo += calcPointLight(N, V, lightPos, lightColor, intensity, fragWorldPos, albedo, metallic, roughness, F0);
    }
    */

    // ========================================
    // 4. Ambient Lighting (Hemisphere)
    // ========================================
    vec3 skyColor = vec3(0.3, 0.4, 0.6);    // Upper hemisphere (sky)
    vec3 groundColor = vec3(0.2, 0.15, 0.1); // Lower hemisphere (ground bounce)
    float hemisphere = dot(N, vec3(0, 1, 0)) * 0.5 + 0.5;
    vec3 ambient = mix(groundColor, skyColor, hemisphere) * albedo * 0.4;  // ⬆️ Increased ambient

    // ========================================
    // 5. Final Color Composition
    // ========================================
    vec3 color = ambient + Lo;

    // ========================================
    // 6. Tone Mapping (ACES Filmic)
    // ========================================
    // Better than Reinhard for high dynamic range
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);

    // ========================================
    // 7. Gamma Correction (if not using SRGB framebuffer)
    // ========================================
    // Uncomment if your swapchain is NOT sRGB:
    // color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, texColor.a * pc.baseColor.a);
}