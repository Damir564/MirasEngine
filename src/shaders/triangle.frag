#version 460
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in mat3 TBN;
layout(location = 6) in vec4 fragPosLightSpace;

// Set 0: Frame UBO
layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    float time;
    float shadowBias;
} ubo;

// Set 1: Base Color
layout(set = 1, binding = 0) uniform sampler2D baseColorSampler;
// Set 2: Normal Map
layout(set = 2, binding = 0) uniform sampler2D normalMapSampler;
// Set 3: Metallic-Roughness Map
layout(set = 3, binding = 0) uniform sampler2D mrSampler;
// Set 4: Shadow Map
layout(set = 4, binding = 0) uniform sampler2DShadow shadowMapSampler;

layout(push_constant) uniform MeshData {
    mat4 modelMatrix;
    vec4 baseColor;
    float metallic;
    float roughness;
    float alphaCutoff;
    int alphaMode;
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;
const int ALPHA_MODE_OPAQUE = 0;
const int ALPHA_MODE_MASK = 1;
const int ALPHA_MODE_BLEND = 2;

float calculateShadow(vec4 fragPosLightSpace, vec3 N, vec3 L) {
    // Perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform from [-1,1] to [0,1] for X and Y only
    // Z is already in [0,1] for Vulkan after the ortho projection with Y-flip
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    // DEBUG OUTPUT - uncomment to test
     // Return RED if outside X bounds
//     if (projCoords.x < 0.0 || projCoords.x > 1.0) return -1.0;
     // Return GREEN if outside Y bounds  
//     if (projCoords.y < 0.0 || projCoords.y > 1.0) return -2.0;
    //  Return BLUE if outside Z bounds
//     if (projCoords.z < 0.0 || projCoords.z > 1.0) return -3.0;
    
    // Check if outside shadow map - return fully lit
    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 1.0;
    }
    
    // If behind the light's far plane or in front of near plane, return lit
    if (projCoords.z > 1.0 || projCoords.z < 0.0) {
        return 1.0;
    }

    // Slope-scale bias
    float bias = max(ubo.shadowBias * (1.0 - dot(N, L)), ubo.shadowBias * 0.1);
    
    // PCF sampling
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMapSampler, 0);
    
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(x, y) * texelSize;
            // For sampler2DShadow, the comparison is: texture returns 1.0 if projCoords.z - bias < sampled depth
            shadow += texture(shadowMapSampler, vec3(projCoords.xy + offset, projCoords.z - bias));
        }
    }
    
    return shadow / 9.0;
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 calcLight(vec3 N, vec3 V, vec3 L, vec3 lightColor, float lightIntensity,
               vec3 albedo, float metallic, float roughness, vec3 F0) {
    vec3 H = normalize(V + L);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);

    vec3 kD = (1.0 - F) * (1.0 - metallic);
    float NdotL = max(dot(N, L), 0.0);

    return (kD * albedo / PI + specular) * lightColor * lightIntensity * NdotL;
}

void main() {
    vec4 texColor = texture(baseColorSampler, fragTexCoord);
    float finalAlpha = texColor.a * pc.baseColor.a;

    if (pc.alphaMode == ALPHA_MODE_MASK) {
        if (finalAlpha < pc.alphaCutoff) {
            discard;
        }
        finalAlpha = 1.0;
    } else if (pc.alphaMode == ALPHA_MODE_OPAQUE) {
        finalAlpha = 1.0;
    }

    if (finalAlpha < 0.001) {
        discard;
    }

    vec3 albedo = pc.baseColor.rgb * texColor.rgb;

    vec3 normalMapValue = texture(normalMapSampler, fragTexCoord).rgb * 2.0 - 1.0;
    vec3 N = normalize(TBN * normalMapValue);

    vec4 mrSample = texture(mrSampler, fragTexCoord);
    float metallic = pc.metallic * mrSample.b;
    float roughness = max(pc.roughness * mrSample.g, 0.05);

    vec3 V = normalize(ubo.cameraPos.xyz - fragWorldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 sunDir = normalize(-ubo.lightDir.xyz);
    float shadow = calculateShadow(fragPosLightSpace, N, sunDir);

     // OPTION 1: Show shadow value directly (white=lit, black=shadow)
//     outColor = vec4(vec3(shadow), 1.0);
//     return;

// OPTION 2: Show light space UV coordinates
//     vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
//     projCoords.xy = projCoords.xy * 0.5 + 0.5;
//     outColor = vec4(projCoords.xy, 0.0, 1.0);
//     return;

    // OPTION 3: Show light space depth
//     vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
//     projCoords.xy = projCoords.xy * 0.5 + 0.5;
//     outColor = vec4(vec3(projCoords.z), 1.0);
//     return;

    // OPTION 4: Show out-of-bounds areas
//     vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
//     projCoords.xy = projCoords.xy * 0.5 + 0.5;
//     vec3 debugColor = vec3(0.0, 1.0, 0.0); // green = in bounds
//     if (projCoords.x < 0.0 || projCoords.x > 1.0) debugColor = vec3(1.0, 0.0, 0.0); // red
//     if (projCoords.y < 0.0 || projCoords.y > 1.0) debugColor = vec3(0.0, 0.0, 1.0); // blue
//     if (projCoords.z < 0.0 || projCoords.z > 1.0) debugColor = vec3(1.0, 1.0, 0.0); // yellow
//     outColor = vec4(debugColor, 1.0);
//     return;

    vec3 Lo = vec3(0.0);
    Lo += calcLight(N, V, sunDir, vec3(1.0, 0.98, 0.95), 3.0, albedo, metallic, roughness, F0) * shadow;
    Lo += calcLight(N, V, normalize(vec3(-0.7, 0.5, -0.5)), vec3(0.6, 0.7, 1.0), 1.5, albedo, metallic, roughness, F0);
    Lo += calcLight(N, V, normalize(vec3(0.0, 0.3, -1.0)), vec3(1.0, 0.9, 0.8), 1.0, albedo, metallic, roughness, F0);
    Lo += calcLight(N, V, normalize(vec3(0.0, 1.0, 0.0)), vec3(0.8, 0.9, 1.0), 0.8, albedo, metallic, roughness, F0);

    vec3 ambient = mix(vec3(0.2, 0.15, 0.1), vec3(0.3, 0.4, 0.6), dot(N, vec3(0,1,0)) * 0.5 + 0.5) * albedo * 0.4;
    vec3 color = ambient + Lo;

    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);

    outColor = vec4(color, finalAlpha);
}