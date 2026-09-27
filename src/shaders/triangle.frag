#version 460
#extension GL_GOOGLE_include_directive : require

#include "frame_ubo.glsl"
#include "draw_data.glsl"

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in mat3 TBN;
layout(location = 6) in vec4 fragPosLightSpace;
layout(location = 7) flat in uint fragDrawIndex;

layout(set = 1, binding = 0) uniform sampler2D baseColorSampler;
layout(set = 2, binding = 0) uniform sampler2D normalMapSampler;
layout(set = 3, binding = 0) uniform sampler2D mrSampler;
layout(set = 4, binding = 0) uniform sampler2DShadow shadowMapSampler;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;
const int ALPHA_MODE_OPAQUE = 0;
const int ALPHA_MODE_MASK = 1;

const vec2 POISSON[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367), vec2(0.14383161, -0.14100790));

// Per-pixel pseudo-random angle; turns PCF banding into fine noise.
float interleavedGradientNoise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float calculateShadow(vec3 N, vec3 L, float distanceToCamera) {
    if (ubo.shadowParams.y < 0.5)
        return 1.0;

    float fadeDistance = ubo.shadowParams.z;
    float fade = smoothstep(fadeDistance * 0.8, fadeDistance, distanceToCamera);
    if (fade >= 1.0)
        return 1.0;

    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    if (any(lessThan(projCoords, vec3(0.0))) || any(greaterThan(projCoords, vec3(1.0))))
        return 1.0;

    float bias = max(ubo.shadowParams.x * (1.0 - dot(N, L)), ubo.shadowParams.x * 0.1);
    float angle = interleavedGradientNoise(gl_FragCoord.xy) * 2.0 * PI;
    mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    float radius = ubo.shadowParams.w * 1.75;

    float shadow = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec2 offset = rotation * POISSON[i] * radius;
        shadow += texture(shadowMapSampler, vec3(projCoords.xy + offset, projCoords.z - bias));
    }
    shadow /= 16.0;
    return mix(shadow, 1.0, fade);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

// 2. Geometry Function (Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

// 3. Fresnel Equation (Schlick)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 calcLight(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, float metallic, float roughness, vec3 F0) {
    vec3 H = normalize(V + L);
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    return (kD * albedo / PI + specular) * radiance * max(dot(N, L), 0.0);
}

void main() {
    DrawData d = draws[fragDrawIndex];
    vec4 texColor = texture(baseColorSampler, fragTexCoord);
    float finalAlpha = texColor.a * d.baseColor.a;

    if (d.alphaMode == ALPHA_MODE_MASK) {
        if (finalAlpha < d.alphaCutoff)
            discard;
        finalAlpha = 1.0;
    } else if (d.alphaMode == ALPHA_MODE_OPAQUE) {
        finalAlpha = 1.0;
    }
    if (finalAlpha < 0.001)
        discard;

    vec3 albedo = d.baseColor.rgb * texColor.rgb;

    vec3 normalMapValue = texture(normalMapSampler, fragTexCoord).rgb * 2.0 - 1.0;
    vec3 N = TBN * normalMapValue;
    N = dot(N, N) > 1e-12 ? normalize(N) : normalize(fragNormal);

    vec4 mrSample = texture(mrSampler, fragTexCoord);
    float metallic = clamp(d.metallic * mrSample.b, 0.0, 1.0);
    float roughness = clamp(d.roughness * mrSample.g, 0.05, 1.0);

    vec3 toCamera = ubo.cameraPos.xyz - fragWorldPos;
    float viewDistance = length(toCamera);
    vec3 V = toCamera / max(viewDistance, 1e-6);
    // Geometry is drawn without culling; light back faces as if seen from the front.
    if (!gl_FrontFacing && dot(N, V) < 0.0)
        N = -N;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 sunDir = normalize(-ubo.lightDir.xyz);
    float shadow = calculateShadow(N, sunDir, viewDistance);
    vec3 sunRadiance = ubo.sunColor.rgb * ubo.sunColor.a;
    vec3 Lo = calcLight(N, V, sunDir, sunRadiance, albedo, metallic, roughness, F0) * shadow;

    // Hemisphere ambient: sky light from above, bounced ground light from below.
    float up = N.y * 0.5 + 0.5;
    vec3 skyAmbient = mix(ubo.groundColor.rgb * 0.6, mix(ubo.skyHorizon.rgb, ubo.skyZenith.rgb, 0.5), up);
    vec3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    vec3 kD = (1.0 - kS) * (1.0 - metallic);
    float smoothness = 1.0 - roughness;
    vec3 specAmbient = skyGradient(reflect(-V, N)) * kS * smoothness * smoothness;
    float ambientOcclusion = mix(0.75, 1.0, shadow);
    vec3 ambient = (kD * albedo * skyAmbient * 0.55 + specAmbient * 0.3) * ambientOcclusion;

    vec3 color = ambient + Lo;

    // ACES filmic approximation
    const float a = 2.51, b = 0.03, c = 2.43, dd = 0.59, e = 0.14;
    color = clamp((color * (a * color + b)) / (color * (c * color + dd) + e), 0.0, 1.0);

    // The sky is not tone mapped, so fog blends towards it after tone mapping.
    color = mix(color, skyGradient(-V), fogFactor(viewDistance));
    outColor = vec4(color, finalAlpha);
}
