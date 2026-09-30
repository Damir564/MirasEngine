// Shared FrameUBO block. Must match FrameUBO in engine/RenderTypes.h (std140).
layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    mat4 invViewProj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;     // direction the sunlight travels
    vec4 sunColor;     // rgb, a = intensity
    vec4 skyZenith;
    vec4 skyHorizon;
    vec4 groundColor;
    vec4 fogParams;    // x = enabled, y = distance where fog reaches ~63%
    vec4 shadowParams; // x = bias, y = enabled, z = fade distance, w = 1 / map size
    float time;
    float nearPlane;
    float farPlane;
    float _pad0;
} ubo;

// Sky gradient without the sun disk; also the fog color, so distant geometry melts into the sky.
vec3 skyGradient(vec3 dir) {
    float h = dir.y;
    vec3 sky = mix(ubo.skyHorizon.rgb, ubo.skyZenith.rgb, pow(clamp(h, 0.0, 1.0), 0.45));
    vec3 ground = mix(ubo.skyHorizon.rgb, ubo.groundColor.rgb, clamp(-h * 3.0, 0.0, 1.0));
    vec3 color = h >= 0.0 ? sky : ground;
    // Warm haze around the sun.
    float sunAmount = max(dot(dir, -normalize(ubo.lightDir.xyz)), 0.0);
    color += ubo.sunColor.rgb * pow(sunAmount, 8.0) * 0.25;
    return color;
}

float fogFactor(float distance) {
    if (ubo.fogParams.x < 0.5)
        return 0.0;
    float d = distance / ubo.fogParams.y;
    return 1.0 - exp(-d * d);
}
