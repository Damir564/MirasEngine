// Point and spot lights. Must match GpuLight / GpuLightHeader in engine/RenderTypes.h (std430).
#define MAX_LIGHTS 64u
#define MAX_LIGHT_SHADOW_LAYERS 16

struct LightData {
    vec4 positionRange; // xyz = world position, w = range (m)
    vec4 color;         // rgb = linear color * intensity, w = 0 point / 1 spot
    vec4 direction;     // xyz = where a spot points, w = cos(outer cone angle)
    vec4 params;        // x = cos(inner cone angle), y = first shadow layer (-1 = none),
                        // z = shadow texel size per meter of distance
};

layout(std430, set = 0, binding = 3) readonly buffer LightBuffer {
    uvec4 lightInfo; // x = light count
    mat4 lightShadowMatrices[MAX_LIGHT_SHADOW_LAYERS];
    LightData lights[];
};

uint lightCount() {
    return min(lightInfo.x, MAX_LIGHTS);
}

// Inverse-square falloff, windowed so it reaches zero at the range (Karis, "Real Shading in UE4").
float lightAttenuation(float distance, float range) {
    float ratio = distance / range;
    float window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    return window * window / max(distance * distance, 0.01);
}

// 1 inside the inner cone, fading to 0 at the outer one. L points from the surface to the light.
float spotFactor(LightData light, vec3 L) {
    float cosAngle = dot(-L, light.direction.xyz);
    return smoothstep(light.direction.w, max(light.params.x, light.direction.w + 1e-4), cosAngle);
}

// The light's irradiance at a surface point before the N.L term (zero out of range or outside a spot's
// cone); L is set to the direction towards the light.
vec3 lightIrradiance(LightData light, vec3 worldPos, out vec3 L, out float distance) {
    vec3 toLight = light.positionRange.xyz - worldPos;
    distance = length(toLight);
    L = toLight / max(distance, 1e-4);
    float range = light.positionRange.w;
    if (distance >= range)
        return vec3(0.0);
    float attenuation = lightAttenuation(distance, range);
    if (light.color.w > 0.5)
        attenuation *= spotFactor(light, L);
    return light.color.rgb * attenuation;
}

// The shadow layer for this point: a spot has one; a point light has six (+X, -X, +Y, -Y, +Z, -Z), one per
// cube face, picked by the major axis of the direction from the light.
int lightShadowLayer(LightData light, vec3 worldPos) {
    int first = int(light.params.y);
    if (first < 0 || light.color.w > 0.5)
        return first;
    vec3 d = worldPos - light.positionRange.xyz;
    vec3 a = abs(d);
    int face = a.x >= a.y && a.x >= a.z ? (d.x > 0.0 ? 0 : 1) : a.y >= a.z ? (d.y > 0.0 ? 2 : 3) : (d.z > 0.0 ? 4 : 5);
    return first + face;
}
