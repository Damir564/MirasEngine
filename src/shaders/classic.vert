#version 460
#extension GL_GOOGLE_include_directive : require

#include "frame_ubo.glsl"
#include "draw_data.glsl"
#include "classic.glsl"

// Classic pipeline: lighting and fog are computed per vertex, the fragment shader only samples the base
// color texture.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragLight;
layout(location = 1) out vec4 fragFog;    // rgb = fog color, a = fog amount
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) flat out uint fragDrawIndex;

// Matches triangle.vert, so the selection mask and level grid depth-test against the same positions.
invariant gl_Position;

void main() {
    uint drawIndex = gl_InstanceIndex;
    TransformData t = transforms[draws[drawIndex].transformIndex];

    vec4 worldPosition = t.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPosition;

    vec3 N = mat3(t.normal) * inNormal;
    N = dot(N, N) > 1e-12 ? normalize(N) : vec3(0.0, 1.0, 0.0);

    // Two-sided lighting: geometry is drawn without culling, so light back faces as seen from the camera.
    vec3 toCamera = ubo.cameraPos.xyz - worldPosition.xyz;
    float distance = length(toCamera);
    if (dot(N, toCamera) < 0.0)
        N = -N;

    vec3 L = -ubo.lightDir.xyz;
    // sunColor is irradiance; dividing by PI gives the radiance of a white Lambertian surface.
    vec3 sun = ubo.sunColor.rgb / PI * max(dot(N, L), 0.0);
    fragLight = classicAmbient(N) + sun;

    vec3 V = toCamera / max(distance, 1e-6);
    vec3 fogDir = normalize(vec3(-V.x, max(-V.y, 0.02), -V.z));
    fragFog = vec4(classicSkyColor(fogDir), fogAmount(worldPosition.xyz, distance));

    fragTexCoord = inTexCoord;
    fragDrawIndex = drawIndex;
}
