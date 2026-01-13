#version 460
#extension GL_KHR_vulkan_glsl : enable

// Vertex attributes from VertexBuffer
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;   
layout(location = 2) in vec2 inTexCoord; // [cite: 30]
layout(location = 3) in vec3 inTangent; // [cite: 30]
layout(location = 4) in vec3 inOffset;   // [cite: 31]

layout(push_constant) uniform MeshData {
    mat4 view;
    mat4 proj;
    vec4 baseColor;
    float metallic;
    float roughness;
    float time;
} pc;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out mat3 TBN; // NEW: Output TBN matrix

void main()
{
   // Create Y-axis rotation matrix from time
    float s = sin(pc.time);
    float c = cos(pc.time);
    mat3 rotY = mat3(
        c, 0, s,
        0, 1, 0,
       -s, 0, c
    );
    rotY = mat3(1.0);
    // 1. Rotate the vertex position before adding offset
    vec3 rotatedPos = rotY * inPosition;
    vec3 pos = rotatedPos + inOffset;
    vec4 worldPosition = vec4(pos, 1.0);

    gl_Position = pc.proj * pc.view * worldPosition;

    // 2. Transform Normal and Tangent by the same rotation
    // This ensures the TBN frame rotates with the geometry
    vec3 worldNormal = normalize(inNormal);
    vec3 worldTangent = normalize(inTangent);
    vec3 worldBitangent = cross(worldNormal, worldTangent);

    // 3. Set Outputs
    fragWorldPos = vec3(worldPosition);
    fragNormal = worldNormal;
    fragTexCoord = inTexCoord;
    TBN = mat3(worldTangent, worldBitangent, worldNormal);
}