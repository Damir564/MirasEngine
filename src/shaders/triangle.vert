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
} pc;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out mat3 TBN; // NEW: Output TBN matrix

void main()
{
    // add instance offset
    vec3 pos = inPosition * 1.0 + inOffset; // [cite: 34]
    vec4 worldPosition = vec4(pos, 1.0);

    gl_Position = pc.proj * pc.view * worldPosition;

    fragWorldPos = vec3(worldPosition);
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
    // --- Create TBN Matrix ---
    vec3 T = normalize(inTangent);
    vec3 N = normalize(inNormal);
    
    // Gram-Schmidt process: re-orthogonalize T with respect to N
    T = normalize(T - dot(T, N) * N);
    
    // Calculate Bitangent (B)
    vec3 B = cross(N, T);

    // If you see inverted normals later, you might need to flip B:
    // B = -cross(N, T); 

    TBN = mat3(T, B, N);
}