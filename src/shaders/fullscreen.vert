#version 460

// One triangle covering the viewport; outputs NDC so fragment shaders can unproject.
layout(location = 0) out vec2 outNdc;

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    outNdc = uv * 2.0 - 1.0;
    // On the far plane, so with depth testing the sky only fills pixels no geometry covered.
    gl_Position = vec4(outNdc, 1.0, 1.0);
}
