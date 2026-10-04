#version 460

// Grid lines on the surfaces of the level shape being edited, in its object space, so the lines
// follow the shape's transform and match where its vertices snap.

layout(location = 0) in vec3 objectPos;
layout(location = 1) in vec3 objectNormal;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform GridPush {
    float cellSize;
    float majorEvery; // cells between major lines
    float _pad0;
    float _pad1;
} pc;

// Coverage of grid lines spaced `spacing` apart, about one pixel wide (as in grid.frag).
float gridLines(vec2 coord, float spacing) {
    vec2 c = coord / spacing;
    vec2 derivative = max(fwidth(c), vec2(1e-6));
    vec2 g = abs(fract(c - 0.5) - 0.5) / derivative;
    float line = 1.0 - min(min(g.x, g.y), 1.0);
    float density = max(derivative.x, derivative.y);
    return line * (1.0 - smoothstep(0.15, 0.5, density));
}

void main() {
    // Project onto the object plane the face is most aligned with; faces are flat, so this is constant per face.
    vec3 n = abs(objectNormal);
    vec2 coord = n.x >= n.y && n.x >= n.z ? objectPos.yz
        : n.y >= n.z ? objectPos.xz : objectPos.xy;

    float minor = gridLines(coord, pc.cellSize);
    float major = gridLines(coord, pc.cellSize * pc.majorEvery);
    float alpha = max(minor * 0.45, major * 0.75);
    if (alpha < 0.002)
        discard;
    outColor = vec4(vec3(0.02), alpha);
}
