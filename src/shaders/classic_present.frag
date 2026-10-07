#version 460

// Classic pipeline: upscales the scene, rendered into a scaled-down corner of the scene target, to the
// viewport. No tone mapping: the scene is already in display range and was dithered when it was drawn.
layout(set = 1, binding = 0) uniform sampler2D sceneColor;

layout(push_constant) uniform ClassicPresentPush {
    vec4 transform; // source UV = gl_FragCoord.xy * xy + zw
    vec4 uvClamp;   // source UVs are clamped to [xy, zw] so filtering does not pick up texels outside it
} pc;

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = clamp(gl_FragCoord.xy * pc.transform.xy + pc.transform.zw, pc.uvClamp.xy, pc.uvClamp.zw);
    outColor = vec4(textureLod(sceneColor, uv, 0.0).rgb, 1.0);
}
