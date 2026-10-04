#version 460

// Resolved single-sample scene depth.
layout(set = 1, binding = 0) uniform sampler2D sceneDepth;

layout(push_constant) uniform MaskPush {
    // Scene depth pixel = gl_FragCoord.xy * xy + zw; differs from identity when the Classic pipeline
    // renders the scene at a lower resolution than the mask.
    vec4 depthTransform;
} pc;

layout(location = 0) out vec2 outMask;

// R = selection silhouette (ignores depth), G = the part of it that is visible in the scene.
void main() {
    ivec2 depthPixel = ivec2(gl_FragCoord.xy * pc.depthTransform.xy + pc.depthTransform.zw);
    float scene = texelFetch(sceneDepth, depthPixel, 0).r;
    // A depth texel covers 1 / scale mask pixels, so the depth slope across it is that much larger.
    float tolerance = fwidth(gl_FragCoord.z) * 2.0 / min(pc.depthTransform.x, pc.depthTransform.y) + 1e-6;
    float visible = gl_FragCoord.z <= scene + tolerance ? 1.0 : 0.0;
    outMask = vec2(1.0, visible);
}
