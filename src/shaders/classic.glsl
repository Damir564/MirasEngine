// Shared by the Classic pipeline shaders. Include after frame_ubo.glsl.

// Ambient light from above and below; the sun's color at the top of the atmosphere tints and dims it, so a
// sunset or a dim sun darkens the whole scene like the time-of-day palettes of older games.
vec3 classicAmbient(vec3 N) {
    vec3 sunTint = ubo.sunTopColor.rgb / 3.2;
    float daylight = clamp(-ubo.lightDir.y * 4.0 + 0.3, 0.15, 1.0);
    vec3 sky = vec3(0.30, 0.35, 0.42) * sunTint * daylight;
    vec3 ground = vec3(0.16, 0.14, 0.12) * sunTint * daylight;
    return mix(ground, sky, N.y * 0.5 + 0.5);
}

// Cheap analytic sky (no LUT): horizon to zenith gradient plus a glow around the sun. In the solid
// background mode it is just the background color, so fog fades into it.
vec3 classicSkyColor(vec3 dir) {
    if (!realisticSky())
        return ubo.backgroundColor.rgb;
    vec3 sunTint = ubo.sunTopColor.rgb / 3.2;
    vec3 toSun = -ubo.lightDir.xyz;
    float daylight = clamp(toSun.y * 4.0 + 0.3, 0.05, 1.0);
    // Low sun: warmer horizon.
    vec3 horizon = mix(vec3(0.85, 0.55, 0.35), vec3(0.70, 0.78, 0.86), clamp(toSun.y * 3.0, 0.0, 1.0));
    vec3 zenith = vec3(0.18, 0.36, 0.72);
    float t = pow(clamp(dir.y, 0.0, 1.0), 0.5);
    vec3 color = mix(horizon, zenith, t) * daylight;
    // Below the horizon: a dim ground haze instead of the sky.
    color = dir.y < 0.0 ? mix(horizon * daylight, horizon * daylight * 0.5, clamp(-dir.y * 4.0, 0.0, 1.0)) : color;
    float sunAmount = max(dot(dir, toSun), 0.0);
    color += pow(sunAmount, 64.0) * 0.6 + pow(sunAmount, 2000.0) * 4.0;
    return color * sunTint;
}
