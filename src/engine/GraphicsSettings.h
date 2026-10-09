#pragma once
#include <glm/glm.hpp>
#include <string>

enum class BackgroundMode {
    SolidColor, // flat color behind the scene; lighting still comes from the sky model
    Realistic,  // physically based sky with the sun disk
};

enum class RenderPipeline {
    Standard, // full PBR with shadows, AO, bloom
    Classic,  // cheap forward pass: per-vertex lighting, base color only (GTA SA style)
};

enum class Tonemapper {
    Aces,
    AgX,
    Neutral,  // Khronos PBR Neutral: keeps material colors close to their base color
    Reinhard,
};

// The look of one scene, saved in its .scn file: sky, sun, fog and grading. Defaults match what every scene
// looked like before scenes had their own settings.
struct SceneSettings {
    // Environment
    BackgroundMode background = BackgroundMode::Realistic;
    glm::vec3 backgroundColor{ 0.24f, 0.26f, 0.29f }; // sRGB
    bool sun = true;           // off: no sun disk or direct sunlight (and no shadows); the sky stays lit
    float sunAzimuth = 135.0f; // degrees clockwise from -Z (north) towards +X (east)
    float sunElevation = 35.0f; // degrees above the horizon
    float sunIntensity = 1.0f;
    glm::vec3 sunColor{ 1.0f }; // sRGB tint; the atmosphere already reddens a low sun
    float haze = 1.0f;         // aerosol density of the atmosphere: whiter sky, softer sun
    bool fog = true;
    float fogDensity = 1.0f;   // multiplier of the base height fog density

    // Look (the effects themselves are switched on in GraphicsSettings)
    float aoRadius = 1.0f;     // meters
    float aoIntensity = 1.0f;
    float bloomIntensity = 0.5f;
    float exposure = 0.0f;     // EV compensation
    Tonemapper tonemapper = Tonemapper::Aces;
    float contrast = 1.0f;
    float saturation = 1.0f;
    float vignette = 0.25f;

    bool operator==(const SceneSettings&) const = default;
};

// What the renderer draws with: the user's global settings (settings.json: quality, display, audio) with the
// open scene's SceneSettings applied over the base (Application does that every frame).
struct GraphicsSettings : SceneSettings {
    // Pipeline
    RenderPipeline pipeline = RenderPipeline::Standard;
    float renderScale = 1.0f;  // Classic only: internal resolution relative to the window, 0.25..1
    // Parts of objects whose bounding sphere covers fewer pixels than this on screen are not drawn; 0 = off.
    float detailCulling = 0.0f;

    // Display
    bool vsync = true;
    int maxFps = 0;            // 0 = unlimited
    int msaaSamples = 4;       // 1, 2, 4 or 8; clamped to what the device supports
    bool fxaa = false;         // post-process anti-aliasing; also smooths edges MSAA misses (alpha test, shading)

    // Shadows
    bool shadows = true;
    int shadowMapSize = 2048;  // per cascade: 1024, 2048 or 4096
    int shadowCascades = 4;    // 1..4; more cascades keep near shadows sharp over a long distance
    float shadowDistance = 150.0f;
    bool softShadows = true;   // penumbrae widen with the distance to the caster (PCSS)
    bool contactShadows = true; // screen-space shadows for small details the shadow map is too coarse for
    bool lightShadows = true;  // point and spot lights that ask for shadows get them (Standard only)

    // Effects
    int ambientOcclusion = 2;  // 0 = off, 1 = low, 2 = medium, 3 = high
    bool bloom = true;
    float viewDistance = 5000.0f; // camera far plane

    // Audio (kept here so one settings file covers the game's options)
    float masterVolume = 0.8f;  // 0..1
    float musicVolume = 0.5f;   // ambience loops, relative to the master volume

    bool operator==(const GraphicsSettings&) const = default;

    SceneSettings& scene() { return *this; }
    const SceneSettings& scene() const { return *this; }
    // The global part equals the other's (the scene part is ignored).
    bool sameGlobal(const GraphicsSettings& other) const;
};

struct RenderCapabilities {
    int maxMsaaSamples = 1;
    float maxAnisotropy = 1.0f;
};

// settings.json in the shared config folder (ConfigPaths.h), so every build uses the same settings.
std::string graphicsSettingsPath();

// Lowest = the Classic pipeline at a reduced render scale, for integrated GPUs.
enum class QualityPreset { Lowest, Low, Medium, High, Ultra };
inline constexpr const char* kQualityPresetNames[] = { "Lowest", "Low", "Medium", "High", "Ultra" };

// Presets only touch the performance-relevant options, not the look (sun, exposure, grading...).
void applyQualityPreset(GraphicsSettings& settings, QualityPreset preset, int maxMsaaSamples);
// Clamps every field into its valid range (unknown MSAA/shadow sizes snap to the nearest valid value).
GraphicsSettings sanitizeGraphicsSettings(GraphicsSettings settings);
SceneSettings sanitizeSceneSettings(SceneSettings settings);
// Only the global part is read and written; the scene part keeps its defaults.
// A missing or unreadable file yields defaults; missing keys keep their default value.
GraphicsSettings loadGraphicsSettings(const std::string& path = graphicsSettingsPath());
bool saveGraphicsSettings(const GraphicsSettings& settings, const std::string& path = graphicsSettingsPath());
// SceneSettings as JSON text, as the scene file stores them. Unknown keys are ignored and missing ones keep
// their defaults, so files from newer or older builds still load.
std::string sceneSettingsToJson(const SceneSettings& settings);
SceneSettings sceneSettingsFromJson(const std::string& text);
