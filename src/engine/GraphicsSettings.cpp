#include "GraphicsSettings.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include "ConfigPaths.h"
#include "Log.h"

NLOHMANN_JSON_SERIALIZE_ENUM(BackgroundMode, {
    { BackgroundMode::Realistic, "realistic" },
    { BackgroundMode::SolidColor, "solid" },
})

NLOHMANN_JSON_SERIALIZE_ENUM(RenderPipeline, {
    { RenderPipeline::Standard, "standard" },
    { RenderPipeline::Classic, "classic" },
})

NLOHMANN_JSON_SERIALIZE_ENUM(Tonemapper, {
    { Tonemapper::Aces, "aces" },
    { Tonemapper::AgX, "agx" },
    { Tonemapper::Neutral, "neutral" },
    { Tonemapper::Reinhard, "reinhard" },
})

namespace {

template <typename T>
void readField(const nlohmann::json& json, const char* key, T& out)
{
    const auto it = json.find(key);
    if (it == json.end())
        return;
    try {
        out = it->get<T>();
    }
    catch (const nlohmann::json::exception&) {
        LOG_ERROR("[SETTINGS] Ignoring invalid value for '" << key << "'\n");
    }
}

// Colors are stored as [r, g, b].
void readField(const nlohmann::json& json, const char* key, glm::vec3& out)
{
    std::array<float, 3> rgb{ out.r, out.g, out.b };
    readField(json, key, rgb);
    out = glm::vec3(rgb[0], rgb[1], rgb[2]);
}

nlohmann::json toJson(const glm::vec3& color)
{
    return nlohmann::json::array({ color.r, color.g, color.b });
}

int snapToPowerOfTwo(int value, int minValue, int maxValue)
{
    int result = minValue;
    while (result < maxValue && result * 2 <= value)
        result *= 2;
    return result;
}

float clampFinite(float value, float minValue, float maxValue, float fallback)
{
    return std::isfinite(value) ? std::clamp(value, minValue, maxValue) : fallback;
}

glm::vec3 clampColor(const glm::vec3& color)
{
    return glm::vec3(clampFinite(color.r, 0.0f, 1.0f, 1.0f), clampFinite(color.g, 0.0f, 1.0f, 1.0f),
        clampFinite(color.b, 0.0f, 1.0f, 1.0f));
}

} // namespace

void applyQualityPreset(GraphicsSettings& s, QualityPreset preset, int maxMsaaSamples)
{
    s.pipeline = RenderPipeline::Standard;
    s.renderScale = 1.0f;
    s.shadows = true;
    s.lightShadows = true;
    switch (preset) {
    case QualityPreset::Lowest:
        // Shadows and effects are off too, so switching back to Standard stays cheap.
        s.pipeline = RenderPipeline::Classic; s.renderScale = 0.75f; s.detailCulling = 3.0f;
        s.msaaSamples = 1; s.fxaa = false; s.shadows = false; s.shadowMapSize = 1024; s.shadowCascades = 1;
        s.softShadows = false; s.contactShadows = false; s.ambientOcclusion = 0; s.bloom = false;
        s.lightShadows = false;
        break;
    case QualityPreset::Low:
        s.detailCulling = 2.0f;
        s.msaaSamples = 1; s.fxaa = true; s.shadowMapSize = 1024; s.shadowCascades = 2;
        s.softShadows = false; s.contactShadows = false; s.ambientOcclusion = 0; s.bloom = false;
        s.lightShadows = false;
        break;
    case QualityPreset::Medium:
        s.detailCulling = 1.0f;
        s.msaaSamples = 2; s.fxaa = false; s.shadowMapSize = 2048; s.shadowCascades = 3;
        s.softShadows = false; s.contactShadows = false; s.ambientOcclusion = 1; s.bloom = true;
        break;
    case QualityPreset::High:
        s.detailCulling = 0.0f;
        s.msaaSamples = 4; s.fxaa = false; s.shadowMapSize = 2048; s.shadowCascades = 4;
        s.softShadows = true; s.contactShadows = true; s.ambientOcclusion = 2; s.bloom = true;
        break;
    case QualityPreset::Ultra:
        s.detailCulling = 0.0f;
        s.msaaSamples = 8; s.fxaa = false; s.shadowMapSize = 4096; s.shadowCascades = 4;
        s.softShadows = true; s.contactShadows = true; s.ambientOcclusion = 3; s.bloom = true;
        break;
    }
    s.msaaSamples = std::min(s.msaaSamples, maxMsaaSamples);
}

bool GraphicsSettings::sameGlobal(const GraphicsSettings& other) const
{
    GraphicsSettings copy = other;
    copy.scene() = scene();
    return copy == *this;
}

GraphicsSettings sanitizeGraphicsSettings(GraphicsSettings s)
{
    const GraphicsSettings defaults;
    s.scene() = sanitizeSceneSettings(s.scene());
    s.renderScale = clampFinite(s.renderScale, 0.25f, 1.0f, defaults.renderScale);
    s.detailCulling = clampFinite(s.detailCulling, 0.0f, 16.0f, defaults.detailCulling);
    s.maxFps = std::clamp(s.maxFps, 0, 1000);
    s.msaaSamples = snapToPowerOfTwo(s.msaaSamples, 1, 8);
    s.shadowMapSize = snapToPowerOfTwo(s.shadowMapSize, 1024, 4096);
    s.shadowCascades = std::clamp(s.shadowCascades, 1, 4);
    s.shadowDistance = clampFinite(s.shadowDistance, 10.0f, 1000.0f, defaults.shadowDistance);
    s.ambientOcclusion = std::clamp(s.ambientOcclusion, 0, 3);
    s.viewDistance = clampFinite(s.viewDistance, 100.0f, 20000.0f, defaults.viewDistance);
    s.masterVolume = clampFinite(s.masterVolume, 0.0f, 1.0f, defaults.masterVolume);
    s.musicVolume = clampFinite(s.musicVolume, 0.0f, 1.0f, defaults.musicVolume);
    return s;
}

SceneSettings sanitizeSceneSettings(SceneSettings s)
{
    const SceneSettings defaults;
    s.aoRadius = clampFinite(s.aoRadius, 0.1f, 5.0f, defaults.aoRadius);
    s.aoIntensity = clampFinite(s.aoIntensity, 0.0f, 4.0f, defaults.aoIntensity);
    s.bloomIntensity = clampFinite(s.bloomIntensity, 0.0f, 1.0f, defaults.bloomIntensity);
    s.exposure = clampFinite(s.exposure, -5.0f, 5.0f, defaults.exposure);
    s.contrast = clampFinite(s.contrast, 0.5f, 1.5f, defaults.contrast);
    s.saturation = clampFinite(s.saturation, 0.0f, 2.0f, defaults.saturation);
    s.vignette = clampFinite(s.vignette, 0.0f, 1.0f, defaults.vignette);
    s.backgroundColor = clampColor(s.backgroundColor);
    s.sunAzimuth = std::isfinite(s.sunAzimuth) ? std::fmod(std::fmod(s.sunAzimuth, 360.0f) + 360.0f, 360.0f)
                                               : defaults.sunAzimuth;
    s.sunElevation = clampFinite(s.sunElevation, -10.0f, 90.0f, defaults.sunElevation);
    s.sunIntensity = clampFinite(s.sunIntensity, 0.0f, 10.0f, defaults.sunIntensity);
    s.sunColor = clampColor(s.sunColor);
    s.haze = clampFinite(s.haze, 0.0f, 10.0f, defaults.haze);
    s.fogDensity = clampFinite(s.fogDensity, 0.0f, 100.0f, defaults.fogDensity);
    return s;
}

std::string graphicsSettingsPath()
{
    return configPath("settings.json");
}

GraphicsSettings loadGraphicsSettings(const std::string& path)
{
    GraphicsSettings settings;
    std::ifstream file(path);
    if (!file)
        return settings;

    const nlohmann::json json = nlohmann::json::parse(file, nullptr, false);
    if (json.is_discarded() || !json.is_object()) {
        LOG_ERROR("[SETTINGS] " << path << " is not valid JSON; using defaults\n");
        return settings;
    }
    readField(json, "pipeline", settings.pipeline);
    readField(json, "renderScale", settings.renderScale);
    readField(json, "detailCulling", settings.detailCulling);
    readField(json, "vsync", settings.vsync);
    readField(json, "maxFps", settings.maxFps);
    readField(json, "msaaSamples", settings.msaaSamples);
    readField(json, "fxaa", settings.fxaa);
    readField(json, "shadows", settings.shadows);
    readField(json, "shadowMapSize", settings.shadowMapSize);
    readField(json, "shadowCascades", settings.shadowCascades);
    readField(json, "shadowDistance", settings.shadowDistance);
    readField(json, "softShadows", settings.softShadows);
    readField(json, "contactShadows", settings.contactShadows);
    readField(json, "lightShadows", settings.lightShadows);
    readField(json, "ambientOcclusion", settings.ambientOcclusion);
    readField(json, "bloom", settings.bloom);
    readField(json, "viewDistance", settings.viewDistance);
    readField(json, "masterVolume", settings.masterVolume);
    readField(json, "musicVolume", settings.musicVolume);
    return sanitizeGraphicsSettings(settings);
}

bool saveGraphicsSettings(const GraphicsSettings& settings, const std::string& path)
{
    const nlohmann::json json = {
        { "pipeline", settings.pipeline },
        { "renderScale", settings.renderScale },
        { "detailCulling", settings.detailCulling },
        { "vsync", settings.vsync },
        { "maxFps", settings.maxFps },
        { "msaaSamples", settings.msaaSamples },
        { "fxaa", settings.fxaa },
        { "shadows", settings.shadows },
        { "shadowMapSize", settings.shadowMapSize },
        { "shadowCascades", settings.shadowCascades },
        { "shadowDistance", settings.shadowDistance },
        { "softShadows", settings.softShadows },
        { "contactShadows", settings.contactShadows },
        { "lightShadows", settings.lightShadows },
        { "ambientOcclusion", settings.ambientOcclusion },
        { "bloom", settings.bloom },
        { "viewDistance", settings.viewDistance },
        { "masterVolume", settings.masterVolume },
        { "musicVolume", settings.musicVolume },
    };
    std::ofstream file(path);
    if (!file) {
        LOG_ERROR("[SETTINGS] Failed to write " << path << "\n");
        return false;
    }
    file << json.dump(4) << "\n";
    return static_cast<bool>(file);
}

std::string sceneSettingsToJson(const SceneSettings& settings)
{
    const nlohmann::json json = {
        { "background", settings.background },
        { "backgroundColor", toJson(settings.backgroundColor) },
        { "sun", settings.sun },
        { "sunAzimuth", settings.sunAzimuth },
        { "sunElevation", settings.sunElevation },
        { "sunIntensity", settings.sunIntensity },
        { "sunColor", toJson(settings.sunColor) },
        { "haze", settings.haze },
        { "fog", settings.fog },
        { "fogDensity", settings.fogDensity },
        { "aoRadius", settings.aoRadius },
        { "aoIntensity", settings.aoIntensity },
        { "bloomIntensity", settings.bloomIntensity },
        { "exposure", settings.exposure },
        { "tonemapper", settings.tonemapper },
        { "contrast", settings.contrast },
        { "saturation", settings.saturation },
        { "vignette", settings.vignette },
    };
    return json.dump();
}

SceneSettings sceneSettingsFromJson(const std::string& text)
{
    SceneSettings settings;
    const nlohmann::json json = nlohmann::json::parse(text, nullptr, false);
    if (json.is_discarded() || !json.is_object()) {
        LOG_ERROR("[SETTINGS] Scene settings are not valid JSON; using defaults\n");
        return settings;
    }
    readField(json, "background", settings.background);
    readField(json, "backgroundColor", settings.backgroundColor);
    readField(json, "sun", settings.sun);
    readField(json, "sunAzimuth", settings.sunAzimuth);
    readField(json, "sunElevation", settings.sunElevation);
    readField(json, "sunIntensity", settings.sunIntensity);
    readField(json, "sunColor", settings.sunColor);
    readField(json, "haze", settings.haze);
    readField(json, "fog", settings.fog);
    readField(json, "fogDensity", settings.fogDensity);
    readField(json, "aoRadius", settings.aoRadius);
    readField(json, "aoIntensity", settings.aoIntensity);
    readField(json, "bloomIntensity", settings.bloomIntensity);
    readField(json, "exposure", settings.exposure);
    readField(json, "tonemapper", settings.tonemapper);
    readField(json, "contrast", settings.contrast);
    readField(json, "saturation", settings.saturation);
    readField(json, "vignette", settings.vignette);
    return sanitizeSceneSettings(settings);
}
