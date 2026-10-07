#include "EditorPrefs.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include "engine/Log.h"

NLOHMANN_JSON_SERIALIZE_ENUM(EditorStyle::Theme, {
    { EditorStyle::Theme::Dark, "dark" },
    { EditorStyle::Theme::Midnight, "midnight" },
    { EditorStyle::Theme::Light, "light" },
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
        LOG_ERROR("[PREFS] Ignoring invalid value for '" << key << "'\n");
    }
}

void readField(const nlohmann::json& json, const char* key, glm::vec3& out)
{
    std::array<float, 3> rgb{ out.r, out.g, out.b };
    readField(json, key, rgb);
    out = glm::vec3(rgb[0], rgb[1], rgb[2]);
}

float clampFinite(float value, float minValue, float maxValue, float fallback)
{
    return std::isfinite(value) ? std::clamp(value, minValue, maxValue) : fallback;
}

} // namespace

EditorPrefs sanitizeEditorPrefs(EditorPrefs p)
{
    const EditorPrefs defaults;
    for (int i = 0; i < 3; ++i)
        p.accent[i] = clampFinite(p.accent[i], 0.0f, 1.0f, defaults.accent[i]);
    p.uiScale = clampFinite(p.uiScale, 0.75f, 2.0f, defaults.uiScale);
    p.rounding = clampFinite(p.rounding, 0.0f, 12.0f, defaults.rounding);
    p.fieldOfView = clampFinite(p.fieldOfView, 30.0f, 120.0f, defaults.fieldOfView);
    p.lookSensitivity = clampFinite(p.lookSensitivity, 0.01f, 1.0f, defaults.lookSensitivity);
    p.cameraSpeed = clampFinite(p.cameraSpeed, 0.5f, 200.0f, defaults.cameraSpeed);
    p.autosaveMinutes = std::clamp(p.autosaveMinutes, 0, 60);
    std::erase(p.recentScenes, std::string());
    if (p.recentScenes.size() > kMaxRecentScenes)
        p.recentScenes.resize(kMaxRecentScenes);
    return p;
}

void addRecentScene(EditorPrefs& prefs, const std::string& path)
{
    if (path.empty())
        return;
    std::erase(prefs.recentScenes, path);
    prefs.recentScenes.insert(prefs.recentScenes.begin(), path);
    if (prefs.recentScenes.size() > kMaxRecentScenes)
        prefs.recentScenes.resize(kMaxRecentScenes);
}

EditorPrefs loadEditorPrefs(const std::string& path)
{
    EditorPrefs prefs;
    std::ifstream file(path);
    if (!file)
        return prefs;
    const nlohmann::json json = nlohmann::json::parse(file, nullptr, false);
    if (json.is_discarded() || !json.is_object()) {
        LOG_ERROR("[PREFS] " << path << " is not valid JSON; using defaults\n");
        return prefs;
    }
    readField(json, "theme", prefs.theme);
    readField(json, "accent", prefs.accent);
    readField(json, "uiScale", prefs.uiScale);
    readField(json, "rounding", prefs.rounding);
    readField(json, "compact", prefs.compact);
    readField(json, "fieldOfView", prefs.fieldOfView);
    readField(json, "lookSensitivity", prefs.lookSensitivity);
    readField(json, "invertLookY", prefs.invertLookY);
    readField(json, "cameraSpeed", prefs.cameraSpeed);
    readField(json, "showViewportHelp", prefs.showViewportHelp);
    readField(json, "showOrientationGizmo", prefs.showOrientationGizmo);
    readField(json, "autosaveMinutes", prefs.autosaveMinutes);
    readField(json, "recentScenes", prefs.recentScenes);
    return sanitizeEditorPrefs(prefs);
}

bool saveEditorPrefs(const EditorPrefs& prefs, const std::string& path)
{
    const nlohmann::json json = {
        { "theme", prefs.theme },
        { "accent", { prefs.accent.r, prefs.accent.g, prefs.accent.b } },
        { "uiScale", prefs.uiScale },
        { "rounding", prefs.rounding },
        { "compact", prefs.compact },
        { "fieldOfView", prefs.fieldOfView },
        { "lookSensitivity", prefs.lookSensitivity },
        { "invertLookY", prefs.invertLookY },
        { "cameraSpeed", prefs.cameraSpeed },
        { "showViewportHelp", prefs.showViewportHelp },
        { "showOrientationGizmo", prefs.showOrientationGizmo },
        { "autosaveMinutes", prefs.autosaveMinutes },
        { "recentScenes", prefs.recentScenes },
    };
    std::ofstream file(path);
    if (!file) {
        LOG_ERROR("[PREFS] Failed to write " << path << "\n");
        return false;
    }
    file << json.dump(4) << "\n";
    return static_cast<bool>(file);
}
