#include "GraphicsSettings.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>

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
        std::cerr << "[SETTINGS] Ignoring invalid value for '" << key << "'\n";
    }
}

int snapToPowerOfTwo(int value, int minValue, int maxValue)
{
    int result = minValue;
    while (result < maxValue && result * 2 <= value)
        result *= 2;
    return result;
}

} // namespace

GraphicsSettings sanitizeGraphicsSettings(GraphicsSettings s)
{
    s.maxFps = std::clamp(s.maxFps, 0, 1000);
    s.msaaSamples = snapToPowerOfTwo(s.msaaSamples, 1, 8);
    s.shadowMapSize = snapToPowerOfTwo(s.shadowMapSize, 1024, 4096);
    s.shadowDistance = std::clamp(s.shadowDistance, 10.0f, 1000.0f);
    s.viewDistance = std::clamp(s.viewDistance, 100.0f, 20000.0f);
    return s;
}

GraphicsSettings loadGraphicsSettings(const std::string& path)
{
    GraphicsSettings settings;
    std::ifstream file(path);
    if (!file)
        return settings;

    const nlohmann::json json = nlohmann::json::parse(file, nullptr, false);
    if (json.is_discarded() || !json.is_object()) {
        std::cerr << "[SETTINGS] " << path << " is not valid JSON; using defaults\n";
        return settings;
    }
    readField(json, "vsync", settings.vsync);
    readField(json, "maxFps", settings.maxFps);
    readField(json, "msaaSamples", settings.msaaSamples);
    readField(json, "shadows", settings.shadows);
    readField(json, "shadowMapSize", settings.shadowMapSize);
    readField(json, "shadowDistance", settings.shadowDistance);
    readField(json, "viewDistance", settings.viewDistance);
    readField(json, "fog", settings.fog);
    return sanitizeGraphicsSettings(settings);
}

bool saveGraphicsSettings(const GraphicsSettings& settings, const std::string& path)
{
    const nlohmann::json json = {
        { "vsync", settings.vsync },
        { "maxFps", settings.maxFps },
        { "msaaSamples", settings.msaaSamples },
        { "shadows", settings.shadows },
        { "shadowMapSize", settings.shadowMapSize },
        { "shadowDistance", settings.shadowDistance },
        { "viewDistance", settings.viewDistance },
        { "fog", settings.fog },
    };
    std::ofstream file(path);
    if (!file) {
        std::cerr << "[SETTINGS] Failed to write " << path << "\n";
        return false;
    }
    file << json.dump(4) << "\n";
    return static_cast<bool>(file);
}
