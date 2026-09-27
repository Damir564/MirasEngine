#pragma once
#include <string>

struct GraphicsSettings {
    bool vsync = true;
    int maxFps = 0;            // 0 = unlimited
    int msaaSamples = 4;       // 1, 2, 4 or 8; clamped to what the device supports
    bool shadows = true;
    int shadowMapSize = 2048;  // 1024, 2048 or 4096
    float shadowDistance = 150.0f;
    float viewDistance = 5000.0f; // camera far plane
    bool fog = true;

    bool operator==(const GraphicsSettings&) const = default;
};

struct RenderCapabilities {
    int maxMsaaSamples = 1;
    float maxAnisotropy = 1.0f;
};

inline constexpr const char* kGraphicsSettingsPath = "settings.json";

// Clamps every field into its valid range (unknown MSAA/shadow sizes snap to the nearest valid value).
GraphicsSettings sanitizeGraphicsSettings(GraphicsSettings settings);
// A missing or unreadable file yields defaults; missing keys keep their default value.
GraphicsSettings loadGraphicsSettings(const std::string& path = kGraphicsSettingsPath);
bool saveGraphicsSettings(const GraphicsSettings& settings, const std::string& path = kGraphicsSettingsPath);
