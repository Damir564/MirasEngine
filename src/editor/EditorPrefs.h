#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "EditorStyle.h"

// Per-user editor preferences (not part of a scene). Panel visibility is kept with the dock layout in
// imgui.ini instead, so saved layouts restore it too.
struct EditorPrefs {
    // Interface
    EditorStyle::Theme theme = EditorStyle::Theme::Dark;
    glm::vec3 accent{ 0.26f, 0.52f, 0.92f }; // sRGB
    float uiScale = 1.0f;     // fonts and sizes, 0.75..2
    float rounding = 3.0f;    // corner radius in pixels, 0..12
    bool compact = false;

    // Viewport
    float fieldOfView = 60.0f;      // vertical, degrees
    float lookSensitivity = 0.1f;   // degrees per pixel of mouse movement
    bool invertLookY = false;
    float cameraSpeed = 10.0f;      // meters per second
    bool showViewportHelp = true;   // the overlay in the top-left corner
    bool showOrientationGizmo = true;

    // Files
    int autosaveMinutes = 5;               // unsaved changes go to kAutosavePath this often; 0 = never
    std::vector<std::string> recentScenes; // most recent first

    bool operator==(const EditorPrefs&) const = default;

    EditorStyle::StyleOptions styleOptions() const
    {
        return { theme, ImVec4(accent.r, accent.g, accent.b, 1.0f), rounding, compact, uiScale };
    }
};

inline constexpr const char* kEditorPrefsPath = "editor_prefs.json";
inline constexpr const char* kAutosavePath = "autosave.scn";
inline constexpr size_t kMaxRecentScenes = 10;

// Puts `path` at the front of the recent scenes list.
void addRecentScene(EditorPrefs& prefs, const std::string& path);

// Clamps every field into its valid range.
EditorPrefs sanitizeEditorPrefs(EditorPrefs prefs);
// A missing or unreadable file yields defaults; missing keys keep their default value.
EditorPrefs loadEditorPrefs(const std::string& path = kEditorPrefsPath);
bool saveEditorPrefs(const EditorPrefs& prefs, const std::string& path = kEditorPrefsPath);
