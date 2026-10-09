#pragma once
#include <span>
#include <string>
#include <string_view>

// One setting of an entity type, stored as key=value in ModelInstance::entityParams. The editor shows a
// widget per setting; the game reads them with entityParam()/entityParamText().
struct EntityParamInfo {
    enum class Kind { Number, Toggle, Text };
    const char* key;
    const char* label;
    Kind kind;
    float defaultValue; // Number/Toggle (0 or 1)
    float min;
    float max;
    const char* format; // Number: printf format for the widget
    const char* tooltip;
};

// A named set of values, e.g. an enemy variant. Keys it leaves out keep their current values.
// Built-in presets come from EntityTypes.cpp; the user's own are kept in entity_presets.json.
struct EntityPreset {
    std::string label;
    std::string params;
    float color[3] = { 1.0f, 1.0f, 1.0f }; // tint the editor gives the object (the game draws enemies with it)
    bool user = false;                     // from the presets file (can be deleted)
};

// Kinds of game entity a scene object can stand for (ModelInstance::entity). The editor places them as
// their marker model; the game hides the marker and spawns the entity in its place, configured by the
// object's entityParams ("key=value" pairs separated by spaces). Entities face +Z, turned by rotation Y.
struct EntityTypeInfo {
    const char* id;
    const char* label;
    const char* model;         // marker model next to the executable, also what the game shows
    const char* defaultParams;
    const char* description;
    std::span<const EntityParamInfo> params;
    std::span<const EntityPreset> presets; // built-ins merged with the presets file; changes when it is reloaded
};

inline constexpr const char* kPlayerStartEntity = "player_start";
inline constexpr const char* kEnemyEntity = "enemy";
inline constexpr const char* kHealthEntity = "health";
inline constexpr const char* kAmmoEntity = "ammo";
inline constexpr const char* kExitEntity = "exit";
// A point or spot light (Lights.h); the renderer lights the scene with it, the game has nothing to spawn.
inline constexpr const char* kLightEntity = "light";

std::span<const EntityTypeInfo> entityTypes();
const EntityTypeInfo* findEntityType(std::string_view id);

// The user's presets file in the shared config folder.
std::string entityPresetsPath();
// Reads the presets file over the built-ins: a user preset replaces the built-in with the same label. A
// missing file leaves only the built-ins; false when the file exists but can't be read.
bool loadEntityPresets(const std::string& path = entityPresetsPath());
// Adds the preset to the type's user presets (replacing one with the same label) and writes the file.
bool saveEntityPreset(std::string_view type, EntityPreset preset, const std::string& path = entityPresetsPath());
// Removes a user preset and writes the file; built-ins can't be removed.
bool deleteEntityPreset(std::string_view type, std::string_view label, const std::string& path = entityPresetsPath());

// Value of `key` in "key=value key2=value2", or the fallback when it is missing (or not a number).
float entityParam(std::string_view params, std::string_view key, float fallback);
std::string entityParamText(std::string_view params, std::string_view key, std::string_view fallback = {});
// The params with `key` set to `value` (replaced where it is, else appended); an empty value removes it.
std::string setEntityParam(std::string_view params, std::string_view key, std::string_view value);
// The number as entityParams stores it: no trailing zeros, at most 3 decimals.
std::string formatEntityNumber(float value);
