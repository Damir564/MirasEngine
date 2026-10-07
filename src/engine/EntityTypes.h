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
struct EntityPreset {
    const char* label;
    const char* params;
    float color[3]; // tint the editor gives the object (the game draws enemies with it)
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
    std::span<const EntityPreset> presets;
};

inline constexpr const char* kPlayerStartEntity = "player_start";
inline constexpr const char* kEnemyEntity = "enemy";
inline constexpr const char* kHealthEntity = "health";
inline constexpr const char* kAmmoEntity = "ammo";
inline constexpr const char* kExitEntity = "exit";

std::span<const EntityTypeInfo> entityTypes();
const EntityTypeInfo* findEntityType(std::string_view id);

// Value of `key` in "key=value key2=value2", or the fallback when it is missing (or not a number).
float entityParam(std::string_view params, std::string_view key, float fallback);
std::string entityParamText(std::string_view params, std::string_view key, std::string_view fallback = {});
// The params with `key` set to `value` (replaced where it is, else appended); an empty value removes it.
std::string setEntityParam(std::string_view params, std::string_view key, std::string_view value);
// The number as entityParams stores it: no trailing zeros, at most 3 decimals.
std::string formatEntityNumber(float value);
