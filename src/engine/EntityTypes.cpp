#include "EntityTypes.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <iterator>

namespace {

using Kind = EntityParamInfo::Kind;

const EntityParamInfo kEnemyParams[] = {
    { "health", "Health", Kind::Number, 60.0f, 1.0f, 1000.0f, "%.0f", "Damage it takes to kill (a close shotgun blast does about 100)" },
    { "speed", "Speed", Kind::Number, 3.4f, 0.0f, 12.0f, "%.1f m/s", "Running speed while chasing; the player walks at about 6" },
    { "damage", "Damage", Kind::Number, 12.0f, 0.0f, 200.0f, "%.0f", "Health the player loses per hit (the player has 100)" },
    { "cooldown", "Attack delay", Kind::Number, 0.35f, 0.0f, 5.0f, "%.2f s", "Pause after each swing before the next one" },
    { "sight", "Sight range", Kind::Number, 30.0f, 1.0f, 100.0f, "%.0f m", "How far it sees the player (in front of it until alerted)" },
    { "hearing", "Hearing range", Kind::Number, 24.0f, 0.0f, 100.0f, "%.0f m", "How far away a shot alerts it" },
    { "size", "Size", Kind::Number, 1.0f, 0.5f, 2.5f, "%.2fx", "Scale of its body: model, collision and reach" },
    { "alert", "Hunter", Kind::Toggle, 0.0f, 0.0f, 1.0f, "", "Always knows where the player is and comes for them, instead of waiting to see or hear them" },
};
const EntityPreset kEnemyPresets[] = {
    { "Grunt", "health=60 speed=3.4 damage=12 cooldown=0.35 size=1", { 1.0f, 1.0f, 1.0f } },
    { "Runner", "health=35 speed=6 damage=8 cooldown=0.2 size=0.85", { 1.0f, 0.85f, 0.4f } },
    { "Brute", "health=180 speed=2.4 damage=30 cooldown=0.8 size=1.35", { 0.75f, 0.35f, 0.3f } },
    { "Hunter", "health=70 speed=4.2 damage=14 cooldown=0.3 size=1 alert=1 sight=60", { 0.5f, 0.6f, 1.0f } },
};
const EntityParamInfo kHealthParams[] = {
    { "amount", "Health", Kind::Number, 25.0f, 1.0f, 100.0f, "%.0f", "Health restored (not above 100)" },
};
const EntityParamInfo kAmmoParams[] = {
    { "amount", "Shells", Kind::Number, 8.0f, 1.0f, 100.0f, "%.0f", "Shells added to the reserve" },
};
const EntityParamInfo kExitParams[] = {
    { "next", "Next level", Kind::Text, 0.0f, 0.0f, 0.0f, "", "The .scn file to play after this one (next to the executable); empty ends the game" },
};

const EntityTypeInfo kEntityTypes[] = {
    { kPlayerStartEntity, "Player start", "models/player_start.glb", "",
        "Where the player spawns, looking where the arrow points.", {}, {} },
    { kEnemyEntity, "Enemy", "models/enemy_grunt.glb", "health=60 speed=3.4 damage=12",
        "Chases and hits the player once it sees them, is shot or hears a shot nearby.", kEnemyParams, kEnemyPresets },
    { kHealthEntity, "Health pack", "models/medkit.glb", "amount=25",
        "Restores health when the player walks over it (not above 100).", kHealthParams, {} },
    { kAmmoEntity, "Shotgun shells", "models/shells.glb", "amount=8",
        "Adds shells to the player's reserve.", kAmmoParams, {} },
    { kExitEntity, "Level exit", "models/exit_portal.glb", "next=",
        "Ends the level when touched, once every enemy is dead.", kExitParams, {} },
};

// The value text of `key`, or nothing.
std::string_view findParam(std::string_view params, std::string_view key)
{
    size_t pos = 0;
    while (pos < params.size()) {
        while (pos < params.size() && params[pos] == ' ')
            ++pos;
        const size_t end = std::min(params.find(' ', pos), params.size());
        const std::string_view pair = params.substr(pos, end - pos);
        const size_t equals = pair.find('=');
        if (equals != std::string_view::npos && pair.substr(0, equals) == key)
            return pair.substr(equals + 1);
        pos = end;
    }
    return {};
}

} // namespace

std::span<const EntityTypeInfo> entityTypes()
{
    return kEntityTypes;
}

const EntityTypeInfo* findEntityType(std::string_view id)
{
    for (const EntityTypeInfo& type : kEntityTypes)
        if (id == type.id)
            return &type;
    return nullptr;
}

float entityParam(std::string_view params, std::string_view key, float fallback)
{
    const std::string_view text = findParam(params, key);
    float value = 0.0f;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    return text.empty() || ec != std::errc() || end != text.data() + text.size() ? fallback : value;
}

std::string entityParamText(std::string_view params, std::string_view key, std::string_view fallback)
{
    const std::string_view text = findParam(params, key);
    return std::string(text.empty() ? fallback : text);
}

std::string setEntityParam(std::string_view params, std::string_view key, std::string_view value)
{
    std::string result;
    bool placed = false;
    size_t pos = 0;
    while (pos < params.size()) {
        while (pos < params.size() && params[pos] == ' ')
            ++pos;
        const size_t end = std::min(params.find(' ', pos), params.size());
        const std::string_view pair = params.substr(pos, end - pos);
        pos = end;
        if (pair.empty())
            continue;
        const size_t equals = pair.find('=');
        std::string_view kept = pair;
        if (equals != std::string_view::npos && pair.substr(0, equals) == key) {
            if (placed || value.empty())
                continue;
            placed = true;
            kept = {};
        }
        if (!result.empty())
            result += ' ';
        if (kept.empty())
            result.append(key).append("=").append(value);
        else
            result.append(kept);
    }
    if (!placed && !value.empty()) {
        if (!result.empty())
            result += ' ';
        result.append(key).append("=").append(value);
    }
    return result;
}

std::string formatEntityNumber(float value)
{
    char text[32];
    snprintf(text, sizeof(text), "%.3f", std::round(value * 1000.0f) / 1000.0f);
    std::string result = text;
    while (!result.empty() && result.back() == '0')
        result.pop_back();
    if (!result.empty() && result.back() == '.')
        result.pop_back();
    return result == "-0" ? "0" : result;
}
