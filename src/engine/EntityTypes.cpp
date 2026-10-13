#include "EntityTypes.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>
#include <nlohmann/json.hpp>
#include "ConfigPaths.h"
#include "Log.h"

namespace {

using Kind = EntityParamInfo::Kind;

const EntityParamInfo kEnemyParams[] = {
    { "health", "Health", Kind::Number, 60.0f, 1.0f, 1000.0f, "%.0f", "Damage it takes to kill (a close shotgun blast does about 100)" },
    { "speed", "Speed", Kind::Number, 3.4f, 0.0f, 12.0f, "%.1f m/s", "Running speed while chasing; the player walks at about 6" },
    { "damage", "Damage", Kind::Number, 12.0f, 0.0f, 200.0f, "%.0f", "Per hit: under 25 costs the player one of their three wounds, 25+ two, 50+ all three" },
    { "cooldown", "Attack delay", Kind::Number, 0.35f, 0.0f, 5.0f, "%.2f s", "Pause after each swing before the next one" },
    { "sight", "Sight range", Kind::Number, 30.0f, 1.0f, 100.0f, "%.0f m", "How far it sees the player (in front of it until alerted)" },
    { "hearing", "Hearing range", Kind::Number, 24.0f, 0.0f, 100.0f, "%.0f m", "How far away a shot alerts it" },
    { "size", "Size", Kind::Number, 1.0f, 0.5f, 2.5f, "%.2fx", "Scale of its body: model, collision and reach" },
    { "alert", "Hunter", Kind::Toggle, 0.0f, 0.0f, 1.0f, "", "Always knows where the player is and comes for them, instead of waiting to see or hear them" },
};
struct BuiltinPreset {
    const char* type;
    const char* label;
    const char* params;
    float color[3];
};
const BuiltinPreset kBuiltinPresets[] = {
    { kEnemyEntity, "Grunt", "health=60 speed=3.4 damage=12 cooldown=0.35 size=1", { 1.0f, 1.0f, 1.0f } },
    { kEnemyEntity, "Runner", "health=35 speed=6 damage=8 cooldown=0.2 size=0.85", { 1.0f, 0.85f, 0.4f } },
    { kEnemyEntity, "Brute", "health=180 speed=2.4 damage=30 cooldown=0.8 size=1.35", { 0.75f, 0.35f, 0.3f } },
    { kEnemyEntity, "Hunter", "health=70 speed=4.2 damage=14 cooldown=0.3 size=1 alert=1 sight=60", { 0.5f, 0.6f, 1.0f } },
    { kLightEntity, "Lamp", "intensity=15 range=12 spot=0 shadows=0", { 1.0f, 0.86f, 0.68f } },
    { kLightEntity, "Shadow lamp", "intensity=15 range=12 spot=0 shadows=1", { 1.0f, 0.86f, 0.68f } },
    { kLightEntity, "Spotlight", "intensity=40 range=20 spot=1 cone=30 softness=0.3 shadows=1", { 1.0f, 0.96f, 0.9f } },
    { kLightEntity, "Torch", "intensity=6 range=7 spot=0 shadows=0", { 1.0f, 0.55f, 0.22f } },
};
const EntityParamInfo kHealthParams[] = {
    { "amount", "Health", Kind::Number, 25.0f, 1.0f, 100.0f, "%.0f", "Health restored (not above 100)" },
};
const EntityParamInfo kAmmoParams[] = {
    { "amount", "Shells", Kind::Number, 8.0f, 1.0f, 100.0f, "%.0f", "Shells added to the reserve" },
};
const EntityParamInfo kRoundsParams[] = {
    { "amount", "Rounds", Kind::Number, 16.0f, 1.0f, 100.0f, "%.0f", "Loose pistol rounds added to the right pocket" },
};
const EntityParamInfo kExitParams[] = {
    { "next", "Next level", Kind::Text, 0.0f, 0.0f, 0.0f, "", "The .scn file to play after this one (next to the executable); empty ends the game" },
    { "needs_clear", "Needs clear", Kind::Toggle, 1.0f, 0.0f, 1.0f, "", "Opens once every enemy is dead; off: only an unlock input opens it" },
};
const EntityParamInfo kLightParams[] = {
    { "intensity", "Intensity", Kind::Number, 15.0f, 0.0f, 500.0f, "%.1f", "Brightness; it falls off with the square of the distance. The object's color is the light's color" },
    { "range", "Range", Kind::Number, 12.0f, 0.5f, 100.0f, "%.1f m", "Distance where the light has faded out completely" },
    { "spot", "Spotlight", Kind::Toggle, 0.0f, 0.0f, 1.0f, "", "A cone along the object's -Y axis (straight down until rotated) instead of every direction" },
    { "cone", "Cone angle", Kind::Number, 35.0f, 1.0f, 89.0f, "%.0f\xC2\xB0", "Spotlight: angle between its axis and the edge of the cone" },
    { "softness", "Edge softness", Kind::Number, 0.25f, 0.0f, 1.0f, "%.2f", "Spotlight: part of the cone that fades out towards its edge" },
    { "shadows", "Shadows", Kind::Toggle, 0.0f, 0.0f, 1.0f, "", "Casts shadows. The lights nearest the camera get them first: 16 shadow views, a point light takes 6, a spotlight 1" },
};

const EntityParamInfo kTriggerParams[] = {
    { "size_x", "Width", Kind::Number, 2.0f, 0.1f, 100.0f, "%.2f m", "Size along the object's X axis" },
    { "size_y", "Height", Kind::Number, 2.0f, 0.1f, 100.0f, "%.2f m", "Height above the object's position" },
    { "size_z", "Depth", Kind::Number, 2.0f, 0.1f, 100.0f, "%.2f m", "Size along the object's Z axis" },
    { "once", "Once", Kind::Toggle, 1.0f, 0.0f, 1.0f, "", "Disables itself after the first on_enter" },
    { "enabled", "Enabled", Kind::Toggle, 1.0f, 0.0f, 1.0f, "", "Starts enabled; the enable / disable inputs change it" },
};
const EntityParamInfo kRelayParams[] = {
    { "enabled", "Enabled", Kind::Toggle, 1.0f, 0.0f, 1.0f, "", "A disabled relay ignores trigger" },
};
const EntityParamInfo kTimerParams[] = {
    { "interval", "Interval", Kind::Number, 1.0f, 0.05f, 600.0f, "%.2f s", "Time between on_timer outputs" },
    { "repeat", "Repeat", Kind::Toggle, 1.0f, 0.0f, 1.0f, "", "Keeps firing; off: fires once, then stops" },
    { "running", "Running", Kind::Toggle, 0.0f, 0.0f, 1.0f, "", "Runs from the level start, without a start input" },
};
const EntityParamInfo kCounterParams[] = {
    { "target", "Target", Kind::Number, 3.0f, 1.0f, 1000.0f, "%.0f", "Fires on_reached when the count gets here" },
};

const EntityParamInfo kDoorParams[] = {
    { "group", "Group", Kind::Text, 0.0f, 0.0f, 0.0f, "", "The group of level objects that move (Ctrl+G groups the selection)" },
    { "angle", "Swing angle", Kind::Number, 0.0f, -180.0f, 180.0f, "%.0f\xC2\xB0", "Swings this far about the door object's position (the hinge, a vertical axis); 0 slides instead" },
    { "move_x", "Slide X", Kind::Number, 0.0f, -20.0f, 20.0f, "%.2f m", "Slide along the door object's X axis when open (with swing angle 0)" },
    { "move_y", "Slide Y", Kind::Number, 3.0f, -20.0f, 20.0f, "%.2f m", "Slide up (or down) when open" },
    { "move_z", "Slide Z", Kind::Number, 0.0f, -20.0f, 20.0f, "%.2f m", "Slide along the door object's Z axis when open" },
    { "time", "Open time", Kind::Number, 1.0f, 0.05f, 20.0f, "%.2f s", "Seconds to open or close fully" },
    { "use", "Player opens", Kind::Toggle, 1.0f, 0.0f, 1.0f, "", "The player opens and closes it with the use key when close" },
    { "locked", "Locked", Kind::Toggle, 0.0f, 0.0f, 1.0f, "", "Starts locked: the player can't open it until an unlock input" },
    { "open", "Starts open", Kind::Toggle, 0.0f, 0.0f, 1.0f, "", "Open at the level start" },
};
const char* const kEnemyOutputs[] = { "on_alert", "on_death" };
const char* const kDoorOutputs[] = { "on_opened", "on_closed" };
const char* const kDoorInputs[] = { "open", "close", "toggle", "lock", "unlock" };
const char* const kExitOutputs[] = { "on_unlock" };
const char* const kExitInputs[] = { "unlock", "lock" };
const char* const kEnemyInputs[] = { "alert", "kill" };
const char* const kPickupOutputs[] = { "on_pickup" };
const char* const kTriggerOutputs[] = { "on_enter", "on_exit" };
const char* const kToggleInputs[] = { "enable", "disable", "toggle" };
const char* const kRelayOutputs[] = { "on_trigger" };
const char* const kRelayInputs[] = { "trigger", "enable", "disable" };
const char* const kTimerOutputs[] = { "on_timer" };
const char* const kTimerInputs[] = { "start", "stop", "toggle", "fire" };
const char* const kCounterOutputs[] = { "on_reached" };
const char* const kCounterInputs[] = { "add", "subtract", "reset" };

// Presets are filled in by Registry from the built-ins and the presets file.
EntityTypeInfo kEntityTypes[] = {
    { kPlayerStartEntity, "Player start", "models/player_start.glb", "",
        "Where the player spawns, looking where the arrow points.", {}, {} },
    { kEnemyEntity, "Enemy", "models/enemy_grunt.glb", "health=60 speed=3.4 damage=12",
        "Chases and hits the player once it sees them, is shot or hears a shot nearby.", kEnemyParams, {},
        kEnemyOutputs, kEnemyInputs },
    { kHealthEntity, "Health pack", "models/medkit.glb", "amount=25",
        "Restores health when the player picks it up (not above 100).", kHealthParams, {}, kPickupOutputs },
    { kAmmoEntity, "Shotgun shells", "models/shells.glb", "amount=8",
        "Adds shells to the player's reserve when picked up.", kAmmoParams, {}, kPickupOutputs },
    { kRoundsEntity, "Pistol rounds", "models/rounds_box.glb", "amount=16",
        "Adds loose pistol rounds (loaded into magazines by hand) when picked up.", kRoundsParams, {},
        kPickupOutputs },
    { kExitEntity, "Level exit", "models/exit_portal.glb", "next=",
        "Ends the level when touched, once open: when every enemy is dead, or by an unlock input.", kExitParams, {},
        kExitOutputs, kExitInputs },
    { kLightEntity, "Light", "models/light.glb", "intensity=15 range=12",
        "Lights its surroundings in the object's color. A spotlight shines along the object's -Y axis; rotate it to aim.",
        kLightParams, {} },
    { kTriggerEntity, "Trigger", "models/trigger.glb", "size_x=2 size_y=2 size_z=2 once=1",
        "A box volume standing on the object's position, turned by rotation Y: fires on_enter when the player walks in "
        "and on_exit when they leave.", kTriggerParams, {}, kTriggerOutputs, kToggleInputs },
    { kRelayEntity, "Relay", "models/relay.glb", "",
        "Passes a trigger input on as on_trigger: one place to fan out to many targets, or to add a delay.",
        kRelayParams, {}, kRelayOutputs, kRelayInputs },
    { kTimerEntity, "Timer", "models/timer.glb", "interval=1 repeat=1 running=0",
        "Fires on_timer every interval while running (start / stop / toggle; fire fires now).", kTimerParams, {},
        kTimerOutputs, kTimerInputs },
    { kCounterEntity, "Counter", "models/counter.glb", "target=3",
        "Counts add / subtract inputs and fires on_reached when the count reaches the target (once, until reset).",
        kCounterParams, {}, kCounterOutputs, kCounterInputs },
    { kDoorEntity, "Door", "models/door.glb", "group= angle=0 move_y=3 time=1 use=1",
        "Moves the level objects of a group: swings them about this object's position (a vertical hinge) by the swing "
        "angle, or slides them by the slide offset (in this object's frame). The game hides this marker.",
        kDoorParams, {}, kDoorOutputs, kDoorInputs },
};
constexpr size_t kTypeCount = std::size(kEntityTypes);

int typeIndex(std::string_view id)
{
    for (size_t i = 0; i < kTypeCount; ++i)
        if (id == kEntityTypes[i].id)
            return static_cast<int>(i);
    return -1;
}

// The presets of every type: the built-ins with the user's presets merged in.
struct Registry {
    std::vector<EntityPreset> user[kTypeCount];
    std::vector<EntityPreset> merged[kTypeCount];

    Registry() { rebuild(); }

    void rebuild()
    {
        for (size_t t = 0; t < kTypeCount; ++t) {
            std::vector<EntityPreset>& out = merged[t];
            out.clear();
            for (const BuiltinPreset& builtin : kBuiltinPresets) {
                if (std::string_view(builtin.type) != kEntityTypes[t].id)
                    continue;
                EntityPreset& preset = out.emplace_back();
                preset.label = builtin.label;
                preset.params = builtin.params;
                std::copy(std::begin(builtin.color), std::end(builtin.color), preset.color);
            }
            for (const EntityPreset& preset : user[t]) {
                const auto same = std::find_if(out.begin(), out.end(),
                    [&](const EntityPreset& p) { return p.label == preset.label; });
                if (same != out.end())
                    *same = preset;
                else
                    out.push_back(preset);
            }
            kEntityTypes[t].presets = out;
        }
    }

    bool save(const std::string& path) const
    {
        nlohmann::json json = nlohmann::json::object();
        for (size_t t = 0; t < kTypeCount; ++t) {
            if (user[t].empty())
                continue;
            nlohmann::json& list = json[kEntityTypes[t].id];
            for (const EntityPreset& preset : user[t])
                list.push_back({ { "label", preset.label }, { "params", preset.params },
                    { "color", { preset.color[0], preset.color[1], preset.color[2] } } });
        }
        return writeDataFile(path, json.dump(4) + "\n");
    }
};

Registry& registry()
{
    static Registry instance;
    return instance;
}

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
    registry();
    return kEntityTypes;
}

const EntityTypeInfo* findEntityType(std::string_view id)
{
    registry();
    const int index = typeIndex(id);
    return index >= 0 ? &kEntityTypes[index] : nullptr;
}

std::string entityPresetsPath()
{
    return dataPath("entity_presets.json");
}

bool loadEntityPresets(const std::string& path)
{
    Registry& reg = registry();
    for (std::vector<EntityPreset>& list : reg.user)
        list.clear();
    std::ifstream file(path);
    if (!file) {
        reg.rebuild();
        return true;
    }
    const nlohmann::json json = nlohmann::json::parse(file, nullptr, false);
    if (json.is_discarded() || !json.is_object()) {
        LOG_ERROR("[ENTITY] " << path << " is not valid JSON; using the built-in presets\n");
        reg.rebuild();
        return false;
    }
    size_t count = 0;
    for (const auto& [type, list] : json.items()) {
        const int t = typeIndex(type);
        if (t < 0 || !list.is_array()) {
            LOG_ERROR("[ENTITY] " << path << ": ignoring presets of unknown entity type '" << type << "'\n");
            continue;
        }
        for (const nlohmann::json& item : list) {
            if (!item.is_object() || !item.contains("label") || !item["label"].is_string())
                continue;
            EntityPreset preset;
            preset.user = true;
            preset.label = item["label"].get<std::string>();
            if (preset.label.empty())
                continue;
            if (const auto params = item.find("params"); params != item.end() && params->is_string())
                preset.params = params->get<std::string>();
            if (const auto color = item.find("color"); color != item.end() && color->is_array() && color->size() == 3)
                for (int c = 0; c < 3; ++c)
                    if ((*color)[c].is_number())
                        preset.color[c] = std::clamp((*color)[c].get<float>(), 0.0f, 1.0f);
            std::erase_if(reg.user[t], [&](const EntityPreset& p) { return p.label == preset.label; });
            reg.user[t].push_back(std::move(preset));
            ++count;
        }
    }
    reg.rebuild();
    LOG_INFO("[ENTITY] " << count << " presets from " << path << "\n");
    return true;
}

bool saveEntityPreset(std::string_view type, EntityPreset preset, const std::string& path)
{
    const int t = typeIndex(type);
    if (t < 0 || preset.label.empty())
        return false;
    Registry& reg = registry();
    preset.user = true;
    std::vector<EntityPreset>& list = reg.user[t];
    const auto same = std::find_if(list.begin(), list.end(), [&](const EntityPreset& p) { return p.label == preset.label; });
    if (same != list.end())
        *same = std::move(preset);
    else
        list.push_back(std::move(preset));
    reg.rebuild();
    return reg.save(path);
}

bool deleteEntityPreset(std::string_view type, std::string_view label, const std::string& path)
{
    const int t = typeIndex(type);
    if (t < 0)
        return false;
    Registry& reg = registry();
    if (std::erase_if(reg.user[t], [&](const EntityPreset& p) { return p.label == label; }) == 0)
        return false;
    reg.rebuild();
    return reg.save(path);
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

std::vector<EntityConnection> parseEntityConnections(std::string_view text)
{
    std::vector<EntityConnection> out;
    while (!text.empty()) {
        const size_t end = std::min(text.find('\n'), text.size());
        std::string_view line = text.substr(0, end);
        text.remove_prefix(std::min(end + 1, text.size()));
        std::string_view fields[4];
        size_t count = 0;
        while (count < 4) {
            const size_t tab = std::min(line.find('\t'), line.size());
            fields[count++] = line.substr(0, tab);
            if (tab == line.size())
                break;
            line.remove_prefix(tab + 1);
        }
        if (count < 3 || fields[0].empty() || fields[1].empty() || fields[2].empty())
            continue;
        EntityConnection& connection = out.emplace_back();
        connection.output = fields[0];
        connection.target = fields[1];
        connection.input = fields[2];
        if (count == 4)
            std::from_chars(fields[3].data(), fields[3].data() + fields[3].size(), connection.delay);
        connection.delay = std::max(connection.delay, 0.0f);
    }
    return out;
}

std::string formatEntityConnections(std::span<const EntityConnection> connections)
{
    std::string out;
    for (const EntityConnection& c : connections) {
        if (c.output.empty() || c.target.empty() || c.input.empty())
            continue;
        out += c.output + '\t' + c.target + '\t' + c.input + '\t' + formatEntityNumber(c.delay) + '\n';
    }
    return out;
}
