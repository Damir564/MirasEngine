#include "WeaponTuning.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include "engine/ConfigPaths.h"
#include "engine/Log.h"

namespace {

constexpr const char* kFileName = "weapon_tuning.json";

// One object of the file. Reads values into the tuning; a missing or mistyped key gets the default
// written back, so the file always lists every setting.
class Section {
public:
    Section(nlohmann::json& parent, const char* name, bool& changed) : m_changed(changed)
    {
        nlohmann::json& section = parent[name];
        if (!section.is_object()) {
            section = nlohmann::json::object();
            m_changed = true;
        }
        m_json = &section;
    }

    void number(const char* key, float& value) { read(key, value, [](const nlohmann::json& j) { return j.is_number(); }); }
    void integer(const char* key, int& value) { read(key, value, [](const nlohmann::json& j) { return j.is_number_integer(); }); }
    void flag(const char* key, bool& value) { read(key, value, [](const nlohmann::json& j) { return j.is_boolean(); }); }
    nlohmann::json& json() { return *m_json; }

private:
    template <typename T, typename Valid>
    void read(const char* key, T& value, Valid valid)
    {
        const auto it = m_json->find(key);
        if (it != m_json->end() && valid(*it)) {
            value = it->get<T>();
            return;
        }
        if (it != m_json->end())
            LOG_ERROR("[GAME] " << kFileName << ": '" << key << "' has the wrong type; using the default\n");
        (*m_json)[key] = value;
        m_changed = true;
    }

    nlohmann::json* m_json = nullptr;
    bool& m_changed;
};

void readZone(Section& zones, const char* name, ZoneConfig& zone, bool& changed)
{
    Section section(zones.json(), name, changed);
    section.number("accessTime", zone.accessTime);
    section.integer("shells", zone.shells);
    section.integer("rounds", zone.rounds);
    section.integer("magazines", zone.magazines);
}

void readCancelWindows(nlohmann::json& root, std::vector<CancelWindow>& windows, bool& changed)
{
    nlohmann::json& list = root["cancelWindows"];
    if (!list.is_array()) {
        list = nlohmann::json::array();
        for (const CancelWindow& window : windows)
            list.push_back({ { "task", window.task }, { "by", window.by }, { "beforeEvent", window.beforeEvent } });
        changed = true;
        return;
    }
    windows.clear();
    for (const nlohmann::json& entry : list) {
        if (!entry.is_object() || !entry.contains("task") || !entry.contains("by") || !entry["task"].is_string() ||
            !entry["by"].is_string()) {
            LOG_ERROR("[GAME] " << kFileName << ": a cancel window needs \"task\" and \"by\" strings; skipped\n");
            continue;
        }
        CancelWindow window;
        window.task = entry["task"].get<std::string>();
        window.by = entry["by"].get<std::string>();
        window.beforeEvent = entry.value("beforeEvent", true);
        windows.push_back(window);
    }
}

} // namespace

WeaponTuning loadWeaponTuning()
{
    WeaponTuning tuning;
    const std::string path = configPath(kFileName);
    nlohmann::json root = nlohmann::json::object();
    if (std::ifstream file(path); file) {
        root = nlohmann::json::parse(file, nullptr, false);
        if (root.is_discarded() || !root.is_object()) {
            // Leave a broken file alone so the edit isn't lost.
            LOG_ERROR("[GAME] " << path << " is not valid JSON; using the default weapon tuning\n");
            return WeaponTuning{};
        }
    }

    bool changed = false;
    Section sway(root, "sway", changed);
    sway.number("turnSway", tuning.turnSway);
    sway.number("maxSwayDegrees", tuning.maxSwayDegrees);
    sway.number("swayRecovery", tuning.swayRecovery);
    sway.number("moveTurnSway", tuning.moveTurnSway);
    sway.number("stepSwayYawDegrees", tuning.stepSwayYawDegrees);
    sway.number("stepSwayPitchDegrees", tuning.stepSwayPitchDegrees);
    sway.number("bobSide", tuning.bobSide);
    sway.number("bobUp", tuning.bobUp);
    sway.number("sprintBob", tuning.sprintBob);

    ArmsConfig& arms = tuning.arms;
    Section pistol(root, "pistol", changed);
    pistol.integer("magazineCapacity", arms.magazineCapacity);
    pistol.number("fireRate", arms.pistolFireRate);
    pistol.number("recoilDegrees", arms.pistolRecoilDegrees);
    pistol.number("drawTime", arms.drawTime);
    pistol.number("stowTime", arms.stowTime);
    pistol.number("ejectTime", arms.ejectTime);
    pistol.number("ejectFrame", arms.ejectFrame);
    pistol.number("insertTime", arms.insertTime);
    pistol.number("insertFrame", arms.insertFrame);
    pistol.number("rackTime", arms.rackTime);
    pistol.number("rackFrame", arms.rackFrame);
    pistol.flag("autoRack", arms.autoRack);
    pistol.number("autoRackDelay", arms.autoRackDelay);
    pistol.number("roundInsertTime", arms.roundInsertTime);
    pistol.number("roundInsertFrame", arms.roundInsertFrame);
    pistol.number("handoffTime", arms.handoffTime);

    Section shotgun(root, "shotgun", changed);
    shotgun.integer("tubeSize", arms.tubeSize);
    shotgun.number("pumpTime", arms.pumpTime);
    shotgun.number("pumpEjectFrame", arms.pumpEjectFrame);
    shotgun.number("shellInsertTime", arms.shellInsertTime);
    shotgun.number("shellInsertFrame", arms.shellInsertFrame);
    shotgun.number("recoilDegrees", arms.shotgunRecoilDegrees);
    shotgun.flag("pumpWhileDualWielding", arms.pumpWhileDualWielding);

    Section hands(root, "hands", changed);
    hands.number("pickUpTime", arms.pickUpTime);
    hands.number("pickUpFrame", arms.pickUpFrame);
    hands.number("restockTime", arms.restockTime);
    hands.flag("forceReloadDropsGun", arms.forceReloadDropsGun);

    Section zones(root, "zones", changed);
    readZone(zones, "bandolier", arms.zones[static_cast<int>(Zone::Bandolier)], changed);
    readZone(zones, "jacketPocket", arms.zones[static_cast<int>(Zone::JacketPocket)], changed);
    readZone(zones, "rightPocket", arms.zones[static_cast<int>(Zone::RightPocket)], changed);

    readCancelWindows(root, arms.cancelWindows, changed);

    if (changed) {
        std::ofstream out(path);
        if (out)
            out << root.dump(4) << "\n";
        else
            LOG_ERROR("[GAME] Can't write " << path << "\n");
    }
    return tuning;
}
