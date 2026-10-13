#include "ArmsPoses.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <string>
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>
#include "engine/ConfigPaths.h"
#include "engine/Log.h"

namespace {

constexpr const char* kFileName = "arms_poses.json";

constexpr ArmsPoseField kPoses[] = {
    { "handReady", "Camera", &ArmsPoses::handReady },
    { "handLow", "Camera", &ArmsPoses::handLow },
    { "rightHandLow", "Camera", &ArmsPoses::rightHandLow },
    { "magazineHeld", "Camera", &ArmsPoses::magazineHeld },
    { "handBandolier", "Jacket", &ArmsPoses::handBandolier },
    { "handPocket", "Jacket", &ArmsPoses::handPocket },
    { "handRightPocket", "Jacket", &ArmsPoses::handRightPocket },
    { "handGround", "Body", &ArmsPoses::handGround },
    { "handHolster", "Body", &ArmsPoses::handHolster },
    { "handSling", "Body", &ArmsPoses::handSling },
    { "shotgunSlung", "Body", &ArmsPoses::shotgunSlung },
    { "shellInHand", "In palm", &ArmsPoses::shellInHand },
    { "roundInHand", "In palm", &ArmsPoses::roundInHand },
    { "magazineInHand", "In palm", &ArmsPoses::magazineInHand },
    { "rightOnShotgun", "On item", &ArmsPoses::rightOnShotgun },
    { "rightOnPistol", "On item", &ArmsPoses::rightOnPistol },
    { "rightOnMagazine", "On item", &ArmsPoses::rightOnMagazine },
    { "leftSupportsPistol", "On item", &ArmsPoses::leftSupportsPistol },
    { "leftAtMagazine", "On item", &ArmsPoses::leftAtMagazine },
    { "leftOnShotgunGrip", "On item", &ArmsPoses::leftOnShotgunGrip },
};

constexpr ArmsVectorField kVectors[] = {
    { "gunRest", "Aim", &ArmsPoses::gunRest },
    { "pistolRest", "Aim", &ArmsPoses::pistolRest },
    { "shotgunKick", "Kick", &ArmsPoses::shotgunKick },
    { "pistolKick", "Kick", &ArmsPoses::pistolKick },
    { "shotgunLower", "Lower", &ArmsPoses::shotgunLower },
    { "pistolLower", "Lower", &ArmsPoses::pistolLower },
    { "shotgunTilt", "Shotgun", &ArmsPoses::shotgunTilt },
    { "shotgunPumpMove", "Shotgun", &ArmsPoses::shotgunPumpMove },
    { "pumpRest", "Parts", &ArmsPoses::pumpRest },
    { "palmBelowPump", "Parts", &ArmsPoses::palmBelowPump },
    { "swapLower", "Swap", &ArmsPoses::swapLower },
    { "drawRise", "Swap", &ArmsPoses::drawRise },
};

constexpr ArmsNumberField kNumbers[] = {
    { "convergence", "Aim", &ArmsPoses::convergence },
    { "shotgunSightHeight", "Aim", &ArmsPoses::shotgunSightHeight },
    { "pistolSightHeight", "Aim", &ArmsPoses::pistolSightHeight },
    { "shotgunKickDegrees", "Kick", &ArmsPoses::shotgunKickDegrees },
    { "pistolKickDegrees", "Kick", &ArmsPoses::pistolKickDegrees },
    { "lowerDegrees", "Lower", &ArmsPoses::lowerDegrees },
    { "shotgunTiltRollDegrees", "Shotgun", &ArmsPoses::shotgunTiltRollDegrees },
    { "shotgunPumpRollDegrees", "Shotgun", &ArmsPoses::shotgunPumpRollDegrees },
    { "pumpTravel", "Parts", &ArmsPoses::pumpTravel },
    { "slideTravel", "Parts", &ArmsPoses::slideTravel },
    { "slideLocked", "Parts", &ArmsPoses::slideLocked },
    { "magazineDrop", "Parts", &ArmsPoses::magazineDrop },
    { "magEjectRollDegrees", "Parts", &ArmsPoses::magEjectRollDegrees },
    { "swapLowerDegrees", "Swap", &ArmsPoses::swapLowerDegrees },
    { "jacketOpenDegrees", "Body", &ArmsPoses::jacketOpenDegrees },
};

// Rounded to 0.01 mm / 0.00001 degrees, so the file shows 0.12 rather than the float's 0.11999999731779099.
double rounded(float value)
{
    return std::round(static_cast<double>(value) * 1e5) / 1e5;
}

nlohmann::json toJson(const glm::vec3& v)
{
    return nlohmann::json::array({ rounded(v.x), rounded(v.y), rounded(v.z) });
}

bool readVector(const nlohmann::json& j, glm::vec3& out)
{
    if (!j.is_array() || j.size() != 3)
        return false;
    for (const nlohmann::json& value : j)
        if (!value.is_number())
            return false;
    out = glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
    return true;
}

nlohmann::json toJson(const PoseEntry& pose)
{
    return { { "t", toJson(pose.translation) }, { "r", toJson(pose.degrees) } };
}

bool readPose(const nlohmann::json& j, PoseEntry& out)
{
    PoseEntry pose;
    if (!j.is_object() || !j.contains("t") || !j.contains("r") || !readVector(j["t"], pose.translation) ||
        !readVector(j["r"], pose.degrees))
        return false;
    out = pose;
    return true;
}

constexpr const char* kTrackNames[] = {
    "fetch", "store", "insertShell", "pushRound", "restock", "pickUp", "magInsert", "rack", "drawOffGun", "stowOffGun", "swap"
};
constexpr const char* kAnchorNames[] = {
    "rest", "ready", "low", "zone", "bandolier", "pocket", "rightPocket", "ground", "hip", "mainItem", "mainHand",
    "magSeat", "slideRear"
};
constexpr const char* kTimeBaseNames[] = { "event", "event2", "duration" };
static_assert(std::size(kTrackNames) == kArmsTrackCount);
static_assert(std::size(kAnchorNames) == static_cast<size_t>(ArmsAnchor::Count));
static_assert(std::size(kTimeBaseNames) == static_cast<size_t>(ArmsTimeBase::Count));

template <typename Enum, size_t N>
bool readName(const nlohmann::json& j, const char* const (&names)[N], Enum& out)
{
    if (!j.is_string())
        return false;
    const std::string name = j.get<std::string>();
    for (size_t i = 0; i < N; ++i) {
        if (name == names[i]) {
            out = static_cast<Enum>(i);
            return true;
        }
    }
    return false;
}

nlohmann::json toJson(const ArmsTrack& track)
{
    nlohmann::json keys = nlohmann::json::array();
    for (const ArmsKey& key : track.keys) {
        nlohmann::json k = { { "at", rounded(key.at) }, { "of", kTimeBaseNames[static_cast<int>(key.base)] },
            { "anchor", kAnchorNames[static_cast<int>(key.anchor)] }, { "t", toJson(key.offset.translation) },
            { "r", toJson(key.offset.degrees) } };
        if (key.needsZone)
            k["needsZone"] = true;
        keys.push_back(std::move(k));
    }
    return keys;
}

bool readTrack(const nlohmann::json& j, ArmsTrack& out)
{
    if (!j.is_array())
        return false;
    ArmsTrack track;
    for (const nlohmann::json& k : j) {
        ArmsKey key;
        if (!k.is_object() || !k.contains("at") || !k["at"].is_number() || !k.contains("of") ||
            !readName(k["of"], kTimeBaseNames, key.base) || !k.contains("anchor") ||
            !readName(k["anchor"], kAnchorNames, key.anchor) || !readPose(k, key.offset))
            return false;
        key.at = k["at"].get<float>();
        key.needsZone = k.value("needsZone", false);
        track.keys.push_back(key);
    }
    out = std::move(track);
    return true;
}

nlohmann::json toJson(const ArmsPoses& poses)
{
    nlohmann::json root = nlohmann::json::object();
    nlohmann::json& trackSection = root["tracks"] = nlohmann::json::object();
    for (size_t i = 0; i < kArmsTrackCount; ++i)
        trackSection[kTrackNames[i]] = toJson(poses.tracks[i]);
    nlohmann::json& poseSection = root["poses"] = nlohmann::json::object();
    for (const ArmsPoseField& field : kPoses)
        poseSection[field.name] = toJson(poses.*field.member);
    nlohmann::json& vectorSection = root["offsets"] = nlohmann::json::object();
    for (const ArmsVectorField& field : kVectors)
        vectorSection[field.name] = toJson(poses.*field.member);
    nlohmann::json& numberSection = root["values"] = nlohmann::json::object();
    for (const ArmsNumberField& field : kNumbers)
        numberSection[field.name] = rounded(poses.*field.member);
    return root;
}

// Reads every field of one section; a missing or broken one gets its default written back.
template <typename Field, typename Read, typename Write>
void readSection(nlohmann::json& root, const char* name, std::span<const Field> fields, ArmsPoses& poses, bool& changed,
    Read read, Write write)
{
    nlohmann::json& section = root[name];
    if (!section.is_object()) {
        section = nlohmann::json::object();
        changed = true;
    }
    for (const Field& field : fields) {
        const auto it = section.find(field.name);
        if (it != section.end() && read(*it, poses.*field.member))
            continue;
        if (it != section.end())
            LOG_ERROR("[GAME] " << kFileName << ": '" << field.name << "' is malformed; using the default\n");
        section[field.name] = write(poses.*field.member);
        changed = true;
    }
}

bool writeFile(const std::string& path, const nlohmann::json& root)
{
    return writeDataFile(path, root.dump(4) + "\n");
}

} // namespace

NodePose makePose(const glm::vec3& translation, float pitchDegrees, float yawDegrees, float rollDegrees)
{
    const glm::vec3 up(0.0f, 1.0f, 0.0f), x(1.0f, 0.0f, 0.0f), z(0.0f, 0.0f, 1.0f);
    NodePose pose;
    pose.translation = translation;
    pose.rotation = glm::angleAxis(glm::radians(yawDegrees), up) * glm::angleAxis(glm::radians(pitchDegrees), x) *
        glm::angleAxis(glm::radians(rollDegrees), z);
    return pose;
}

glm::mat4 matrixOf(const NodePose& pose)
{
    return glm::translate(glm::mat4(1.0f), pose.translation) * glm::mat4_cast(pose.rotation) *
        glm::scale(glm::mat4(1.0f), pose.scale);
}

NodePose poseOf(const glm::mat4& matrix)
{
    NodePose pose;
    pose.translation = glm::vec3(matrix[3]);
    pose.rotation = glm::quat_cast(glm::mat3(matrix));
    return pose;
}

NodePose easePose(const NodePose& a, const NodePose& b, float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    t = t * t * (3.0f - 2.0f * t);
    NodePose pose;
    pose.translation = glm::mix(a.translation, b.translation, t);
    pose.rotation = glm::slerp(a.rotation, b.rotation, t);
    return pose;
}

glm::mat4 offset(const glm::vec3& translation, float pitchDegrees, float yawDegrees, float rollDegrees)
{
    return matrixOf(makePose(translation, pitchDegrees, yawDegrees, rollDegrees));
}

NodePose PoseEntry::pose() const
{
    return makePose(translation, degrees.x, degrees.y, degrees.z);
}

glm::mat4 PoseEntry::matrix() const
{
    return matrixOf(pose());
}

std::span<const ArmsPoseField> armsPoseFields() { return kPoses; }
std::span<const ArmsVectorField> armsVectorFields() { return kVectors; }
std::span<const ArmsNumberField> armsNumberFields() { return kNumbers; }

ArmsPoses loadArmsPoses()
{
    ArmsPoses poses;
    const std::string path = dataPath(kFileName);
    nlohmann::json root = nlohmann::json::object();
    if (std::ifstream file(path); file) {
        root = nlohmann::json::parse(file, nullptr, false);
        if (root.is_discarded() || !root.is_object()) {
            // Leave a broken file alone so the edit isn't lost.
            LOG_ERROR("[GAME] " << path << " is not valid JSON; using the default arms poses\n");
            return ArmsPoses{};
        }
    }

    bool changed = false;
    readSection<ArmsPoseField>(root, "poses", kPoses, poses, changed, readPose,
        [](const PoseEntry& p) { return toJson(p); });
    readSection<ArmsVectorField>(root, "offsets", kVectors, poses, changed, readVector,
        [](const glm::vec3& v) { return toJson(v); });
    readSection<ArmsNumberField>(root, "values", kNumbers, poses, changed,
        [](const nlohmann::json& j, float& value) {
            if (!j.is_number())
                return false;
            value = j.get<float>();
            return true;
        },
        [](float value) { return nlohmann::json(rounded(value)); });

    nlohmann::json& tracks = root["tracks"];
    if (!tracks.is_object()) {
        tracks = nlohmann::json::object();
        changed = true;
    }
    for (size_t i = 0; i < kArmsTrackCount; ++i) {
        const auto it = tracks.find(kTrackNames[i]);
        if (it != tracks.end() && readTrack(*it, poses.tracks[i]))
            continue;
        if (it != tracks.end())
            LOG_ERROR("[GAME] " << kFileName << ": track '" << kTrackNames[i] << "' is malformed; using the default\n");
        tracks[kTrackNames[i]] = toJson(poses.tracks[i]);
        changed = true;
    }
    if (changed)
        writeFile(path, root);
    return poses;
}

bool saveArmsPoses(const ArmsPoses& poses)
{
    return writeFile(dataPath(kFileName), toJson(poses));
}

// ---------------------------------------------------------------------------------------------
// Tracks
// ---------------------------------------------------------------------------------------------

ArmsTrackId armsTrackOf(Task task)
{
    switch (task) {
    case Task::Fetch: return ArmsTrackId::Fetch;
    case Task::Store: return ArmsTrackId::Store;
    case Task::InsertShell: return ArmsTrackId::InsertShell;
    case Task::PushRound: return ArmsTrackId::PushRound;
    case Task::Restock: return ArmsTrackId::Restock;
    case Task::PickUp: return ArmsTrackId::PickUp;
    case Task::MagInsert: return ArmsTrackId::MagInsert;
    case Task::Rack: return ArmsTrackId::Rack;
    case Task::DrawOffGun: return ArmsTrackId::DrawOffGun;
    case Task::StowOffGun: return ArmsTrackId::StowOffGun;
    case Task::Swap: return ArmsTrackId::Swap;
    default: return ArmsTrackId::Count;
    }
}

const char* armsTrackName(ArmsTrackId id) { return kTrackNames[static_cast<size_t>(id)]; }
const char* armsAnchorName(ArmsAnchor anchor) { return kAnchorNames[static_cast<size_t>(anchor)]; }
const char* armsTimeBaseName(ArmsTimeBase base) { return kTimeBaseNames[static_cast<size_t>(base)]; }

std::array<ArmsTrack, kArmsTrackCount> defaultArmsTracks()
{
    using A = ArmsAnchor;
    using B = ArmsTimeBase;
    const auto key = [](float at, B base, A anchor, PoseEntry offset = {}, bool needsZone = false) {
        return ArmsKey{ at, base, anchor, offset, needsZone };
    };
    std::array<ArmsTrack, kArmsTrackCount> t;
    const auto set = [&](ArmsTrackId id, std::vector<ArmsKey> keys) { t[static_cast<size_t>(id)].keys = std::move(keys); };
    set(ArmsTrackId::Fetch, { key(1.0f, B::Event, A::Zone), key(1.0f, B::Duration, A::Ready) });
    set(ArmsTrackId::Store, { key(1.0f, B::Event, A::Zone), key(1.0f, B::Duration, A::Rest) });
    set(ArmsTrackId::InsertShell, { key(0.55f, B::Event, A::MainItem, { { 0.0f, -0.08f, -0.06f } }),  // under the port
        key(1.0f, B::Event, A::MainItem, { { 0.0f, -0.064f, -0.075f } }),                              // pushing it in
        key(1.0f, B::Duration, A::Rest) });
    set(ArmsTrackId::PushRound, { key(1.0f, B::Event, A::Zone, {}, true),
        key(1.0f, B::Event2, A::MainItem, { { -0.01f, 0.02f, 0.01f } }),                               // the feed lips
        key(1.0f, B::Duration, A::Rest) });
    set(ArmsTrackId::Restock, { key(1.0f, B::Event, A::Pocket), key(1.0f, B::Event2, A::Bandolier) });
    set(ArmsTrackId::PickUp, { key(1.0f, B::Event, A::Ground), key(1.0f, B::Duration, A::Ready) });
    // The magazine lines up under the grip, then goes up into it.
    set(ArmsTrackId::MagInsert, { key(0.45f, B::Event, A::MagSeat, { { 0.0f, -0.16f, 0.0f } }),
        key(1.0f, B::Event, A::MagSeat), key(1.0f, B::Duration, A::Rest) });
    // Grab the slide's rear and pull it back with it.
    const PoseEntry slideRear{ { 0.0f, 0.072f, 0.03f }, { 0.0f, 0.0f, 90.0f } };
    set(ArmsTrackId::Rack, { key(0.4f, B::Event, A::SlideRear, slideRear), key(1.0f, B::Event, A::SlideRear, slideRear),
        key(1.0f, B::Duration, A::Rest) });
    set(ArmsTrackId::DrawOffGun, { key(0.5f, B::Duration, A::Hip), key(1.0f, B::Duration, A::Low) });
    set(ArmsTrackId::StowOffGun, { key(1.0f, B::Duration, A::Hip) });
    // Handing the magazine over to the main hand.
    set(ArmsTrackId::Swap, { key(1.0f, B::Duration, A::MainHand, { { -0.06f, 0.0f, 0.0f } }) });
    return t;
}

float ArmsTaskTimes::base(ArmsTimeBase b) const
{
    switch (b) {
    case ArmsTimeBase::Event: return eventAt;
    case ArmsTimeBase::Event2: return event2At >= 0.0f ? event2At : eventAt;
    default: return duration;
    }
}

ArmsTaskTimes ArmsTaskTimes::of(const HandState& hand)
{
    return { hand.duration, hand.eventAt, hand.event2At, hand.zoneUsed };
}

NodePose armsTrackTarget(const ArmsPoses& p, ArmsTrackId track, const ArmsTaskTimes& times, float t,
    const ArmsAnchors& a, int& key, float& phaseLeft)
{
    const std::vector<ArmsKey>& keys = p.tracks[static_cast<size_t>(track)].keys;
    int chosen = -1;
    for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
        if (keys[i].needsZone && !times.zoneUsed)
            continue;
        chosen = i;
        if (t < keys[i].at * times.base(keys[i].base))
            break;
    }
    if (chosen < 0) {
        key = 100 + a.restKey;
        phaseLeft = times.duration - t;
        return poseOf(a.rest);
    }
    const ArmsKey& k = keys[chosen];
    phaseLeft = k.at * times.base(k.base) - t;
    int extra = 0;
    glm::mat4 anchor(1.0f);
    switch (k.anchor) {
    case ArmsAnchor::Rest: anchor = a.rest; extra = a.restKey; break;
    case ArmsAnchor::Ready: anchor = p.handReady.matrix(); break;
    case ArmsAnchor::Low: anchor = p.handLow.matrix(); break;
    case ArmsAnchor::Zone:
        extra = static_cast<int>(a.zone);
        switch (a.zone) {
        case Zone::Bandolier: anchor = a.flapL * p.handBandolier.matrix(); break;
        case Zone::JacketPocket: anchor = a.flapL * p.handPocket.matrix(); break;
        default: anchor = a.flapR * p.handRightPocket.matrix(); break;
        }
        break;
    case ArmsAnchor::Bandolier: anchor = a.flapL * p.handBandolier.matrix(); break;
    case ArmsAnchor::Pocket: anchor = a.flapL * p.handPocket.matrix(); break;
    case ArmsAnchor::RightPocket: anchor = a.flapR * p.handRightPocket.matrix(); break;
    case ArmsAnchor::Ground: anchor = a.bodyToCamera * p.handGround.matrix(); break;
    case ArmsAnchor::Hip: anchor = a.bodyToCamera * (a.hipPistol ? p.handHolster : p.handSling).matrix(); break;
    case ArmsAnchor::MainItem: anchor = a.mainItem; break;
    case ArmsAnchor::MainHand: anchor = a.mainHand; break;
    case ArmsAnchor::MagSeat:
        // The offset places the magazine; the hand holds it by the base.
        key = 1000 + (static_cast<int>(track) * 32 + chosen) * 16;
        return poseOf(a.mainItem * a.magSeat * k.offset.matrix() * glm::inverse(p.magazineInHand.matrix()));
    case ArmsAnchor::SlideRear: anchor = a.mainItem * offset(glm::vec3(0.0f, 0.0f, p.slideTravel * a.slide)); break;
    default: break;
    }
    key = 1000 + (static_cast<int>(track) * 32 + chosen) * 16 + extra;
    return poseOf(anchor * k.offset.matrix());
}

float armsRackPull(const ArmsPoses& p, const ArmsTaskTimes& times, float t)
{
    if (times.eventAt <= 0.0f)
        return 1.0f;
    const std::vector<ArmsKey>& keys = p.tracks[static_cast<size_t>(ArmsTrackId::Rack)].keys;
    const float grab = keys.empty() ? 0.0f
                                    : std::clamp(keys[0].at * times.base(keys[0].base), 0.0f, times.eventAt * 0.99f);
    return std::clamp((t - grab) / std::max(times.eventAt - grab, 1e-3f), 0.0f, 1.0f);
}

void ArmsRigs::find(const AnimatedModelData* shotgun, const AnimatedModelData* pistol, const AnimatedModelData* magazine,
    const AnimatedModelData* hands)
{
    *this = {};
    const auto valid = [](const AnimatedModelData* model) { return model && !model->nodes.empty(); };
    const auto nodeTranslation = [](const AnimatedModelData& model, int node, glm::vec3 fallback) {
        return node >= 0 ? model.nodes[node].translation : fallback;
    };
    if (valid(shotgun)) {
        shotgunGun = shotgun->findNode("gun");
        shotgunPump = shotgun->findNode("pump");
        shotgunFlash = shotgun->findNode("flash");
        shotgunMuzzle = nodeTranslation(*shotgun, shotgun->findNode("muzzle"), shotgunMuzzle);
    }
    if (valid(pistol)) {
        pistolSlide = pistol->findNode("slide");
        pistolFlash = pistol->findNode("flash");
        pistolMuzzle = nodeTranslation(*pistol, pistol->findNode("muzzle"), pistolMuzzle);
        if (const int seat = pistol->findNode("magseat"); seat >= 0) {
            NodePose pose;
            pose.translation = pistol->nodes[seat].translation;
            pose.rotation = pistol->nodes[seat].rotation;
            pistolMagSeat = matrixOf(pose);
        }
    }
    if (valid(magazine)) {
        magazineFollower = magazine->findNode("follower");
        for (size_t i = 0; i < magazineRounds.size(); ++i)
            magazineRounds[i] = magazine->findNode("round" + std::to_string(i));
    }
    if (valid(hands)) {
        handR = hands->findNode("hand_r");
        handL = hands->findNode("hand_l");
    }
}

glm::mat4 bodyWorldFrom(const glm::mat4& cameraWorld)
{
    const auto flat = [](const glm::vec3& v) { return glm::vec3(v.x, 0.0f, v.z); };
    glm::vec3 forward = -glm::vec3(cameraWorld[2]);
    if (glm::length(flat(forward)) < 0.1f)
        forward = glm::vec3(cameraWorld[1]); // looking straight down: the view's up points ahead
    forward = glm::normalize(flat(forward));
    return glm::rotate(glm::translate(glm::mat4(1.0f), glm::vec3(cameraWorld[3])), std::atan2(-forward.x, -forward.z),
        glm::vec3(0.0f, 1.0f, 0.0f));
}

NodePose shotgunPose(const ArmsPoses& p, float kick, float pumpAmount, float tilt, const glm::vec2& sway, float lower,
    float side)
{
    const glm::vec3 up(0.0f, 1.0f, 0.0f), x(1.0f, 0.0f, 0.0f), z(0.0f, 0.0f, 1.0f);
    const glm::vec3 mirror(side, 1.0f, 1.0f);
    const float yaw = side * std::atan(p.gunRest.x / p.convergence) + sway.x;
    const float pitch = std::atan((-p.gunRest.y - p.shotgunSightHeight) / p.convergence) + sway.y +
        glm::radians(p.shotgunKickDegrees) * kick - glm::radians(p.lowerDegrees) * lower;
    const float roll = side * (glm::radians(p.shotgunPumpRollDegrees) * pumpAmount + glm::radians(p.shotgunTiltRollDegrees) * tilt);
    NodePose pose;
    pose.translation = p.gunRest * mirror + p.shotgunKick * kick + p.shotgunTilt * mirror * tilt +
        p.shotgunPumpMove * pumpAmount + p.shotgunLower * mirror * lower;
    pose.rotation = glm::angleAxis(yaw, up) * glm::angleAxis(pitch, x) * glm::angleAxis(roll, z);
    return pose;
}

NodePose pistolPose(const ArmsPoses& p, float kick, const glm::vec2& sway, float lower, float side)
{
    const glm::vec3 up(0.0f, 1.0f, 0.0f), x(1.0f, 0.0f, 0.0f);
    const glm::vec3 mirror(side, 1.0f, 1.0f);
    const float yaw = side * std::atan(p.pistolRest.x / p.convergence) + sway.x;
    const float pitch = std::atan((-p.pistolRest.y - p.pistolSightHeight) / p.convergence) + sway.y +
        glm::radians(p.pistolKickDegrees) * kick - glm::radians(p.lowerDegrees) * lower;
    NodePose pose;
    pose.translation = p.pistolRest * mirror + p.pistolKick * kick + p.pistolLower * mirror * lower;
    pose.rotation = glm::angleAxis(yaw, up) * glm::angleAxis(pitch, x);
    return pose;
}

void poseShotgun(const AnimatedModelData& model, const ArmsRigs& rigs, const ArmsPoses& p, float pump, float flash,
    float flashSpin, std::vector<NodePose>& pose)
{
    restPose(model.nodes, pose);
    if (rigs.shotgunGun >= 0)
        pose[rigs.shotgunGun] = NodePose{};
    if (rigs.shotgunPump >= 0)
        pose[rigs.shotgunPump].translation = p.pumpRest + glm::vec3(0.0f, 0.0f, p.pumpTravel * pump);
    if (rigs.shotgunFlash >= 0) {
        pose[rigs.shotgunFlash].scale = flash > 0.0f ? glm::vec3(1.2f * flash, 1.2f * flash, 1.4f) : glm::vec3(0.0001f);
        pose[rigs.shotgunFlash].rotation = glm::angleAxis(flashSpin, glm::vec3(0.0f, 0.0f, 1.0f));
    }
}

void posePistol(const AnimatedModelData& model, const ArmsRigs& rigs, const ArmsPoses& p, float slide, float flash,
    float flashSpin, std::vector<NodePose>& pose)
{
    restPose(model.nodes, pose);
    if (rigs.pistolSlide >= 0)
        pose[rigs.pistolSlide].translation += glm::vec3(0.0f, 0.0f, p.slideTravel * slide);
    if (rigs.pistolFlash >= 0) {
        pose[rigs.pistolFlash].scale = flash > 0.0f ? glm::vec3(0.6f * flash, 0.6f * flash, 0.7f) : glm::vec3(0.0001f);
        pose[rigs.pistolFlash].rotation = glm::angleAxis(flashSpin, glm::vec3(0.0f, 0.0f, 1.0f));
    }
}

void poseMagazine(const AnimatedModelData& model, const ArmsRigs& rigs, int rounds, std::vector<NodePose>& pose)
{
    restPose(model.nodes, pose);
    // The rounds stack from the top; the follower sits under the last one.
    rounds = std::clamp(rounds, 0, static_cast<int>(rigs.magazineRounds.size()));
    for (size_t i = 0; i < rigs.magazineRounds.size(); ++i)
        if (rigs.magazineRounds[i] >= 0)
            pose[rigs.magazineRounds[i]].scale = glm::vec3(static_cast<int>(i) < rounds ? 1.0f : 0.0001f);
    if (rigs.magazineFollower >= 0)
        pose[rigs.magazineFollower].translation.y = -0.004f - 0.012f * rounds;
}

void poseHands(const AnimatedModelData& model, const ArmsRigs& rigs, const NodePose& right, const NodePose& left,
    std::vector<NodePose>& pose)
{
    restPose(model.nodes, pose);
    if (rigs.handR >= 0)
        pose[rigs.handR] = right;
    if (rigs.handL >= 0)
        pose[rigs.handL] = left;
}
