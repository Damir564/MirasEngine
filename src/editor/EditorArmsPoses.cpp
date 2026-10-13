// Arms Poses panel: the first-person hands and guns of the game (game/ArmsPoses.h, arms_poses.json),
// edited with a live preview drawn in front of the editor camera the way GameArms.cpp draws them. The
// Tasks tab plays one support hand task along its track of keys, with the timing weapon_tuning.json gives.
#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "engine/ConfigPaths.h"
#include "game/WeaponTuning.h"

namespace {

// What the preview shows; the int fields of Editor::ArmsPreviewState index these.
enum class ArmsMain { None, Shotgun, Pistol, Magazine };
constexpr const char* kMainNames[] = { "Nothing", "Shotgun", "Pistol", "Magazine" };

// Where the support hand is when no task plays.
enum class ArmsOff { Rest, Ready, Bandolier, Pocket, RightPocket, Ground, Holster, Sling, OwnPistol, OwnShotgun };
constexpr const char* kOffNames[] = {
    "On the main item", "Ready (holding)", "Bandolier", "Jacket pocket", "Right pocket", "Floor", "Holster", "Sling",
    "Own pistol", "Own shotgun"
};

enum class ArmsHeld { None, Shell, Round, Magazine };
constexpr const char* kHeldNames[] = { "Nothing", "Shell", "Round", "Magazine" };

constexpr const char* kZoneNames[] = { "Bandolier", "Jacket pocket", "Right pocket" };
constexpr const char* kGunNames[] = { "Pistol", "Shotgun" };

// Selecting a pose sets up the preview where it is used.
struct ArmsPreset {
    const char* pose;
    ArmsMain main;
    ArmsOff off;
    ArmsHeld held;
    float jacket;
};
constexpr ArmsPreset kPresets[] = {
    { "handReady", ArmsMain::Shotgun, ArmsOff::Ready, ArmsHeld::Shell, 0.0f },
    { "handLow", ArmsMain::None, ArmsOff::Rest, ArmsHeld::None, 0.0f },
    { "rightHandLow", ArmsMain::None, ArmsOff::Rest, ArmsHeld::None, 0.0f },
    { "magazineHeld", ArmsMain::Magazine, ArmsOff::Rest, ArmsHeld::None, 0.0f },
    { "handBandolier", ArmsMain::Shotgun, ArmsOff::Bandolier, ArmsHeld::None, 0.0f },
    { "handPocket", ArmsMain::Shotgun, ArmsOff::Pocket, ArmsHeld::None, 1.0f },
    { "handRightPocket", ArmsMain::Pistol, ArmsOff::RightPocket, ArmsHeld::None, 1.0f },
    { "handGround", ArmsMain::Shotgun, ArmsOff::Ground, ArmsHeld::None, 0.0f },
    { "handHolster", ArmsMain::Shotgun, ArmsOff::Holster, ArmsHeld::None, 0.0f },
    { "handSling", ArmsMain::Pistol, ArmsOff::Sling, ArmsHeld::None, 0.0f },
    { "shotgunSlung", ArmsMain::Pistol, ArmsOff::Rest, ArmsHeld::None, 0.0f },
    { "shellInHand", ArmsMain::Shotgun, ArmsOff::Ready, ArmsHeld::Shell, 0.0f },
    { "roundInHand", ArmsMain::Magazine, ArmsOff::Ready, ArmsHeld::Round, 0.0f },
    { "magazineInHand", ArmsMain::Pistol, ArmsOff::Ready, ArmsHeld::Magazine, 0.0f },
    { "rightOnShotgun", ArmsMain::Shotgun, ArmsOff::Rest, ArmsHeld::None, 0.0f },
    { "rightOnPistol", ArmsMain::Pistol, ArmsOff::Rest, ArmsHeld::None, 0.0f },
    { "rightOnMagazine", ArmsMain::Magazine, ArmsOff::Rest, ArmsHeld::None, 0.0f },
    { "leftSupportsPistol", ArmsMain::Pistol, ArmsOff::Rest, ArmsHeld::None, 0.0f },
    { "leftAtMagazine", ArmsMain::Magazine, ArmsOff::Rest, ArmsHeld::None, 0.0f },
    { "leftOnShotgunGrip", ArmsMain::Pistol, ArmsOff::OwnShotgun, ArmsHeld::None, 0.0f },
};

// What the main hand and the support hand hold when a task is picked.
struct ArmsTaskPreset {
    ArmsMain main;
    ArmsHeld held;
};
constexpr ArmsTaskPreset kTaskPresets[] = {
    { ArmsMain::Shotgun, ArmsHeld::Shell },     // Fetch
    { ArmsMain::Shotgun, ArmsHeld::Shell },     // Store
    { ArmsMain::Shotgun, ArmsHeld::Shell },     // InsertShell
    { ArmsMain::Magazine, ArmsHeld::Round },    // PushRound
    { ArmsMain::Shotgun, ArmsHeld::None },      // Restock
    { ArmsMain::Shotgun, ArmsHeld::Shell },     // PickUp
    { ArmsMain::Pistol, ArmsHeld::Magazine },   // MagInsert
    { ArmsMain::Pistol, ArmsHeld::None },       // Rack
    { ArmsMain::Shotgun, ArmsHeld::None },      // DrawOffGun
    { ArmsMain::Shotgun, ArmsHeld::None },      // StowOffGun
    { ArmsMain::Pistol, ArmsHeld::Magazine },   // Swap
};
static_assert(std::size(kTaskPresets) == kArmsTrackCount);

constexpr GameModelId kPreviewModels[] = {
    GameModelId::Hands, GameModelId::Shotgun, GameModelId::Pistol, GameModelId::Magazine, GameModelId::ShellItem,
    GameModelId::RoundItem, GameModelId::PlayerBody
};

constexpr glm::vec3 kUp(0.0f, 1.0f, 0.0f);
constexpr glm::vec3 kXAxis(1.0f, 0.0f, 0.0f);
// The game's frame step is close enough for the preview's replay of a task.
constexpr float kTaskStep = 1.0f / 120.0f;

template <size_t N>
bool combo(const char* label, int& value, const char* const (&names)[N])
{
    return ImGui::Combo(label, &value, names, static_cast<int>(N));
}

template <typename Enum>
bool enumCombo(const char* label, Enum& value, const char* (*name)(Enum))
{
    bool changed = false;
    if (ImGui::BeginCombo(label, name(value))) {
        for (int i = 0; i < static_cast<int>(Enum::Count); ++i) {
            const Enum option = static_cast<Enum>(i);
            if (ImGui::Selectable(name(option), option == value)) {
                value = option;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

bool usesZone(ArmsTrackId track)
{
    return track == ArmsTrackId::Fetch || track == ArmsTrackId::Store || track == ArmsTrackId::PushRound;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Preview models
// ---------------------------------------------------------------------------------------------

void Editor::updateArmsPreviewAssets()
{
    const bool wanted = m_showArmsPoses && m_armsPreview;
    // A scene load or new scene unloads every model, these included.
    if (m_armsAssetsLoaded && (!wanted || !m_armsAssets.intact(m_models)))
        releaseArmsPreview();
    if (!wanted || m_armsAssetsLoaded)
        return;

    for (GameModelId id : kPreviewModels)
        m_armsAssets.loadModel(m_models, id);
    m_armsAssetsLoaded = true;
    m_armsRigs.find(&m_armsAssets.model(GameModelId::Shotgun).data, &m_armsAssets.model(GameModelId::Pistol).data,
        &m_armsAssets.model(GameModelId::Magazine).data, &m_armsAssets.model(GameModelId::Hands).data);
    const AnimatedModelData& body = m_armsAssets.model(GameModelId::PlayerBody).data;
    m_armsFlapL = body.findNode("flap_l");
    m_armsFlapR = body.findNode("flap_r");
    m_armsHolster = body.findNode("holster_gun");
}

void Editor::releaseArmsPreview()
{
    m_armsDrawables.clear();
    if (!m_armsAssetsLoaded)
        return;
    m_armsAssets.release(m_models);
    m_armsAssetsLoaded = false;
}

// ---------------------------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------------------------

ArmsTaskTimes Editor::armsTaskTimes() const
{
    const ArmsConfig& c = m_armsConfig;
    const float access = c.zones[std::clamp(m_armsTask.zone, 0, 2)].accessTime;
    const auto timed = [](float duration, float fraction) {
        ArmsTaskTimes times;
        times.duration = std::max(duration, 0.0f);
        times.eventAt = times.duration * std::clamp(fraction, 0.0f, 1.0f);
        return times;
    };
    switch (static_cast<ArmsTrackId>(m_armsTask.track)) {
    case ArmsTrackId::Fetch:
    case ArmsTrackId::Store: return timed(access, 0.5f);
    case ArmsTrackId::InsertShell: return timed(c.shellInsertTime, c.shellInsertFrame);
    case ArmsTrackId::PushRound: {
        if (!m_armsTask.fromStorage)
            return timed(c.roundInsertTime, c.roundInsertFrame);
        // As Arms: to the rounds and back (half the access time each way), then pressed in.
        ArmsTaskTimes times = timed(access + c.roundInsertTime, 0.0f);
        times.eventAt = access * 0.5f;
        times.event2At = access + c.roundInsertTime * c.roundInsertFrame;
        times.zoneUsed = true;
        return times;
    }
    case ArmsTrackId::Restock: {
        ArmsTaskTimes times = timed(c.restockTime, 1.0f / 3.0f);
        times.event2At = c.restockTime * 2.0f / 3.0f;
        return times;
    }
    case ArmsTrackId::PickUp: return timed(c.pickUpTime, c.pickUpFrame);
    case ArmsTrackId::MagInsert: return timed(c.insertTime, c.insertFrame);
    case ArmsTrackId::Rack: return timed(c.rackTime, c.rackFrame);
    case ArmsTrackId::DrawOffGun: return timed(c.drawTime, 1.0f);
    case ArmsTrackId::StowOffGun: return timed(c.stowTime, 1.0f);
    case ArmsTrackId::Swap: return timed(c.handoffTime, 1.0f);
    default: return timed(1.0f, 0.5f);
    }
}

void Editor::selectArmsTask(int track)
{
    m_armsTask.track = std::clamp(track, 0, static_cast<int>(kArmsTrackCount) - 1);
    m_armsTask.key = -1;
    m_armsTask.time = 0.0f;
    const ArmsTaskPreset& preset = kTaskPresets[m_armsTask.track];
    m_armsState.main = static_cast<int>(preset.main);
    m_armsState.held = static_cast<int>(preset.held);
}

void Editor::updateArmsTask(float dt)
{
    if (!m_showArmsPoses || !m_armsTaskMode || !m_armsTask.playing)
        return;
    const float duration = armsTaskTimes().duration;
    m_armsTask.time += dt * m_armsTask.speed;
    if (m_armsTask.time <= duration)
        return;
    if (m_armsTask.loop && duration > 0.0f) {
        m_armsTask.time = std::fmod(m_armsTask.time, duration);
    }
    else {
        m_armsTask.time = duration;
        m_armsTask.playing = false;
    }
}

// ---------------------------------------------------------------------------------------------
// Posing (mirrors GameWorld::poseArms() / collectArms() for one moment)
// ---------------------------------------------------------------------------------------------

void Editor::buildArmsPreview()
{
    m_armsDrawables.clear();
    if (!m_armsAssetsLoaded || !m_showArmsPoses || !m_armsPreview || !m_armsAssets.intact(m_models))
        return;
    const ArmsPoses& p = m_armsPoses;
    const ArmsPreviewState& s = m_armsState;
    const ArmsMain main = static_cast<ArmsMain>(s.main);
    const auto model = [&](GameModelId id) -> const GameModel& { return m_armsAssets.model(id); };

    // What the moment is: the sliders, or the task at its time.
    const bool task = m_armsTaskMode;
    const ArmsTrackId track = static_cast<ArmsTrackId>(m_armsTask.track);
    const ArmsTaskTimes times = armsTaskTimes();
    const float t = std::clamp(m_armsTask.time, 0.0f, times.duration);
    const Zone zone = static_cast<Zone>(std::clamp(m_armsTask.zone, 0, 2));
    float jacket = s.jacket, pump = s.pump, slide = s.slide, kick = s.kick, tilt = s.tilt, swap = s.swap;
    if (task) {
        // The pockets are reached with the jacket open; nothing else moves but the task's parts.
        const bool pocket = track == ArmsTrackId::Restock || (usesZone(track) && zone != Zone::Bandolier &&
            (track != ArmsTrackId::PushRound || m_armsTask.fromStorage));
        jacket = pocket ? 1.0f : 0.0f;
        pump = kick = tilt = swap = 0.0f;
        slide = 0.0f;
        if (track == ArmsTrackId::Rack)
            slide = t < times.eventAt ? armsRackPull(p, times, t)
                                      : 1.0f - (t - times.eventAt) / std::max(times.duration - times.eventAt, 1e-3f);
    }

    const glm::mat4 cameraWorld = glm::inverse(getView(m_camera));
    const glm::mat4 bodyWorld = bodyWorldFrom(cameraWorld);
    const glm::mat4 bodyToCamera = glm::inverse(cameraWorld) * bodyWorld;
    const float open = jacket * jacket * (3.0f - 2.0f * jacket);

    // The body with its jacket panels opened.
    const GameModel& body = model(GameModelId::PlayerBody);
    m_armsBodyMatrices.clear();
    if (body.valid()) {
        restPose(body.data.nodes, m_armsScratchPose);
        const float angle = glm::radians(p.jacketOpenDegrees) * open;
        if (m_armsFlapL >= 0)
            m_armsScratchPose[m_armsFlapL].rotation = glm::angleAxis(angle, kUp);
        if (m_armsFlapR >= 0)
            m_armsScratchPose[m_armsFlapR].rotation = glm::angleAxis(-angle, kUp);
        poseToMatrices(body.data.nodes, m_armsScratchPose, m_armsBodyMatrices);
    }
    const auto bodyNode = [&](int node, const glm::vec3& fallback) {
        return node >= 0 && node < static_cast<int>(m_armsBodyMatrices.size()) ? m_armsBodyMatrices[node] : offset(fallback);
    };

    // The main hand's item, and the hand on it.
    NodePose mainPose;
    switch (main) {
    case ArmsMain::Shotgun: mainPose = shotgunPose(p, kick, pump, tilt, glm::vec2(0.0f), open, 1.0f); break;
    case ArmsMain::Pistol: mainPose = pistolPose(p, kick, glm::vec2(0.0f), open, 1.0f); break;
    case ArmsMain::Magazine: mainPose = p.magazineHeld.pose(); break;
    case ArmsMain::None: mainPose = p.rightHandLow.pose(); break;
    }
    mainPose.translation += p.swapLower * swap;
    mainPose.rotation = mainPose.rotation * glm::angleAxis(glm::radians(p.swapLowerDegrees) * swap, kXAxis);
    const glm::mat4 mainItem = matrixOf(mainPose);
    NodePose rightHand = mainPose;
    switch (main) {
    case ArmsMain::Shotgun: rightHand = poseOf(mainItem * p.rightOnShotgun.matrix()); break;
    case ArmsMain::Pistol: rightHand = poseOf(mainItem * p.rightOnPistol.matrix()); break;
    case ArmsMain::Magazine: rightHand = poseOf(mainItem * p.rightOnMagazine.matrix()); break;
    case ArmsMain::None: break;
    }

    ArmsAnchors anchors;
    anchors.restKey = s.main;
    switch (main) {
    case ArmsMain::Shotgun:
        anchors.rest = mainItem * offset(p.pumpRest + glm::vec3(0.0f, 0.0f, p.pumpTravel * pump) + p.palmBelowPump);
        break;
    case ArmsMain::Pistol: anchors.rest = mainItem * p.leftSupportsPistol.matrix(); break;
    case ArmsMain::Magazine: anchors.rest = mainItem * p.leftAtMagazine.matrix(); break;
    case ArmsMain::None: anchors.rest = p.handLow.matrix(); break;
    }
    anchors.mainItem = mainItem;
    anchors.mainHand = matrixOf(rightHand);
    anchors.flapL = bodyToCamera * bodyNode(m_armsFlapL, glm::vec3(-0.21f, -0.55f, -0.16f));
    anchors.flapR = bodyToCamera * bodyNode(m_armsFlapR, glm::vec3(0.21f, -0.55f, -0.16f));
    anchors.bodyToCamera = bodyToCamera;
    anchors.magSeat = m_armsRigs.pistolMagSeat;
    anchors.zone = zone;
    anchors.hipPistol = m_armsTask.gun == 0;

    // The support hand, what it holds and whether it has a gun of its own.
    NodePose leftHand;
    ArmsHeld held = static_cast<ArmsHeld>(s.held);
    enum class OwnGun { None, Pistol, Shotgun } ownGun = OwnGun::None;
    bool gunInHand = false; // the own gun is carried by the palm (on its way to the hip)
    NodePose offGun;
    bool pistolHasMagazine = true;
    bool holsterEmpty = false, slingEmpty = false;
    const auto ownGunPose = [&](bool shotgun, float rise) {
        NodePose gun = shotgun ? shotgunPose(p, 0.0f, 0.0f, 0.0f, glm::vec2(0.0f), open, -1.0f)
                               : pistolPose(p, 0.0f, glm::vec2(0.0f), open, -1.0f);
        gun.translation += p.drawRise * rise;
        return gun;
    };

    if (task) {
        // Replays the task from its start, the way poseArms() moves the hand from key to key.
        NodePose pose = poseOf(anchors.rest), from = pose;
        int key = -1;
        float moveTime = 0.0f, moveDuration = 0.06f;
        for (float step = 0.0f;; step = std::min(step + kTaskStep, t)) {
            if (track == ArmsTrackId::Rack)
                anchors.slide = armsRackPull(p, times, step);
            int stepKey = 0;
            float phaseLeft = 0.1f;
            const NodePose target = armsTrackTarget(p, track, times, step, anchors, stepKey, phaseLeft);
            if (stepKey != key) {
                from = pose; // the task starts with the hand at rest
                key = stepKey;
                moveTime = 0.0f;
                moveDuration = std::max(phaseLeft, 0.06f);
            }
            moveTime += kTaskStep;
            pose = easePose(from, target, moveTime / moveDuration);
            if (step >= t)
                break;
        }
        leftHand = pose;

        // What is in the hand before and after the event, as Arms does it.
        const bool afterEvent = t >= times.eventAt;
        const ArmsHeld chosen = held;
        switch (track) {
        case ArmsTrackId::Fetch:
        case ArmsTrackId::PickUp: held = afterEvent ? chosen : ArmsHeld::None; break;
        case ArmsTrackId::Store: held = afterEvent ? ArmsHeld::None : chosen; break;
        case ArmsTrackId::InsertShell: held = afterEvent ? ArmsHeld::None : ArmsHeld::Shell; break;
        case ArmsTrackId::PushRound:
            held = t >= times.base(ArmsTimeBase::Event2) || (times.zoneUsed && !afterEvent) ? ArmsHeld::None : ArmsHeld::Round;
            break;
        case ArmsTrackId::MagInsert:
            held = afterEvent ? ArmsHeld::None : ArmsHeld::Magazine;
            pistolHasMagazine = afterEvent;
            break;
        case ArmsTrackId::Restock:
        case ArmsTrackId::Rack: held = ArmsHeld::None; break;
        case ArmsTrackId::Swap: held = ArmsHeld::Magazine; break;
        case ArmsTrackId::DrawOffGun:
        case ArmsTrackId::StowOffGun: {
            held = ArmsHeld::None;
            const bool shotgun = m_armsTask.gun == 1;
            const float half = times.duration * 0.5f;
            if (track == ArmsTrackId::DrawOffGun && t >= half) {
                // Up from the hip: poseArms() puts the gun at its place and the hand on its grip.
                offGun = ownGunPose(shotgun, 1.0f - std::clamp((t - half) / std::max(half, 1e-3f), 0.0f, 1.0f));
                leftHand = poseOf(matrixOf(offGun) * (shotgun ? p.leftOnShotgunGrip : p.rightOnPistol).matrix());
                ownGun = shotgun ? OwnGun::Shotgun : OwnGun::Pistol;
            }
            else if (track == ArmsTrackId::StowOffGun) {
                ownGun = shotgun ? OwnGun::Shotgun : OwnGun::Pistol;
                gunInHand = true;
            }
            (shotgun ? slingEmpty : holsterEmpty) = ownGun != OwnGun::None;
            break;
        }
        default: break;
        }
    }
    else {
        const ArmsOff off = static_cast<ArmsOff>(s.off);
        switch (off) {
        case ArmsOff::Rest: leftHand = poseOf(anchors.rest); break;
        case ArmsOff::Ready: leftHand = p.handReady.pose(); break;
        case ArmsOff::Bandolier: leftHand = poseOf(anchors.flapL * p.handBandolier.matrix()); break;
        case ArmsOff::Pocket: leftHand = poseOf(anchors.flapL * p.handPocket.matrix()); break;
        case ArmsOff::RightPocket: leftHand = poseOf(anchors.flapR * p.handRightPocket.matrix()); break;
        case ArmsOff::Ground: leftHand = poseOf(bodyToCamera * p.handGround.matrix()); break;
        case ArmsOff::Holster: leftHand = poseOf(bodyToCamera * p.handHolster.matrix()); break;
        case ArmsOff::Sling: leftHand = poseOf(bodyToCamera * p.handSling.matrix()); break;
        case ArmsOff::OwnPistol:
        case ArmsOff::OwnShotgun: {
            const bool shotgun = off == ArmsOff::OwnShotgun;
            offGun = ownGunPose(shotgun, s.rise);
            leftHand = poseOf(matrixOf(offGun) * (shotgun ? p.leftOnShotgunGrip : p.rightOnPistol).matrix());
            ownGun = shotgun ? OwnGun::Shotgun : OwnGun::Pistol;
            (shotgun ? slingEmpty : holsterEmpty) = true;
            held = ArmsHeld::None;
            break;
        }
        }
    }

    // Drawing, as GameWorld::collectArms().
    std::vector<DynamicInstance>& out = m_armsDrawables;
    const auto emit = [&](GameModelId id, const glm::mat4& world) {
        const GameModel& m = model(id);
        poseToMatrices(m.data.nodes, m_armsScratchPose, m_armsScratchMatrices);
        emitModelNodes(m, m_armsScratchMatrices, world, glm::vec3(1.0f), false, out);
    };
    const auto drawMagazine = [&](const glm::mat4& world) {
        if (const GameModel& m = model(GameModelId::Magazine); m.valid()) {
            poseMagazine(m.data, m_armsRigs, 8, m_armsScratchPose);
            emit(GameModelId::Magazine, world);
        }
    };
    const auto drawShotgun = [&](const glm::mat4& world, float pumpAmount, float flash) {
        if (const GameModel& m = model(GameModelId::Shotgun); m.valid()) {
            poseShotgun(m.data, m_armsRigs, p, pumpAmount, flash, 0.0f, m_armsScratchPose);
            emit(GameModelId::Shotgun, world);
        }
    };
    const auto drawPistol = [&](const glm::mat4& world, float slideAmount, float flash, bool withMagazine) {
        if (const GameModel& m = model(GameModelId::Pistol); m.valid()) {
            posePistol(m.data, m_armsRigs, p, slideAmount, flash, 0.0f, m_armsScratchPose);
            emit(GameModelId::Pistol, world);
            if (withMagazine)
                drawMagazine(world * m_armsRigs.pistolMagSeat);
        }
    };
    const auto drawItem = [&](GameModelId id, const glm::mat4& world) {
        if (const GameModel& m = model(id); m.valid()) {
            restPose(m.data.nodes, m_armsScratchPose);
            emit(id, world);
        }
    };

    if (s.body && body.valid()) {
        emitModelNodes(body, m_armsBodyMatrices, bodyWorld, glm::vec3(1.0f), false, out);
        if (main != ArmsMain::Pistol && !holsterEmpty && m_armsHolster >= 0)
            drawPistol(bodyWorld * bodyNode(m_armsHolster, glm::vec3(0.0f)), 0.0f, 0.0f, true);
        if (main != ArmsMain::Shotgun && !slingEmpty)
            drawShotgun(bodyWorld * p.shotgunSlung.matrix(), 0.0f, 0.0f);
    }
    if (const GameModel& hands = model(GameModelId::Hands); hands.valid()) {
        poseHands(hands.data, m_armsRigs, rightHand, leftHand, m_armsScratchPose);
        emit(GameModelId::Hands, cameraWorld);
    }
    const glm::mat4 mainWorld = cameraWorld * mainItem;
    const float flash = task ? 0.0f : s.flash;
    switch (main) {
    case ArmsMain::Shotgun: drawShotgun(mainWorld, pump, flash); break;
    case ArmsMain::Pistol: drawPistol(mainWorld, std::clamp(slide, 0.0f, 1.0f), flash, pistolHasMagazine); break;
    case ArmsMain::Magazine: drawMagazine(mainWorld); break;
    case ArmsMain::None: break;
    }
    const glm::mat4 palm = cameraWorld * matrixOf(leftHand);
    if (ownGun != OwnGun::None) {
        const glm::mat4 gunWorld = gunInHand ? palm : cameraWorld * matrixOf(offGun);
        if (ownGun == OwnGun::Pistol)
            drawPistol(gunWorld, 0.0f, flash, true);
        else
            drawShotgun(gunWorld, 0.0f, flash);
    }
    switch (held) {
    case ArmsHeld::Shell: drawItem(GameModelId::ShellItem, palm * p.shellInHand.matrix()); break;
    case ArmsHeld::Round: drawItem(GameModelId::RoundItem, palm * p.roundInHand.matrix()); break;
    case ArmsHeld::Magazine: drawMagazine(palm * p.magazineInHand.matrix()); break;
    case ArmsHeld::None: break;
    }
}

// ---------------------------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------------------------

void Editor::applyArmsPreset(const char* poseName)
{
    for (const ArmsPreset& preset : kPresets) {
        if (std::strcmp(preset.pose, poseName) != 0)
            continue;
        m_armsState.main = static_cast<int>(preset.main);
        m_armsState.off = static_cast<int>(preset.off);
        m_armsState.held = static_cast<int>(preset.held);
        m_armsState.jacket = preset.jacket;
        return;
    }
}

void Editor::drawArmsPosesPanel()
{
    if (!m_armsPosesRead) {
        m_armsPoses = m_armsPosesSaved = loadArmsPoses();
        m_armsConfig = loadWeaponTuning().arms;
        m_armsPosesRead = true;
    }
    m_armsTaskMode = false;
    if (!ImGui::Begin("Arms Poses", &m_showArmsPoses)) {
        ImGui::End();
        return;
    }

    const bool dirty = !(m_armsPoses == m_armsPosesSaved);
    ImGui::BeginDisabled(!dirty);
    if (ImGui::Button("Save")) {
        if (saveArmsPoses(m_armsPoses)) {
            m_armsPosesSaved = m_armsPoses;
            setStatus("Saved " + configPath("arms_poses.json"));
        }
        else {
            setStatus("Can't write " + configPath("arms_poses.json"), true);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Revert"))
        m_armsPoses = m_armsPosesSaved;
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Reload")) {
        // Edits made to the files by hand.
        m_armsPoses = m_armsPosesSaved = loadArmsPoses();
        m_armsConfig = loadWeaponTuning().arms;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Show in view", &m_armsPreview);
    if (dirty) {
        ImGui::SameLine();
        ImGui::TextDisabled("(unsaved)");
    }
    ImGui::TextDisabled("arms_poses.json; the game reads it at level start and Restart Level.");

    if (ImGui::CollapsingHeader("Preview", ImGuiTreeNodeFlags_DefaultOpen))
        drawArmsPreviewControls();

    if (ImGui::BeginTabBar("ArmsPosesTabs")) {
        if (ImGui::BeginTabItem("Poses")) {
            drawArmsPoseList();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Tasks")) {
            m_armsTaskMode = true;
            drawArmsTasks();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Motion")) {
            drawArmsMotion();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void Editor::drawArmsPreviewControls()
{
    ArmsPreviewState& s = m_armsState;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    combo("Main hand", s.main, kMainNames);
    // Last frame's tab: the Tasks tab is drawn after this.
    if (m_armsTaskMode) {
        combo("Holding", s.held, kHeldNames);
        ImGui::Checkbox("Body", &s.body);
        ImGui::PopItemWidth();
        ImGui::TextDisabled("The task drives the support hand.");
        return;
    }
    combo("Support hand", s.off, kOffNames);
    combo("Holding", s.held, kHeldNames);
    ImGui::SliderFloat("Jacket open / lower", &s.jacket, 0.0f, 1.0f);
    ImGui::SliderFloat("Kick", &s.kick, 0.0f, 1.0f);
    ImGui::SliderFloat("Flash", &s.flash, 0.0f, 1.0f);
    const ArmsMain main = static_cast<ArmsMain>(s.main);
    const ArmsOff off = static_cast<ArmsOff>(s.off);
    if (main == ArmsMain::Shotgun) {
        ImGui::SliderFloat("Pump", &s.pump, 0.0f, 1.0f);
        ImGui::SliderFloat("Tilt", &s.tilt, 0.0f, 1.0f);
    }
    if (main == ArmsMain::Pistol)
        ImGui::SliderFloat("Slide", &s.slide, 0.0f, 1.0f);
    if (off == ArmsOff::OwnPistol || off == ArmsOff::OwnShotgun)
        ImGui::SliderFloat("Drawing (from hip)", &s.rise, 0.0f, 1.0f);
    ImGui::SliderFloat("Swapping (lowered)", &s.swap, 0.0f, 1.0f);
    ImGui::Checkbox("Body", &s.body);
    ImGui::SameLine();
    if (ImGui::Button("Reset sliders")) {
        s.kick = s.pump = s.slide = s.tilt = s.swap = s.rise = s.flash = 0.0f;
        s.jacket = ArmsPreviewState{}.jacket;
    }
    ImGui::PopItemWidth();
    ImGui::TextDisabled("Look down (fly the camera) to see the body and the jacket.");
}

void Editor::drawArmsPoseList()
{
    const auto fields = armsPoseFields();
    m_armsSelected = std::clamp(m_armsSelected, 0, static_cast<int>(fields.size()) - 1);
    const ArmsPoses defaults;

    if (ImGui::BeginChild("ArmsPoseList", ImVec2(ImGui::GetContentRegionAvail().x * 0.4f, 0.0f), ImGuiChildFlags_Borders)) {
        const char* group = nullptr;
        for (int i = 0; i < static_cast<int>(fields.size()); ++i) {
            const ArmsPoseField& field = fields[i];
            if (!group || std::strcmp(group, field.group) != 0) {
                group = field.group;
                ImGui::SeparatorText(group);
            }
            const bool changed = !(m_armsPoses.*field.member == m_armsPosesSaved.*field.member);
            const std::string label = std::string(field.name) + (changed ? " *" : "");
            if (ImGui::Selectable(label.c_str(), m_armsSelected == i)) {
                m_armsSelected = i;
                applyArmsPreset(field.name);
            }
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();

    if (ImGui::BeginChild("ArmsPoseDetails", ImVec2(0.0f, 0.0f))) {
        const ArmsPoseField& field = fields[m_armsSelected];
        PoseEntry& pose = m_armsPoses.*field.member;
        ImGui::TextUnformatted(field.name);
        ImGui::TextDisabled("%s space", field.group);
        ImGui::DragFloat3("Position (m)", &pose.translation.x, 0.001f, -5.0f, 5.0f, "%.3f");
        ImGui::DragFloat3("Pitch Yaw Roll", &pose.degrees.x, 0.25f, -360.0f, 360.0f, "%.1f");
        if (ImGui::Button("Show"))
            applyArmsPreset(field.name);
        ImGui::SameLine();
        ImGui::BeginDisabled(pose == m_armsPosesSaved.*field.member);
        if (ImGui::Button("Revert pose"))
            pose = m_armsPosesSaved.*field.member;
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(pose == defaults.*field.member);
        if (ImGui::Button("Default"))
            pose = defaults.*field.member;
        ImGui::EndDisabled();
    }
    ImGui::EndChild();
}

void Editor::drawArmsTimeline(const ArmsTaskTimes& times)
{
    const std::vector<ArmsKey>& keys = m_armsPoses.tracks[m_armsTask.track].keys;
    const float width = std::max(ImGui::GetContentRegionAvail().x, 100.0f);
    const float height = 34.0f * ImGui::GetIO().FontGlobalScale;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("ArmsTimeline", ImVec2(width, height));
    const float duration = std::max(times.duration, 1e-3f);
    const auto xOf = [&](float time) { return origin.x + 6.0f + (width - 12.0f) * std::clamp(time / duration, 0.0f, 1.0f); };
    if (ImGui::IsItemActive()) {
        // Scrubbing.
        const float x = (ImGui::GetIO().MousePos.x - origin.x - 6.0f) / (width - 12.0f);
        m_armsTask.time = std::clamp(x, 0.0f, 1.0f) * times.duration;
        m_armsTask.playing = false;
    }

    ImDrawList* draw = ImGui::GetWindowDrawList();
    const float midY = origin.y + height * 0.5f;
    draw->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height), ImGui::GetColorU32(ImGuiCol_FrameBg), 4.0f);
    draw->AddLine(ImVec2(xOf(0.0f), midY), ImVec2(xOf(duration), midY), ImGui::GetColorU32(ImGuiCol_Border), 2.0f);
    const auto marker = [&](float time, ImU32 color) {
        draw->AddLine(ImVec2(xOf(time), origin.y + 3.0f), ImVec2(xOf(time), origin.y + height - 3.0f), color, 2.0f);
    };
    marker(times.eventAt, IM_COL32(240, 150, 40, 255));
    if (times.event2At >= 0.0f)
        marker(times.event2At, IM_COL32(240, 220, 60, 255));
    for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
        const ArmsKey& key = keys[i];
        const bool active = !key.needsZone || times.zoneUsed;
        const float x = xOf(key.at * times.base(key.base));
        const float r = i == m_armsTask.key ? 7.0f : 5.0f;
        const ImU32 color = !active ? IM_COL32(120, 120, 120, 160)
            : i == m_armsTask.key ? IM_COL32(110, 200, 255, 255) : IM_COL32(220, 220, 230, 255);
        draw->AddQuadFilled(ImVec2(x, midY - r), ImVec2(x + r, midY), ImVec2(x, midY + r), ImVec2(x - r, midY), color);
    }
    const float playX = xOf(m_armsTask.time);
    draw->AddLine(ImVec2(playX, origin.y), ImVec2(playX, origin.y + height), IM_COL32(255, 80, 80, 255), 2.0f);
    ImGui::TextDisabled("Orange: the event, yellow: the second event; diamonds: keys (grey: needs a zone).");
}

void Editor::drawArmsTasks()
{
    ArmsTaskPreview& task = m_armsTask;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    if (ImGui::BeginCombo("Task", armsTrackName(static_cast<ArmsTrackId>(task.track)))) {
        for (int i = 0; i < static_cast<int>(kArmsTrackCount); ++i) {
            const bool changed = !(m_armsPoses.tracks[i] == m_armsPosesSaved.tracks[i]);
            const std::string label = std::string(armsTrackName(static_cast<ArmsTrackId>(i))) + (changed ? " *" : "");
            if (ImGui::Selectable(label.c_str(), task.track == i))
                selectArmsTask(i);
        }
        ImGui::EndCombo();
    }
    const ArmsTrackId track = static_cast<ArmsTrackId>(task.track);
    if (usesZone(track))
        combo("Zone", task.zone, kZoneNames);
    if (track == ArmsTrackId::PushRound)
        ImGui::Checkbox("Round from storage", &task.fromStorage);
    if (track == ArmsTrackId::DrawOffGun || track == ArmsTrackId::StowOffGun)
        combo("Gun", task.gun, kGunNames);
    ImGui::PopItemWidth();

    const ArmsTaskTimes times = armsTaskTimes();
    task.time = std::clamp(task.time, 0.0f, times.duration);
    if (ImGui::Button(task.playing ? "Pause" : "Play")) {
        if (!task.playing && task.time >= times.duration)
            task.time = 0.0f;
        task.playing = !task.playing;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Loop", &task.loop);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    ImGui::SliderFloat("Speed", &task.speed, 0.1f, 2.0f, "%.2fx");
    ImGui::SameLine();
    if (times.event2At >= 0.0f)
        ImGui::TextDisabled("%.2f / %.2f s (events %.2f, %.2f s)", task.time, times.duration, times.eventAt, times.event2At);
    else
        ImGui::TextDisabled("%.2f / %.2f s (event %.2f s)", task.time, times.duration, times.eventAt);
    drawArmsTimeline(times);

    std::vector<ArmsKey>& keys = m_armsPoses.tracks[task.track].keys;
    task.key = std::clamp(task.key, -1, static_cast<int>(keys.size()) - 1);
    const auto keyTime = [&](const ArmsKey& key) { return key.at * times.base(key.base); };
    if (ImGui::BeginTable("ArmsKeys", 5, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 28.0f);
        ImGui::TableSetupColumn("At");
        ImGui::TableSetupColumn("Of");
        ImGui::TableSetupColumn("Anchor");
        ImGui::TableSetupColumn("Zone only", ImGuiTableColumnFlags_WidthFixed, 64.0f);
        ImGui::TableHeadersRow();
        for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
            ArmsKey& key = keys[i];
            ImGui::PushID(i);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::Selectable(std::to_string(i + 1).c_str(), task.key == i, ImGuiSelectableFlags_SpanAllColumns |
                    ImGuiSelectableFlags_AllowOverlap)) {
                task.key = i;
                task.time = std::clamp(keyTime(key), 0.0f, times.duration);
                task.playing = false;
            }
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::DragFloat("##at", &key.at, 0.005f, 0.0f, 1.0f, "%.3f");
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            enumCombo("##of", key.base, armsTimeBaseName);
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-FLT_MIN);
            enumCombo("##anchor", key.anchor, armsAnchorName);
            ImGui::TableNextColumn();
            ImGui::Checkbox("##zone", &key.needsZone);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    if (ImGui::Button("Add key")) {
        // A copy of the selected key (or the last), right after it.
        const int at = task.key >= 0 ? task.key + 1 : static_cast<int>(keys.size());
        const ArmsKey copy = keys.empty() ? ArmsKey{} : keys[std::max(at - 1, 0)];
        keys.insert(keys.begin() + at, copy);
        task.key = at;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(task.key < 0);
    if (ImGui::Button("Delete key")) {
        keys.erase(keys.begin() + task.key);
        task.key = std::min(task.key, static_cast<int>(keys.size()) - 1);
    }
    ImGui::SameLine();
    if (ImGui::ArrowButton("##up", ImGuiDir_Up) && task.key > 0) {
        std::swap(keys[task.key], keys[task.key - 1]);
        --task.key;
    }
    ImGui::SameLine();
    if (ImGui::ArrowButton("##down", ImGuiDir_Down) && task.key >= 0 && task.key + 1 < static_cast<int>(keys.size())) {
        std::swap(keys[task.key], keys[task.key + 1]);
        ++task.key;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    const ArmsTrack defaultTrack = defaultArmsTracks()[task.track];
    ImGui::BeginDisabled(m_armsPoses.tracks[task.track] == defaultTrack);
    if (ImGui::Button("Default track")) {
        m_armsPoses.tracks[task.track] = defaultTrack;
        task.key = -1;
    }
    ImGui::EndDisabled();

    if (task.key >= 0) {
        ArmsKey& key = keys[task.key];
        ImGui::SeparatorText(("Key " + std::to_string(task.key + 1) + ": offset from " + armsAnchorName(key.anchor)).c_str());
        ImGui::DragFloat3("Position (m)", &key.offset.translation.x, 0.001f, -5.0f, 5.0f, "%.3f");
        ImGui::DragFloat3("Pitch Yaw Roll", &key.offset.degrees.x, 0.25f, -360.0f, 360.0f, "%.1f");
        if (ImGui::Button("Zero offset"))
            key.offset = {};
        ImGui::SameLine();
        if (ImGui::Button("Go to key")) {
            task.time = std::clamp(keyTime(key), 0.0f, times.duration);
            task.playing = false;
        }
    }
    ImGui::TextDisabled("The hand reaches each key's anchor x offset at its time, keys in order.");
}

void Editor::drawArmsMotion()
{
    const ArmsPoses defaults;
    // Offsets and numbers interleaved by group, in the order the file lists them.
    const auto vectors = armsVectorFields();
    const auto numbers = armsNumberFields();
    std::vector<const char*> groups;
    const auto addGroup = [&](const char* group) {
        for (const char* g : groups)
            if (std::strcmp(g, group) == 0)
                return;
        groups.push_back(group);
    };
    for (const ArmsVectorField& field : vectors)
        addGroup(field.group);
    for (const ArmsNumberField& field : numbers)
        addGroup(field.group);

    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.55f);
    for (const char* group : groups) {
        ImGui::SeparatorText(group);
        for (const ArmsVectorField& field : vectors) {
            if (std::strcmp(field.group, group) != 0)
                continue;
            glm::vec3& value = m_armsPoses.*field.member;
            ImGui::DragFloat3(field.name, &value.x, 0.001f, -5.0f, 5.0f, "%.3f");
            if (ImGui::BeginPopupContextItem(field.name)) {
                if (ImGui::MenuItem("Default", nullptr, false, value != defaults.*field.member))
                    value = defaults.*field.member;
                ImGui::EndPopup();
            }
        }
        for (const ArmsNumberField& field : numbers) {
            if (std::strcmp(field.group, group) != 0)
                continue;
            float& value = m_armsPoses.*field.member;
            const bool degrees = std::strstr(field.name, "Degrees") != nullptr;
            ImGui::DragFloat(field.name, &value, degrees ? 0.25f : 0.001f, -360.0f, 360.0f, degrees ? "%.1f" : "%.3f");
            if (ImGui::BeginPopupContextItem(field.name)) {
                if (ImGui::MenuItem("Default", nullptr, false, value != defaults.*field.member))
                    value = defaults.*field.member;
                ImGui::EndPopup();
            }
        }
    }
    ImGui::PopItemWidth();
    ImGui::TextDisabled("Right-click a value to reset it to its default.");
}
