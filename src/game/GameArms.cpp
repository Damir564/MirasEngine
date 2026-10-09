// The player's hands and guns as seen and heard: GameWorld's side of Arms (Arms.h holds the rules and the
// state). Input becomes Arms requests; Arms events become sounds, shots, messages and dropped objects;
// the hands and items are posed in camera space from the Arms tasks' progress and drawn with the camera.
#include "GameWorld.h"
#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "engine/Log.h"
#include "engine/ModelManager.h"
#include "engine/Physics.h"
#include "engine/Renderer.h"

namespace {

constexpr float kPi = 3.14159265f;
constexpr glm::vec3 kUp(0.0f, 1.0f, 0.0f);
constexpr glm::vec3 kXAxis(1.0f, 0.0f, 0.0f);
constexpr glm::vec3 kZAxis(0.0f, 0.0f, 1.0f);

// Shots
constexpr int kPellets = 9;
constexpr float kSpreadDegrees = 5.0f;
constexpr float kShotRange = 70.0f;
constexpr float kPelletDamageNear = 13.0f;
constexpr float kPelletDamageFar = 4.0f;
constexpr float kPistolSpreadDegrees = 0.6f;
constexpr float kPistolDamageNear = 35.0f;
constexpr float kPistolDamageFar = 22.0f;
constexpr float kGunshotHearing = 24.0f;
constexpr float kPickUpReach = 2.2f; // meters from the feet

// Where things are held, camera space (+X right, +Y up, looking along -Z). The guns are held low and to
// the side; their barrels cross the middle of the view this far away, and the shots follow the barrels.
constexpr float kConvergence = 9.0f;
constexpr glm::vec3 kGunRest(0.2f, -0.22f, -0.34f);
constexpr glm::vec3 kPistolRest(0.15f, -0.19f, -0.38f);
constexpr glm::vec3 kPumpRest(0.0f, -0.022f, -0.3f);
constexpr glm::vec3 kPalmBelowPump(0.0f, -0.04f, 0.0f);
constexpr float kPumpTravel = 0.11f;
constexpr float kSlideTravel = 0.05f;
constexpr float kSlideLocked = 0.7f; // of kSlideTravel
constexpr float kJacketOpenDegrees = 72.0f; // opened this far, the panels' insides face the eyes
const glm::vec3 kFlapLRest(-0.21f, -0.55f, -0.16f); // where the panels hang if the body model is missing
const glm::vec3 kFlapRRest(0.21f, -0.55f, -0.16f);

glm::vec3 flat(const glm::vec3& v)
{
    return glm::vec3(v.x, 0.0f, v.z);
}

NodePose makePose(const glm::vec3& translation, float pitchDegrees, float yawDegrees, float rollDegrees)
{
    NodePose pose;
    pose.translation = translation;
    pose.rotation = glm::angleAxis(glm::radians(yawDegrees), kUp) * glm::angleAxis(glm::radians(pitchDegrees), kXAxis) *
        glm::angleAxis(glm::radians(rollDegrees), kZAxis);
    return pose;
}

NodePose poseOf(const glm::mat4& matrix)
{
    NodePose pose;
    pose.translation = glm::vec3(matrix[3]);
    pose.rotation = glm::quat_cast(glm::mat3(matrix));
    return pose;
}

glm::mat4 matrixOf(const NodePose& pose)
{
    return glm::translate(glm::mat4(1.0f), pose.translation) * glm::mat4_cast(pose.rotation) *
        glm::scale(glm::mat4(1.0f), pose.scale);
}

glm::mat4 offset(const glm::vec3& translation, float pitchDegrees = 0.0f, float yawDegrees = 0.0f, float rollDegrees = 0.0f)
{
    return matrixOf(makePose(translation, pitchDegrees, yawDegrees, rollDegrees));
}

// Eased move from a to b, t in 0..1.
NodePose blend(const NodePose& a, const NodePose& b, float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    t = t * t * (3.0f - 2.0f * t);
    NodePose pose;
    pose.translation = glm::mix(a.translation, b.translation, t);
    pose.rotation = glm::slerp(a.rotation, b.rotation, t);
    return pose;
}

// Support hand poses: ready below the guns holding something (camera space); at the bandolier and in the
// pouch (the left panel's space, player_body.glb `flap_l`), in the right pocket (`flap_r`), so the hand
// follows the panels when the jacket opens.
const NodePose kHandReady = makePose(glm::vec3(0.06f, -0.27f, -0.4f), 25.0f, 15.0f, -10.0f);
const NodePose kHandLow = makePose(glm::vec3(-0.15f, -0.42f, -0.3f), 10.0f, 0.0f, 0.0f);
const NodePose kHandBandolier = makePose(glm::vec3(0.11f, 0.06f, -0.09f), -35.0f, 0.0f, 25.0f);
const NodePose kHandPocket = makePose(glm::vec3(0.1f, 0.17f, 0.03f), -60.0f, 0.0f, 10.0f);
const NodePose kHandRightPocket = makePose(glm::vec3(-0.11f, 0.12f, 0.03f), -60.0f, 0.0f, -10.0f);
// Body space (eyes, facing -Z): reaching down to the floor, the hips (holster / sling).
const NodePose kHandGround = makePose(glm::vec3(0.0f, -1.45f, -0.5f), -80.0f, 0.0f, 0.0f);
const NodePose kHandHolster = makePose(glm::vec3(0.22f, -0.8f, -0.05f), -70.0f, 0.0f, 0.0f);
const NodePose kHandSling = makePose(glm::vec3(-0.25f, -0.45f, 0.1f), -30.0f, 0.0f, 0.0f);
// The held magazine: its slotted side (+X) turned towards the eyes so the rounds show.
const NodePose kMagazineHeld = makePose(glm::vec3(0.07f, -0.2f, -0.34f), 10.0f, -70.0f, 0.0f);

// Small items in the support hand, relative to the palm.
const glm::mat4 kShellInHand = offset(glm::vec3(0.0f, 0.022f, -0.01f));
const glm::mat4 kRoundInHand = offset(glm::vec3(0.0f, 0.02f, -0.01f));
const glm::mat4 kMagazineInHand = offset(glm::vec3(0.0f, 0.115f, -0.005f));
// Hands on the guns and the magazine, relative to the item.
const glm::mat4 kRightOnShotgun = offset(glm::vec3(0.0f, -0.085f, 0.065f), -18.0f);
const glm::mat4 kRightOnPistol = offset(glm::vec3(0.0f), -15.0f);
const glm::mat4 kRightOnMagazine = offset(glm::vec3(0.0f, -0.105f, 0.0f));
const glm::mat4 kLeftSupportsPistol = offset(glm::vec3(-0.026f, -0.03f, 0.01f), -15.0f, 0.0f, 70.0f);
const glm::mat4 kLeftAtMagazine = offset(glm::vec3(-0.03f, -0.02f, 0.0f));
const glm::mat4 kLeftOnShotgunGrip = offset(glm::vec3(0.0f, -0.085f, 0.065f), -18.0f);

// A gun in a hand: aimed to cross the view's center at kConvergence, kicked up by a shot (kick 0..1),
// lagging behind turns (sway), lowered out of the way while the jacket is open (lower 0..1). side = 1 for
// the right hand, -1 mirrors it to the left.
NodePose shotgunPose(float kick, float pumpAmount, float tilt, const glm::vec2& sway, float lower, float side)
{
    const glm::vec3 rest(kGunRest.x * side, kGunRest.y, kGunRest.z);
    const float yaw = side * std::atan(kGunRest.x / kConvergence) + sway.x;
    const float pitch = std::atan((-kGunRest.y - 0.022f) / kConvergence) + sway.y + glm::radians(12.0f) * kick -
        glm::radians(30.0f) * lower;
    const float roll = side * (glm::radians(-4.0f) * pumpAmount + glm::radians(28.0f) * tilt);
    NodePose pose;
    pose.translation = rest + glm::vec3(0.0f, 0.02f, 0.07f) * kick + glm::vec3(-0.03f * side, -0.03f, 0.04f) * tilt +
        glm::vec3(0.0f, -0.006f, 0.01f) * pumpAmount + glm::vec3(0.08f * side, -0.1f, 0.08f) * lower;
    pose.rotation = glm::angleAxis(yaw, kUp) * glm::angleAxis(pitch, kXAxis) * glm::angleAxis(roll, kZAxis);
    return pose;
}

NodePose pistolPose(float kick, const glm::vec2& sway, float lower, float side)
{
    const glm::vec3 rest(kPistolRest.x * side, kPistolRest.y, kPistolRest.z);
    const float yaw = side * std::atan(kPistolRest.x / kConvergence) + sway.x;
    const float pitch = std::atan((-kPistolRest.y - 0.07f) / kConvergence) + sway.y + glm::radians(10.0f) * kick -
        glm::radians(30.0f) * lower;
    NodePose pose;
    pose.translation = rest + glm::vec3(0.0f, 0.012f, 0.04f) * kick + glm::vec3(0.06f * side, -0.1f, 0.08f) * lower;
    pose.rotation = glm::angleAxis(yaw, kUp) * glm::angleAxis(pitch, kXAxis);
    return pose;
}

float kickCurve(float shotTime)
{
    return shotTime < 0.04f ? shotTime / 0.04f : std::exp(-(shotTime - 0.04f) / 0.1f);
}

glm::mat4 cameraWorldFrom(const glm::vec3& eye, const glm::vec3& forward)
{
    const glm::vec3 up = std::abs(glm::normalize(forward).y) > 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : kUp;
    return glm::inverse(glm::lookAt(eye, eye + forward, up));
}

// The body's space: at the eyes, turned with the view's yaw only (facing -Z like the camera).
glm::mat4 bodyWorldFrom(const glm::mat4& cameraWorld)
{
    glm::vec3 forward = -glm::vec3(cameraWorld[2]);
    if (glm::length(flat(forward)) < 0.1f)
        forward = glm::vec3(cameraWorld[1]); // looking straight down: the view's up points ahead
    forward = glm::normalize(flat(forward));
    return glm::rotate(glm::translate(glm::mat4(1.0f), glm::vec3(cameraWorld[3])), std::atan2(-forward.x, -forward.z), kUp);
}

// How far each kind sits above the floor lying down, and the axis it lies along.
float restHeight(ItemKind item)
{
    switch (item) {
    case ItemKind::Shell: return 0.011f;
    case ItemKind::Round: return 0.005f;
    case ItemKind::Magazine: return 0.012f;
    case ItemKind::Pistol: return 0.014f;
    case ItemKind::Shotgun: return 0.03f;
    default: return 0.01f;
    }
}

const char* dropSound(ItemKind item)
{
    switch (item) {
    case ItemKind::Magazine: return "mag_drop";
    case ItemKind::Pistol:
    case ItemKind::Shotgun: return "gun_drop";
    default: return "shell_drop";
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------------------------

void GameWorld::resetArms()
{
    // Read on every (re)start, so edits to the file show after "Restart Level".
    m_tuning = loadWeaponTuning();
    m_arms.setConfig(m_tuning.arms);
    m_arms.reset();
    m_view = {};

    m_rigs = {};
    m_rigs.magazineRounds.fill(-1);
    const auto nodeTranslation = [](const GameModel& model, int node, glm::vec3 fallback) {
        return node >= 0 ? model.data.nodes[node].translation : fallback;
    };
    if (const GameModel& shotgun = m_assets.model(GameModelId::Shotgun); shotgun.valid()) {
        m_rigs.shotgunGun = shotgun.data.findNode("gun");
        m_rigs.shotgunPump = shotgun.data.findNode("pump");
        m_rigs.shotgunFlash = shotgun.data.findNode("flash");
        m_rigs.shotgunMuzzle = nodeTranslation(shotgun, shotgun.data.findNode("muzzle"), m_rigs.shotgunMuzzle);
    }
    if (const GameModel& pistol = m_assets.model(GameModelId::Pistol); pistol.valid()) {
        m_rigs.pistolSlide = pistol.data.findNode("slide");
        m_rigs.pistolFlash = pistol.data.findNode("flash");
        m_rigs.pistolMuzzle = nodeTranslation(pistol, pistol.data.findNode("muzzle"), m_rigs.pistolMuzzle);
        if (const int seat = pistol.data.findNode("magseat"); seat >= 0) {
            NodePose pose;
            pose.translation = pistol.data.nodes[seat].translation;
            pose.rotation = pistol.data.nodes[seat].rotation;
            m_rigs.pistolMagSeat = matrixOf(pose);
        }
    }
    if (const GameModel& magazine = m_assets.model(GameModelId::Magazine); magazine.valid()) {
        m_rigs.magazineFollower = magazine.data.findNode("follower");
        for (size_t i = 0; i < m_rigs.magazineRounds.size(); ++i)
            m_rigs.magazineRounds[i] = magazine.data.findNode("round" + std::to_string(i));
    }
    if (const GameModel& hands = m_assets.model(GameModelId::Hands); hands.valid()) {
        m_rigs.handR = hands.data.findNode("hand_r");
        m_rigs.handL = hands.data.findNode("hand_l");
    }

    m_body = {};
    m_body.band.fill(-1);
    m_body.pocket.fill(-1);
    m_body.boxRounds.fill(-1);
    m_jacket = 0.0f;
    if (const GameModel& body = m_assets.model(GameModelId::PlayerBody); body.valid()) {
        const auto find = [&](const std::string& name) { return body.data.findNode(name); };
        m_body.flapL = find("flap_l");
        m_body.flapR = find("flap_r");
        m_body.legL = find("leg_l");
        m_body.legR = find("leg_r");
        m_body.holsterGun = find("holster_gun");
        for (int i = 0; i < kBandolierLoops; ++i)
            m_body.band[i] = find("band" + std::to_string(i));
        for (int i = 0; i < kPocketSlots; ++i)
            m_body.pocket[i] = find("pocket" + std::to_string(i));
        for (int i = 0; i < kBoxRounds; ++i)
            m_body.boxRounds[i] = find("rround" + std::to_string(i));
        m_body.bandMags = { find("bandmag0"), find("bandmag1") };
        m_body.pocketMags = { find("pocketmag0"), find("pocketmag1") };
        m_body.rightPocketMag = find("rpmag0");
    }
    poseBody();
    poseArms(0.0f, glm::mat4(1.0f));
}

void GameWorld::addShells(int amount, int& added)
{
    // The bandolier's empty loops first, then the jacket pocket; what doesn't fit stays behind.
    added = 0;
    ArmsState& state = m_arms.mutableState();
    for (Zone zone : { Zone::Bandolier, Zone::JacketPocket }) {
        const int z = static_cast<int>(zone);
        const int room = std::max(m_arms.config().zones[z].shells - state.zones[z].shells, 0);
        const int n = std::min(room, amount - added);
        state.zones[z].shells += n;
        added += n;
    }
}

glm::vec3 GameWorld::weaponBob() const
{
    return glm::vec3(std::sin(m_view.bobPhase) * m_tuning.bobSide,
        -std::abs(std::cos(m_view.bobPhase)) * m_tuning.bobUp, 0.0f) * m_view.bobAmount;
}

glm::mat4 GameWorld::bodyNode(int node) const
{
    return node >= 0 && node < static_cast<int>(m_bodyMatrices.size()) ? m_bodyMatrices[node] : glm::mat4(1.0f);
}

// ---------------------------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------------------------

void GameWorld::updateWeapon(const Input& input)
{
    const float dt = input.dt;
    ArmsView& view = m_view;
    view.time += dt;
    view.messageCooldown -= dt;
    for (float& t : view.shotTime)
        t += dt;
    const float targetBob = input.moving ? (input.sprinting ? m_tuning.sprintBob : 1.0f) : 0.0f;
    view.bobAmount += (targetBob - view.bobAmount) * std::min(dt * 8.0f, 1.0f);
    view.bobPhase += dt * (input.sprinting ? 11.0f : 8.0f) * (input.moving ? 1.0f : 0.0f);
    const glm::mat4 steadyCamera = cameraWorldFrom(input.eye, input.forward);
    const glm::mat4 cameraWorld = steadyCamera * glm::translate(glm::mat4(1.0f), weaponBob());
    view.bodyToCamera = glm::inverse(cameraWorld) * bodyWorldFrom(steadyCamera);
    m_jacket = input.jacket;

    // The guns lag behind turns, and the shots go where they point.
    const glm::vec3 forward = glm::normalize(input.forward);
    if (glm::length(view.lastForward) > 0.5f) {
        const float yaw = std::atan2(forward.z, forward.x), lastYaw = std::atan2(view.lastForward.z, view.lastForward.x);
        const float pitch = std::asin(std::clamp(forward.y, -1.0f, 1.0f));
        const float lastPitch = std::asin(std::clamp(view.lastForward.y, -1.0f, 1.0f));
        // Turning while on the move throws the guns around more.
        const float swayGain = m_tuning.turnSway * (1.0f + m_tuning.moveTurnSway * view.bobAmount);
        view.sway.x += std::remainder(yaw - lastYaw, 2.0f * kPi) * swayGain;
        view.sway.y -= (pitch - lastPitch) * swayGain;
    }
    view.lastForward = forward;
    const float maxSway = glm::radians(std::max(m_tuning.maxSwayDegrees, 0.0f));
    view.sway = glm::clamp(view.sway * std::exp(-dt * m_tuning.swayRecovery), glm::vec2(-maxSway), glm::vec2(maxSway));

    m_arms.setRestockHeld(input.restock && input.jacket > 0.5f);
    requestFromInput(input);
    m_arms.update(dt);
    const float tiltTarget = m_arms.state().hand(Hand::Off).task == Task::InsertShell ? 1.0f : 0.0f;
    view.tilt += (tiltTarget - view.tilt) * std::min(dt * 14.0f, 1.0f);

    poseBody(); // first: the hand reaches into the jacket
    poseArms(dt, cameraWorld);
    handleArmsEvents(cameraWorld);
}

void GameWorld::requestFromInput(const Input& input)
{
    const auto request = [&](ActionKind kind, Zone zone = Zone::Bandolier, ItemKind item = ItemKind::Empty, int ground = -1) {
        Action action;
        action.kind = kind;
        action.zone = zone;
        action.item = item;
        action.ground = ground;
        m_arms.request(action);
    };
    // The trigger: one shot per pull.
    const bool pulled = input.fire && !m_view.triggerHeld;
    m_view.triggerHeld = input.fire;
    if (pulled)
        request(ActionKind::FireMain);
    if (input.secondary)
        request(m_arms.secondaryAction());
    if (input.swapTo != ItemKind::Empty)
        request(ActionKind::Swap, Zone::Bandolier, input.swapTo);
    if (input.toggleOffGun)
        request(ActionKind::ToggleOffGun);
    if (input.unload)
        request(ActionKind::Unload);
    if (input.reload)
        request(ActionKind::Reload);
    if (input.forceReload)
        request(ActionKind::ForceReload);
    if (input.zone >= 0 && input.zone < kZoneCount) {
        // With the jacket open, the bandolier's button refills it from the pocket.
        if (input.jacket > 0.5f && input.zone == static_cast<int>(Zone::Bandolier))
            request(ActionKind::Restock);
        else
            request(ActionKind::ZoneButton, static_cast<Zone>(input.zone));
    }
    if (input.putBack)
        request(ActionKind::PutBack);
    if (input.pickUp)
        request(ActionKind::PickUp, Zone::Bandolier, ItemKind::Empty, nearestGroundItem());
}

int GameWorld::nearestGroundItem() const
{
    int best = -1;
    float bestDistance = kPickUpReach;
    for (const Debris& debris : m_debris) {
        if (debris.ground < 0 || !m_arms.state().groundItem(debris.ground))
            continue;
        const glm::vec3 offset = debris.position - m_lastPlayerFeet;
        const float distance = glm::length(flat(offset));
        if (distance < bestDistance && std::abs(offset.y) < 1.6f) {
            best = debris.ground;
            bestDistance = distance;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------------------------

void GameWorld::handleArmsEvents(const glm::mat4& cameraWorld)
{
    const ArmsState& s = m_arms.state();
    const auto worldPoint = [&](const glm::mat4& local, const glm::vec3& point) {
        return glm::vec3(cameraWorld * local * glm::vec4(point, 1.0f));
    };
    const auto worldDirection = [&](const glm::vec3& direction) {
        return glm::vec3(cameraWorld * glm::vec4(direction, 0.0f));
    };
    const auto message = [&](const std::string& text) {
        // The same refusal again and again (a held key) shows once.
        if (text == m_view.lastMessage && m_view.messageCooldown > 0.0f)
            return;
        showMessage(text);
        m_view.lastMessage = text;
        m_view.messageCooldown = 1.0f;
    };
    const glm::mat4 mainItem = matrixOf(m_view.items[0]);
    const glm::mat4 offItem = matrixOf(m_view.items[1]);
    const glm::mat4 offHand = matrixOf(m_view.hands[1].pose);

    for (const ArmsEvent& event : m_arms.takeEvents()) {
        switch (event.type) {
        case EventType::Fired:
            shoot(event.hand, event.item, cameraWorld);
            break;

        case EventType::DryClick: {
            play(event.item == ItemKind::Pistol ? "pistol_dry" : "shotgun_empty", 0.8f);
            if (event.item == ItemKind::Shotgun) {
                message(s.shotgun.tube > 0 ? "Click. Pump it (right mouse)"
                    : "Click. Take a shell (Q / E), push it in (R)");
            }
            else {
                const Magazine* magazine = s.magazine(s.pistol.magazine);
                if (s.pistol.slideLocked && magazine && magazine->rounds > 0)
                    message("Click. Rack the slide (right mouse)");
                else if (!magazine || magazine->rounds == 0)
                    message("Click. Empty: R swaps the magazine");
                else
                    message("Click. Rack the slide (right mouse)");
            }
            break;
        }

        case EventType::MagEject: play("mag_eject", 0.8f); break;
        case EventType::MagInsert: play("mag_insert", 0.9f); break;
        case EventType::SlideRack: play("slide_rack", 0.85f); break;
        case EventType::SlideLock: play("slide_lock", 0.6f); break;
        case EventType::Pumped: play("shotgun_pump", 0.8f, random(0.97f, 1.03f)); break;
        case EventType::ShellInsert: play("shell_insert", 0.8f, random(0.95f, 1.05f)); break;
        case EventType::RoundInsert: play("round_insert", 0.7f, random(0.95f, 1.05f)); break;
        case EventType::Drawn:
        case EventType::Holstered: play("holster", 0.6f, random(0.9f, 1.1f)); break;
        case EventType::Stored: play("shell_grab", 0.45f, random(1.1f, 1.25f)); break;

        case EventType::ShellEjected:
            // A spent hull out of the shotgun's port, to the right.
            spawnDebris(ItemKind::Shell, -1, worldPoint(mainItem, glm::vec3(0.05f, 0.015f, -0.05f)),
                worldDirection(glm::vec3(random(1.8f, 2.6f), random(1.2f, 1.8f), random(0.0f, 0.6f))),
                glm::vec3(0.45f, 0.4f, 0.38f));
            break;

        case EventType::Grab:
            play("shell_grab", 0.7f, random(0.9f, 1.1f));
            // The pocket's rattle tells how much is left in it.
            play(event.fill > 0.5f ? "pocket_rattle_full" : "pocket_rattle_low", 0.5f, random(0.95f, 1.05f));
            break;

        case EventType::ZoneEmpty:
            play("pocket_empty", 0.7f);
            message(std::string("The ") + zoneName(event.zone) + " is empty");
            break;

        case EventType::Dropped: {
            // Where it falls from depends on what dropped it.
            glm::vec3 position, velocity;
            if (event.item == ItemKind::Magazine && event.hand == Hand::Main) {
                position = worldPoint(mainItem * m_rigs.pistolMagSeat, glm::vec3(0.0f, -0.06f, 0.0f));
                velocity = worldDirection(glm::vec3(0.0f, -0.5f, 0.0f));
            }
            else if (event.item == ItemKind::Round && event.hand == Hand::Main) {
                position = worldPoint(mainItem, glm::vec3(0.02f, 0.075f, -0.07f)); // racked out of the port
                velocity = worldDirection(glm::vec3(random(1.2f, 1.8f), random(1.0f, 1.6f), random(0.0f, 0.4f)));
            }
            else if (event.item == ItemKind::Shell && s.hand(Hand::Off).task == Task::Pump) {
                position = worldPoint(mainItem, glm::vec3(0.05f, 0.015f, -0.05f)); // a live shell pumped out
                velocity = worldDirection(glm::vec3(random(1.8f, 2.6f), random(1.2f, 1.8f), random(0.0f, 0.6f)));
            }
            else if (isGun(event.item)) {
                position = worldPoint(offItem, glm::vec3(0.0f));
                velocity = worldDirection(glm::vec3(-0.4f, 0.2f, -0.3f));
                message(std::string("Dropped the ") + itemName(event.item));
            }
            else {
                position = worldPoint(offHand, glm::vec3(0.0f, 0.03f, 0.0f));
                velocity = worldDirection(glm::vec3(random(-0.4f, 0.4f), 0.5f, random(-0.6f, 0.2f)));
                message(std::string("Dropped a ") + itemName(event.item));
            }
            spawnDebris(event.item, event.ground, position, velocity);
            break;
        }

        case EventType::PickedUp:
            std::erase_if(m_debris, [&](const Debris& d) { return d.ground == event.ground; });
            play("shell_grab", 0.7f, 0.9f);
            break;

        case EventType::Rejected:
            // Nothing happened, and the hand's shrug and the thud say so; the reason is shown too.
            play("reject", 0.6f, random(0.95f, 1.05f));
            message(event.reason);
            break;
        }
    }
}

void GameWorld::shoot(Hand hand, ItemKind gun, const glm::mat4& cameraWorld)
{
    const int h = static_cast<int>(hand);
    const ArmsConfig& config = m_arms.config();
    const bool shotgun = gun == ItemKind::Shotgun;
    m_view.shotTime[h] = 0.0f;
    m_recoil += shotgun ? config.shotgunRecoilDegrees : config.pistolRecoilDegrees;
    play(shotgun ? "shotgun_fire" : "pistol_fire", 1.0f, random(0.96f, 1.04f));
    // Everyone nearby hears it.
    alertNearby(glm::vec3(cameraWorld[3]), kGunshotHearing);
    if (!shotgun) {
        // The empty case flies out of the pistol's port.
        const glm::mat4 item = matrixOf(m_view.items[h]);
        spawnDebris(ItemKind::Round, -1, glm::vec3(cameraWorld * item * glm::vec4(0.02f, 0.075f, -0.07f, 1.0f)),
            glm::vec3(cameraWorld * glm::vec4(random(1.2f, 1.8f), random(1.0f, 1.6f), random(0.0f, 0.4f), 0.0f)),
            glm::vec3(0.95f, 0.75f, 0.4f));
    }

    // Shots leave along the barrel. The ray starts back from the muzzle so one poking through a wall doesn't
    // shoot from its far side.
    const glm::vec3 forward = m_view.barrelDirection[h];
    const glm::vec3 origin = m_view.barrelOrigin[h] - forward * (shotgun ? 0.6f : 0.25f);
    const glm::vec3 right = glm::normalize(glm::cross(forward, std::abs(forward.y) > 0.99f ? kXAxis : kUp));
    const glm::vec3 up = glm::cross(right, forward);
    const float spread = std::tan(glm::radians(shotgun ? kSpreadDegrees : kPistolSpreadDegrees));
    const int pellets = shotgun ? kPellets : 1;
    const float damageNear = shotgun ? kPelletDamageNear : kPistolDamageNear;
    const float damageFar = shotgun ? kPelletDamageFar : kPistolDamageFar;
    std::vector<float> damage(m_entities.size(), 0.0f);
    bool impactPlayed = false;
    for (int pellet = 0; pellet < pellets; ++pellet) {
        // Uniform over the cone's cross-section.
        const float radius = spread * std::sqrt(random(0.0f, 1.0f));
        const float angle = random(0.0f, 2.0f * kPi);
        const glm::vec3 dir = glm::normalize(forward + right * (radius * std::cos(angle)) + up * (radius * std::sin(angle)));

        float wallDistance = kShotRange;
        glm::vec3 normal(0.0f, 1.0f, 0.0f);
        const bool wall = m_physics.castRay(origin, dir, kShotRange, wallDistance, &normal);
        int target = -1;
        float targetDistance = wallDistance;
        for (size_t i = 0; i < m_entities.size(); ++i) {
            const Entity& entity = m_entities[i];
            if (entity.kind != Entity::Kind::Enemy || entity.state == EnemyState::Dead)
                continue;
            const float t = rayHitsEnemy(origin, dir, entity);
            if (t >= 0.0f && t < targetDistance) {
                target = static_cast<int>(i);
                targetDistance = t;
            }
        }
        if (pellet % 3 == 0)
            m_tracers.push_back({ m_view.barrelOrigin[h] + dir * 0.1f, origin + dir * targetDistance, 0.07f });
        if (target >= 0) {
            const Entity& enemy = m_entities[target];
            const glm::vec3 point = origin + dir * targetDistance;
            const float falloff = std::clamp((targetDistance - 6.0f) / 24.0f, 0.0f, 1.0f);
            float amount = damageNear + (damageFar - damageNear) * falloff;
            if (point.y > enemy.position.y + 1.55f)
                amount *= 1.6f; // head
            damage[target] += amount;
            spawnParticles(point, -dir, glm::vec3(0.45f, 0.03f, 0.02f), 4, 2.5f, 0.06f, 0.5f, true);
        }
        else if (wall) {
            const glm::vec3 point = origin + dir * wallDistance + normal * 0.02f;
            spawnParticles(point, normal, glm::vec3(1.0f, 0.72f, 0.3f), 3, 4.0f, 0.025f, 0.25f, true);
            spawnParticles(point, normal, glm::vec3(0.5f, 0.48f, 0.45f), 2, 1.0f, 0.07f, 0.6f, false);
            if (!impactPlayed) {
                playAt("impact", point, 0.6f, random(0.8f, 1.2f));
                impactPlayed = true;
            }
        }
    }
    for (size_t i = 0; i < damage.size(); ++i) {
        if (damage[i] <= 0.0f)
            continue;
        m_player.hitMarker = 1.0f;
        playAt("flesh_hit", m_entities[i].position + glm::vec3(0.0f, 1.2f, 0.0f), 0.9f, random(0.9f, 1.1f));
        damageEnemy(m_entities[i], damage[i]);
    }
}

// ---------------------------------------------------------------------------------------------
// Dropped things
// ---------------------------------------------------------------------------------------------

void GameWorld::spawnDebris(ItemKind item, int ground, const glm::vec3& position, const glm::vec3& velocity,
    const glm::vec3& tint)
{
    Debris debris;
    debris.item = item;
    debris.ground = ground;
    debris.position = position;
    debris.velocity = velocity;
    debris.spin = isGun(item) ? glm::vec3(random(-3.0f, 3.0f), random(-2.0f, 2.0f), random(-3.0f, 3.0f))
                              : glm::vec3(random(-14.0f, 14.0f), random(-6.0f, 6.0f), random(-14.0f, 14.0f));
    debris.tint = tint;
    float distance = 0.0f;
    debris.floor = (m_physics.castRay(position, glm::vec3(0.0f, -1.0f, 0.0f), 4.0f, distance)
        ? position.y - distance : m_lastPlayerFeet.y) + restHeight(item);
    debris.life = 20.0f;
    m_debris.push_back(debris);
    // Cosmetic ones make room; dropped items stay.
    int cosmetic = 0;
    for (const Debris& d : m_debris)
        cosmetic += d.ground < 0 ? 1 : 0;
    if (cosmetic > 40) {
        auto oldest = std::find_if(m_debris.begin(), m_debris.end(), [](const Debris& d) { return d.ground < 0; });
        m_debris.erase(oldest);
    }
}

void GameWorld::updateDebris(float dt)
{
    for (Debris& debris : m_debris) {
        if (debris.ground < 0)
            debris.life -= dt;
        if (debris.landed)
            continue;
        debris.velocity.y -= 9.81f * dt;
        debris.position += debris.velocity * dt;
        const glm::vec3 spin = debris.spin * dt;
        if (glm::length(spin) > 1e-6f)
            debris.rotation = glm::normalize(glm::angleAxis(glm::length(spin), glm::normalize(spin)) * debris.rotation);
        if (debris.position.y <= debris.floor && debris.velocity.y < 0.0f) {
            playAt(dropSound(debris.item), debris.position, isGun(debris.item) ? 0.8f : 0.5f, random(0.9f, 1.15f));
            // One bounce, then it lies down: turned about Y only, along its long axis.
            if (debris.velocity.y < -2.0f && !isGun(debris.item)) {
                debris.velocity = glm::vec3(debris.velocity.x * 0.3f, -debris.velocity.y * 0.25f, debris.velocity.z * 0.3f);
                debris.spin *= 0.5f;
            }
            else {
                debris.landed = true;
                const glm::vec3 along = debris.item == ItemKind::Shell ? kXAxis : kZAxis;
                const glm::vec3 axis = debris.rotation * along;
                const float yaw = debris.item == ItemKind::Shell ? std::atan2(-axis.z, axis.x) : std::atan2(-axis.x, -axis.z);
                debris.rotation = glm::angleAxis(yaw, kUp);
                // Magazines and guns lie on their side.
                if (debris.item == ItemKind::Magazine || isGun(debris.item))
                    debris.rotation = debris.rotation * glm::angleAxis(glm::radians(90.0f), kZAxis);
            }
            debris.position.y = debris.floor;
        }
    }
    std::erase_if(m_debris, [](const Debris& d) { return d.ground < 0 && d.life <= 0.0f; });
}

// ---------------------------------------------------------------------------------------------
// Poses
// ---------------------------------------------------------------------------------------------

ItemKind GameWorld::shownItem(Hand hand) const
{
    const HandState& h = m_arms.state().hand(hand);
    if (hand == Hand::Main && h.task == Task::Swap && h.eventDone)
        return h.target; // on its way in
    if (hand == Hand::Off && h.task == Task::DrawOffGun && h.time >= h.duration * 0.5f)
        return h.target;
    return h.item.kind;
}

void GameWorld::poseBody()
{
    const GameModel& body = m_assets.model(GameModelId::PlayerBody);
    if (!body.valid()) {
        m_bodyMatrices.clear();
        return;
    }
    const ArmsState& s = m_arms.state();
    restPose(body.data.nodes, m_bodyPose);
    const float open = m_jacket * m_jacket * (3.0f - 2.0f * m_jacket);
    const float angle = glm::radians(kJacketOpenDegrees) * open;
    if (m_body.flapL >= 0)
        m_bodyPose[m_body.flapL].rotation = glm::angleAxis(angle, kUp);
    if (m_body.flapR >= 0)
        m_bodyPose[m_body.flapR].rotation = glm::angleAxis(-angle, kUp);
    // Legs swing with the steps.
    const float swing = std::sin(m_view.bobPhase) * 0.45f * std::min(m_view.bobAmount, 1.0f);
    if (m_body.legL >= 0)
        m_bodyPose[m_body.legL].rotation = glm::angleAxis(swing, kXAxis);
    if (m_body.legR >= 0)
        m_bodyPose[m_body.legR].rotation = glm::angleAxis(-swing, kXAxis);
    // The ammunition itself is the counter: empty loops are gaps; the jacket's insides only show open.
    const auto show = [&](int node, bool visible) {
        if (node >= 0)
            m_bodyPose[node].scale = glm::vec3(visible ? 1.0f : 0.0001f);
    };
    const ZoneState& bandolier = s.zones[static_cast<int>(Zone::Bandolier)];
    const ZoneState& pocket = s.zones[static_cast<int>(Zone::JacketPocket)];
    const ZoneState& rightPocket = s.zones[static_cast<int>(Zone::RightPocket)];
    for (int i = 0; i < kBandolierLoops; ++i)
        show(m_body.band[i], i < bandolier.shells);
    for (int i = 0; i < kPocketSlots; ++i)
        show(m_body.pocket[i], i < pocket.shells && open > 0.05f);
    for (int i = 0; i < kBoxRounds; ++i)
        show(m_body.boxRounds[i], i < rightPocket.rounds && open > 0.05f);
    poseToMatrices(body.data.nodes, m_bodyPose, m_bodyMatrices);
}

NodePose GameWorld::offHandTarget(int& key, float& phaseLeft) const
{
    const ArmsState& s = m_arms.state();
    const HandState& off = s.hand(Hand::Off);
    const ArmsConfig& config = m_arms.config();
    const glm::mat4& toCamera = m_view.bodyToCamera;
    const glm::mat4 flapL = toCamera * (m_body.flapL >= 0 && !m_bodyMatrices.empty() ? bodyNode(m_body.flapL)
                                                                                     : offset(kFlapLRest));
    const glm::mat4 flapR = toCamera * (m_body.flapR >= 0 && !m_bodyMatrices.empty() ? bodyNode(m_body.flapR)
                                                                                     : offset(kFlapRRest));
    const glm::mat4 mainItem = matrixOf(m_view.items[0]);
    const ItemKind mainKind = shownItem(Hand::Main);
    const float t = off.time;
    phaseLeft = off.duration - t;

    const auto zoneAnchor = [&](Zone zone) {
        switch (zone) {
        case Zone::Bandolier: return poseOf(flapL * matrixOf(kHandBandolier));
        case Zone::JacketPocket: return poseOf(flapL * matrixOf(kHandPocket));
        default: return poseOf(flapR * matrixOf(kHandRightPocket));
        }
    };
    // Where the hand rests when it isn't doing anything: on what the main hand holds.
    const auto rest = [&]() {
        key = 100 + static_cast<int>(mainKind);
        switch (mainKind) {
        case ItemKind::Shotgun: {
            float pump = 0.0f;
            if (off.task == Task::Pump) {
                const float p = off.progress(), f = std::clamp(config.pumpEjectFrame, 0.05f, 0.95f);
                pump = p < f ? p / f : (1.0f - p) / (1.0f - f);
            }
            return poseOf(mainItem * offset(kPumpRest + glm::vec3(0.0f, 0.0f, kPumpTravel * pump) + kPalmBelowPump));
        }
        case ItemKind::Pistol: return poseOf(mainItem * kLeftSupportsPistol);
        case ItemKind::Magazine: return poseOf(mainItem * kLeftAtMagazine);
        default: return kHandLow;
        }
    };
    const auto phase = [&](float end, int phaseKey) {
        key = phaseKey;
        phaseLeft = end - t;
    };

    switch (off.task) {
    case Task::None:
    case Task::Pump:
        if (off.item.kind == ItemKind::Empty)
            return rest();
        key = 1;
        phaseLeft = 0.15f;
        return kHandReady;

    case Task::Fetch:
        if (t < off.eventAt) {
            phase(off.eventAt, 20 + static_cast<int>(off.zone));
            return zoneAnchor(off.zone);
        }
        key = 1;
        return kHandReady;

    case Task::Store:
        if (t < off.eventAt) {
            phase(off.eventAt, 20 + static_cast<int>(off.zone));
            return zoneAnchor(off.zone);
        }
        return rest();

    case Task::InsertShell: {
        const glm::mat4 port = mainItem * offset(glm::vec3(0.0f, -0.08f, -0.06f));
        if (t < off.eventAt * 0.55f) {
            phase(off.eventAt * 0.55f, 30);
            return poseOf(port);
        }
        if (t < off.eventAt) {
            phase(off.eventAt, 31);
            return poseOf(mainItem * offset(glm::vec3(0.0f, -0.064f, -0.075f)));
        }
        return rest();
    }

    case Task::PushRound: {
        const glm::mat4 lips = mainItem * offset(glm::vec3(-0.01f, 0.02f, 0.01f));
        if (off.zoneUsed && t < off.eventAt) {
            phase(off.eventAt, 20 + static_cast<int>(off.zone));
            return zoneAnchor(off.zone);
        }
        const float insertAt = off.zoneUsed ? off.event2At : off.eventAt;
        if (t < insertAt) {
            phase(insertAt, 47);
            return poseOf(lips);
        }
        return rest();
    }

    case Task::Restock:
        if (t < off.eventAt) {
            phase(off.eventAt, 21);
            return zoneAnchor(Zone::JacketPocket);
        }
        phase(off.event2At, 20);
        return zoneAnchor(Zone::Bandolier);

    case Task::PickUp:
        if (t < off.eventAt) {
            phase(off.eventAt, 40);
            return poseOf(toCamera * matrixOf(kHandGround));
        }
        key = 1;
        return kHandReady;

    case Task::MagInsert: {
        // The magazine lines up under the grip, then goes up into it; the hand holds its base.
        const glm::mat4 seat = mainItem * m_rigs.pistolMagSeat;
        const glm::mat4 handFromMagazine = glm::inverse(kMagazineInHand);
        const float lineUp = off.eventAt * 0.45f;
        if (t < lineUp) {
            phase(lineUp, 42);
            return poseOf(seat * offset(glm::vec3(0.0f, -0.16f, 0.0f)) * handFromMagazine);
        }
        if (t < off.eventAt) {
            phase(off.eventAt, 43);
            const float depth = 0.16f * (1.0f - (t - lineUp) / std::max(off.eventAt - lineUp, 1e-3f));
            return poseOf(seat * offset(glm::vec3(0.0f, -depth, 0.0f)) * handFromMagazine);
        }
        return rest();
    }

    case Task::Rack: {
        // Grab the slide's rear and pull it back with it.
        const float grab = off.eventAt * 0.4f;
        const float slide = off.eventAt > 0.0f ? std::clamp((t - grab) / std::max(off.eventAt - grab, 1e-3f), 0.0f, 1.0f) : 1.0f;
        const glm::mat4 rear = mainItem * offset(glm::vec3(0.0f, 0.072f, 0.03f + kSlideTravel * slide), 0.0f, 0.0f, 90.0f);
        if (t < grab) {
            phase(grab, 44);
            return poseOf(rear);
        }
        if (t < off.eventAt) {
            phase(off.eventAt, 45);
            return poseOf(rear);
        }
        return rest();
    }

    case Task::DrawOffGun:
    case Task::StowOffGun: {
        const ItemKind gun = off.task == Task::DrawOffGun ? off.target : off.item.kind;
        const NodePose& hip = gun == ItemKind::Pistol ? kHandHolster : kHandSling;
        if (off.task == Task::DrawOffGun && t >= off.duration * 0.5f) {
            key = 50 + static_cast<int>(gun); // up to the side; poseArms puts the gun there
            return kHandLow;
        }
        phase(off.task == Task::DrawOffGun ? off.duration * 0.5f : off.duration, 41);
        return poseOf(toCamera * matrixOf(hip));
    }

    case Task::Swap:
        // Handing the magazine over to the main hand.
        key = 48;
        return poseOf(matrixOf(m_view.hands[0].pose) * offset(glm::vec3(-0.06f, 0.0f, 0.0f)));

    case Task::MagEject:
        return rest();
    }
    return rest();
}

void GameWorld::poseArms(float dt, const glm::mat4& cameraWorld)
{
    const ArmsState& s = m_arms.state();
    const ArmsConfig& config = m_arms.config();
    ArmsView& view = m_view;
    const HandState& main = s.hand(Hand::Main);
    const HandState& off = s.hand(Hand::Off);

    // Shaking the hand that was refused something.
    const auto shake = [&](const HandState& hand) {
        if (hand.rejected > 0.3f)
            return glm::vec3(0.0f);
        const float fade = 1.0f - hand.rejected / 0.3f;
        return glm::vec3(std::sin(view.time * 70.0f) * 0.008f, std::sin(view.time * 53.0f) * 0.004f, 0.0f) * fade;
    };
    const auto pumpAmount = [&]() {
        if (off.task != Task::Pump)
            return 0.0f;
        const float p = off.progress(), f = std::clamp(config.pumpEjectFrame, 0.05f, 0.95f);
        return p < f ? p / f : (1.0f - p) / (1.0f - f);
    };

    // The main hand's item: a gun aimed off the view's center, or the magazine held up to look at.
    const ItemKind mainKind = shownItem(Hand::Main);
    const float mainKick = kickCurve(view.shotTime[0]);
    NodePose mainPose;
    switch (mainKind) {
    case ItemKind::Shotgun: mainPose = shotgunPose(mainKick, pumpAmount(), view.tilt, view.sway, m_jacket, 1.0f); break;
    case ItemKind::Pistol: mainPose = pistolPose(mainKick, view.sway, m_jacket, 1.0f); break;
    case ItemKind::Magazine:
        mainPose = kMagazineHeld;
        mainPose.rotation = glm::angleAxis(view.sway.x, kUp) * glm::angleAxis(view.sway.y, kXAxis) * mainPose.rotation;
        break;
    default: mainPose = kHandLow; break;
    }
    // Swapping: down out of view with the old item, up with the new one.
    if (main.task == Task::Swap) {
        const float e = std::max(main.eventAt, 1e-3f);
        const float lower = main.time < main.eventAt ? main.time / e
                                                     : 1.0f - (main.time - main.eventAt) / std::max(main.duration - main.eventAt, 1e-3f);
        const float l = std::clamp(lower, 0.0f, 1.0f);
        mainPose.translation += glm::vec3(0.05f, -0.28f, 0.12f) * l;
        mainPose.rotation = mainPose.rotation * glm::angleAxis(glm::radians(-50.0f) * l, kXAxis);
    }
    // A slight turn of the pistol while the magazine drops.
    if (main.task == Task::MagEject)
        mainPose.rotation = mainPose.rotation * glm::angleAxis(glm::radians(15.0f) * std::sin(kPi * main.progress()), kZAxis);
    mainPose.translation += shake(main);
    view.items[0] = mainPose;
    const glm::mat4 mainItem = matrixOf(mainPose);
    switch (mainKind) {
    case ItemKind::Shotgun: view.hands[0].pose = poseOf(mainItem * kRightOnShotgun); break;
    case ItemKind::Pistol: view.hands[0].pose = poseOf(mainItem * kRightOnPistol); break;
    case ItemKind::Magazine: view.hands[0].pose = poseOf(mainItem * kRightOnMagazine); break;
    default: view.hands[0].pose = mainPose; break;
    }

    // The off hand: a gun of its own, or moving between the targets of its task.
    const ItemKind offKind = shownItem(Hand::Off);
    HandView& offView = view.hands[1];
    if (isGun(offKind) && off.task != Task::StowOffGun) {
        const float kick = kickCurve(view.shotTime[1]);
        NodePose gun = offKind == ItemKind::Shotgun ? shotgunPose(kick, 0.0f, 0.0f, view.sway, m_jacket, -1.0f)
                                                    : pistolPose(kick, view.sway, m_jacket, -1.0f);
        if (off.task == Task::DrawOffGun) {
            // Coming up from the hip.
            const float rise = 1.0f - std::clamp((off.time - off.duration * 0.5f) / std::max(off.duration * 0.5f, 1e-3f), 0.0f, 1.0f);
            gun.translation += glm::vec3(-0.05f, -0.28f, 0.12f) * rise;
        }
        gun.translation += shake(off);
        view.items[1] = gun;
        const glm::mat4 grip = offKind == ItemKind::Shotgun ? kLeftOnShotgunGrip : kRightOnPistol;
        const NodePose target = poseOf(matrixOf(gun) * grip);
        if (offView.key != 50 + static_cast<int>(offKind)) {
            offView.from = offView.key < 0 ? target : offView.pose;
            offView.key = 50 + static_cast<int>(offKind);
            offView.time = 0.0f;
            offView.duration = 0.12f;
        }
        offView.time += dt;
        offView.pose = blend(offView.from, target, offView.time / offView.duration);
    }
    else {
        int key = 0;
        float phaseLeft = 0.1f;
        const NodePose target = offHandTarget(key, phaseLeft);
        if (key != offView.key) {
            offView.from = offView.key < 0 ? target : offView.pose;
            offView.key = key;
            offView.time = 0.0f;
            offView.duration = std::max(phaseLeft, 0.06f);
        }
        offView.time += dt;
        offView.pose = blend(offView.from, target, offView.time / offView.duration);
        offView.pose.translation += shake(off);
        // What it holds sits in the palm.
        const glm::mat4 hand = matrixOf(offView.pose);
        switch (offKind) {
        case ItemKind::Shell: view.items[1] = poseOf(hand * kShellInHand); break;
        case ItemKind::Round: view.items[1] = poseOf(hand * kRoundInHand); break;
        case ItemKind::Magazine: view.items[1] = poseOf(hand * kMagazineInHand); break;
        default: view.items[1] = offView.pose; break; // a gun on its way to the hip
        }
    }

    // Barrels, in world space, for the shots.
    for (int h = 0; h < 2; ++h) {
        const ItemKind kind = h == 0 ? mainKind : offKind;
        const glm::mat4 gun = cameraWorld * matrixOf(view.items[h]);
        if (kind == ItemKind::Shotgun) {
            view.barrelOrigin[h] = glm::vec3(gun * glm::vec4(m_rigs.shotgunMuzzle, 1.0f));
            view.barrelDirection[h] = glm::normalize(-glm::vec3(gun[2]));
        }
        else if (kind == ItemKind::Pistol) {
            view.barrelOrigin[h] = glm::vec3(gun * glm::vec4(m_rigs.pistolMuzzle, 1.0f));
            view.barrelDirection[h] = glm::normalize(-glm::vec3(gun[2]));
        }
        else {
            view.barrelOrigin[h] = glm::vec3(cameraWorld[3]);
            view.barrelDirection[h] = -glm::vec3(cameraWorld[2]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------------------------

void GameWorld::drawItem(GameModelId id, const glm::mat4& world, const glm::vec3& tint, std::vector<DynamicInstance>& out)
{
    const GameModel& model = m_assets.model(id);
    if (!model.valid())
        return;
    restPose(model.data.nodes, m_scratchPose);
    poseToMatrices(model.data.nodes, m_scratchPose, m_scratchMatrices);
    emitNodes(model, m_scratchMatrices, world, tint, false, out);
}

void GameWorld::drawShotgun(const glm::mat4& world, float pump, float flash, std::vector<DynamicInstance>& out)
{
    // `world` places the gun node itself.
    const GameModel& model = m_assets.model(GameModelId::Shotgun);
    if (!model.valid())
        return;
    restPose(model.data.nodes, m_scratchPose);
    if (m_rigs.shotgunGun >= 0)
        m_scratchPose[m_rigs.shotgunGun] = NodePose{};
    if (m_rigs.shotgunPump >= 0)
        m_scratchPose[m_rigs.shotgunPump].translation = kPumpRest + glm::vec3(0.0f, 0.0f, kPumpTravel * pump);
    if (m_rigs.shotgunFlash >= 0) {
        m_scratchPose[m_rigs.shotgunFlash].scale = flash > 0.0f ? glm::vec3(1.2f * flash, 1.2f * flash, 1.4f) : glm::vec3(0.0001f);
        m_scratchPose[m_rigs.shotgunFlash].rotation = glm::angleAxis(m_levelTime * 50.0f, kZAxis);
    }
    poseToMatrices(model.data.nodes, m_scratchPose, m_scratchMatrices);
    emitNodes(model, m_scratchMatrices, world, glm::vec3(1.0f), false, out);
}

void GameWorld::drawPistol(const glm::mat4& world, float slide, float flash, bool withMagazine,
    std::vector<DynamicInstance>& out)
{
    const GameModel& model = m_assets.model(GameModelId::Pistol);
    if (!model.valid())
        return;
    restPose(model.data.nodes, m_scratchPose);
    if (m_rigs.pistolSlide >= 0)
        m_scratchPose[m_rigs.pistolSlide].translation += glm::vec3(0.0f, 0.0f, kSlideTravel * slide);
    if (m_rigs.pistolFlash >= 0) {
        m_scratchPose[m_rigs.pistolFlash].scale = flash > 0.0f ? glm::vec3(0.6f * flash, 0.6f * flash, 0.7f) : glm::vec3(0.0001f);
        m_scratchPose[m_rigs.pistolFlash].rotation = glm::angleAxis(m_levelTime * 50.0f, kZAxis);
    }
    poseToMatrices(model.data.nodes, m_scratchPose, m_scratchMatrices);
    emitNodes(model, m_scratchMatrices, world, glm::vec3(1.0f), false, out);

    const ArmsState& s = m_arms.state();
    if (!withMagazine || s.pistol.magazine < 0)
        return;
    // Sliding out while the release is pressed.
    float drop = 0.0f;
    const HandState& main = s.hand(Hand::Main);
    if (main.task == Task::MagEject && main.item.kind == ItemKind::Pistol)
        drop = 0.03f * std::clamp(main.time / std::max(main.eventAt, 1e-3f), 0.0f, 1.0f);
    drawMagazine(world * m_rigs.pistolMagSeat * offset(glm::vec3(0.0f, -drop, 0.0f)), s.pistol.magazine, out);
}

void GameWorld::drawMagazine(const glm::mat4& world, int magazineId, std::vector<DynamicInstance>& out)
{
    const GameModel& model = m_assets.model(GameModelId::Magazine);
    const Magazine* magazine = m_arms.state().magazine(magazineId);
    if (!model.valid() || !magazine)
        return;
    restPose(model.data.nodes, m_scratchPose);
    // The rounds stack from the top; the follower sits under the last one.
    const int rounds = std::clamp(magazine->rounds, 0, static_cast<int>(m_rigs.magazineRounds.size()));
    for (size_t i = 0; i < m_rigs.magazineRounds.size(); ++i)
        if (m_rigs.magazineRounds[i] >= 0)
            m_scratchPose[m_rigs.magazineRounds[i]].scale = glm::vec3(static_cast<int>(i) < rounds ? 1.0f : 0.0001f);
    if (m_rigs.magazineFollower >= 0)
        m_scratchPose[m_rigs.magazineFollower].translation.y = -0.004f - 0.012f * rounds;
    poseToMatrices(model.data.nodes, m_scratchPose, m_scratchMatrices);
    emitNodes(model, m_scratchMatrices, world, glm::vec3(1.0f), false, out);
}

void GameWorld::collectArms(const glm::mat4& cameraWorld, std::vector<DynamicInstance>& out)
{
    const ArmsState& s = m_arms.state();
    const float open = m_jacket * m_jacket * (3.0f - 2.0f * m_jacket);
    const auto slideOf = [&](Hand hand) {
        // Kicked back by a shot, pulled by the off hand, or locked open.
        float slide = s.pistol.slideLocked ? kSlideLocked : 0.0f;
        const float shot = m_view.shotTime[static_cast<int>(hand)];
        if (shot < 0.07f)
            slide = std::max(slide, 1.0f - shot / 0.07f);
        const HandState& off = s.hand(Hand::Off);
        if (off.task == Task::Rack && hand == Hand::Main) {
            const float grab = off.eventAt * 0.4f;
            slide = off.time < off.eventAt
                ? std::max(slide, std::clamp((off.time - grab) / std::max(off.eventAt - grab, 1e-3f), 0.0f, 1.0f))
                : (s.pistol.slideLocked ? kSlideLocked : 1.0f - (off.time - off.eventAt) / std::max(off.duration - off.eventAt, 1e-3f));
        }
        return std::clamp(slide, 0.0f, 1.0f);
    };
    const auto flashOf = [&](Hand hand) {
        const float shot = m_view.shotTime[static_cast<int>(hand)];
        return shot < 0.06f ? 1.0f - shot * 6.0f : 0.0f;
    };

    // Dropped and spent things.
    for (const Debris& debris : m_debris) {
        const glm::mat4 world = glm::translate(glm::mat4(1.0f), debris.position) * glm::mat4_cast(debris.rotation);
        switch (debris.item) {
        case ItemKind::Shell: drawItem(GameModelId::ShellItem, world, debris.tint, out); break;
        case ItemKind::Round: drawItem(GameModelId::RoundItem, world, debris.tint, out); break;
        case ItemKind::Magazine:
            if (const GroundItem* item = s.groundItem(debris.ground))
                drawMagazine(world, item->item.magazine, out);
            break;
        case ItemKind::Pistol: drawPistol(world, s.pistol.slideLocked ? kSlideLocked : 0.0f, 0.0f, true, out); break;
        case ItemKind::Shotgun: drawShotgun(world, 0.0f, 0.0f, out); break;
        default: break;
        }
    }
    if (!m_player.alive)
        return;

    // The body, the magazines stored on it and the holstered pistol.
    const glm::mat4 bodyWorld = bodyWorldFrom(cameraWorld);
    const GameModel& body = m_assets.model(GameModelId::PlayerBody);
    if (body.valid() && !m_bodyMatrices.empty()) {
        emitNodes(body, m_bodyMatrices, bodyWorld, glm::vec3(1.0f), false, out);
        const auto magazinesAt = [&](Zone zone, const int* anchors, int count, bool visible) {
            const std::vector<int>& magazines = s.zones[static_cast<int>(zone)].magazines;
            for (int i = 0; i < count && i < static_cast<int>(magazines.size()); ++i)
                if (anchors[i] >= 0 && visible)
                    drawMagazine(bodyWorld * bodyNode(anchors[i]), magazines[i], out);
        };
        magazinesAt(Zone::Bandolier, m_body.bandMags.data(), 2, true);
        magazinesAt(Zone::JacketPocket, m_body.pocketMags.data(), 2, open > 0.05f);
        magazinesAt(Zone::RightPocket, &m_body.rightPocketMag, 1, open > 0.05f);
        // The pistol is in its holster unless a hand has it (or is just taking it), or it lies somewhere.
        const HandState& main = s.hand(Hand::Main);
        const HandState& off = s.hand(Hand::Off);
        const bool beingDrawn = (main.task == Task::Swap && main.target == ItemKind::Pistol && main.eventDone) ||
            (off.task == Task::DrawOffGun && off.target == ItemKind::Pistol && off.time >= off.duration * 0.5f);
        if (s.handHolding(ItemKind::Pistol) < 0 && !s.onGround(ItemKind::Pistol) && !beingDrawn && m_body.holsterGun >= 0)
            drawPistol(bodyWorld * bodyNode(m_body.holsterGun), s.pistol.slideLocked ? kSlideLocked : 0.0f, 0.0f, true, out);
    }

    // The hands and what they hold, with the camera.
    const glm::mat4 view = cameraWorld * glm::translate(glm::mat4(1.0f), weaponBob());
    if (const GameModel& hands = m_assets.model(GameModelId::Hands); hands.valid()) {
        restPose(hands.data.nodes, m_scratchPose);
        if (m_rigs.handR >= 0)
            m_scratchPose[m_rigs.handR] = m_view.hands[0].pose;
        if (m_rigs.handL >= 0)
            m_scratchPose[m_rigs.handL] = m_view.hands[1].pose;
        poseToMatrices(hands.data.nodes, m_scratchPose, m_scratchMatrices);
        emitNodes(hands, m_scratchMatrices, view, glm::vec3(1.0f), false, out);
    }
    for (Hand hand : { Hand::Main, Hand::Off }) {
        const int h = static_cast<int>(hand);
        const glm::mat4 world = view * matrixOf(m_view.items[h]);
        const HandState& state = s.hand(hand);
        switch (shownItem(hand)) {
        case ItemKind::Shotgun: {
            float pump = 0.0f;
            const HandState& off = s.hand(Hand::Off);
            if (hand == Hand::Main && off.task == Task::Pump) {
                const float p = off.progress(), f = std::clamp(m_arms.config().pumpEjectFrame, 0.05f, 0.95f);
                pump = p < f ? p / f : (1.0f - p) / (1.0f - f);
            }
            drawShotgun(world, pump, flashOf(hand), out);
            break;
        }
        case ItemKind::Pistol: drawPistol(world, slideOf(hand), flashOf(hand), true, out); break;
        case ItemKind::Magazine: {
            // The one in the hand, or the one on its way in (a swap takes it at the end).
            int id = state.item.magazine;
            if (id < 0 && hand == Hand::Main && s.hand(Hand::Off).task == Task::Swap)
                id = s.hand(Hand::Off).item.magazine;
            if (id < 0 && hand == Hand::Main && state.task == Task::Swap) {
                const std::vector<int>& stored = s.zones[static_cast<int>(state.zone)].magazines;
                if (!stored.empty())
                    id = *std::min_element(stored.begin(), stored.end(), [&](int a, int b) {
                        return s.magazine(a)->rounds < s.magazine(b)->rounds;
                    });
            }
            drawMagazine(world, id, out);
            break;
        }
        case ItemKind::Shell: drawItem(GameModelId::ShellItem, world, glm::vec3(1.0f), out); break;
        case ItemKind::Round: drawItem(GameModelId::RoundItem, world, glm::vec3(1.0f), out); break;
        default: break;
        }
    }
}
