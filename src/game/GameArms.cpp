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
constexpr float kPistolSpreadDegrees = 0.6f;
constexpr float kGunshotHearing = 24.0f;
constexpr float kPickUpReach = 2.2f; // meters from the feet

// Where things are held and how they move: ArmsPoses (arms_poses.json).
const glm::vec3 kFlapLRest(-0.21f, -0.55f, -0.16f); // where the panels hang if the body model is missing
const glm::vec3 kFlapRRest(0.21f, -0.55f, -0.16f);

glm::vec3 flat(const glm::vec3& v)
{
    return glm::vec3(v.x, 0.0f, v.z);
}

// Where looking down points at things to bind them (body space), for what has no node of its own.
const glm::vec3 kShotgunBindPoint(-0.3f, -0.8f, 0.0f);

float kickCurve(float shotTime)
{
    return shotTime < 0.04f ? shotTime / 0.04f : std::exp(-(shotTime - 0.04f) / 0.1f);
}

glm::mat4 cameraWorldFrom(const glm::vec3& eye, const glm::vec3& forward)
{
    const glm::vec3 up = std::abs(glm::normalize(forward).y) > 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : kUp;
    return glm::inverse(glm::lookAt(eye, eye + forward, up));
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
    m_poses = loadArmsPoses();
    m_arms.setConfig(m_tuning.arms);
    m_arms.reset();
    m_view = {};

    m_rigs.find(&m_assets.model(GameModelId::Shotgun).data, &m_assets.model(GameModelId::Pistol).data,
        &m_assets.model(GameModelId::Magazine).data, &m_assets.model(GameModelId::Hands).data);

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

void GameWorld::addRounds(int amount, int& added)
{
    // Loose, into every zone that takes rounds (the right pocket), quickest first.
    added = 0;
    ArmsState& state = m_arms.mutableState();
    for (int z = 0; z < kZoneCount; ++z) {
        const int room = std::max(m_arms.config().zones[z].rounds - state.zones[z].rounds, 0);
        const int n = std::min(room, amount - added);
        state.zones[z].rounds += n;
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
    view.bodyWorld = bodyWorldFrom(steadyCamera);
    view.cameraWorld = cameraWorld;
    view.bodyToCamera = glm::inverse(cameraWorld) * view.bodyWorld;
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

    m_arms.setRestockHeld(input.restockHeld && input.jacket > 0.5f);
    m_arms.setReloadHeld(input.reloadHeld && input.jacket <= 0.5f);
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
    // The left mouse button fires the left hand's gun, or the right hand's when the left has none; the right
    // button only matters with a gun in each hand (the right one). One shot per pull. Pumping and racking
    // are on R.
    const ArmsState& s = m_arms.state();
    const bool leftArmed = isGun(s.hand(Hand::Off).item.kind);
    const bool rightArmed = isGun(s.hand(Hand::Main).item.kind);
    const bool pulled = input.fire && !m_view.triggerHeld;
    m_view.triggerHeld = input.fire;
    if (pulled)
        request(leftArmed ? ActionKind::FireOff : ActionKind::FireMain);
    if (input.secondary && leftArmed && rightArmed)
        request(ActionKind::FireMain);
    // Slot keys and clicks in the jacket view. A press does one thing; held, taking (or putting away) goes on.
    using Repeat = ArmsView::Repeat;
    const ArmsConfig& config = m_arms.config();
    const auto repeatOf = [&](const Action& action) {
        const Verdict verdict = CanPerformAction(s, config, action);
        return !verdict.ok ? Repeat::None
            : verdict.resolved.kind == ActionKind::Fetch ? Repeat::Take
            : verdict.resolved.kind == ActionKind::Store ? Repeat::Put : Repeat::None;
    };
    const auto press = [&](ActionKind kind, const SlotBind& bind) {
        Action action;
        action.kind = kind;
        action.zone = bind.zone;
        action.item = kind == ActionKind::Store ? s.hand(Hand::Off).item.kind : bind.item;
        m_view.repeat = repeatOf(action);
        m_view.repeatBind = bind;
        request(action.kind, action.zone, action.item);
    };
    if (input.use.item != ItemKind::Empty)
        press(ActionKind::Slot, input.use);
    if (input.take.item != ItemKind::Empty) {
        // A gun is drawn (or put away) like with its key; a magazine goes into the empty right hand like with
        // its key; everything else into the left hand, onto what it already holds of the same.
        const bool emptyHands = s.hand(Hand::Main).item.kind == ItemKind::Empty && s.hand(Hand::Off).item.kind == ItemKind::Empty;
        const bool slotLike = isGun(input.take.item) || (input.take.item == ItemKind::Magazine && emptyHands);
        press(slotLike ? ActionKind::Slot : ActionKind::Fetch, input.take);
    }
    if (input.put.item != ItemKind::Empty && !isGun(input.put.item))
        press(ActionKind::Store, input.put);
    // Still held: another one, once the hand is free, as long as it fits / there is some left. (Checked first,
    // so running out ends it quietly.) A mouse button follows the cursor to another target.
    const SlotBind& held = input.useHeld.item != ItemKind::Empty ? input.useHeld
        : input.takeHeld.item != ItemKind::Empty ? input.takeHeld : input.putHeld;
    if (held.item == ItemKind::Empty || isGun(held.item)) {
        m_view.repeat = Repeat::None;
    }
    else if (m_view.repeat != Repeat::None && !s.hand(Hand::Off).busy()) {
        if (input.useHeld.item == ItemKind::Empty)
            m_view.repeatBind = held;
        const SlotBind& bind = m_view.repeatBind;
        const ZoneState& zone = s.zones[static_cast<int>(bind.zone)];
        Action action;
        action.zone = bind.zone;
        if (m_view.repeat == Repeat::Take) {
            const int left = bind.item == ItemKind::Shell ? zone.shells : bind.item == ItemKind::Round ? zone.rounds : 0;
            action.kind = ActionKind::Fetch;
            action.item = bind.item;
            if (stackable(bind.item) && left > 0 && m_arms.offHandRoom(bind.item) > 0 &&
                CanPerformAction(s, config, action).ok)
                m_arms.request(action);
        }
        else {
            action.kind = ActionKind::Store;
            action.item = s.hand(Hand::Off).item.kind;
            if ((stackable(action.item) || action.item == ItemKind::Magazine) && CanPerformAction(s, config, action).ok)
                m_arms.request(action);
        }
    }
    // R: tap, double tap or hold (Game tells them apart); with the jacket open it refills the bandolier.
    if (input.restock)
        request(ActionKind::Restock);
    if (input.reloadTap)
        request(ActionKind::ReloadTap);
    if (input.reloadDoubleTap)
        request(ActionKind::QuickEject);
    if (input.reloadHold)
        request(ActionKind::ReloadHold);
    if (input.interact) {
        // The nearer of a pickup (taken at once) and a dropped item (the left hand picks it up).
        float pickupDistance = 0.0f, groundDistance = 0.0f;
        const int pickup = nearestPickup(&pickupDistance);
        const int ground = nearestGroundItem(&groundDistance);
        if (pickup >= 0 && (ground < 0 || pickupDistance <= groundDistance))
            collectPickup(m_entities[pickup]);
        else
            request(ActionKind::PickUp, Zone::Bandolier, ItemKind::Empty, ground);
    }
}

int GameWorld::nearestGroundItem(float* distance) const
{
    int best = -1;
    float bestDistance = kPickUpReach;
    for (const Debris& debris : m_debris) {
        if (debris.ground < 0 || !m_arms.state().groundItem(debris.ground))
            continue;
        const glm::vec3 offset = debris.position - m_lastPlayerFeet;
        const float d = glm::length(flat(offset));
        if (d < bestDistance && std::abs(offset.y) < 1.6f) {
            best = debris.ground;
            bestDistance = d;
        }
    }
    if (distance)
        *distance = bestDistance;
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
                message(s.shotgun.tube > 0 ? "Click. Pump it (tap R)"
                    : "Click. Take a shell (Z X C V), hold R to push it in");
            }
            else {
                const Magazine* magazine = s.magazine(s.pistol.magazine);
                if (!magazine || magazine->rounds == 0)
                    message("Click. Empty: double-tap R drops the magazine, hold R takes it out");
                else
                    message("Click. Rack the slide (tap R)");
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
            if (event.item == ItemKind::Magazine && (event.hand == Hand::Main || shownItem(event.hand) == ItemKind::Pistol)) {
                // Out of the pistol's grip, in whichever hand it is.
                const glm::mat4 gun = event.hand == Hand::Main ? mainItem : offItem;
                position = worldPoint(gun * m_rigs.pistolMagSeat, glm::vec3(0.0f, -0.06f, 0.0f));
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
    const float damageNear = shotgun ? m_tuning.pelletDamageNear : m_tuning.pistolDamageNear;
    const float damageFar = shotgun ? m_tuning.pelletDamageFar : m_tuning.pistolDamageFar;
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
                amount *= m_tuning.headMultiplier;
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
    const float angle = glm::radians(m_poses.jacketOpenDegrees) * open;
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
    phaseLeft = off.duration - off.time;

    // Where the hand rests when it isn't doing anything: on what the main hand holds.
    ArmsAnchors anchors;
    anchors.restKey = static_cast<int>(mainKind);
    switch (mainKind) {
    case ItemKind::Shotgun: {
        float pump = 0.0f;
        if (off.task == Task::Pump) {
            const float p = off.progress(), f = std::clamp(config.pumpEjectFrame, 0.05f, 0.95f);
            pump = p < f ? p / f : (1.0f - p) / (1.0f - f);
        }
        anchors.rest = mainItem * offset(m_poses.pumpRest + glm::vec3(0.0f, 0.0f, m_poses.pumpTravel * pump) +
            m_poses.palmBelowPump);
        break;
    }
    case ItemKind::Pistol: anchors.rest = mainItem * m_poses.leftSupportsPistol.matrix(); break;
    case ItemKind::Magazine: anchors.rest = mainItem * m_poses.leftAtMagazine.matrix(); break;
    default: anchors.rest = m_poses.handLow.matrix(); break;
    }
    const auto rest = [&]() {
        key = 100 + anchors.restKey;
        return poseOf(anchors.rest);
    };

    switch (off.task) {
    case Task::None:
    case Task::Pump:
        if (off.item.kind == ItemKind::Empty)
            return rest();
        key = 1;
        phaseLeft = 0.15f;
        return m_poses.handReady.pose();
    case Task::MagEject:
        return rest();
    default:
        break;
    }

    const ArmsTrackId track = armsTrackOf(off.task);
    if (track == ArmsTrackId::Count)
        return rest();
    const ArmsTaskTimes times = ArmsTaskTimes::of(off);
    anchors.mainItem = mainItem;
    anchors.mainHand = matrixOf(m_view.hands[0].pose);
    anchors.flapL = flapL;
    anchors.flapR = flapR;
    anchors.bodyToCamera = toCamera;
    anchors.magSeat = m_rigs.pistolMagSeat;
    anchors.slide = off.task == Task::Rack ? armsRackPull(m_poses, times, off.time) : 0.0f;
    anchors.zone = off.zone;
    anchors.hipPistol = (off.task == Task::DrawOffGun ? off.target : off.item.kind) == ItemKind::Pistol;
    return armsTrackTarget(m_poses, track, times, off.time, anchors, key, phaseLeft);
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
    case ItemKind::Shotgun: mainPose = shotgunPose(m_poses, mainKick, pumpAmount(), view.tilt, view.sway, m_jacket, 1.0f); break;
    case ItemKind::Pistol: mainPose = pistolPose(m_poses, mainKick, view.sway, m_jacket, 1.0f); break;
    case ItemKind::Magazine:
        mainPose = m_poses.magazineHeld.pose();
        mainPose.rotation = glm::angleAxis(view.sway.x, kUp) * glm::angleAxis(view.sway.y, kXAxis) * mainPose.rotation;
        break;
    default: mainPose = m_poses.rightHandLow.pose(); break;
    }
    // Swapping: down out of view with the old item, up with the new one.
    if (main.task == Task::Swap) {
        const float e = std::max(main.eventAt, 1e-3f);
        const float lower = main.time < main.eventAt ? main.time / e
                                                     : 1.0f - (main.time - main.eventAt) / std::max(main.duration - main.eventAt, 1e-3f);
        const float l = std::clamp(lower, 0.0f, 1.0f);
        mainPose.translation += m_poses.swapLower * l;
        mainPose.rotation = mainPose.rotation * glm::angleAxis(glm::radians(m_poses.swapLowerDegrees) * l, kXAxis);
    }
    // A slight turn of the pistol while the magazine drops.
    if (main.task == Task::MagEject)
        mainPose.rotation = mainPose.rotation *
            glm::angleAxis(glm::radians(m_poses.magEjectRollDegrees) * std::sin(kPi * main.progress()), kZAxis);
    mainPose.translation += shake(main);
    view.items[0] = mainPose;
    const glm::mat4 mainItem = matrixOf(mainPose);
    switch (mainKind) {
    case ItemKind::Shotgun: view.hands[0].pose = poseOf(mainItem * m_poses.rightOnShotgun.matrix()); break;
    case ItemKind::Pistol: view.hands[0].pose = poseOf(mainItem * m_poses.rightOnPistol.matrix()); break;
    case ItemKind::Magazine: view.hands[0].pose = poseOf(mainItem * m_poses.rightOnMagazine.matrix()); break;
    default: view.hands[0].pose = mainPose; break;
    }

    // The off hand: a gun of its own, or moving between the targets of its task.
    const ItemKind offKind = shownItem(Hand::Off);
    HandView& offView = view.hands[1];
    if (isGun(offKind) && off.task != Task::StowOffGun) {
        const float kick = kickCurve(view.shotTime[1]);
        NodePose gun = offKind == ItemKind::Shotgun ? shotgunPose(m_poses, kick, 0.0f, 0.0f, view.sway, m_jacket, -1.0f)
                                                    : pistolPose(m_poses, kick, view.sway, m_jacket, -1.0f);
        if (off.task == Task::DrawOffGun) {
            // Coming up from the hip.
            const float rise = 1.0f - std::clamp((off.time - off.duration * 0.5f) / std::max(off.duration * 0.5f, 1e-3f), 0.0f, 1.0f);
            gun.translation += m_poses.drawRise * rise;
        }
        gun.translation += shake(off);
        view.items[1] = gun;
        const glm::mat4 grip = (offKind == ItemKind::Shotgun ? m_poses.leftOnShotgunGrip : m_poses.rightOnPistol).matrix();
        const NodePose target = poseOf(matrixOf(gun) * grip);
        if (offView.key != 50 + static_cast<int>(offKind)) {
            offView.from = offView.key < 0 ? target : offView.pose;
            offView.key = 50 + static_cast<int>(offKind);
            offView.time = 0.0f;
            offView.duration = 0.12f;
        }
        offView.time += dt;
        offView.pose = easePose(offView.from, target, offView.time / offView.duration);
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
        offView.pose = easePose(offView.from, target, offView.time / offView.duration);
        offView.pose.translation += shake(off);
        // What it holds sits in the palm.
        const glm::mat4 hand = matrixOf(offView.pose);
        switch (offKind) {
        case ItemKind::Shell: view.items[1] = poseOf(hand * m_poses.shellInHand.matrix()); break;
        case ItemKind::Round: view.items[1] = poseOf(hand * m_poses.roundInHand.matrix()); break;
        case ItemKind::Magazine: view.items[1] = poseOf(hand * m_poses.magazineInHand.matrix()); break;
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
    emitModelNodes(model, m_scratchMatrices, world, tint, false, out);
}

void GameWorld::drawShotgun(const glm::mat4& world, float pump, float flash, std::vector<DynamicInstance>& out)
{
    // `world` places the gun node itself.
    const GameModel& model = m_assets.model(GameModelId::Shotgun);
    if (!model.valid())
        return;
    poseShotgun(model.data, m_rigs, m_poses, pump, flash, m_levelTime * 50.0f, m_scratchPose);
    poseToMatrices(model.data.nodes, m_scratchPose, m_scratchMatrices);
    emitModelNodes(model, m_scratchMatrices, world, glm::vec3(1.0f), false, out);
}

void GameWorld::drawPistol(const glm::mat4& world, float slide, float flash, bool withMagazine,
    std::vector<DynamicInstance>& out)
{
    const GameModel& model = m_assets.model(GameModelId::Pistol);
    if (!model.valid())
        return;
    posePistol(model.data, m_rigs, m_poses, slide, flash, m_levelTime * 50.0f, m_scratchPose);
    poseToMatrices(model.data.nodes, m_scratchPose, m_scratchMatrices);
    emitModelNodes(model, m_scratchMatrices, world, glm::vec3(1.0f), false, out);

    const ArmsState& s = m_arms.state();
    if (!withMagazine || s.pistol.magazine < 0)
        return;
    // Sliding out while the release is pressed.
    float drop = 0.0f;
    for (const HandState& hand : s.hands)
        if (hand.task == Task::MagEject && hand.item.kind == ItemKind::Pistol)
            drop = m_poses.magazineDrop * std::clamp(hand.time / std::max(hand.eventAt, 1e-3f), 0.0f, 1.0f);
    drawMagazine(world * m_rigs.pistolMagSeat * offset(glm::vec3(0.0f, -drop, 0.0f)), s.pistol.magazine, out);
}

void GameWorld::drawMagazine(const glm::mat4& world, int magazineId, std::vector<DynamicInstance>& out)
{
    const GameModel& model = m_assets.model(GameModelId::Magazine);
    const Magazine* magazine = m_arms.state().magazine(magazineId);
    if (!model.valid() || !magazine)
        return;
    poseMagazine(model.data, m_rigs, magazine->rounds, m_scratchPose);
    poseToMatrices(model.data.nodes, m_scratchPose, m_scratchMatrices);
    emitModelNodes(model, m_scratchMatrices, world, glm::vec3(1.0f), false, out);
}

void GameWorld::collectArms(const glm::mat4& cameraWorld, std::vector<DynamicInstance>& out)
{
    const ArmsState& s = m_arms.state();
    const float open = m_jacket * m_jacket * (3.0f - 2.0f * m_jacket);
    const auto slideOf = [&](Hand hand) {
        // Kicked back by a shot, pulled by the off hand, or locked open.
        float slide = s.pistol.slideLocked ? m_poses.slideLocked : 0.0f;
        const float shot = m_view.shotTime[static_cast<int>(hand)];
        if (shot < 0.07f)
            slide = std::max(slide, 1.0f - shot / 0.07f);
        const HandState& off = s.hand(Hand::Off);
        if (off.task == Task::Rack && hand == Hand::Main) {
            slide = off.time < off.eventAt
                ? std::max(slide, armsRackPull(m_poses, ArmsTaskTimes::of(off), off.time))
                : (s.pistol.slideLocked ? m_poses.slideLocked : 1.0f - (off.time - off.eventAt) / std::max(off.duration - off.eventAt, 1e-3f));
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
        case ItemKind::Pistol: drawPistol(world, s.pistol.slideLocked ? m_poses.slideLocked : 0.0f, 0.0f, true, out); break;
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
        emitModelNodes(body, m_bodyMatrices, bodyWorld, glm::vec3(1.0f), false, out);
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
            drawPistol(bodyWorld * bodyNode(m_body.holsterGun), s.pistol.slideLocked ? m_poses.slideLocked : 0.0f, 0.0f, true, out);
        // The same for the shotgun on its sling.
        const bool shotgunDrawn = (main.task == Task::Swap && main.target == ItemKind::Shotgun && main.eventDone) ||
            (off.task == Task::DrawOffGun && off.target == ItemKind::Shotgun && off.time >= off.duration * 0.5f);
        if (s.handHolding(ItemKind::Shotgun) < 0 && !s.onGround(ItemKind::Shotgun) && !shotgunDrawn)
            drawShotgun(bodyWorld * m_poses.shotgunSlung.matrix(), 0.0f, 0.0f, out);
    }

    // The hands and what they hold, with the camera.
    const glm::mat4 view = cameraWorld * glm::translate(glm::mat4(1.0f), weaponBob());
    if (const GameModel& hands = m_assets.model(GameModelId::Hands); hands.valid()) {
        poseHands(hands.data, m_rigs, m_view.hands[0].pose, m_view.hands[1].pose, m_scratchPose);
        poseToMatrices(hands.data.nodes, m_scratchPose, m_scratchMatrices);
        emitModelNodes(hands, m_scratchMatrices, view, glm::vec3(1.0f), false, out);
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
        case ItemKind::Shell:
        case ItemKind::Round: {
            // A handful: side by side in the palm (shells lie along X, rounds along Z).
            const bool shell = shownItem(hand) == ItemKind::Shell;
            const glm::vec3 step = shell ? glm::vec3(0.0f, 0.004f, 0.024f) : glm::vec3(0.012f, 0.002f, 0.0f);
            const int count = std::max(state.item.count, 1);
            for (int i = 0; i < count; ++i)
                drawItem(shell ? GameModelId::ShellItem : GameModelId::RoundItem,
                    world * offset(step * (static_cast<float>(i) - 0.5f * static_cast<float>(count - 1))), glm::vec3(1.0f), out);
            break;
        }
        default: break;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Slot binding
// ---------------------------------------------------------------------------------------------

void GameWorld::bindTargets(std::vector<BindTarget>& out) const
{
    out.clear();
    if (!m_player.alive)
        return;
    const ArmsState& s = m_arms.state();
    const bool open = m_jacket > 0.5f;
    const auto bodyPoint = [&](const glm::vec3& point) { return glm::vec3(m_view.bodyWorld * glm::vec4(point, 1.0f)); };
    // The middle of a group of body nodes (shell loops, the cartridge box); false when the model lacks them.
    const auto nodesCenter = [&](const int* nodes, int count, glm::vec3& center) {
        glm::vec3 sum(0.0f);
        int found = 0;
        for (int i = 0; i < count; ++i) {
            if (nodes[i] < 0 || m_bodyMatrices.empty())
                continue;
            sum += glm::vec3(bodyNode(nodes[i])[3]);
            ++found;
        }
        if (found > 0)
            center = bodyPoint(sum / static_cast<float>(found));
        return found > 0;
    };
    const auto add = [&](ItemKind item, Zone zone, const glm::vec3& position) {
        out.push_back({ { item, zone }, position });
    };
    // The guns: in a hand, holstered / slung.
    for (ItemKind gun : { ItemKind::Shotgun, ItemKind::Pistol }) {
        if (const int h = s.handHolding(gun); h >= 0)
            add(gun, Zone::Bandolier, glm::vec3(m_view.cameraWorld * glm::vec4(m_view.items[h].translation, 1.0f)));
        else if (!s.onGround(gun))
            add(gun, Zone::Bandolier, gun == ItemKind::Shotgun || m_body.holsterGun < 0 || m_bodyMatrices.empty()
                ? bodyPoint(kShotgunBindPoint) : bodyPoint(glm::vec3(bodyNode(m_body.holsterGun)[3])));
    }
    glm::vec3 center;
    if (nodesCenter(m_body.band.data(), kBandolierLoops, center))
        add(ItemKind::Shell, Zone::Bandolier, center);
    if (nodesCenter(m_body.bandMags.data(), 2, center))
        add(ItemKind::Magazine, Zone::Bandolier, center);
    if (!open)
        return; // the rest is inside the jacket
    if (nodesCenter(m_body.pocket.data(), kPocketSlots, center))
        add(ItemKind::Shell, Zone::JacketPocket, center);
    if (nodesCenter(m_body.pocketMags.data(), 2, center))
        add(ItemKind::Magazine, Zone::JacketPocket, center);
    if (nodesCenter(m_body.boxRounds.data(), kBoxRounds, center))
        add(ItemKind::Round, Zone::RightPocket, center);
    if (nodesCenter(&m_body.rightPocketMag, 1, center))
        add(ItemKind::Magazine, Zone::RightPocket, center);
}
