#pragma once
#include <array>
#include <span>
#include <vector>
#include <glm/glm.hpp>
#include "engine/Animation.h"
#include "Arms.h"

// A place for a hand or an item: a translation and pitch / yaw / roll in degrees (applied yaw, then pitch,
// then roll, like the camera).
struct PoseEntry {
    glm::vec3 translation{ 0.0f };
    glm::vec3 degrees{ 0.0f }; // pitch, yaw, roll

    NodePose pose() const;
    glm::mat4 matrix() const;
    bool operator==(const PoseEntry&) const = default;
};

NodePose makePose(const glm::vec3& translation, float pitchDegrees, float yawDegrees = 0.0f, float rollDegrees = 0.0f);
glm::mat4 matrixOf(const NodePose& pose);
NodePose poseOf(const glm::mat4& matrix); // drops scale
// Eased move from a to b, t in 0..1 (scale 1).
NodePose easePose(const NodePose& a, const NodePose& b, float t);
glm::mat4 offset(const glm::vec3& translation, float pitchDegrees = 0.0f, float yawDegrees = 0.0f, float rollDegrees = 0.0f);

// The support hand's tasks that move along a track of keys (the rest are posed in code).
enum class ArmsTrackId { Fetch, Store, InsertShell, PushRound, Restock, PickUp, MagInsert, Rack, DrawOffGun, StowOffGun, Swap, Count };
inline constexpr size_t kArmsTrackCount = static_cast<size_t>(ArmsTrackId::Count);
// The track of an Arms task, or Count when it has none.
ArmsTrackId armsTrackOf(Task task);

// What a key's offset is relative to (camera space unless noted):
//   Rest       where the hand rests on what the main hand holds (the pump, the pistol, ...)
//   Ready      `handReady`; Low `handLow`
//   Zone       the task's zone pose (`handBandolier` / `handPocket` / `handRightPocket` on its jacket panel)
//   Bandolier / Pocket / RightPocket   that zone's pose
//   Ground     `handGround`; Hip `handHolster` for the pistol, `handSling` for the shotgun (body space)
//   MainItem   the main hand's item; MainHand the main hand
//   MagSeat    the pistol's magazine seat: the offset moves the magazine, the hand holds its base
//   SlideRear  the pistol's slide as far back as it is now
enum class ArmsAnchor { Rest, Ready, Low, Zone, Bandolier, Pocket, RightPocket, Ground, Hip, MainItem, MainHand, MagSeat, SlideRear, Count };
// A key's time is `at` times the task's event, second event (the first one if it has none) or duration.
enum class ArmsTimeBase { Event, Event2, Duration, Count };

// The hand reaches anchor x offset at its time, moving there (eased) from where it was when the previous
// key was reached. needsZone: only when the task goes to a zone (PushRound from storage).
struct ArmsKey {
    float at = 1.0f;
    ArmsTimeBase base = ArmsTimeBase::Event;
    ArmsAnchor anchor = ArmsAnchor::Rest;
    PoseEntry offset;
    bool needsZone = false;
    bool operator==(const ArmsKey&) const = default;
};
struct ArmsTrack {
    std::vector<ArmsKey> keys;
    bool operator==(const ArmsTrack&) const = default;
};
std::array<ArmsTrack, kArmsTrackCount> defaultArmsTracks();
const char* armsTrackName(ArmsTrackId id);
const char* armsAnchorName(ArmsAnchor anchor);
const char* armsTimeBaseName(ArmsTimeBase base);

// Where the player's hands and items are held and how they move: the first-person "animation" GameArms
// builds from the Arms tasks. Editable in arms_poses.json in the config folder (%APPDATA%\MirasEngine), read
// at every level start and restart; missing keys are written back with these defaults. Camera space is
// +X right, +Y up, looking along -Z; body space is the same at the eyes, turned with the view's yaw only.
struct ArmsPoses {
    // Camera space.
    PoseEntry handReady{ { 0.04f, -0.27f, -0.4f }, { 25.0f, 15.0f, -10.0f } };     // support hand holding something
    PoseEntry handLow{ { -0.15f, -0.42f, -0.3f }, { 10.0f, 0.0f, 0.0f } };
    PoseEntry rightHandLow{ { 0.17f, -0.42f, -0.3f }, { 10.0f, 0.0f, 0.0f } };     // nothing in it
    PoseEntry magazineHeld{ { 0.07f, -0.2f, -0.34f }, { 10.0f, -70.0f, 0.0f } };   // slotted side to the eyes
    // The jacket panels' space (player_body.glb `flap_l`, `flap_r` for the right pocket).
    PoseEntry handBandolier{ { 0.11f, 0.06f, -0.09f }, { -35.0f, 0.0f, 25.0f } };
    PoseEntry handPocket{ { 0.1f, 0.17f, 0.03f }, { -60.0f, 0.0f, 10.0f } };
    PoseEntry handRightPocket{ { -0.11f, 0.12f, 0.03f }, { -60.0f, 0.0f, -10.0f } };
    // Body space.
    PoseEntry handGround{ { 0.0f, -1.45f, -0.5f }, { -80.0f, 0.0f, 0.0f } };
    PoseEntry handHolster{ { 0.22f, -0.8f, -0.05f }, { -70.0f, 0.0f, 0.0f } };
    PoseEntry handSling{ { -0.25f, -0.45f, 0.1f }, { -30.0f, 0.0f, 0.0f } };
    PoseEntry shotgunSlung{ { -0.3f, -0.75f, 0.05f }, { 100.0f, 0.0f, 0.0f } };     // grip at the left hip
    // Small items in the support hand, relative to the palm.
    PoseEntry shellInHand{ { 0.0f, 0.022f, -0.01f } };
    PoseEntry roundInHand{ { 0.0f, 0.02f, -0.01f } };
    PoseEntry magazineInHand{ { 0.0f, 0.115f, -0.005f } };
    // Hands on the items, relative to the item.
    PoseEntry rightOnShotgun{ { 0.0f, -0.085f, 0.065f }, { -18.0f, 0.0f, 0.0f } };
    PoseEntry rightOnPistol{ { 0.0f, 0.0f, 0.0f }, { -15.0f, 0.0f, 0.0f } };
    PoseEntry rightOnMagazine{ { 0.0f, -0.105f, 0.0f } };
    PoseEntry leftSupportsPistol{ { -0.026f, -0.03f, 0.01f }, { -15.0f, 0.0f, 70.0f } };
    PoseEntry leftAtMagazine{ { -4.0f, -4.02f, 4.0f } };                           // out of view
    PoseEntry leftOnShotgunGrip{ { 0.0f, -0.085f, 0.065f }, { -18.0f, 0.0f, 0.0f } };

    // Where the guns are held (camera space): aimed to cross the view's center `convergence` m away.
    glm::vec3 gunRest{ 0.12f, -0.2f, -0.34f };
    glm::vec3 pistolRest{ 0.08f, -0.17f, -0.38f };
    float convergence = 9.0f;
    float shotgunSightHeight = 0.022f; // the barrel's height above the gun node
    float pistolSightHeight = 0.07f;
    // A shot's kick at its peak, the gun lowered while the jacket is open (move + degrees down).
    glm::vec3 shotgunKick{ 0.0f, 0.02f, 0.07f };
    float shotgunKickDegrees = 12.0f;
    glm::vec3 pistolKick{ 0.0f, 0.012f, 0.04f };
    float pistolKickDegrees = 10.0f;
    glm::vec3 shotgunLower{ 0.08f, -0.1f, 0.08f };
    glm::vec3 pistolLower{ 0.06f, -0.1f, 0.08f };
    float lowerDegrees = 30.0f;
    // The shotgun tilted (tilt 1) and pumped (pump 1).
    glm::vec3 shotgunTilt{ -0.03f, -0.03f, 0.04f };
    float shotgunTiltRollDegrees = 28.0f;
    glm::vec3 shotgunPumpMove{ 0.0f, -0.006f, 0.01f };
    float shotgunPumpRollDegrees = -4.0f;
    // Moving parts.
    glm::vec3 pumpRest{ 0.0f, -0.022f, -0.3f }; // the pump node in the gun
    glm::vec3 palmBelowPump{ 0.0f, -0.04f, 0.0f };
    float pumpTravel = 0.11f;
    float slideTravel = 0.05f;
    float slideLocked = 0.7f; // of slideTravel
    float magazineDrop = 0.03f;   // MagEject: slides out this far while the release is pressed
    float magEjectRollDegrees = 15.0f;
    // Swapping guns: the main hand goes down out of view; the off hand's gun comes up from the hip.
    glm::vec3 swapLower{ 0.05f, -0.28f, 0.12f };
    float swapLowerDegrees = -50.0f;
    glm::vec3 drawRise{ -0.05f, -0.28f, 0.12f };
    float jacketOpenDegrees = 72.0f; // opened this far, the panels' insides face the eyes

    // The support hand's tasks, by ArmsTrackId.
    std::array<ArmsTrack, kArmsTrackCount> tracks = defaultArmsTracks();

    bool operator==(const ArmsPoses&) const = default;
};

// Every value of ArmsPoses by name, for the file and the editor.
struct ArmsPoseField {
    const char* name;
    const char* group;
    PoseEntry ArmsPoses::* member;
};
struct ArmsVectorField {
    const char* name;
    const char* group;
    glm::vec3 ArmsPoses::* member;
};
struct ArmsNumberField {
    const char* name;
    const char* group;
    float ArmsPoses::* member;
};
std::span<const ArmsPoseField> armsPoseFields();
std::span<const ArmsVectorField> armsVectorFields();
std::span<const ArmsNumberField> armsNumberFields();

// Reads the file (writing it with the defaults when it doesn't exist or lacks keys).
ArmsPoses loadArmsPoses();
bool saveArmsPoses(const ArmsPoses& poses);

// Node indices in the hands and item models (-1 when missing), and the points read from them.
struct ArmsRigs {
    int shotgunGun = -1, shotgunPump = -1, shotgunFlash = -1;
    int pistolSlide = -1, pistolFlash = -1;
    int magazineFollower = -1;
    std::array<int, 8> magazineRounds{ -1, -1, -1, -1, -1, -1, -1, -1 };
    int handR = -1, handL = -1;
    glm::vec3 shotgunMuzzle{ 0.0f, 0.022f, -0.71f }; // in `gun` space
    glm::vec3 pistolMuzzle{ 0.0f, 0.07f, -0.155f };
    glm::mat4 pistolMagSeat{ 1.0f };

    // Finds the nodes by name; a null or empty model leaves its part unset.
    void find(const AnimatedModelData* shotgun, const AnimatedModelData* pistol, const AnimatedModelData* magazine,
        const AnimatedModelData* hands);
};

// The body's space: at the eyes, turned with the view's yaw only (facing -Z like the camera).
glm::mat4 bodyWorldFrom(const glm::mat4& cameraWorld);

// The guns as held, camera space: aimed to cross the view's center at `convergence`, kicked up by a shot
// (kick 0..1), lagging behind turns (sway, radians), lowered out of the way while the jacket is open
// (lower 0..1). side = 1 for the right hand, -1 mirrors it to the left.
NodePose shotgunPose(const ArmsPoses& p, float kick, float pumpAmount, float tilt, const glm::vec2& sway, float lower,
    float side);
NodePose pistolPose(const ArmsPoses& p, float kick, const glm::vec2& sway, float lower, float side);

// Node poses of the models: the rest pose with the moving parts set. The shotgun is placed by its `gun`
// node; pump and slide 0..1 (slide locked open = slideLocked); flash 0..1 with the flash turned by
// flashSpin radians; rounds shown in the magazine; the hands' palms in the hands model's space.
void poseShotgun(const AnimatedModelData& model, const ArmsRigs& rigs, const ArmsPoses& p, float pump, float flash,
    float flashSpin, std::vector<NodePose>& pose);
void posePistol(const AnimatedModelData& model, const ArmsRigs& rigs, const ArmsPoses& p, float slide, float flash,
    float flashSpin, std::vector<NodePose>& pose);
void poseMagazine(const AnimatedModelData& model, const ArmsRigs& rigs, int rounds, std::vector<NodePose>& pose);
void poseHands(const AnimatedModelData& model, const ArmsRigs& rigs, const NodePose& right, const NodePose& left,
    std::vector<NodePose>& pose);

// A task's timing (from its HandState, or made up by the editor's preview).
struct ArmsTaskTimes {
    float duration = 0.0f;
    float eventAt = 0.0f;
    float event2At = -1.0f;
    bool zoneUsed = false;

    float base(ArmsTimeBase b) const;
    static ArmsTaskTimes of(const HandState& hand);
};

// The anchors of this moment, camera space.
struct ArmsAnchors {
    glm::mat4 rest{ 1.0f };
    int restKey = 0;              // changes with what the rest pose is on, so the hand moves when it does
    glm::mat4 mainItem{ 1.0f };
    glm::mat4 mainHand{ 1.0f };
    glm::mat4 flapL{ 1.0f }, flapR{ 1.0f };
    glm::mat4 bodyToCamera{ 1.0f };
    glm::mat4 magSeat{ 1.0f };    // ArmsRigs::pistolMagSeat
    float slide = 0.0f;           // the pistol's slide, 0..1
    Zone zone = Zone::Bandolier;  // the task's
    bool hipPistol = true;        // Hip: the holster, else the sling
};

// Where the hand heads at time t of the task: the first key not yet reached (the last one once all are).
// key identifies it (a new key starts a new move), phaseLeft is the time until it is reached.
NodePose armsTrackTarget(const ArmsPoses& p, ArmsTrackId track, const ArmsTaskTimes& times, float t,
    const ArmsAnchors& anchors, int& key, float& phaseLeft);
// The pistol's slide while racking: pulled from the Rack track's first key to the event (0..1).
float armsRackPull(const ArmsPoses& p, const ArmsTaskTimes& times, float t);
