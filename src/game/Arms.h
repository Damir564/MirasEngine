#pragma once
#include <array>
#include <string>
#include <vector>

// The player's two hands and everything they can hold: the shotgun, the pistol, its magazines, loose
// shells and rounds, the storage zones on the body and whatever was dropped on the ground.
//
// Pure game logic: no rendering, audio, physics or input devices. GameWorld turns key presses into
// request() calls, advances update() every frame, plays the events it reports (sounds, dropped objects,
// shots) and poses the hands from the tasks' progress. Every request goes through CanPerformAction(),
// which also says why a refused one was refused. Items are never created or destroyed except by firing:
// they move between hands, guns, zones and the ground.

enum class Hand { Main, Off };
enum class ItemKind { Empty, Pistol, Shotgun, Magazine, Shell, Round };
// Body storage. Closer to the support hand = quicker to reach (ZoneConfig::accessTime).
enum class Zone { Bandolier, JacketPocket, RightPocket, Count };
constexpr int kZoneCount = static_cast<int>(Zone::Count);

enum class Chamber { Empty, Loaded, Spent };
enum class PistolState { Holstered, Drawing, Ready, Firing, SlideBack, MagEjecting, AwaitingMag, MagInserting, Racking };

// What a hand is busy with. Each task takes `duration`; its gameplay effect happens at the event point(s)
// (the "animation events": grabbing, the insert frame, the eject frame), from the animation through
// animationEvent() or, as the fallback, when the task's time reaches the configured fraction.
enum class Task {
    None,
    Swap,        // main hand: put the held item away, take another (gun from holster/sling, magazine)
    Fetch,       // off hand: to a zone and back with a shell, round or magazine
    Store,       // off hand: put the held shell, round or magazine into a zone
    InsertShell, // off hand: push the held shell into the shotgun's tube
    PushRound,   // off hand: take a loose round and press it into the magazine in the main hand
    Restock,     // off hand: jacket pocket -> bandolier, one shell per loop (jacket open)
    PickUp,      // off hand: an item from the ground
    Pump,        // off hand on the shotgun's pump
    MagEject,    // main hand: magazine release (the pistol is MagEjecting)
    MagInsert,   // off hand: magazine into the pistol (MagInserting)
    Rack,        // off hand: pulls the pistol's slide (Racking)
    DrawOffGun,  // off hand: takes the other gun (dual wielding)
    StowOffGun,  // off hand: puts it back
};

const char* itemName(ItemKind item);
const char* zoneName(Zone zone);
const char* taskName(Task task);
const char* pistolStateName(PistolState state);
bool isGun(ItemKind item);

struct ZoneConfig {
    float accessTime = 0.5f; // to the zone and back, seconds
    int shells = 0;          // capacities
    int rounds = 0;
    int magazines = 0;
};

// Lets `by` (an ActionKind name, e.g. "Swap") interrupt `task` (a taskName, e.g. "MagInsert"); with
// beforeEvent only until the task's event happened. Whatever the hand carries stays in it.
struct CancelWindow {
    std::string task;
    std::string by;
    bool beforeEvent = true;
};

struct ArmsConfig {
    // Pistol
    int magazineCapacity = 8;
    float pistolFireRate = 360.0f;      // rounds per minute at most (semi-automatic)
    float pistolRecoilDegrees = 3.0f;   // view kick per shot
    float drawTime = 0.35f;             // a gun out of the holster/sling into a hand
    float stowTime = 0.3f;              // and back
    float ejectTime = 0.25f;
    float ejectFrame = 0.4f;            // fraction of ejectTime where the magazine leaves the grip
    float insertTime = 0.45f;
    float insertFrame = 0.7f;           // the magazine clicks in
    float rackTime = 0.35f;
    float rackFrame = 0.5f;             // slide fully back: the chamber empties, then a round goes in
    bool autoRack = false;              // after a magazine goes into a locked-open pistol
    float autoRackDelay = 0.2f;
    float roundInsertTime = 0.3f;       // pressing one loose round into a held magazine (after fetching it)
    float roundInsertFrame = 0.6f;
    float handoffTime = 0.25f;          // a magazine from the off hand to the main hand
    // Shotgun
    int tubeSize = 5;
    float pumpTime = 0.4f;
    float pumpEjectFrame = 0.5f;
    float shellInsertTime = 0.3f;
    float shellInsertFrame = 0.67f;
    float shotgunRecoilDegrees = 3.0f;
    // Hands
    float pickUpTime = 0.5f;
    float pickUpFrame = 0.6f;
    float restockTime = 0.45f;
    bool pumpWhileDualWielding = false; // the pump needs the hand that holds the other gun
    bool forceReloadDropsGun = true;    // the force-reload key drops the off hand's gun to free it
    std::array<ZoneConfig, kZoneCount> zones{ {
        { 0.3f, 8, 0, 2 },   // Bandolier: shell loops and two magazine pouches on the chest
        { 0.8f, 16, 0, 2 },  // JacketPocket: the pouch inside the jacket
        { 0.6f, 0, 30, 1 },  // RightPocket: loose pistol rounds and one magazine
    } };
    std::vector<CancelWindow> cancelWindows{
        { "MagInsert", "Swap", true },     // the magazine stays in the off hand
        { "Fetch", "Pump", false },        // pumping drops what the hand brought
        { "InsertShell", "Pump", true },
        { "Restock", "Pump", false },
        { "PushRound", "Swap", false },
    };
};

struct Magazine {
    int id = 0;
    int capacity = 8;
    int rounds = 0;
};

struct HeldItem {
    ItemKind kind = ItemKind::Empty;
    int magazine = -1; // Magazine id when kind == Magazine
};

struct HandState {
    HeldItem item;
    bool hasOrigin = false; // where the held shell/round/magazine was taken from (put back goes there)
    Zone origin = Zone::Bandolier;
    Task task = Task::None;
    float time = 0.0f;
    float duration = 0.0f;
    float eventAt = 0.0f;   // seconds into the task
    float event2At = -1.0f; // a second event (PushRound: the round goes in); < 0 = none
    bool eventDone = false;
    bool event2Done = false;
    Zone zone = Zone::Bandolier; // Fetch / Store / PushRound / Swap source or target zone
    bool zoneUsed = false;       // Swap: the zone is involved (a magazine to or from storage)
    ItemKind target = ItemKind::Empty; // Swap: what comes into the hand; Fetch: what to take
    int ground = -1;             // PickUp: the ground item
    bool drop = false;           // Swap: the magazine being put away had no room and falls
    float rejected = 0.0f;       // seconds since the last refused action involving this hand (for a shake)

    bool busy() const { return task != Task::None; }
    float progress() const { return duration > 0.0f ? time / duration : 1.0f; }
};

struct ZoneState {
    int shells = 0;
    int rounds = 0;
    std::vector<int> magazines; // ids, in the order they were put in
};

struct GroundItem {
    int id = 0;
    HeldItem item;
};

struct PistolGun {
    PistolState state = PistolState::Holstered;
    bool chambered = true;
    int magazine = -1;       // inserted magazine id
    bool slideLocked = false;
    float time = 0.0f;       // in Firing
    float rackDelay = -1.0f; // auto-rack countdown; < 0 = none
};

struct ShotgunGun {
    Chamber chamber = Chamber::Loaded;
    int tube = 5;
};

enum class ActionKind {
    FireMain,     // trigger of the gun in the main hand
    FireOff,      // trigger of the gun in the off hand
    Pump,
    Rack,
    Reload,       // context: shell in, magazine in/out/fetch, round into the held magazine
    ForceReload,  // like Reload, but frees the off hand by dropping its gun (config flag)
    Unload,       // pistol: magazine out
    Load,         // pistol: magazine from the off hand in
    Fetch,        // zone -> off hand
    Store,        // off hand -> zone
    ZoneButton,   // Fetch with an empty hand, Store with a full one
    PutBack,      // the held item back where it came from
    InsertShell,
    PushRound,
    Swap,         // main hand takes `item`
    ToggleOffGun, // the other gun into / out of the off hand
    PickUp,
    Restock,
};
const char* actionName(ActionKind action);

struct Action {
    ActionKind kind = ActionKind::FireMain;
    Zone zone = Zone::Bandolier;
    ItemKind item = ItemKind::Empty;
    int ground = -1;
};

struct Verdict {
    bool ok = false;
    std::string reason; // why not; empty = refused without a cue (e.g. the trigger while already firing)
    Action resolved;    // the concrete action (Reload/ZoneButton resolve to another)
};

enum class EventType {
    Fired,         // item: the gun; hand: which hand
    DryClick,
    MagEject,
    MagInsert,
    SlideRack,
    SlideLock,     // the last round went: the slide stays back
    Pumped,        // the pump went back
    ShellEjected,  // a spent shell left the port (cosmetic)
    ShellInsert,
    RoundInsert,
    Grab,          // the off hand took something from a zone; fill: how full the zone is now (0..1)
    ZoneEmpty,     // ...or found nothing
    Stored,
    Dropped,       // item fell: `ground` is the new ground item
    PickedUp,
    Drawn,
    Holstered,
    Rejected,      // reason, hand
};

struct ArmsEvent {
    EventType type = EventType::Fired;
    Hand hand = Hand::Main;
    ItemKind item = ItemKind::Empty;
    Zone zone = Zone::Bandolier;
    int ground = -1;
    float fill = 0.0f;
    std::string reason;
};

struct ArmsState {
    std::array<HandState, 2> hands;
    PistolGun pistol;
    ShotgunGun shotgun;
    std::vector<Magazine> magazines;
    std::array<ZoneState, kZoneCount> zones;
    std::vector<GroundItem> ground;
    int nextGroundId = 1;
    int roundsFired = 0;
    int shellsFired = 0;
    bool pendingShellInsert = false; // R while the shell is still on its way
    bool restockHeld = false;

    HandState& hand(Hand h) { return hands[static_cast<int>(h)]; }
    const HandState& hand(Hand h) const { return hands[static_cast<int>(h)]; }
    Magazine* magazine(int id);
    const Magazine* magazine(int id) const;
    const GroundItem* groundItem(int id) const;
    // Which hand holds a gun, or -1 when it isn't in a hand.
    int handHolding(ItemKind gun) const;
    bool onGround(ItemKind gun) const;
    bool bothHandsHoldGuns() const;
};

// The one place that decides whether an action is allowed now. Contextual actions (Reload, ZoneButton)
// come back resolved to the concrete action that would run.
Verdict CanPerformAction(const ArmsState& state, const ArmsConfig& config, const Action& action);

class Arms {
public:
    explicit Arms(const ArmsConfig& config = {});

    // The starting loadout: shotgun in the main hand (loaded), pistol holstered (loaded), magazines and
    // loose ammunition in the zones.
    void reset();
    void setConfig(const ArmsConfig& config) { m_config = config; }
    const ArmsConfig& config() const { return m_config; }

    // Validates and starts the action; a refused one reports a Rejected event. Returns whether it ran.
    bool request(const Action& action);
    // Advances the tasks.
    void update(float dt);
    // The animation reached a task's event (insert frame, eject frame, grab...) before its timer did.
    void animationEvent(Hand hand);
    // Restocking continues while this is held.
    void setRestockHeld(bool held) { m_state.restockHeld = held; }

    // Contextual buttons as they would resolve now: the secondary button fires the off hand's gun, or
    // pumps / racks the main hand's.
    ActionKind secondaryAction() const;

    std::vector<ArmsEvent> takeEvents();
    const ArmsState& state() const { return m_state; }
    ArmsState& mutableState() { return m_state; } // tests and debugging
    const std::string& lastRejection() const { return m_lastRejection; }

    // Bookkeeping, for tests: everything currently in the world.
    int totalRounds() const;  // in magazines, chambered, loose in zones, in hands, on the ground
    int totalShells() const;  // tube, chamber (loaded), zones, hands, ground
    int magazineCount() const; // distinct magazines found in the world (each must be exactly one place)
    bool magazinesAccountedFor() const;

private:
    void perform(const Action& action);
    void startTask(Hand hand, Task task, float duration, float eventFraction);
    void finishTask(Hand hand);
    void runEvent(Hand hand);
    void runEvent2(Hand hand);
    void abortTask(Hand hand);
    void fire(Hand hand);
    int drop(Hand hand, HeldItem item);
    void emit(EventType type, Hand hand, ItemKind item = ItemKind::Empty);
    void settlePistol();
    void startRack();
    bool storeInto(Zone zone, const HeldItem& item); // false when it doesn't fit
    int zoneFor(const HeldItem& item, int preferred) const; // a zone with room, or -1
    float accessTime(Zone zone) const { return m_config.zones[static_cast<int>(zone)].accessTime; }

    ArmsConfig m_config;
    ArmsState m_state;
    std::vector<ArmsEvent> m_events;
    std::string m_lastRejection;
};
