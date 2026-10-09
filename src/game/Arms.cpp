#include "Arms.h"
#include <algorithm>
#include <initializer_list>
#include <utility>

namespace {

bool storable(ItemKind kind)
{
    return kind == ItemKind::Shell || kind == ItemKind::Round || kind == ItemKind::Magazine;
}

int capacityFor(const ZoneConfig& zone, ItemKind kind)
{
    switch (kind) {
    case ItemKind::Shell: return zone.shells;
    case ItemKind::Round: return zone.rounds;
    case ItemKind::Magazine: return zone.magazines;
    default: return 0;
    }
}

int countIn(const ZoneState& zone, ItemKind kind)
{
    switch (kind) {
    case ItemKind::Shell: return zone.shells;
    case ItemKind::Round: return zone.rounds;
    case ItemKind::Magazine: return static_cast<int>(zone.magazines.size());
    default: return 0;
    }
}

const char* plural(ItemKind kind)
{
    switch (kind) {
    case ItemKind::Shell: return "shells";
    case ItemKind::Round: return "rounds";
    case ItemKind::Magazine: return "magazines";
    default: return "that";
    }
}

std::string the(const char* name)
{
    return std::string("The ") + name;
}

bool cancelAllowed(const ArmsConfig& config, const HandState& hand, ActionKind by)
{
    if (!hand.busy())
        return true;
    for (const CancelWindow& window : config.cancelWindows)
        if (window.task == taskName(hand.task) && window.by == actionName(by) && (!window.beforeEvent || !hand.eventDone))
            return true;
    return false;
}

// Zones sorted by how quickly the hand gets there.
std::array<int, kZoneCount> zonesByAccess(const ArmsConfig& config)
{
    std::array<int, kZoneCount> order{};
    for (int i = 0; i < kZoneCount; ++i)
        order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&](int a, int b) { return config.zones[a].accessTime < config.zones[b].accessTime; });
    return order;
}

// The preferred zone if the item fits there, else the quickest one it fits in; -1 when none has room.
int zoneWithRoom(const ArmsState& state, const ArmsConfig& config, ItemKind kind, int preferred)
{
    const auto fits = [&](int zone) {
        return countIn(state.zones[zone], kind) < capacityFor(config.zones[zone], kind);
    };
    if (preferred >= 0 && fits(preferred))
        return preferred;
    for (int zone : zonesByAccess(config))
        if (fits(zone))
            return zone;
    return -1;
}

// The quickest zone holding a magazine with rounds in it (the fullest one there); -1 when none.
int zoneWithLoadedMagazine(const ArmsState& state, const ArmsConfig& config)
{
    for (int zone : zonesByAccess(config))
        for (int id : state.zones[zone].magazines)
            if (const Magazine* magazine = state.magazine(id); magazine && magazine->rounds > 0)
                return zone;
    return -1;
}

int zoneWithRounds(const ArmsState& state, const ArmsConfig& config)
{
    for (int zone : zonesByAccess(config))
        if (state.zones[zone].rounds > 0)
            return zone;
    return -1;
}

// For loading rounds: the stored magazine with the most room (quickest zone on ties); -1 when none.
int zoneWithEmptiestMagazine(const ArmsState& state, const ArmsConfig& config)
{
    int best = -1, bestRounds = 0;
    for (int zone : zonesByAccess(config)) {
        for (int id : state.zones[zone].magazines) {
            const Magazine* magazine = state.magazine(id);
            if (magazine && (best < 0 || magazine->rounds < bestRounds)) {
                best = zone;
                bestRounds = magazine->rounds;
            }
        }
    }
    return best;
}

// Tasks of the off hand that work on the main hand's item; swapping that item away must not leave them
// hanging.
bool assistsMainHand(Task task)
{
    return task == Task::InsertShell || task == Task::Pump || task == Task::Rack || task == Task::MagInsert ||
        task == Task::PushRound;
}

bool pistolIdle(PistolState state)
{
    return state == PistolState::Ready || state == PistolState::SlideBack || state == PistolState::AwaitingMag;
}

} // namespace

const char* itemName(ItemKind item)
{
    switch (item) {
    case ItemKind::Empty: return "nothing";
    case ItemKind::Pistol: return "pistol";
    case ItemKind::Shotgun: return "shotgun";
    case ItemKind::Magazine: return "magazine";
    case ItemKind::Shell: return "shell";
    case ItemKind::Round: return "round";
    }
    return "?";
}

const char* zoneName(Zone zone)
{
    switch (zone) {
    case Zone::Bandolier: return "bandolier";
    case Zone::JacketPocket: return "jacket pocket";
    case Zone::RightPocket: return "right pocket";
    default: return "?";
    }
}

const char* taskName(Task task)
{
    switch (task) {
    case Task::None: return "None";
    case Task::Swap: return "Swap";
    case Task::Fetch: return "Fetch";
    case Task::Store: return "Store";
    case Task::InsertShell: return "InsertShell";
    case Task::PushRound: return "PushRound";
    case Task::Restock: return "Restock";
    case Task::PickUp: return "PickUp";
    case Task::Pump: return "Pump";
    case Task::MagEject: return "MagEject";
    case Task::MagInsert: return "MagInsert";
    case Task::Rack: return "Rack";
    case Task::DrawOffGun: return "DrawOffGun";
    case Task::StowOffGun: return "StowOffGun";
    }
    return "?";
}

const char* pistolStateName(PistolState state)
{
    switch (state) {
    case PistolState::Holstered: return "Holstered";
    case PistolState::Drawing: return "Drawing";
    case PistolState::Ready: return "Ready";
    case PistolState::Firing: return "Firing";
    case PistolState::SlideBack: return "SlideBack";
    case PistolState::MagEjecting: return "MagEjecting";
    case PistolState::AwaitingMag: return "AwaitingMag";
    case PistolState::MagInserting: return "MagInserting";
    case PistolState::Racking: return "Racking";
    }
    return "?";
}

const char* actionName(ActionKind action)
{
    switch (action) {
    case ActionKind::FireMain: return "FireMain";
    case ActionKind::FireOff: return "FireOff";
    case ActionKind::Pump: return "Pump";
    case ActionKind::Rack: return "Rack";
    case ActionKind::Reload: return "Reload";
    case ActionKind::ForceReload: return "ForceReload";
    case ActionKind::Unload: return "Unload";
    case ActionKind::Load: return "Load";
    case ActionKind::Fetch: return "Fetch";
    case ActionKind::Store: return "Store";
    case ActionKind::ZoneButton: return "ZoneButton";
    case ActionKind::PutBack: return "PutBack";
    case ActionKind::InsertShell: return "InsertShell";
    case ActionKind::PushRound: return "PushRound";
    case ActionKind::Swap: return "Swap";
    case ActionKind::ToggleOffGun: return "ToggleOffGun";
    case ActionKind::PickUp: return "PickUp";
    case ActionKind::Restock: return "Restock";
    }
    return "?";
}

bool isGun(ItemKind item)
{
    return item == ItemKind::Pistol || item == ItemKind::Shotgun;
}

// ---------------------------------------------------------------------------------------------
// State queries
// ---------------------------------------------------------------------------------------------

Magazine* ArmsState::magazine(int id)
{
    for (Magazine& magazine : magazines)
        if (magazine.id == id)
            return &magazine;
    return nullptr;
}

const Magazine* ArmsState::magazine(int id) const
{
    return const_cast<ArmsState*>(this)->magazine(id);
}

const GroundItem* ArmsState::groundItem(int id) const
{
    for (const GroundItem& item : ground)
        if (item.id == id)
            return &item;
    return nullptr;
}

int ArmsState::handHolding(ItemKind gun) const
{
    for (int h = 0; h < 2; ++h)
        if (hands[h].item.kind == gun)
            return h;
    return -1;
}

bool ArmsState::onGround(ItemKind gun) const
{
    return std::any_of(ground.begin(), ground.end(), [&](const GroundItem& item) { return item.item.kind == gun; });
}

bool ArmsState::bothHandsHoldGuns() const
{
    return isGun(hands[0].item.kind) && isGun(hands[1].item.kind);
}

// ---------------------------------------------------------------------------------------------
// The rules
// ---------------------------------------------------------------------------------------------

Verdict CanPerformAction(const ArmsState& s, const ArmsConfig& c, const Action& a)
{
    Verdict v;
    v.resolved = a;
    const auto no = [&](std::string reason) {
        v.ok = false;
        v.reason = std::move(reason);
        return v;
    };
    const auto yes = [&](const Action& resolved) {
        v.ok = true;
        v.reason.clear();
        v.resolved = resolved;
        return v;
    };
    const auto with = [&](ActionKind kind) {
        Action resolved = a;
        resolved.kind = kind;
        return CanPerformAction(s, c, resolved);
    };
    const HandState& main = s.hand(Hand::Main);
    const HandState& off = s.hand(Hand::Off);
    const bool bothGuns = s.bothHandsHoldGuns();
    const char* busy = "The other hand is busy";
    const PistolGun& pistol = s.pistol;

    switch (a.kind) {
    case ActionKind::FireMain:
    case ActionKind::FireOff: {
        const bool mainHand = a.kind == ActionKind::FireMain;
        const HandState& hand = mainHand ? main : off;
        if (!isGun(hand.item.kind))
            return no(mainHand ? "No gun in hand" : "");
        if (mainHand ? main.task == Task::Swap : (off.task == Task::DrawOffGun || off.task == Task::StowOffGun))
            return no("");
        if (hand.item.kind == ItemKind::Shotgun)
            return off.task == Task::Pump ? no("") : yes(a);
        if (pistolIdle(pistol.state))
            return yes(a); // a dry click when nothing is chambered
        return no(pistol.state == PistolState::Firing ? "" : "The pistol is busy");
    }

    case ActionKind::Pump:
        if (main.item.kind != ItemKind::Shotgun)
            return no("No shotgun in hand");
        if (main.busy())
            return no("The hand is busy");
        if (off.task == Task::Pump)
            return no("");
        if (isGun(off.item.kind) && !c.pumpWhileDualWielding)
            return no("Pumping needs both hands");
        if (off.item.kind == ItemKind::Magazine)
            return no("Pumping needs the other hand: it holds a magazine");
        if (!cancelAllowed(c, off, a.kind))
            return no(busy);
        return yes(a);

    case ActionKind::Rack:
        if (main.item.kind != ItemKind::Pistol)
            return no("No pistol in hand");
        if (bothGuns)
            return no("Both hands hold guns");
        if (main.busy() || !pistolIdle(pistol.state))
            return no("The pistol is busy");
        if (off.item.kind != ItemKind::Empty)
            return no("Racking needs the other hand free");
        if (off.busy())
            return no(busy);
        return yes(a);

    case ActionKind::Unload:
        if (main.item.kind != ItemKind::Pistol)
            return no(main.item.kind == ItemKind::Empty ? "No gun in hand" : "Only the pistol has a magazine");
        if (bothGuns)
            return no("Both hands hold guns");
        if (pistol.magazine < 0)
            return no("No magazine in the pistol");
        if (main.busy() || !pistolIdle(pistol.state))
            return no("The pistol is busy");
        return yes(a);

    case ActionKind::Load:
        if (main.item.kind != ItemKind::Pistol)
            return no("No pistol in hand");
        if (bothGuns)
            return no("Both hands hold guns");
        if (off.item.kind != ItemKind::Magazine)
            return no("No magazine in the other hand");
        if (pistol.magazine >= 0)
            return no("A magazine is already in");
        if (main.busy() || pistol.state != PistolState::AwaitingMag)
            return no("The pistol is busy");
        if (off.busy())
            return no(busy);
        return yes(a);

    case ActionKind::Reload: {
        if (bothGuns)
            return no("Both hands hold guns");
        switch (main.item.kind) {
        case ItemKind::Shotgun:
            return with(ActionKind::InsertShell);
        case ItemKind::Magazine:
            return with(ActionKind::PushRound);
        case ItemKind::Pistol: {
            if (main.busy() || !pistolIdle(pistol.state))
                return no("The pistol is busy");
            if (pistol.magazine >= 0) {
                // Locked open (or nothing chambered) with rounds left: rack; otherwise out with the magazine.
                const Magazine* magazine = s.magazine(pistol.magazine);
                if ((pistol.slideLocked || !pistol.chambered) && magazine && magazine->rounds > 0)
                    return with(ActionKind::Rack);
                return with(ActionKind::Unload);
            }
            if (off.item.kind == ItemKind::Magazine) {
                const Magazine* magazine = s.magazine(off.item.magazine);
                // An empty one goes back into storage first.
                return with(magazine && magazine->rounds > 0 ? ActionKind::Load : ActionKind::PutBack);
            }
            if (off.item.kind != ItemKind::Empty)
                return no("The other hand is full");
            if (off.busy())
                return no(busy);
            const int zone = zoneWithLoadedMagazine(s, c);
            if (zone < 0)
                return no("No loaded magazine left");
            Action fetch = a;
            fetch.kind = ActionKind::Fetch;
            fetch.zone = static_cast<Zone>(zone);
            fetch.item = ItemKind::Magazine;
            return CanPerformAction(s, c, fetch);
        }
        default:
            return no("Nothing to reload");
        }
    }

    case ActionKind::ForceReload: {
        if (!bothGuns)
            return with(ActionKind::Reload);
        if (!c.forceReloadDropsGun)
            return no("Both hands hold guns");
        // Only worth dropping the gun if the reload can then go ahead.
        ArmsState freed = s;
        freed.hand(Hand::Off) = {};
        Action reload = a;
        reload.kind = ActionKind::Reload;
        const Verdict after = CanPerformAction(freed, c, reload);
        if (!after.ok)
            return after;
        return yes(a);
    }

    case ActionKind::Fetch:
        if (isGun(off.item.kind))
            return no(the(itemName(off.item.kind)) + " is in the other hand");
        if (off.item.kind != ItemKind::Empty)
            return no("The other hand is full");
        if (off.busy())
            return no(busy);
        return yes(a); // an empty zone: the hand comes back empty, which is the signal

    case ActionKind::Store: {
        if (!storable(off.item.kind))
            return no(off.item.kind == ItemKind::Empty ? "Nothing in hand to put away" : "That doesn't go in a pocket");
        if (off.busy())
            return no(busy);
        const int zone = static_cast<int>(a.zone);
        const int capacity = capacityFor(c.zones[zone], off.item.kind);
        if (capacity <= 0)
            return no(the(zoneName(a.zone)) + " doesn't take " + plural(off.item.kind));
        if (countIn(s.zones[zone], off.item.kind) >= capacity)
            return no(the(zoneName(a.zone)) + " is full");
        return yes(a);
    }

    case ActionKind::ZoneButton:
        return with(storable(off.item.kind) ? ActionKind::Store : ActionKind::Fetch);

    case ActionKind::PutBack: {
        if (!storable(off.item.kind))
            return no("Nothing in hand to put away");
        if (off.busy())
            return no(busy);
        const int zone = zoneWithRoom(s, c, off.item.kind, off.hasOrigin ? static_cast<int>(off.origin) : -1);
        if (zone < 0)
            return no("No room to put it away");
        Action store = a;
        store.kind = ActionKind::Store;
        store.zone = static_cast<Zone>(zone);
        return yes(store);
    }

    case ActionKind::InsertShell:
        if (main.item.kind != ItemKind::Shotgun)
            return no("No shotgun in hand");
        if (bothGuns)
            return no("Both hands hold guns");
        if (s.shotgun.tube >= c.tubeSize)
            return no("The tube is full");
        if (off.task == Task::Fetch && off.target == ItemKind::Shell && !off.eventDone)
            return yes(a); // goes in when it arrives
        if (off.item.kind != ItemKind::Shell)
            return no("No shell in hand: take one with Q / E");
        if (off.busy() || main.busy())
            return no(busy);
        return yes(a);

    case ActionKind::PushRound: {
        if (main.item.kind != ItemKind::Magazine)
            return no("No magazine in hand");
        const Magazine* magazine = s.magazine(main.item.magazine);
        if (!magazine || magazine->rounds >= magazine->capacity)
            return no("The magazine is full");
        if (main.busy() || off.busy())
            return no(busy);
        if (off.item.kind == ItemKind::Round)
            return yes(a);
        if (off.item.kind != ItemKind::Empty)
            return no("The other hand is full");
        const int zone = zoneWithRounds(s, c);
        if (zone < 0)
            return no("No loose rounds left");
        Action push = a;
        push.zone = static_cast<Zone>(zone);
        return yes(push);
    }

    case ActionKind::Swap: {
        if (a.item == main.item.kind)
            return no("");
        if (!isGun(a.item) && a.item != ItemKind::Magazine)
            return no("");
        if (main.busy())
            return no("The hand is busy");
        if (off.busy() && assistsMainHand(off.task) && !cancelAllowed(c, off, a.kind))
            return no(busy);
        if (isGun(a.item)) {
            if (s.onGround(a.item))
                return no(the(itemName(a.item)) + " is on the ground");
            if (off.task == Task::DrawOffGun && off.target == a.item)
                return no(the(itemName(a.item)) + " is going into the other hand");
            if (s.handHolding(a.item) == static_cast<int>(Hand::Off))
                return no(the(itemName(a.item)) + " is in the other hand");
            return yes(a);
        }
        // A magazine: the one in the other hand, or a stored one.
        if (off.item.kind == ItemKind::Magazine && (!off.busy() || off.task == Task::MagInsert))
            return yes(a);
        if (zoneWithEmptiestMagazine(s, c) < 0)
            return no("No spare magazine");
        return yes(a);
    }

    case ActionKind::ToggleOffGun: {
        if (off.busy())
            return no(busy);
        if (isGun(off.item.kind))
            return yes(a);
        if (off.item.kind != ItemKind::Empty)
            return no("The other hand is full");
        const ItemKind other = main.item.kind == ItemKind::Pistol ? ItemKind::Shotgun : ItemKind::Pistol;
        if (s.onGround(other))
            return no(the(itemName(other)) + " is on the ground");
        if (main.item.kind == other || (main.task == Task::Swap && main.target == other))
            return no(the(itemName(other)) + " is in the other hand");
        Action draw = a;
        draw.item = other;
        return yes(draw);
    }

    case ActionKind::PickUp:
        if (!s.groundItem(a.ground))
            return no("Nothing to pick up");
        if (off.item.kind != ItemKind::Empty)
            return no("The other hand is full");
        if (off.busy())
            return no(busy);
        return yes(a);

    case ActionKind::Restock:
        if (off.item.kind != ItemKind::Empty || off.busy())
            return no(busy);
        if (s.zones[static_cast<int>(Zone::Bandolier)].shells >= c.zones[static_cast<int>(Zone::Bandolier)].shells)
            return no("The bandolier is full");
        if (s.zones[static_cast<int>(Zone::JacketPocket)].shells <= 0)
            return no("The jacket pocket has no shells");
        return yes(a);
    }
    return no("");
}

// ---------------------------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------------------------

Arms::Arms(const ArmsConfig& config) : m_config(config)
{
    reset();
}

void Arms::reset()
{
    m_state = {};
    m_events.clear();
    m_lastRejection.clear();
    const int capacity = std::max(m_config.magazineCapacity, 1);
    m_state.magazines = { { 1, capacity, capacity }, { 2, capacity, capacity }, { 3, capacity, capacity },
        { 4, capacity, 0 } };
    m_state.hand(Hand::Main).item = { ItemKind::Shotgun, -1 };
    m_state.shotgun.chamber = Chamber::Loaded;
    m_state.shotgun.tube = m_config.tubeSize;
    m_state.pistol.state = PistolState::Holstered;
    m_state.pistol.chambered = true;
    m_state.pistol.magazine = 1;

    const auto fill = [&](Zone zone, int shells, int rounds, std::vector<int> magazines) {
        ZoneState& z = m_state.zones[static_cast<int>(zone)];
        const ZoneConfig& capacityOf = m_config.zones[static_cast<int>(zone)];
        z.shells = std::min(shells, capacityOf.shells);
        z.rounds = std::min(rounds, capacityOf.rounds);
        for (int id : magazines) {
            // A magazine that doesn't fit anywhere stays out of the game rather than vanishing silently.
            const int target = static_cast<int>(z.magazines.size()) < capacityOf.magazines ? static_cast<int>(zone)
                : zoneWithRoom(m_state, m_config, ItemKind::Magazine, -1);
            if (target >= 0)
                m_state.zones[target].magazines.push_back(id);
            else
                m_state.ground.push_back({ m_state.nextGroundId++, { ItemKind::Magazine, id } });
        }
    };
    fill(Zone::Bandolier, 8, 0, { 2 });
    fill(Zone::JacketPocket, 8, 0, { 3 });
    fill(Zone::RightPocket, 0, 16, { 4 });
}

ActionKind Arms::secondaryAction() const
{
    const HandState& main = m_state.hand(Hand::Main);
    if (isGun(m_state.hand(Hand::Off).item.kind))
        return ActionKind::FireOff;
    return main.item.kind == ItemKind::Pistol ? ActionKind::Rack : ActionKind::Pump;
}

bool Arms::request(const Action& action)
{
    const Verdict verdict = CanPerformAction(m_state, m_config, action);
    if (!verdict.ok) {
        if (!verdict.reason.empty()) {
            const bool offHand = action.kind == ActionKind::FireOff || action.kind == ActionKind::Fetch ||
                action.kind == ActionKind::Store || action.kind == ActionKind::ZoneButton ||
                action.kind == ActionKind::PutBack || action.kind == ActionKind::PickUp ||
                action.kind == ActionKind::ToggleOffGun || action.kind == ActionKind::Restock ||
                action.kind == ActionKind::Pump || action.kind == ActionKind::Rack;
            const Hand hand = offHand ? Hand::Off : Hand::Main;
            m_state.hand(hand).rejected = 0.0f;
            m_lastRejection = std::string(actionName(action.kind)) + ": " + verdict.reason;
            ArmsEvent event;
            event.type = EventType::Rejected;
            event.hand = hand;
            event.reason = verdict.reason;
            m_events.push_back(event);
        }
        return false;
    }
    perform(verdict.resolved);
    return true;
}

void Arms::emit(EventType type, Hand hand, ItemKind item)
{
    ArmsEvent event;
    event.type = type;
    event.hand = hand;
    event.item = item;
    m_events.push_back(event);
}

std::vector<ArmsEvent> Arms::takeEvents()
{
    return std::exchange(m_events, {});
}

void Arms::startTask(Hand hand, Task task, float duration, float eventFraction)
{
    HandState& h = m_state.hand(hand);
    h.task = task;
    h.time = 0.0f;
    h.duration = std::max(duration, 0.0f);
    h.eventAt = h.duration * std::clamp(eventFraction, 0.0f, 1.0f);
    h.event2At = -1.0f;
    h.eventDone = false;
    h.event2Done = false;
}

int Arms::drop(Hand hand, HeldItem item)
{
    GroundItem ground;
    ground.id = m_state.nextGroundId++;
    ground.item = item;
    m_state.ground.push_back(ground);
    ArmsEvent event;
    event.type = EventType::Dropped;
    event.hand = hand;
    event.item = item.kind;
    event.ground = ground.id;
    m_events.push_back(event);
    if (item.kind == ItemKind::Pistol)
        settlePistol();
    return ground.id;
}

bool Arms::storeInto(Zone zone, const HeldItem& item)
{
    const int z = static_cast<int>(zone);
    if (countIn(m_state.zones[z], item.kind) >= capacityFor(m_config.zones[z], item.kind))
        return false;
    ZoneState& state = m_state.zones[z];
    switch (item.kind) {
    case ItemKind::Shell: ++state.shells; break;
    case ItemKind::Round: ++state.rounds; break;
    case ItemKind::Magazine: state.magazines.push_back(item.magazine); break;
    default: return false;
    }
    return true;
}

int Arms::zoneFor(const HeldItem& item, int preferred) const
{
    return zoneWithRoom(m_state, m_config, item.kind, preferred);
}

void Arms::settlePistol()
{
    PistolGun& pistol = m_state.pistol;
    if (m_state.handHolding(ItemKind::Pistol) < 0) {
        pistol.state = PistolState::Holstered; // holstered, or lying on the ground
        return;
    }
    pistol.state = pistol.magazine < 0 ? PistolState::AwaitingMag
        : pistol.slideLocked ? PistolState::SlideBack : PistolState::Ready;
}

void Arms::startRack()
{
    startTask(Hand::Off, Task::Rack, m_config.rackTime, m_config.rackFrame);
    m_state.pistol.state = PistolState::Racking;
    m_state.pistol.rackDelay = -1.0f;
}

void Arms::perform(const Action& action)
{
    ArmsState& s = m_state;
    HandState& main = s.hand(Hand::Main);
    HandState& off = s.hand(Hand::Off);
    switch (action.kind) {
    case ActionKind::FireMain:
        fire(Hand::Main);
        break;
    case ActionKind::FireOff:
        fire(Hand::Off);
        break;

    case ActionKind::Pump:
        abortTask(Hand::Off);
        // The hand goes back to the pump: whatever small thing it held falls.
        if (off.item.kind == ItemKind::Shell || off.item.kind == ItemKind::Round) {
            const HeldItem held = off.item;
            off.item = {};
            off.hasOrigin = false;
            drop(Hand::Off, held);
        }
        startTask(Hand::Off, Task::Pump, m_config.pumpTime, m_config.pumpEjectFrame);
        emit(EventType::Pumped, Hand::Off, ItemKind::Shotgun);
        break;

    case ActionKind::Rack:
        startRack();
        break;

    case ActionKind::Unload:
        startTask(Hand::Main, Task::MagEject, m_config.ejectTime, m_config.ejectFrame);
        s.pistol.state = PistolState::MagEjecting;
        s.pistol.rackDelay = -1.0f;
        break;

    case ActionKind::Load:
        startTask(Hand::Off, Task::MagInsert, m_config.insertTime, m_config.insertFrame);
        s.pistol.state = PistolState::MagInserting;
        break;

    case ActionKind::ForceReload:
        if (s.bothHandsHoldGuns()) {
            // Free the hand by letting its gun fall; it stays on the ground to be picked up.
            abortTask(Hand::Off);
            const HeldItem gun = off.item;
            off.item = {};
            drop(Hand::Off, gun);
        }
        request({ ActionKind::Reload });
        break;

    case ActionKind::Fetch:
        startTask(Hand::Off, Task::Fetch, accessTime(action.zone), 0.5f);
        off.zone = action.zone;
        // What the hand looks for first; it takes something else from the zone if that's all there is.
        off.target = action.item != ItemKind::Empty ? action.item
            : main.item.kind == ItemKind::Pistol ? ItemKind::Magazine
            : main.item.kind == ItemKind::Magazine ? ItemKind::Round
            : ItemKind::Shell;
        break;

    case ActionKind::Store:
        startTask(Hand::Off, Task::Store, accessTime(action.zone), 0.5f);
        off.zone = action.zone;
        break;

    case ActionKind::InsertShell:
        if (off.task == Task::Fetch)
            s.pendingShellInsert = true;
        else
            startTask(Hand::Off, Task::InsertShell, m_config.shellInsertTime, m_config.shellInsertFrame);
        break;

    case ActionKind::PushRound:
        if (off.item.kind == ItemKind::Round) {
            startTask(Hand::Off, Task::PushRound, m_config.roundInsertTime, m_config.roundInsertFrame);
            off.zoneUsed = false;
        }
        else {
            // To the rounds and back (half the access time each way), then pressed in.
            const float access = accessTime(action.zone);
            startTask(Hand::Off, Task::PushRound, access + m_config.roundInsertTime, 0.0f);
            off.eventAt = access * 0.5f;
            off.event2At = access + m_config.roundInsertTime * m_config.roundInsertFrame;
            off.zone = action.zone;
            off.zoneUsed = true;
        }
        break;

    case ActionKind::Swap: {
        if (off.busy() && assistsMainHand(off.task))
            abortTask(Hand::Off);
        // Put away what the hand holds, then take the new item.
        float putAway = 0.0f;
        main.zoneUsed = false;
        main.drop = false;
        if (isGun(main.item.kind)) {
            putAway = m_config.stowTime;
        }
        else if (main.item.kind == ItemKind::Magazine) {
            const int zone = zoneFor(main.item, -1);
            main.drop = zone < 0; // no room anywhere: it falls
            main.zone = zone < 0 ? Zone::Bandolier : static_cast<Zone>(zone);
            main.zoneUsed = zone >= 0;
            putAway = zone < 0 ? 0.1f : accessTime(main.zone);
        }
        float take = m_config.drawTime;
        bool handoff = false;
        if (action.item == ItemKind::Magazine) {
            if (off.item.kind == ItemKind::Magazine) {
                abortTask(Hand::Off);
                take = m_config.handoffTime;
                handoff = true;
            }
            else {
                main.zone = static_cast<Zone>(zoneWithEmptiestMagazine(s, m_config));
                main.zoneUsed = true;
                take = accessTime(main.zone);
            }
        }
        const float duration = putAway + take;
        startTask(Hand::Main, Task::Swap, duration, duration > 0.0f ? putAway / duration : 0.0f);
        main.target = action.item;
        if (handoff) {
            // The other hand passes the magazine over at the end; until then it's busy doing so.
            startTask(Hand::Off, Task::Swap, duration, 1.0f);
            off.target = ItemKind::Magazine;
        }
        break;
    }

    case ActionKind::ToggleOffGun:
        if (isGun(off.item.kind)) {
            startTask(Hand::Off, Task::StowOffGun, m_config.stowTime, 1.0f);
            off.target = off.item.kind;
        }
        else {
            startTask(Hand::Off, Task::DrawOffGun, m_config.drawTime, 1.0f);
            off.target = action.item;
        }
        break;

    case ActionKind::PickUp:
        startTask(Hand::Off, Task::PickUp, m_config.pickUpTime, m_config.pickUpFrame);
        off.ground = action.ground;
        break;

    case ActionKind::Restock:
        startTask(Hand::Off, Task::Restock, m_config.restockTime, 1.0f / 3.0f);
        off.event2At = m_config.restockTime * 2.0f / 3.0f;
        break;

    case ActionKind::Reload:
    case ActionKind::ZoneButton:
        break; // always resolved to something concrete
    }
}

void Arms::fire(Hand hand)
{
    ArmsState& s = m_state;
    const ItemKind gun = s.hand(hand).item.kind;
    if (gun == ItemKind::Shotgun) {
        if (s.shotgun.chamber != Chamber::Loaded) {
            emit(EventType::DryClick, hand, gun);
            return;
        }
        s.shotgun.chamber = Chamber::Spent;
        ++s.shellsFired;
        emit(EventType::Fired, hand, gun);
        return;
    }
    PistolGun& pistol = s.pistol;
    if (!pistol.chambered || pistol.slideLocked) {
        emit(EventType::DryClick, hand, gun);
        return;
    }
    pistol.chambered = false;
    ++s.roundsFired;
    emit(EventType::Fired, hand, gun);
    // Semi-automatic: the slide chambers the next round, or locks back on an empty magazine.
    if (Magazine* magazine = s.magazine(pistol.magazine)) {
        if (magazine->rounds > 0) {
            --magazine->rounds;
            pistol.chambered = true;
        }
        else {
            pistol.slideLocked = true;
            emit(EventType::SlideLock, hand, gun);
        }
    }
    pistol.state = PistolState::Firing;
    pistol.time = 0.0f;
}

void Arms::update(float dt)
{
    ArmsState& s = m_state;
    for (HandState& hand : s.hands)
        hand.rejected += dt;

    PistolGun& pistol = s.pistol;
    if (pistol.state == PistolState::Firing) {
        pistol.time += dt;
        if (pistol.time >= 60.0f / std::max(m_config.pistolFireRate, 1.0f))
            settlePistol();
    }
    if (pistol.rackDelay >= 0.0f) {
        pistol.rackDelay -= dt;
        if (pistol.rackDelay < 0.0f && CanPerformAction(s, m_config, { ActionKind::Rack }).ok)
            startRack();
    }

    for (Hand hand : { Hand::Main, Hand::Off }) {
        HandState& h = s.hand(hand);
        if (!h.busy())
            continue;
        h.time += dt;
        if (!h.eventDone && h.time >= h.eventAt)
            runEvent(hand);
        if (h.busy() && h.event2At >= 0.0f && !h.event2Done && h.time >= h.event2At)
            runEvent2(hand);
        if (h.busy() && h.time >= h.duration) {
            if (!h.eventDone)
                runEvent(hand);
            if (h.event2At >= 0.0f && !h.event2Done)
                runEvent2(hand);
            finishTask(hand);
        }
    }
}

void Arms::animationEvent(Hand hand)
{
    HandState& h = m_state.hand(hand);
    if (!h.busy())
        return;
    if (!h.eventDone)
        runEvent(hand);
    else if (h.event2At >= 0.0f && !h.event2Done)
        runEvent2(hand);
}

void Arms::runEvent(Hand hand)
{
    ArmsState& s = m_state;
    HandState& h = s.hand(hand);
    HandState& main = s.hand(Hand::Main);
    HandState& off = s.hand(Hand::Off);
    h.eventDone = true;
    switch (h.task) {
    case Task::Fetch: {
        ZoneState& zone = s.zones[static_cast<int>(h.zone)];
        const ZoneConfig& capacity = m_config.zones[static_cast<int>(h.zone)];
        ItemKind order[4] = { h.target, ItemKind::Shell, ItemKind::Magazine, ItemKind::Round };
        for (ItemKind kind : order) {
            if (countIn(zone, kind) <= 0)
                continue;
            if (kind == ItemKind::Magazine) {
                // The fullest one: that's the one to load.
                auto best = std::max_element(zone.magazines.begin(), zone.magazines.end(), [&](int a, int b) {
                    const Magazine* ma = s.magazine(a);
                    const Magazine* mb = s.magazine(b);
                    return (ma ? ma->rounds : 0) < (mb ? mb->rounds : 0);
                });
                h.item = { ItemKind::Magazine, *best };
                zone.magazines.erase(best);
            }
            else {
                (kind == ItemKind::Shell ? zone.shells : zone.rounds) -= 1;
                h.item = { kind, -1 };
            }
            h.hasOrigin = true;
            h.origin = h.zone;
            ArmsEvent event;
            event.type = EventType::Grab;
            event.hand = hand;
            event.item = kind;
            event.zone = h.zone;
            const int total = capacityFor(capacity, kind);
            event.fill = total > 0 ? static_cast<float>(countIn(zone, kind)) / total : 0.0f;
            m_events.push_back(event);
            return;
        }
        ArmsEvent event;
        event.type = EventType::ZoneEmpty;
        event.hand = hand;
        event.zone = h.zone;
        m_events.push_back(event);
        s.pendingShellInsert = false;
        break;
    }

    case Task::Store: {
        const HeldItem item = h.item;
        h.item = {};
        h.hasOrigin = false;
        if (storeInto(h.zone, item)) {
            ArmsEvent event;
            event.type = EventType::Stored;
            event.hand = hand;
            event.item = item.kind;
            event.zone = h.zone;
            m_events.push_back(event);
        }
        else {
            drop(hand, item); // filled up meanwhile
        }
        break;
    }

    case Task::InsertShell:
        if (h.item.kind == ItemKind::Shell && s.shotgun.tube < m_config.tubeSize) {
            h.item = {};
            h.hasOrigin = false;
            ++s.shotgun.tube;
            emit(EventType::ShellInsert, hand, ItemKind::Shell);
        }
        break;

    case Task::PushRound:
        if (h.zoneUsed) {
            ZoneState& zone = s.zones[static_cast<int>(h.zone)];
            if (zone.rounds > 0) {
                --zone.rounds;
                h.item = { ItemKind::Round, -1 };
                h.hasOrigin = true;
                h.origin = h.zone;
                ArmsEvent event;
                event.type = EventType::Grab;
                event.hand = hand;
                event.item = ItemKind::Round;
                event.zone = h.zone;
                const int total = m_config.zones[static_cast<int>(h.zone)].rounds;
                event.fill = total > 0 ? static_cast<float>(zone.rounds) / total : 0.0f;
                m_events.push_back(event);
            }
            else {
                ArmsEvent event;
                event.type = EventType::ZoneEmpty;
                event.hand = hand;
                event.zone = h.zone;
                m_events.push_back(event);
            }
        }
        else {
            runEvent2(hand);
        }
        break;

    case Task::Restock: {
        ZoneState& pocket = s.zones[static_cast<int>(Zone::JacketPocket)];
        if (pocket.shells > 0) {
            --pocket.shells;
            h.item = { ItemKind::Shell, -1 };
            emit(EventType::Grab, hand, ItemKind::Shell);
        }
        break;
    }

    case Task::PickUp: {
        auto it = std::find_if(s.ground.begin(), s.ground.end(), [&](const GroundItem& g) { return g.id == h.ground; });
        if (it != s.ground.end() && h.item.kind == ItemKind::Empty) {
            h.item = it->item;
            h.hasOrigin = false;
            ArmsEvent event;
            event.type = EventType::PickedUp;
            event.hand = hand;
            event.item = it->item.kind;
            event.ground = it->id;
            s.ground.erase(it);
            m_events.push_back(event);
            if (h.item.kind == ItemKind::Pistol)
                settlePistol();
        }
        break;
    }

    case Task::Pump:
        // Back: the chamber's shell flies out, spent or live.
        if (s.shotgun.chamber == Chamber::Spent) {
            emit(EventType::ShellEjected, hand, ItemKind::Shell);
        }
        else if (s.shotgun.chamber == Chamber::Loaded) {
            drop(hand, { ItemKind::Shell, -1 });
        }
        s.shotgun.chamber = Chamber::Empty;
        break;

    case Task::MagEject: {
        const int id = s.pistol.magazine;
        s.pistol.magazine = -1;
        emit(EventType::MagEject, hand, ItemKind::Magazine);
        if (id < 0)
            break;
        // Into the free other hand, or onto the ground.
        if (off.item.kind == ItemKind::Empty && !off.busy()) {
            off.item = { ItemKind::Magazine, id };
            off.hasOrigin = false;
        }
        else {
            drop(Hand::Main, { ItemKind::Magazine, id });
        }
        break;
    }

    case Task::MagInsert:
        if (h.item.kind == ItemKind::Magazine && s.pistol.magazine < 0) {
            s.pistol.magazine = h.item.magazine;
            h.item = {};
            h.hasOrigin = false;
            emit(EventType::MagInsert, hand, ItemKind::Magazine);
        }
        break;

    case Task::Rack: {
        PistolGun& pistol = s.pistol;
        emit(EventType::SlideRack, Hand::Main, ItemKind::Pistol);
        if (pistol.chambered) {
            pistol.chambered = false;
            drop(Hand::Main, { ItemKind::Round, -1 }); // a live round flies out
        }
        Magazine* magazine = s.magazine(pistol.magazine);
        if (magazine && magazine->rounds > 0) {
            --magazine->rounds;
            pistol.chambered = true;
            pistol.slideLocked = false;
        }
        else if (magazine) {
            pistol.slideLocked = true;
            emit(EventType::SlideLock, Hand::Main, ItemKind::Pistol);
        }
        else {
            pistol.slideLocked = false;
        }
        break;
    }

    case Task::Swap:
        if (hand == Hand::Off)
            break; // handing a magazine over happens when the main hand finishes
        // The old item is away: holstered / slung, or into its zone.
        if (isGun(main.item.kind)) {
            const ItemKind gun = main.item.kind;
            main.item = {};
            if (gun == ItemKind::Pistol)
                settlePistol();
            emit(EventType::Holstered, Hand::Main, gun);
        }
        else if (main.item.kind == ItemKind::Magazine) {
            const HeldItem magazine = main.item;
            main.item = {};
            if (!main.drop && main.zoneUsed && storeInto(main.zone, magazine)) {
                ArmsEvent event;
                event.type = EventType::Stored;
                event.item = ItemKind::Magazine;
                event.zone = main.zone;
                m_events.push_back(event);
            }
            else if (const int zone = zoneFor(magazine, -1); zone >= 0 && !main.drop) {
                storeInto(static_cast<Zone>(zone), magazine);
            }
            else {
                drop(Hand::Main, magazine);
            }
        }
        if (main.target == ItemKind::Pistol)
            s.pistol.state = PistolState::Drawing;
        break;

    case Task::DrawOffGun:
        h.item = { h.target, -1 };
        if (h.target == ItemKind::Pistol)
            settlePistol();
        emit(EventType::Drawn, hand, h.target);
        break;

    case Task::StowOffGun: {
        const ItemKind gun = h.item.kind;
        h.item = {};
        if (gun == ItemKind::Pistol)
            settlePistol();
        emit(EventType::Holstered, hand, gun);
        break;
    }

    case Task::None:
        break;
    }
}

void Arms::runEvent2(Hand hand)
{
    ArmsState& s = m_state;
    HandState& h = s.hand(hand);
    h.event2Done = true;
    if (h.task == Task::PushRound) {
        Magazine* magazine = s.magazine(s.hand(Hand::Main).item.magazine);
        if (h.item.kind == ItemKind::Round && magazine && s.hand(Hand::Main).item.kind == ItemKind::Magazine &&
            magazine->rounds < magazine->capacity) {
            ++magazine->rounds;
            h.item = {};
            h.hasOrigin = false;
            emit(EventType::RoundInsert, hand, ItemKind::Round);
        }
    }
    else if (h.task == Task::Restock && h.item.kind == ItemKind::Shell) {
        h.item = {};
        // Into a loop; if a pickup filled the bandolier meanwhile, back into the pocket.
        if (!storeInto(Zone::Bandolier, { ItemKind::Shell, -1 }))
            storeInto(Zone::JacketPocket, { ItemKind::Shell, -1 });
        emit(EventType::Stored, hand, ItemKind::Shell);
    }
}

void Arms::finishTask(Hand hand)
{
    ArmsState& s = m_state;
    HandState& h = s.hand(hand);
    HandState& main = s.hand(Hand::Main);
    HandState& off = s.hand(Hand::Off);
    const Task task = h.task;
    h.task = Task::None;
    h.time = h.duration = 0.0f;

    switch (task) {
    case Task::Pump:
        if (s.shotgun.chamber == Chamber::Empty && s.shotgun.tube > 0) {
            --s.shotgun.tube;
            s.shotgun.chamber = Chamber::Loaded;
        }
        break;

    case Task::MagEject:
    case Task::Rack:
        settlePistol();
        break;

    case Task::MagInsert:
        settlePistol();
        if (s.pistol.slideLocked && s.pistol.magazine >= 0 && m_config.autoRack)
            s.pistol.rackDelay = m_config.autoRackDelay;
        break;

    case Task::Fetch:
        if (s.pendingShellInsert) {
            s.pendingShellInsert = false;
            if (h.item.kind == ItemKind::Shell && main.item.kind == ItemKind::Shotgun && s.shotgun.tube < m_config.tubeSize)
                startTask(Hand::Off, Task::InsertShell, m_config.shellInsertTime, m_config.shellInsertFrame);
        }
        break;

    case Task::Restock:
        if (s.restockHeld && CanPerformAction(s, m_config, { ActionKind::Restock }).ok)
            perform({ ActionKind::Restock });
        break;

    case Task::Swap:
        if (hand == Hand::Off)
            break;
        // The new item arrives.
        if (isGun(main.target)) {
            main.item = { main.target, -1 };
            if (main.target == ItemKind::Pistol)
                settlePistol();
            emit(EventType::Drawn, Hand::Main, main.target);
        }
        else if (main.target == ItemKind::Magazine) {
            if (off.task == Task::Swap && off.item.kind == ItemKind::Magazine) {
                main.item = off.item;
                off.item = {};
                off.hasOrigin = false;
                off.task = Task::None;
            }
            else {
                ZoneState& zone = s.zones[static_cast<int>(main.zone)];
                if (!zone.magazines.empty()) {
                    auto least = std::min_element(zone.magazines.begin(), zone.magazines.end(), [&](int a, int b) {
                        const Magazine* ma = s.magazine(a);
                        const Magazine* mb = s.magazine(b);
                        return (ma ? ma->rounds : 0) < (mb ? mb->rounds : 0);
                    });
                    main.item = { ItemKind::Magazine, *least };
                    zone.magazines.erase(least);
                    emit(EventType::Grab, Hand::Main, ItemKind::Magazine);
                }
                else {
                    ArmsEvent event;
                    event.type = EventType::ZoneEmpty;
                    event.zone = main.zone;
                    m_events.push_back(event);
                }
            }
        }
        break;

    default:
        break;
    }
}

void Arms::abortTask(Hand hand)
{
    ArmsState& s = m_state;
    HandState& h = s.hand(hand);
    if (!h.busy())
        return;
    const Task task = h.task;
    h.task = Task::None;
    h.time = h.duration = 0.0f;
    // Whatever the hand carries stays in it; a gun that was being worked on settles back.
    if (task == Task::MagInsert || task == Task::Rack || task == Task::MagEject)
        settlePistol();
    if (task == Task::Fetch)
        s.pendingShellInsert = false;
}

// ---------------------------------------------------------------------------------------------
// Bookkeeping
// ---------------------------------------------------------------------------------------------

int Arms::totalRounds() const
{
    const ArmsState& s = m_state;
    int total = s.pistol.chambered ? 1 : 0;
    for (const Magazine& magazine : s.magazines)
        total += magazine.rounds;
    for (const ZoneState& zone : s.zones)
        total += zone.rounds;
    for (const HandState& hand : s.hands)
        total += hand.item.kind == ItemKind::Round ? 1 : 0;
    for (const GroundItem& item : s.ground)
        total += item.item.kind == ItemKind::Round ? 1 : 0;
    return total;
}

int Arms::totalShells() const
{
    const ArmsState& s = m_state;
    int total = s.shotgun.tube + (s.shotgun.chamber == Chamber::Loaded ? 1 : 0);
    for (const ZoneState& zone : s.zones)
        total += zone.shells;
    for (const HandState& hand : s.hands)
        total += hand.item.kind == ItemKind::Shell ? 1 : 0;
    for (const GroundItem& item : s.ground)
        total += item.item.kind == ItemKind::Shell ? 1 : 0;
    return total;
}

namespace {

std::vector<int> magazinePlaces(const ArmsState& s)
{
    std::vector<int> ids;
    if (s.pistol.magazine >= 0)
        ids.push_back(s.pistol.magazine);
    for (const HandState& hand : s.hands)
        if (hand.item.kind == ItemKind::Magazine)
            ids.push_back(hand.item.magazine);
    for (const ZoneState& zone : s.zones)
        ids.insert(ids.end(), zone.magazines.begin(), zone.magazines.end());
    for (const GroundItem& item : s.ground)
        if (item.item.kind == ItemKind::Magazine)
            ids.push_back(item.item.magazine);
    return ids;
}

} // namespace

int Arms::magazineCount() const
{
    return static_cast<int>(magazinePlaces(m_state).size());
}

bool Arms::magazinesAccountedFor() const
{
    std::vector<int> ids = magazinePlaces(m_state);
    std::sort(ids.begin(), ids.end());
    if (ids.size() != m_state.magazines.size())
        return false;
    for (size_t i = 0; i < ids.size(); ++i)
        if (!m_state.magazine(ids[i]) || (i > 0 && ids[i] == ids[i - 1]))
            return false;
    return true;
}
