// Acceptance tests for the hands, guns and storage logic (Arms.h). Built as its own executable:
//   cmake --build out/build/x64-debug --target arms_tests && out/build/x64-debug/src/arms_tests.exe
// Exit code 0 = all passed.
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include "Arms.h"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        ++g_checks;                                                                    \
        if (!(cond)) {                                                                 \
            std::printf("  FAIL line %d: %s\n", __LINE__, #cond);                      \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

void run(Arms& arms, float seconds)
{
    for (float t = 0.0f; t < seconds; t += 0.01f)
        arms.update(0.01f);
}

bool has(const std::vector<ArmsEvent>& events, EventType type)
{
    for (const ArmsEvent& event : events)
        if (event.type == type)
            return true;
    return false;
}

Action act(ActionKind kind, Zone zone = Zone::Bandolier, ItemKind item = ItemKind::Empty, int ground = -1)
{
    Action action;
    action.kind = kind;
    action.zone = zone;
    action.item = item;
    action.ground = ground;
    return action;
}

const HandState& offHand(const Arms& arms) { return arms.state().hand(Hand::Off); }
const HandState& mainHand(const Arms& arms) { return arms.state().hand(Hand::Main); }

void drawPistol(Arms& arms)
{
    arms.request(act(ActionKind::Swap, Zone::Bandolier, ItemKind::Pistol));
    run(arms, 1.0f);
}

// 1. A full magazine: the last shot locks the slide back, then dry clicks.
void fireFullMagazine()
{
    Arms arms;
    drawPistol(arms);
    CHECK(mainHand(arms).item.kind == ItemKind::Pistol);
    CHECK(arms.state().pistol.state == PistolState::Ready);
    const int shots = 1 + arms.config().magazineCapacity; // chambered + magazine
    for (int i = 0; i < shots; ++i) {
        arms.takeEvents();
        CHECK(arms.request(act(ActionKind::FireMain)));
        CHECK(has(arms.takeEvents(), EventType::Fired));
        run(arms, 0.3f);
    }
    CHECK(arms.state().pistol.slideLocked);
    CHECK(arms.state().pistol.state == PistolState::SlideBack);
    CHECK(arms.state().roundsFired == shots);
    arms.request(act(ActionKind::FireMain));
    const auto events = arms.takeEvents();
    CHECK(has(events, EventType::DryClick));
    CHECK(!has(events, EventType::Fired));
}

// 2. Eject, fetch a full magazine from the bandolier, insert, rack, fire.
void ejectFetchInsertRackFire()
{
    Arms arms;
    drawPistol(arms);
    for (int i = 0; i < 9; ++i) {
        arms.request(act(ActionKind::FireMain));
        run(arms, 0.3f);
    }
    CHECK(arms.request(act(ActionKind::Unload)));
    run(arms, 0.5f);
    CHECK(arms.state().pistol.magazine < 0);
    CHECK(offHand(arms).item.kind == ItemKind::Magazine); // the free hand caught it
    CHECK(arms.request(act(ActionKind::PutBack)));
    run(arms, 1.0f);
    CHECK(offHand(arms).item.kind == ItemKind::Empty);
    CHECK(arms.request(act(ActionKind::Fetch, Zone::Bandolier, ItemKind::Magazine)));
    run(arms, 0.5f);
    CHECK(offHand(arms).item.kind == ItemKind::Magazine);
    const Magazine* fetched = arms.state().magazine(offHand(arms).item.magazine);
    CHECK(fetched && fetched->rounds == fetched->capacity);
    CHECK(arms.request(act(ActionKind::Load)));
    run(arms, 0.6f);
    CHECK(arms.state().pistol.magazine >= 0);
    CHECK(arms.state().pistol.state == PistolState::SlideBack); // still locked open
    arms.takeEvents();
    arms.request(act(ActionKind::FireMain));
    CHECK(has(arms.takeEvents(), EventType::DryClick));
    CHECK(arms.request(act(ActionKind::Rack)));
    run(arms, 0.5f);
    CHECK(arms.state().pistol.state == PistolState::Ready);
    CHECK(arms.state().pistol.chambered);
    arms.request(act(ActionKind::FireMain));
    CHECK(has(arms.takeEvents(), EventType::Fired));
}

// 3. Unloading with the other hand busy drops the magazine; it can be picked up again.
void unloadWithBusyHand()
{
    Arms arms;
    drawPistol(arms);
    // A shell in the off hand (taken from the bandolier so nothing is created).
    ArmsState& s = arms.mutableState();
    --s.zones[static_cast<int>(Zone::Bandolier)].shells;
    s.hand(Hand::Off).item = { ItemKind::Shell, -1 };
    const int magazine = s.pistol.magazine;
    CHECK(arms.request(act(ActionKind::Unload)));
    run(arms, 0.5f);
    const auto events = arms.takeEvents();
    CHECK(has(events, EventType::Dropped));
    CHECK(arms.state().ground.size() == 1);
    const int ground = arms.state().ground.empty() ? -1 : arms.state().ground.front().id;
    CHECK(arms.state().ground.front().item.magazine == magazine);
    CHECK(arms.request(act(ActionKind::PutBack))); // the shell back
    run(arms, 1.0f);
    CHECK(arms.request(act(ActionKind::PickUp, Zone::Bandolier, ItemKind::Empty, ground)));
    run(arms, 1.0f);
    CHECK(offHand(arms).item.kind == ItemKind::Magazine && offHand(arms).item.magazine == magazine);
    CHECK(arms.state().ground.empty());
    CHECK(arms.magazinesAccountedFor());
}

// 4. Fetching with a full off hand is refused with a reason, and nothing changes.
void fetchWithBusyHand()
{
    Arms arms;
    drawPistol(arms);
    arms.request(act(ActionKind::Unload));
    run(arms, 0.5f);
    CHECK(offHand(arms).item.kind == ItemKind::Magazine);
    const ArmsState before = arms.state();
    arms.takeEvents();
    CHECK(!arms.request(act(ActionKind::Fetch, Zone::Bandolier)));
    const auto events = arms.takeEvents();
    CHECK(has(events, EventType::Rejected));
    CHECK(!arms.lastRejection().empty());
    run(arms, 1.0f);
    CHECK(offHand(arms).item.magazine == before.hand(Hand::Off).item.magazine);
    CHECK(offHand(arms).task == Task::None);
    CHECK(arms.state().zones[0].magazines == before.zones[0].magazines);
    CHECK(arms.state().zones[0].shells == before.zones[0].shells);
}

// 5. Pistol in one hand, shotgun in the other: pump and reload refused, both triggers work.
void dualWield()
{
    Arms arms;
    CHECK(arms.request(act(ActionKind::ToggleOffGun)));
    run(arms, 1.0f);
    CHECK(mainHand(arms).item.kind == ItemKind::Shotgun);
    CHECK(offHand(arms).item.kind == ItemKind::Pistol);
    CHECK(arms.state().bothHandsHoldGuns());
    arms.takeEvents();
    CHECK(!arms.request(act(ActionKind::Pump)));
    CHECK(!arms.request(act(ActionKind::Reload)));
    CHECK(!arms.request(act(ActionKind::InsertShell)));
    auto events = arms.takeEvents();
    int rejections = 0;
    for (const ArmsEvent& event : events)
        rejections += event.type == EventType::Rejected ? 1 : 0;
    CHECK(rejections == 3); // none of them silent
    CHECK(arms.request(act(ActionKind::FireMain)));
    CHECK(arms.request(act(ActionKind::FireOff)));
    events = arms.takeEvents();
    int shots = 0;
    for (const ArmsEvent& event : events)
        shots += event.type == EventType::Fired ? 1 : 0;
    CHECK(shots == 2);
    run(arms, 0.3f);
    // The shotgun can't be pumped, so it has fired its one shell.
    arms.request(act(ActionKind::FireMain));
    CHECK(has(arms.takeEvents(), EventType::DryClick));
    CHECK(arms.secondaryAction() == ActionKind::FireOff);
}

// 6. Forced reload while dual wielding drops the off hand's gun (flag on), or does nothing (flag off).
void forcedReload()
{
    for (bool flag : { true, false }) {
        ArmsConfig config;
        config.forceReloadDropsGun = flag;
        Arms arms(config);
        drawPistol(arms);
        arms.request(act(ActionKind::ToggleOffGun)); // the shotgun into the other hand
        run(arms, 1.0f);
        CHECK(offHand(arms).item.kind == ItemKind::Shotgun);
        arms.takeEvents();
        const bool ran = arms.request(act(ActionKind::ForceReload));
        CHECK(ran == flag);
        run(arms, 0.5f);
        if (flag) {
            CHECK(arms.state().onGround(ItemKind::Shotgun));
            CHECK(mainHand(arms).item.kind == ItemKind::Pistol);
            CHECK(arms.state().pistol.magazine < 0); // the reload went ahead: magazine out...
            CHECK(offHand(arms).item.kind == ItemKind::Magazine); // ...into the freed hand
        }
        else {
            CHECK(arms.state().ground.empty());
            CHECK(offHand(arms).item.kind == ItemKind::Shotgun);
            CHECK(has(arms.takeEvents(), EventType::Rejected));
        }
    }
}

// 7. Three loose rounds into a held magazine, one per press; the count stays once it's stored.
void loadRounds()
{
    Arms arms;
    CHECK(arms.request(act(ActionKind::Swap, Zone::Bandolier, ItemKind::Magazine)));
    run(arms, 2.0f);
    CHECK(mainHand(arms).item.kind == ItemKind::Magazine);
    const int id = mainHand(arms).item.magazine;
    const int start = arms.state().magazine(id)->rounds;
    const int loose = arms.state().zones[static_cast<int>(Zone::RightPocket)].rounds;
    for (int i = 0; i < 3; ++i) {
        CHECK(arms.request(act(ActionKind::Reload)));
        run(arms, 1.2f);
        CHECK(arms.state().magazine(id)->rounds == start + i + 1);
    }
    CHECK(arms.state().zones[static_cast<int>(Zone::RightPocket)].rounds == loose - 3);
    CHECK(arms.request(act(ActionKind::Swap, Zone::Bandolier, ItemKind::Shotgun)));
    run(arms, 2.0f);
    bool stored = false;
    for (const ZoneState& zone : arms.state().zones)
        for (int m : zone.magazines)
            stored = stored || m == id;
    CHECK(stored);
    CHECK(arms.state().magazine(id)->rounds == start + 3);
}

// 8. Swapping before the insert frame stops the magazine going in; it stays in the off hand.
void interruptInsert()
{
    Arms arms;
    drawPistol(arms);
    arms.request(act(ActionKind::Unload));
    run(arms, 0.5f);
    const int id = offHand(arms).item.magazine;
    CHECK(arms.request(act(ActionKind::Load)));
    run(arms, arms.config().insertTime * arms.config().insertFrame * 0.5f);
    CHECK(arms.state().pistol.state == PistolState::MagInserting);
    CHECK(arms.request(act(ActionKind::Swap, Zone::Bandolier, ItemKind::Shotgun)));
    CHECK(offHand(arms).item.kind == ItemKind::Magazine && offHand(arms).item.magazine == id);
    run(arms, 1.0f);
    CHECK(mainHand(arms).item.kind == ItemKind::Shotgun);
    CHECK(offHand(arms).item.magazine == id);
    CHECK(arms.state().pistol.magazine < 0);
    CHECK(arms.magazinesAccountedFor());
}

// The animation can trigger a task's event before its timer does.
void animationEvents()
{
    Arms arms;
    drawPistol(arms);
    arms.request(act(ActionKind::Unload));
    run(arms, 0.5f);
    arms.request(act(ActionKind::Load));
    arms.update(0.01f);
    CHECK(arms.state().pistol.magazine < 0);
    arms.animationEvent(Hand::Off); // the insert frame
    CHECK(arms.state().pistol.magazine >= 0);
    run(arms, 1.0f);
    CHECK(arms.state().pistol.state == PistolState::Ready);
}

// 9. A long random sequence never creates or loses magazines, rounds or shells.
void conservation()
{
    for (bool autoRack : { false, true }) {
        ArmsConfig config;
        config.autoRack = autoRack;
        Arms arms(config);
        const int rounds = arms.totalRounds();
        const int shells = arms.totalShells();
        const size_t magazines = arms.state().magazines.size();
        std::mt19937 rng(autoRack ? 7u : 42u);
        const ActionKind kinds[] = { ActionKind::FireMain, ActionKind::FireOff, ActionKind::Pump, ActionKind::Rack,
            ActionKind::Reload, ActionKind::ForceReload, ActionKind::Unload, ActionKind::Load, ActionKind::Fetch,
            ActionKind::Store, ActionKind::ZoneButton, ActionKind::PutBack, ActionKind::InsertShell,
            ActionKind::PushRound, ActionKind::Swap, ActionKind::ToggleOffGun, ActionKind::PickUp, ActionKind::Restock };
        const ItemKind swaps[] = { ItemKind::Pistol, ItemKind::Shotgun, ItemKind::Magazine };
        bool ok = true;
        for (int step = 0; step < 20000 && ok; ++step) {
            Action action = act(kinds[rng() % std::size(kinds)], static_cast<Zone>(rng() % kZoneCount),
                swaps[rng() % std::size(swaps)]);
            if (action.kind == ActionKind::PickUp && !arms.state().ground.empty())
                action.ground = arms.state().ground[rng() % arms.state().ground.size()].id;
            arms.setRestockHeld(rng() % 2 == 0);
            arms.request(action);
            arms.takeEvents();
            arms.update(static_cast<float>(rng() % 30) * 0.01f);
            ok = arms.magazinesAccountedFor() && arms.state().magazines.size() == magazines &&
                arms.totalRounds() + arms.state().roundsFired == rounds &&
                arms.totalShells() + arms.state().shellsFired == shells;
            if (!ok)
                std::printf("  broken after step %d (%s): rounds %d+%d/%d shells %d+%d/%d magazines ok %d\n", step,
                    actionName(action.kind), arms.totalRounds(), arms.state().roundsFired, rounds, arms.totalShells(),
                    arms.state().shellsFired, shells, arms.magazinesAccountedFor());
        }
        CHECK(ok);
    }
}

} // namespace

int main()
{
    const std::pair<const char*, void (*)()> tests[] = {
        { "1 fire a full magazine", fireFullMagazine },
        { "2 eject, fetch, insert, rack, fire", ejectFetchInsertRackFire },
        { "3 unload with the off hand busy", unloadWithBusyHand },
        { "4 fetch with the off hand busy", fetchWithBusyHand },
        { "5 dual wield", dualWield },
        { "6 forced reload", forcedReload },
        { "7 rounds into a held magazine", loadRounds },
        { "8 interrupt the magazine insert", interruptInsert },
        { "  animation events", animationEvents },
        { "9 conservation", conservation },
    };
    for (const auto& [name, test] : tests) {
        const int before = g_failures;
        test();
        std::printf("%s %s\n", g_failures == before ? "PASS" : "FAIL", name);
    }
    std::printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
