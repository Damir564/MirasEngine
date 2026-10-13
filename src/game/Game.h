#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "app/AppMode.h"
#include "engine/Camera.h"
#include "engine/Physics.h"
#include "engine/Renderer.h"
#include "GameAssets.h"
#include "GameWorld.h"

class ModelManager;
class SceneManager;
struct GraphicsSettings;
struct ImDrawList;

// Player mode: main menu, then the level (lvl01.scn by default) as a first-person shooter: the scene's
// entity objects become enemies, pickups and the exit (GameWorld), the player walks with physics and
// carries a shotgun.
class Game final : public AppMode {
public:
    // Plays the scene that is already loaded (the editor's) instead of showing the main menu.
    struct PlayInEditor {};

    static constexpr const char* kDefaultLevel = "lvl01.scn";
    // The level simulates in fixed steps of this length, whatever the frame rate; frames draw between the
    // last two steps.
    static constexpr float kTickSeconds = 1.0f / 60.0f;

    explicit Game(const EngineContext& engine, std::string levelPath = kDefaultLevel);
    Game(const EngineContext& engine, PlayInEditor);
    ~Game() override;

    Game(const Game&) = delete;
    Game& operator=(const Game&) = delete;

    void onEvent(const SDL_Event& event) override;
    void update(float dt) override;
    // The HUD is ImGui too, so the UI stays on while playing (without mouse input).
    bool uiVisible() const override { return true; }
    void drawUi() override;
    void fillFrame(FrameInput& frame) override;
    bool quitRequested() const override { return m_quitRequested; }
    // --autoplay: 1 when the bot failed the level (died, time limit, not every enemy killed, load error).
    int exitCode() const override { return m_exitCode; }
    ModeRequest takeModeRequest() override;

private:
    enum class State {
        MainMenu,
        Loading,
        Playing,
        Paused,
        Settings,
        Controls,
        Dead,
        Complete,
    };

    void setState(State state);
    void startLoading();
    void updateLoading(float dt);
    void failLoading(const std::string& message);
    // The scene has loaded: game models, colliders, entities, player.
    void startLevel();
    // Back to the level's start without loading it again.
    void restartLevel();
    // Drops the level's entities, colliders and game models (before the scene itself is closed).
    void endLevel();
    // At the player start, or above the middle of the scene when there is none.
    void spawnPlayer(const SpawnInfo* start);
    void movePlayer(float dt);
    // R pressed / released (onEvent), and the hold and double-tap timers (update).
    void onReloadKey(bool down);
    void updateReloadKey(float dt);
    void updateWorld(float dt, bool controls);
    // One fixed step of the level (Playing, Dead, Complete): bot, player, world. Consumes the key presses.
    void tick(float dt);
    // How far the frame is between the last two ticks, 0..1.
    float tickBlend() const { return std::clamp(m_tickTime / kTickSeconds, 0.0f, 1.0f); }
    void returnToMainMenu();
    void stopPlayInEditor();
    glm::vec3 eyePosition() const;
    // The camera actually used: the mouse pitch plus the shot's kick, tilted down while the jacket is open.
    Camera viewCamera() const;

    // Autoplay: aims at and shoots the nearest enemy, walks or teleports to the next one and to the exit.
    void updateBot(float dt);

    void drawMainMenu();
    void drawLoadingScreen();
    void drawPauseMenu();
    void drawSettingsScreen();
    void drawControlsScreen();
    // Opens Settings or Controls; Back (or Esc) returns to the current screen.
    void openScreen(State screen);
    void drawDeathScreen();
    void drawCompleteScreen();
    // GameHud.cpp
    void drawHud();
    void drawDebugOverlay();
    // Jacket open: the cursor, what it points at and the slot keys' bindings.
    void drawBindView(ImDrawList* draw, float scale);
    // Full-window ImGui window for a menu screen; returns the result of Begin().
    bool beginScreen(const char* id, float backgroundAlpha);
    bool menuButton(const char* label);

    SDL_Window* m_window;
    Renderer& m_renderer;
    ModelManager& m_models;
    SceneManager& m_scenes;
    GraphicsSettings& m_settings;
    AudioSystem& m_audio;
    std::string m_levelPath;

    State m_state = State::MainMenu;
    State m_screenReturn = State::MainMenu;
    std::string m_error;
    float m_loadingStallTime = 0.0f;
    bool m_quitRequested = false;
    int m_exitCode = 0;
    bool m_playInEditor = false;
    bool m_levelActive = false; // startLevel() ran and endLevel() didn't yet
    ModeRequest m_modeRequest = ModeRequest::None;

    Camera m_camera;
    float m_fieldOfView = 75.0f;
    PhysicsWorld m_physics;
    GameAssets m_assets;
    GameWorld m_world;
    std::vector<DynamicInstance> m_drawables;
    glm::vec3 m_spawnPoint{ 0.0f };
    float m_spawnYaw = -90.0f;
    // Falling below this (off the edge of the level) kills.
    float m_killHeight = -100.0f;
    float m_eyeHeight = 1.65f;
    float m_tickTime = 0.0f;       // frame time not simulated yet (< kTickSeconds)
    glm::vec3 m_eyeFrom{ 0.0f };   // camera position after the second-to-last and the last tick
    glm::vec3 m_eyeTo{ 0.0f };
    bool m_jumpRequested = false;
    // Weapon and inventory key presses since the last update (only the pressed-this-frame fields are used).
    GameWorld::Input m_presses;
    struct ReloadKey {
        bool down = false;
        bool used = false;      // this press already did something (a hold, a double tap, restocking)
        float held = 0.0f;
        bool tapPending = false; // a pistol tap waiting to see whether a second one follows
        float sinceTap = 0.0f;
    };
    ReloadKey m_reloadKey;
    bool m_debugOverlay = false; // F3 in debug builds: hands, pistol and storage state
    // 0..1: Tab held opens the jacket, tilting the view down to the body and slowing the player.
    float m_jacket = 0.0f;
    // While it is open the mouse moves a cursor (pixels from the screen's center) instead of the view; a slot
    // key binds what the cursor points at (GameHud.cpp picks it).
    glm::vec2 m_bindCursor{ 0.0f };
    std::vector<GameWorld::BindTarget> m_bindTargets;
    int m_hoveredBind = -1; // index into m_bindTargets
    float m_stateTime = 0.0f;

    struct Bot {
        bool enabled = false;
        float time = 0.0f;
        int restarts = 0;
        float lostSight = 0.0f;
        float stuckTime = 0.0f;
        glm::vec3 lastPosition{ 0.0f };
        glm::vec3 move{ 0.0f };
        bool fire = false;
        bool pump = false;
        bool reload = false;
        bool reloadWasHeld = false;
        SlotBind use; // a slot key's action (item Empty = none)
        bool interact = false;
        int frames = 0;
        int shots = 0;
        int pumps = 0;
        int teleports = 0;
    };
    Bot m_bot;
};
