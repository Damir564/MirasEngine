#pragma once
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

// Player mode: main menu, then the level (lvl01.scn by default) as a first-person shooter: the scene's
// entity objects become enemies, pickups and the exit (GameWorld), the player walks with physics and
// carries a shotgun.
class Game final : public AppMode {
public:
    // Plays the scene that is already loaded (the editor's) instead of showing the main menu.
    struct PlayInEditor {};

    static constexpr const char* kDefaultLevel = "lvl01.scn";

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
    ModeRequest takeModeRequest() override;

private:
    enum class State {
        MainMenu,
        Loading,
        Playing,
        Paused,
        Settings,
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
    void updateWorld(float dt, bool controls);
    void returnToMainMenu();
    void stopPlayInEditor();
    glm::vec3 eyePosition() const;

    // Autoplay: aims at and shoots the nearest enemy, walks or teleports to the next one and to the exit.
    void updateBot(float dt);

    void drawMainMenu();
    void drawLoadingScreen();
    void drawPauseMenu();
    void drawSettingsScreen();
    void drawDeathScreen();
    void drawCompleteScreen();
    // GameHud.cpp
    void drawHud();
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
    State m_settingsReturn = State::MainMenu;
    std::string m_error;
    float m_loadingStallTime = 0.0f;
    bool m_quitRequested = false;
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
    bool m_jumpRequested = false;
    bool m_reloadRequested = false;
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
        int frames = 0;
        int teleports = 0;
    };
    Bot m_bot;
};
