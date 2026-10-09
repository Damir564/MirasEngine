#include "Game.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <limits>
#include <utility>
#include "imgui.h"
#include "app/SettingsUi.h"
#include "engine/Audio.h"
#include "engine/GraphicsSettings.h"
#include "engine/ModelManager.h"
#include "engine/Renderer.h"
#include "engine/SceneManager.h"
#include "engine/Log.h"

namespace {
constexpr float kButtonWidth = 320.0f;
constexpr float kButtonHeight = 56.0f;
constexpr float kWalkSpeed = 4.5f;
constexpr float kSprintMultiplier = 1.8f;
constexpr float kEyeHeight = 1.65f;
constexpr float kJacketPitch = -62.0f;     // looking down at the open jacket
constexpr float kJacketOpenSeconds = 0.25f;
constexpr float kJacketSpeed = 0.45f;      // walking speed with the jacket fully open
constexpr float kRestartDelay = 0.3f;      // after dying, so a key still held from the fight doesn't restart
// Scene loads that make no progress this long (e.g. every model file missing) are reported as failed.
constexpr float kLoadingStallSeconds = 1.0f;
constexpr float kBotTimeLimit = 300.0f;
constexpr ImGuiWindowFlags kScreenFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBringToFrontOnFocus;
// While playing, ImGui only draws the HUD: it must not react to the (captured) mouse.
constexpr ImGuiConfigFlags kPlayingImGuiFlags = ImGuiConfigFlags_NoMouse | ImGuiConfigFlags_NoMouseCursorChange;
}

Game::Game(const EngineContext& engine, std::string levelPath)
    : m_window(engine.window)
    , m_renderer(engine.renderer)
    , m_models(engine.models)
    , m_scenes(engine.scenes)
    , m_settings(engine.settings)
    , m_audio(engine.audio)
    , m_levelPath(std::move(levelPath))
    , m_world(engine.models, m_physics, engine.audio, m_assets)
{
    m_camera.position = glm::vec3(0.0f, 2.0f, 5.0f);
    m_bot.enabled = engine.autoplay;
    SDL_SetWindowTitle(m_window, "MirasEngine");
    SDL_SetWindowRelativeMouseMode(m_window, false);
    if (m_bot.enabled) {
        LOG_INFO("[AUTOPLAY] Playing " << m_levelPath << "\n");
        startLoading();
    }
}

Game::Game(const EngineContext& engine, PlayInEditor)
    : Game(engine, engine.scenes.currentPath())
{
    m_playInEditor = true;
    SDL_SetWindowTitle(m_window, "MirasEngine - Playing (Esc: pause, F5: stop)");
    // The scene is already open; if models are still streaming in, wait on the loading screen.
    m_loadingStallTime = 0.0f;
    setState(State::Loading);
}

ModeRequest Game::takeModeRequest()
{
    return std::exchange(m_modeRequest, ModeRequest::None);
}

void Game::stopPlayInEditor()
{
    endLevel();
    SDL_SetWindowRelativeMouseMode(m_window, false);
    m_modeRequest = ModeRequest::ReturnToEditor;
}

Game::~Game()
{
    endLevel();
    ImGui::GetIO().ConfigFlags &= ~kPlayingImGuiFlags;
    m_audio.setPaused(false);
    if (m_window)
        SDL_SetWindowRelativeMouseMode(m_window, false);
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

void Game::setState(State state)
{
    const bool wasPlaying = m_state == State::Playing;
    m_state = state;
    m_stateTime = 0.0f;
    const bool playing = state == State::Playing;
    SDL_SetWindowRelativeMouseMode(m_window, playing);
    ImGuiIO& io = ImGui::GetIO();
    if (playing)
        io.ConfigFlags |= kPlayingImGuiFlags;
    else
        io.ConfigFlags &= ~kPlayingImGuiFlags;
    if (playing && !wasPlaying) {
        // ImGui ignores the mouse while playing; release what it thinks is held so the menu does not come
        // back with a stuck button (the click that pressed Resume/Play).
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        io.ClearInputKeys();
    }
    m_audio.setPaused(state == State::Paused || state == State::Settings || state == State::Controls);
}

void Game::startLoading()
{
    m_error.clear();
    if (!std::filesystem::exists(m_levelPath)) {
        std::string message = m_levelPath + " not found";
        if (m_levelPath == kDefaultLevel)
            message += ". Build it in the editor: File > Run Script... > levels/lvl01.mscript";
        failLoading(message);
        return;
    }
    const SceneManager::OpenResult result = m_scenes.open(m_levelPath);
    if (!result.ok) {
        failLoading("Failed to open " + m_levelPath);
        return;
    }
    for (const std::string& missing : result.missingFiles)
        LOG_ERROR("[GAME] Model file not found: " << missing << "\n");
    m_loadingStallTime = 0.0f;
    setState(State::Loading);
}

void Game::updateLoading(float dt)
{
    if (!m_scenes.isLoading()) {
        if (m_models.getInstances().empty()) {
            failLoading("Failed to load " + m_levelPath + " (the scene is empty)");
            return;
        }
        startLevel();
        return;
    }

    const auto& tasks = m_models.getLoadingTasks();
    const bool anyActive = std::any_of(tasks.begin(), tasks.end(),
        [](const LoadingTask& task) { return task.state != LoadingState::Failed; });
    m_loadingStallTime = anyActive ? 0.0f : m_loadingStallTime + dt;
    if (m_loadingStallTime > kLoadingStallSeconds)
        failLoading("Failed to load " + m_levelPath);
}

void Game::failLoading(const std::string& message)
{
    LOG_ERROR("[GAME] " << message << "\n");
    if (m_playInEditor) {
        stopPlayInEditor();
        return;
    }
    endLevel();
    m_scenes.clear();
    m_error = message;
    setState(State::MainMenu);
    if (m_bot.enabled)
        m_quitRequested = true;
}

void Game::startLevel()
{
    m_assets.load(m_models, m_audio);
    m_physics.buildStaticScene(m_models);
    m_levelActive = true;
    restartLevel();
}

void Game::restartLevel()
{
    m_world.clear();
    SpawnInfo start;
    const bool hasStart = m_world.spawnFromScene(m_models.getInstances(), start);
    spawnPlayer(hasStart ? &start : nullptr);
    setState(State::Playing);
}

void Game::endLevel()
{
    if (!m_levelActive)
        return;
    m_levelActive = false;
    m_world.clear();
    m_audio.stopAll();
    m_physics.clear();
    m_drawables.clear();
    m_assets.release(m_models);
}

void Game::returnToMainMenu()
{
    if (m_playInEditor) {
        stopPlayInEditor();
        return;
    }
    endLevel();
    m_scenes.clear();
    m_scenes.setCurrentPath({});
    m_error.clear();
    setState(State::MainMenu);
}

void Game::spawnPlayer(const SpawnInfo* start)
{
    glm::vec3 boundsMin(std::numeric_limits<float>::max());
    glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
    bool any = false;
    for (const ModelInstance& instance : m_models.getInstances()) {
        const GPUModel* model = m_models.getModel(instance.modelIndex);
        if (!instance.visible || !instance.entity.empty() || !model || !model->isValid())
            continue;
        const glm::mat4 transform = instance.getTransformMatrix();
        for (int corner = 0; corner < 8; ++corner) {
            const glm::vec3 local(corner & 1 ? model->boundsMax.x : model->boundsMin.x,
                corner & 2 ? model->boundsMax.y : model->boundsMin.y,
                corner & 4 ? model->boundsMax.z : model->boundsMin.z);
            const glm::vec3 world = glm::vec3(transform * glm::vec4(local, 1.0f));
            boundsMin = glm::min(boundsMin, world);
            boundsMax = glm::max(boundsMax, world);
            any = true;
        }
    }
    m_killHeight = any ? boundsMin.y - 30.0f : -100.0f;
    if (start) {
        // Entities face +Z turned by their yaw; the camera's yaw 0 looks along +X.
        m_spawnPoint = start->position + glm::vec3(0.0f, 0.05f, 0.0f);
        m_spawnYaw = glm::degrees(std::atan2(std::cos(start->yaw), std::sin(start->yaw)));
    }
    else if (any) {
        const glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
        m_spawnPoint = glm::vec3(center.x, boundsMax.y + 1.0f, center.z);
        m_spawnYaw = -90.0f;
    }
    else {
        m_spawnPoint = glm::vec3(0.0f, 2.0f, 0.0f);
        m_spawnYaw = -90.0f;
    }
    m_physics.spawnPlayer(m_spawnPoint);
    m_camera.yaw = m_spawnYaw;
    m_camera.pitch = 0.0f;
    m_eyeHeight = kEyeHeight;
    m_camera.position = eyePosition();
    m_jumpRequested = false;
    m_presses = {};
    m_jacket = 0.0f;
    m_bot.lastPosition = m_spawnPoint;
}

glm::vec3 Game::eyePosition() const
{
    return m_physics.playerPosition() + glm::vec3(0.0f, m_eyeHeight, 0.0f);
}

Camera Game::viewCamera() const
{
    Camera view = m_camera;
    const float open = m_jacket * m_jacket * (3.0f - 2.0f * m_jacket);
    view.pitch = std::clamp(glm::mix(m_camera.pitch, kJacketPitch, open) + m_world.recoil(), -89.0f, 89.0f);
    return view;
}

// ---------------------------------------------------------------------------------------------
// Input and frame
// ---------------------------------------------------------------------------------------------

void Game::onEvent(const SDL_Event& event)
{
    if (m_playInEditor && event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.scancode == SDL_SCANCODE_F5) {
        stopPlayInEditor();
        return;
    }
    if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.scancode == SDL_SCANCODE_ESCAPE) {
        if (m_state == State::Playing) setState(State::Paused);
        else if (m_state == State::Paused) setState(State::Playing);
        else if (m_state == State::Settings || m_state == State::Controls) setState(m_screenReturn);
        return;
    }
    // Dying is quick and so is trying again: one key, no loading.
    if (m_state == State::Dead && m_stateTime > kRestartDelay && event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
        (event.key.scancode == SDL_SCANCODE_R || event.key.scancode == SDL_SCANCODE_SPACE ||
            event.key.scancode == SDL_SCANCODE_RETURN)) {
        restartLevel();
        return;
    }
    if (m_state != State::Playing)
        return;
    if (event.type == SDL_EVENT_WINDOW_FOCUS_LOST && !m_bot.enabled) {
        setState(State::Paused);
        return;
    }
    if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.scancode == SDL_SCANCODE_SPACE)
        m_jumpRequested = true;
    if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
        GameWorld::Input& press = m_presses;
        switch (event.key.scancode) {
        case SDL_SCANCODE_Q: press.zone = static_cast<int>(Zone::Bandolier); break;
        case SDL_SCANCODE_E: press.zone = static_cast<int>(Zone::JacketPocket); break;
        case SDL_SCANCODE_C: press.zone = static_cast<int>(Zone::RightPocket); break;
        case SDL_SCANCODE_R: press.reload = true; break;
        case SDL_SCANCODE_V: press.forceReload = true; break;
        case SDL_SCANCODE_X: press.unload = true; break;
        case SDL_SCANCODE_F: press.putBack = true; break;
        case SDL_SCANCODE_T: press.pickUp = true; break;
        case SDL_SCANCODE_G: press.toggleOffGun = true; break;
        case SDL_SCANCODE_1: press.swapTo = ItemKind::Shotgun; break;
        case SDL_SCANCODE_2: press.swapTo = ItemKind::Pistol; break;
        case SDL_SCANCODE_3: press.swapTo = ItemKind::Magazine; break;
#ifndef NDEBUG
        case SDL_SCANCODE_F3: m_debugOverlay = !m_debugOverlay; break;
#endif
        default: break;
        }
    }
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_RIGHT)
        m_presses.secondary = true;
    if (event.type == SDL_EVENT_MOUSE_MOTION && !m_bot.enabled) {
        m_camera.yaw += event.motion.xrel * m_camera.sensitivity;
        m_camera.pitch = glm::clamp(m_camera.pitch - event.motion.yrel * m_camera.sensitivity, -89.0f, 89.0f);
    }
}

void Game::update(float dt)
{
    // Long hitches (loading, window drags) must not turn into one huge simulation step.
    dt = std::min(dt, 0.1f);
    m_stateTime += dt;
    switch (m_state) {
    case State::Loading:
        updateLoading(dt);
        break;
    case State::Playing:
        if (m_bot.enabled)
            updateBot(dt);
        movePlayer(dt);
        updateWorld(dt, true);
        if (!m_world.player().alive)
            setState(State::Dead);
        else if (m_world.levelComplete())
            setState(State::Complete);
        break;
    case State::Dead:
    case State::Complete:
        // The level goes on behind the screen (enemies finish their animations).
        if (m_state == State::Dead)
            m_eyeHeight = std::max(0.35f, m_eyeHeight - dt * 3.0f);
        m_physics.updatePlayer(dt, glm::vec3(0.0f), false);
        updateWorld(dt, false);
        if (m_bot.enabled)
            updateBot(dt);
        break;
    default:
        break;
    }
    m_jumpRequested = false;
    m_presses = {};
}

void Game::movePlayer(float dt)
{
    glm::vec3 move(0.0f);
    bool sprint = false;
    if (m_bot.enabled) {
        move = m_bot.move;
    }
    else {
        const bool* keys = SDL_GetKeyboardState(nullptr);
        // Walk on the horizontal plane regardless of where the player is looking.
        const float yaw = glm::radians(m_camera.yaw);
        const glm::vec3 forward(std::cos(yaw), 0.0f, std::sin(yaw));
        const glm::vec3 right(-forward.z, 0.0f, forward.x);
        if (keys[SDL_SCANCODE_W]) move += forward;
        if (keys[SDL_SCANCODE_S]) move -= forward;
        if (keys[SDL_SCANCODE_D]) move += right;
        if (keys[SDL_SCANCODE_A]) move -= right;
        sprint = keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT];
    }
    if (glm::length(move) > 0.0f)
        move = glm::normalize(move) * kWalkSpeed * (sprint ? kSprintMultiplier : 1.0f) *
            glm::mix(1.0f, kJacketSpeed, m_jacket);

    m_physics.updatePlayer(dt, move, m_jumpRequested);
    if (m_physics.playerPosition().y < m_killHeight)
        m_world.damagePlayer(1000.0f);
    m_camera.position = eyePosition();
}

void Game::updateWorld(float dt, bool controls)
{
    // The presses since the last update (the keys are read in onEvent).
    GameWorld::Input input = controls ? m_presses : GameWorld::Input{};
    input.dt = dt;
    input.eye = eyePosition();
    const bool* keys = SDL_GetKeyboardState(nullptr);
    const bool jacketHeld = controls && keys[SDL_SCANCODE_TAB];
    m_jacket = std::clamp(m_jacket + (jacketHeld ? dt : -dt) / kJacketOpenSeconds, 0.0f, 1.0f);
    input.jacket = m_jacket;
    input.restock = controls && keys[SDL_SCANCODE_Q];
    input.forward = getFront(viewCamera());
    input.playerFeet = m_physics.playerPosition();
    if (controls) {
        const bool walking = m_bot.enabled ? glm::length(m_bot.move) > 0.01f
            : keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_D];
        input.moving = walking && m_physics.playerOnGround();
        input.sprinting = !m_bot.enabled && (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT]);
        const SDL_MouseButtonFlags buttons = SDL_GetMouseState(nullptr, nullptr);
        if (m_bot.enabled) {
            input.fire = m_bot.fire;
            input.secondary = m_bot.pump;
            input.reload = m_bot.reload;
            input.zone = m_bot.zone;
        }
        else {
            input.fire = (buttons & SDL_BUTTON_LMASK) != 0;
        }
    }
    m_world.update(input);
    m_camera.position = input.eye;
}

void Game::fillFrame(FrameInput& frame)
{
    const Camera view = viewCamera();
    frame.models = &m_models;
    frame.view = getView(view);
    frame.proj = getProjection(frame.windowWidth, frame.windowHeight, kCameraNearPlane, m_settings.viewDistance,
        m_fieldOfView);
    frame.cameraPosition = view.position;
    frame.viewport = { 0.0f, 0.0f, frame.windowWidth, frame.windowHeight };
    frame.highlight = {};
    frame.showPath = false;
    frame.showGrid = false;
    frame.hideEntities = true;
    m_drawables.clear();
    if (m_levelActive)
        m_world.collectDrawables(glm::inverse(frame.view), m_drawables);
    frame.dynamicInstances = m_drawables;
}

// ---------------------------------------------------------------------------------------------
// Autoplay
// ---------------------------------------------------------------------------------------------

void Game::updateBot(float dt)
{
    Bot& bot = m_bot;
    bot.time += dt;
    ++bot.frames;
    bot.move = glm::vec3(0.0f);
    bot.fire = false;
    bot.pump = false;
    bot.reload = false;
    bot.zone = -1;
    const ArmsState& arms = m_world.arms().state();
    const int shellsCarried = arms.zones[static_cast<int>(Zone::Bandolier)].shells +
        arms.zones[static_cast<int>(Zone::JacketPocket)].shells;

    if (m_state == State::Dead) {
        if (m_stateTime < 1.5f)
            return;
        LOG_INFO("[AUTOPLAY] Died after " << bot.time << " s (" << m_world.enemiesKilled() << "/"
            << m_world.enemiesTotal() << " enemies killed; " << bot.shots << " shots, " << bot.pumps << " pumps, chamber "
            << static_cast<int>(arms.shotgun.chamber) << ", tube " << arms.shotgun.tube << ", carried "
            << shellsCarried << ")\n");
        if (bot.restarts++ < 2) {
            LOG_INFO("[AUTOPLAY] Restarting the level\n");
            restartLevel();
        }
        else {
            LOG_INFO("[AUTOPLAY] RESULT: failed (died 3 times)\n");
            m_quitRequested = true;
        }
        return;
    }
    if (m_state == State::Complete) {
        if (m_stateTime < 1.0f)
            return;
        LOG_INFO("[AUTOPLAY] RESULT: level complete in " << m_world.levelTime() << " s, " << m_world.enemiesKilled()
            << "/" << m_world.enemiesTotal() << " enemies, health " << m_world.player().health << ", "
            << bot.teleports << " teleports, " << bot.restarts << " restarts, " << bot.shots << " shots, "
            << bot.pumps << " pumps, " << arms.shotgun.tube << " in the tube + " << shellsCarried
            << " carried, " << static_cast<int>(bot.frames / std::max(bot.time, 0.001f)) << " FPS average\n");
        m_quitRequested = true;
        return;
    }
    if (bot.time > kBotTimeLimit) {
        LOG_INFO("[AUTOPLAY] RESULT: failed (time limit, " << m_world.enemiesKilled() << "/" << m_world.enemiesTotal()
            << " enemies killed)\n");
        m_quitRequested = true;
        return;
    }

    if (bot.frames == 3) {
        // Self-checks of the animation and weapon placement, readable in the log.
        for (const Entity& entity : m_world.entities()) {
            if (entity.kind != Entity::Kind::Enemy || !entity.model)
                continue;
            Animator pose = entity.animator;
            std::vector<glm::mat4> nodes;
            pose.evaluate(nodes);
            const int head = entity.model->data.findNode("head");
            if (head >= 0)
                LOG_INFO("[AUTOPLAY] Check: enemy head " << nodes[head][3].y << " m above its feet (expect ~1.7)\n");
            break;
        }
        const GameModel& shotgun = m_assets.model(GameModelId::Shotgun);
        Camera view = m_camera;
        view.pitch = std::clamp(m_camera.pitch + m_world.recoil(), -89.0f, 89.0f);
        const glm::mat4 viewMatrix = getView(view);
        for (const DynamicInstance& drawable : m_drawables) {
            if (!shotgun.meshModels.empty() && drawable.instance.modelIndex == shotgun.meshModels.front()) {
                const glm::vec3 local(viewMatrix * drawable.transform[3]);
                LOG_INFO("[AUTOPLAY] Check: shotgun at (" << local.x << ", " << local.y << ", " << local.z
                    << ") in view space (expect about 0.2, -0.22, -0.34)\n");
                break;
            }
        }
        LOG_INFO("[AUTOPLAY] Check: " << m_drawables.size() << " dynamic instances drawn\n");
    }

    const glm::vec3 eye = eyePosition();
    const glm::vec3 feet = m_physics.playerPosition();
    glm::vec3 target;
    bool visible = false;
    const bool enemyLeft = m_world.nearestEnemy(eye, target, visible);
    if (!enemyLeft && !m_world.exitPosition(target)) {
        // Nothing left to do; a level without an exit completes on its own.
        return;
    }
    if (!enemyLeft)
        target += glm::vec3(0.0f, 1.0f, 0.0f);
    // An enemy within reach is the target, even when its chest is out of sight (the barrel is lower).
    bool threatened = false;
    float closest = 4.0f;
    for (const Entity& entity : m_world.entities()) {
        if (entity.kind != Entity::Kind::Enemy || entity.state == EnemyState::Dead)
            continue;
        const float d = glm::distance(entity.position, feet);
        if (d < closest) {
            closest = d;
            threatened = visible = true;
            target = entity.position + glm::vec3(0.0f, 1.2f, 0.0f);
        }
    }

    // Turn towards the target, at most 300 degrees per second. In a fight the barrel is what has to point
    // at the enemy; it is held off the view's center, so turn by the barrel's error instead.
    const glm::vec3 toTarget = target - eye;
    const auto yawOf = [](const glm::vec3& v) { return glm::degrees(std::atan2(v.z, v.x)); };
    const auto pitchOf = [](const glm::vec3& v) { return glm::degrees(std::atan2(v.y, glm::length(glm::vec2(v.x, v.z)))); };
    float yawError = std::remainder(yawOf(toTarget) - m_camera.yaw, 360.0f);
    float pitchError = pitchOf(toTarget) - m_camera.pitch;
    if (enemyLeft && visible) {
        const glm::vec3 fromBarrel = target - m_world.barrelOrigin();
        yawError = std::remainder(yawOf(fromBarrel) - yawOf(m_world.barrelDirection()), 360.0f);
        pitchError = pitchOf(fromBarrel) - pitchOf(m_world.barrelDirection());
    }
    const float maxTurn = 300.0f * dt;
    m_camera.yaw += std::clamp(yawError, -maxTurn, maxTurn);
    m_camera.pitch += std::clamp(pitchError, -maxTurn, maxTurn);
    m_camera.pitch = std::clamp(m_camera.pitch, -89.0f, 89.0f);
    const float distance = glm::length(toTarget);
    const glm::vec3 flatDirection = glm::length(glm::vec2(toTarget.x, toTarget.z)) > 1e-3f
        ? glm::normalize(glm::vec3(toTarget.x, 0.0f, toTarget.z)) : glm::vec3(0.0f);

    // Operate the shotgun: shoot when the barrel (not the view) is on target, pump a spent chamber, load
    // shells while nothing is in sight.
    // The bot keeps the shotgun and handles it like a player: shells one at a time from the bandolier or
    // the pocket, pushed in with R, a pump after every shot.
    const HandState& off = arms.hand(Hand::Off);
    const int tubeSize = m_world.arms().config().tubeSize;
    const bool pumping = off.task == Task::Pump;
    const bool handFree = off.item.kind == ItemKind::Empty && !off.busy();
    const bool holdingShell = off.item.kind == ItemKind::Shell && !off.busy();
    const Chamber chamber = arms.shotgun.chamber;
    const int tube = arms.shotgun.tube;
    const bool fighting = (enemyLeft && visible) || threatened;
    const bool chamberReady = chamber == Chamber::Loaded && !pumping;
    const bool wantLoad = tube < tubeSize && shellsCarried > 0 &&
        (!fighting || (tube == 0 && chamber != Chamber::Loaded));
    const bool needPump = chamber != Chamber::Loaded && tube > 0 && !pumping;
    // A shell in hand goes in first (a press of reload); pumping would drop it.
    const bool insertHeld = holdingShell && tube < tubeSize;
    if (needPump && (fighting || !wantLoad) && !insertHeld) {
        bot.pump = handFree || holdingShell;
        bot.pumps += bot.pump ? 1 : 0;
    }
    else if (insertHeld) {
        bot.reload = !bot.reloadWasHeld;
    }
    else if (wantLoad && handFree) {
        // Take a shell (Q / E), then R pushes it in.
        bot.zone = static_cast<int>(arms.zones[static_cast<int>(Zone::Bandolier)].shells > 0 ? Zone::Bandolier
                                                                                            : Zone::JacketPocket);
    }
    bot.reloadWasHeld = bot.reload;

    if (enemyLeft && visible) {
        bot.lostSight = 0.0f;
        const glm::vec3 toAim = glm::normalize(target - m_world.barrelOrigin());
        const float barrelError = glm::degrees(std::acos(std::clamp(glm::dot(m_world.barrelDirection(), toAim), -1.0f, 1.0f)));
        bot.fire = chamberReady && barrelError < 3.0f && distance < 14.0f;
        bot.shots += bot.fire ? 1 : 0;
        // Close in to shotgun range, and keep out of reach while pumping: three hits kill.
        if (distance > 8.0f)
            bot.move = flatDirection;
        else if (distance < 5.5f)
            bot.move = -flatDirection;
    }
    else {
        // Walk towards it; teleport closer when walking gets nowhere.
        bot.lostSight += dt;
        bot.move = flatDirection;
        if (!enemyLeft && distance < 1.5f)
            bot.move = glm::vec3(0.0f);
    }

    bot.stuckTime += dt;
    if (bot.stuckTime > 1.5f) {
        const bool stuck = glm::length(glm::vec2(feet.x - bot.lastPosition.x, feet.z - bot.lastPosition.z)) < 0.5f &&
            glm::length(bot.move) > 0.0f;
        bot.lastPosition = feet;
        bot.stuckTime = 0.0f;
        if (stuck || bot.lostSight > 6.0f) {
            // The exit itself, or a spot on the ground some meters from the enemy that can see it.
            const glm::vec3 base = target - glm::vec3(0.0f, enemyLeft ? 1.2f : 1.0f, 0.0f);
            glm::vec3 place = base + glm::vec3(0.0f, 0.1f, 0.0f);
            if (enemyLeft) {
                bool found = false;
                for (int i = 0; i < 16 && !found; ++i) {
                    const float angle = std::atan2(-flatDirection.z, -flatDirection.x) + glm::radians(22.5f * i);
                    const glm::vec3 candidate = base + glm::vec3(std::cos(angle), 0.0f, std::sin(angle)) * 6.0f;
                    float ground = 0.0f, blocked = 0.0f;
                    const glm::vec3 top = candidate + glm::vec3(0.0f, 2.5f, 0.0f);
                    if (!m_physics.castRay(top, glm::vec3(0.0f, -1.0f, 0.0f), 5.0f, ground) || ground < 0.5f)
                        continue; // no floor, or inside something
                    const glm::vec3 feetCandidate = top - glm::vec3(0.0f, ground - 0.05f, 0.0f);
                    const glm::vec3 eyeCandidate = feetCandidate + glm::vec3(0.0f, kEyeHeight, 0.0f);
                    if (m_physics.castRay(eyeCandidate, target - eyeCandidate, glm::length(target - eyeCandidate) - 0.3f, blocked))
                        continue; // can't see the enemy from there
                    place = feetCandidate;
                    found = true;
                }
            }
            m_physics.spawnPlayer(place);
            bot.lastPosition = place;
            bot.lostSight = 0.0f;
            ++bot.teleports;
            LOG_INFO("[AUTOPLAY] Teleported to (" << place.x << ", " << place.y << ", " << place.z << ")\n");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------------------------

void Game::drawUi()
{
    switch (m_state) {
    case State::MainMenu: drawMainMenu(); break;
    case State::Loading:  drawLoadingScreen(); break;
    case State::Paused:   drawHud(); drawPauseMenu(); break;
    case State::Settings: drawSettingsScreen(); break;
    case State::Controls: drawControlsScreen(); break;
    case State::Playing:  drawHud(); break;
    case State::Dead:     drawHud(); drawDeathScreen(); break;
    case State::Complete: drawCompleteScreen(); break;
    }
}

bool Game::beginScreen(const char* id, float backgroundAlpha)
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.065f, 0.08f, backgroundAlpha));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    const bool open = ImGui::Begin(id, nullptr, kScreenFlags);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    return open;
}

bool Game::menuButton(const char* label)
{
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - kButtonWidth) * 0.5f);
    const bool pressed = ImGui::Button(label, ImVec2(kButtonWidth, kButtonHeight));
    ImGui::Dummy(ImVec2(0.0f, 6.0f));
    return pressed;
}

namespace {

void centeredText(const char* text, float scale, const ImVec4& color)
{
    ImGui::SetWindowFontScale(scale);
    const float width = ImGui::CalcTextSize(text).x;
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - width) * 0.5f);
    ImGui::TextColored(color, "%s", text);
    ImGui::SetWindowFontScale(1.0f);
}

// Vertically centers a block of the given height in the current window.
void centerBlock(float height)
{
    ImGui::SetCursorPosY(std::max((ImGui::GetWindowHeight() - height) * 0.5f, 20.0f));
}

std::string formatTime(float seconds)
{
    const int total = static_cast<int>(seconds);
    char text[32];
    snprintf(text, sizeof(text), "%d:%02d", total / 60, total % 60);
    return text;
}

} // namespace

void Game::drawMainMenu()
{
    if (beginScreen("##MainMenu", 1.0f)) {
        const float blockHeight = 4.0f * (kButtonHeight + 14.0f) + 160.0f;
        centerBlock(blockHeight);
        centeredText("MirasEngine", 3.0f, ImVec4(0.95f, 0.95f, 0.97f, 1.0f));
        centeredText("Level 1: The Yard", 1.3f, ImVec4(0.7f, 0.72f, 0.78f, 1.0f));
        ImGui::Dummy(ImVec2(0.0f, 30.0f));

        ImGui::SetWindowFontScale(1.4f);
        if (menuButton("Play")) startLoading();
        if (menuButton("Controls")) openScreen(State::Controls);
        if (menuButton("Settings")) openScreen(State::Settings);
        if (menuButton("Exit")) m_quitRequested = true;
        ImGui::SetWindowFontScale(1.0f);

        if (!m_error.empty()) {
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            centeredText(m_error.c_str(), 1.2f, ImVec4(1.0f, 0.4f, 0.35f, 1.0f));
        }
    }
    ImGui::End();
}

void Game::drawLoadingScreen()
{
    if (beginScreen("##Loading", 1.0f)) {
        centerBlock(60.0f);
        static constexpr char kSpinner[] = "|/-\\";
        const char spin = kSpinner[static_cast<int>(ImGui::GetTime() * 8.0) % 4];
        const std::string text = std::string("Loading ") + m_levelPath + "...  " + spin;
        centeredText(text.c_str(), 1.6f, ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
    }
    ImGui::End();
}

void Game::drawPauseMenu()
{
    if (beginScreen("##Paused", 0.6f)) {
        centerBlock(6.0f * (kButtonHeight + 14.0f) + 110.0f);
        centeredText("Paused", 2.2f, ImVec4(0.95f, 0.95f, 0.97f, 1.0f));
        ImGui::Dummy(ImVec2(0.0f, 30.0f));
        ImGui::SetWindowFontScale(1.4f);
        if (menuButton("Resume")) setState(State::Playing);
        if (menuButton("Restart Level")) restartLevel();
        if (menuButton("Controls")) openScreen(State::Controls);
        if (menuButton("Settings")) openScreen(State::Settings);
        if (m_playInEditor) {
            if (menuButton("Stop (Back to Editor)")) stopPlayInEditor();
        }
        else {
            if (menuButton("Main Menu")) returnToMainMenu();
            if (menuButton("Exit")) m_quitRequested = true;
        }
        ImGui::SetWindowFontScale(1.0f);
    }
    ImGui::End();
}

void Game::drawDeathScreen()
{
    // The red fade shows the scene behind for a moment before the buttons appear.
    const float alpha = std::clamp(m_stateTime / 0.4f, 0.0f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.25f, 0.0f, 0.0f, 0.55f * alpha));
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    if (ImGui::Begin("##Dead", nullptr, kScreenFlags)) {
        centerBlock(3.0f * (kButtonHeight + 14.0f) + 120.0f);
        centeredText("You died", 3.0f, ImVec4(1.0f, 0.35f, 0.3f, alpha));
        const std::string stats = std::to_string(m_world.enemiesKilled()) + " of " +
            std::to_string(m_world.enemiesTotal()) + " enemies killed";
        centeredText(stats.c_str(), 1.2f, ImVec4(0.9f, 0.85f, 0.85f, alpha));
        centeredText("R / Space: try again", 1.2f, ImVec4(1.0f, 0.92f, 0.6f, m_stateTime > kRestartDelay ? alpha : 0.0f));
        ImGui::Dummy(ImVec2(0.0f, 30.0f));
        if (m_stateTime > 0.6f) {
            ImGui::SetWindowFontScale(1.4f);
            if (menuButton("Try Again")) restartLevel();
            if (m_playInEditor) {
                if (menuButton("Back to Editor")) stopPlayInEditor();
            }
            else if (menuButton("Main Menu")) {
                returnToMainMenu();
            }
            ImGui::SetWindowFontScale(1.0f);
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void Game::drawCompleteScreen()
{
    if (beginScreen("##Complete", 0.7f)) {
        centerBlock(4.0f * (kButtonHeight + 14.0f) + 170.0f);
        centeredText("Level complete", 3.0f, ImVec4(0.55f, 1.0f, 0.7f, 1.0f));
        const std::string stats = "Time " + formatTime(m_world.levelTime()) + "    Enemies " +
            std::to_string(m_world.enemiesKilled()) + "/" + std::to_string(m_world.enemiesTotal()) + "    Health " +
            std::to_string(static_cast<int>(m_world.player().health));
        centeredText(stats.c_str(), 1.3f, ImVec4(0.9f, 0.92f, 0.95f, 1.0f));
        ImGui::Dummy(ImVec2(0.0f, 30.0f));
        ImGui::SetWindowFontScale(1.4f);
        const std::string next = m_world.nextLevel();
        if (!next.empty() && !m_playInEditor && std::filesystem::exists(next) && menuButton("Next Level")) {
            endLevel();
            m_levelPath = next;
            startLoading();
        }
        if (menuButton("Play Again")) restartLevel();
        if (m_playInEditor) {
            if (menuButton("Back to Editor")) stopPlayInEditor();
        }
        else {
            if (menuButton("Main Menu")) returnToMainMenu();
            if (menuButton("Exit")) m_quitRequested = true;
        }
        ImGui::SetWindowFontScale(1.0f);
    }
    ImGui::End();
}

void Game::drawSettingsScreen()
{
    // Over the paused scene the settings stay translucent so their effect is visible.
    const float alpha = m_screenReturn == State::Paused ? 0.75f : 1.0f;
    if (beginScreen("##Settings", alpha)) {
        const float panelWidth = 480.0f;
        const float panelHeight = std::clamp(ImGui::GetWindowHeight() - 260.0f, 150.0f, 460.0f);
        centerBlock(panelHeight + 120.0f);
        centeredText("Settings", 2.2f, ImVec4(0.95f, 0.95f, 0.97f, 1.0f));
        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - panelWidth) * 0.5f);
        if (ImGui::BeginChild("##settingsPanel", ImVec2(panelWidth, panelHeight), ImGuiChildFlags_Borders))
            drawGraphicsSettings(m_settings, m_renderer.capabilities());
        ImGui::EndChild();
        ImGui::Dummy(ImVec2(0.0f, 16.0f));
        ImGui::SetWindowFontScale(1.4f);
        if (menuButton("Back")) setState(m_screenReturn);
        ImGui::SetWindowFontScale(1.0f);
    }
    ImGui::End();
}

void Game::openScreen(State screen)
{
    m_screenReturn = m_state;
    setState(screen);
}

void Game::drawControlsScreen()
{
    struct Row {
        const char* keys;
        const char* action;
    };
    // Headings have no keys.
    static constexpr Row kRows[] = {
        { nullptr, "Moving" },
        { "W A S D", "Walk" },
        { "Shift", "Sprint" },
        { "Space", "Jump" },
        { "Mouse", "Look; the gun follows, a little behind" },
        { nullptr, "Hands" },
        { "1 / 2 / 3", "Right hand takes the shotgun / the pistol / a magazine (to load rounds into)" },
        { "G", "Left hand takes the other gun, or puts it away (a gun in each hand: no reloading)" },
        { "Left click", "Fire the gun in the right hand" },
        { "Right click", "Fire the gun in the left hand; otherwise pump the shotgun / rack the pistol" },
        { "T", "Pick up the nearest dropped item" },
        { nullptr, "Shotgun" },
        { "Q / E", "Take a shell from the bandolier (quick) / the jacket pocket (slow)" },
        { "R", "Push the shell in your left hand into the tube" },
        { nullptr, "Pistol" },
        { "X", "Drop the magazine out: into the free left hand, or onto the floor" },
        { "R", "Reload step by step: magazine out, put an empty one away, fetch the fullest, push it in, rack" },
        { "V", "Reload even with a gun in each hand: the left hand drops its gun first" },
        { "Holding a magazine", "R presses one loose round in (from the right pocket)" },
        { nullptr, "Inventory" },
        { "Q / E / C", "With the left hand empty: take from the bandolier / jacket pocket / right pocket" },
        { "Q / E / C", "With something in the left hand: put it there" },
        { "F", "Put what the left hand holds back where it came from" },
        { "Tab (hold)", "Open the jacket and look at what you carry" },
        { "Tab + Q (hold)", "Move shells from the pocket to the bandolier" },
        { nullptr, "Game" },
        { "Esc", "Pause" },
        { "R / Space", "Try again after dying" },
    };
    const float alpha = m_screenReturn == State::Paused ? 0.85f : 1.0f;
    if (beginScreen("##Controls", alpha)) {
        const float panelWidth = 640.0f;
        const float panelHeight = std::clamp(ImGui::GetWindowHeight() - 260.0f, 150.0f, 520.0f);
        centerBlock(panelHeight + 120.0f);
        centeredText("Controls", 2.2f, ImVec4(0.95f, 0.95f, 0.97f, 1.0f));
        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - panelWidth) * 0.5f);
        if (ImGui::BeginChild("##controlsPanel", ImVec2(panelWidth, panelHeight), ImGuiChildFlags_Borders)) {
            ImGui::SetWindowFontScale(1.15f);
            if (ImGui::BeginTable("##controls", 2, ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableSetupColumn("keys", ImGuiTableColumnFlags_WidthFixed, 170.0f);
                ImGui::TableSetupColumn("action", ImGuiTableColumnFlags_WidthStretch);
                for (const Row& row : kRows) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    if (!row.keys) {
                        ImGui::Dummy(ImVec2(0.0f, 4.0f));
                        ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.45f, 1.0f), "%s", row.action);
                        continue;
                    }
                    ImGui::TextColored(ImVec4(0.75f, 0.85f, 1.0f, 1.0f), "%s", row.keys);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextWrapped("%s", row.action);
                }
                ImGui::EndTable();
            }
            ImGui::SetWindowFontScale(1.0f);
        }
        ImGui::EndChild();
        ImGui::Dummy(ImVec2(0.0f, 16.0f));
        ImGui::SetWindowFontScale(1.4f);
        if (menuButton("Back")) setState(m_screenReturn);
        ImGui::SetWindowFontScale(1.0f);
    }
    ImGui::End();
}
