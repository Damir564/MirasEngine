#pragma once
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

class VulkanContext;
class Renderer;
class ModelManager;
class SceneManager;
struct FrameInput;
struct GraphicsSettings;
class MaterialLibrary;
class AudioSystem;

// Engine objects owned by Application that a mode works with. They outlive the mode.
struct EngineContext {
    SDL_Window* window;
    VulkanContext& vulkan;
    Renderer& renderer;
    ModelManager& models;
    SceneManager& scenes;
    // Modes may edit this; Application applies and saves it after the UI ran each frame.
    GraphicsSettings& settings;
    // Shared materials (.mat files); ModelManager resolves level slots and overrides through it.
    MaterialLibrary& materials;
    // Sound playback; silent when there is no audio device. Volumes follow the settings.
    AudioSystem& audio;
    // Launch flags the modes care about.
    bool autoplay = false; // game: a bot plays the level (automated tests)
};

// Asks Application to switch modes after the current frame.
enum class ModeRequest {
    None,
    PlayScene,       // editor: play the open scene in a game session, keeping the editor suspended
    ReturnToEditor,  // game started from the editor: stop and resume the editor
};

// What the application runs: the editor or the game. Application drives the
// per-frame order: onEvent* -> update -> (drawUi if uiVisible) -> lateUpdate -> fillFrame.
class AppMode {
public:
    virtual ~AppMode() = default;

    virtual void onEvent(const SDL_Event& event) = 0;
    virtual void update(float dt) = 0;
    // Runs after the UI so that anything the UI changed this frame can still be overridden.
    virtual void lateUpdate(float /*dt*/) {}
    // While false, ImGui receives no events and no ImGui frame is built.
    virtual bool uiVisible() const = 0;
    // Right before ImGui::NewFrame(), outside any ImGui frame: where the style, font size or a saved dock
    // layout may be changed.
    virtual void beforeUiFrame() {}
    virtual void drawUi() = 0;
    // Application has already set the window size, ImGui draw data and time.
    virtual void fillFrame(FrameInput& frame) = 0;
    virtual bool quitRequested() const = 0;
    // The window is being closed: true to quit now; false when the mode asks the user first (and then
    // reports quitRequested() itself).
    virtual bool confirmQuit() { return true; }
    // Returns and clears the pending request.
    virtual ModeRequest takeModeRequest() { return ModeRequest::None; }
    // Called when a suspended mode becomes active again.
    virtual void onResume() {}
};
