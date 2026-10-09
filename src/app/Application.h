#pragma once
#include <memory>
#include "AppOptions.h"
#include "engine/Audio.h"
#include "engine/GraphicsSettings.h"
#include "engine/MaterialLibrary.h"
#include "engine/Renderer.h"
#include "engine/VulkanContext.h"

struct SDL_Window;
class AppMode;
struct EngineContext;
class ModelManager;
class SceneManager;

// Owns the window, the engine objects and the ImGui context, and runs the active AppMode.
class Application {
public:
    Application();
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    // Returns the process exit code.
    int run(const AppOptions& options);

private:
    bool init(const AppOptions& options);
    bool initWindow();
    void initImGui();
    bool createModelManager();
    void createMode(const AppOptions& options);
    EngineContext engineContext();
    void handleModeRequest();
    // Ends a game started from the editor and resumes the editor; nothing when no mode is suspended.
    void resumeSuspendedMode();
    void shutdown();

    int mainLoop(int exitAfterFrames);
    void pollEvents();
    // False while there is nothing to render into (minimized, zero-sized, swapchain not rebuildable).
    bool readyToRender();
    ImDrawData* buildUi();
    // Applies the open scene's SceneSettings over m_settings and pushes what changed this frame to the
    // renderer; changes to the global part also go to settings.json.
    void applyChangedSettings();
    // Sleeps until the frame that started at frameStartNs has lasted 1 / maxFps.
    void limitFrameRate(uint64_t frameStartNs) const;

    SDL_Window* m_window = nullptr;
    bool m_sdlInitialized = false;
    bool m_imguiInitialized = false;
    bool m_quit = false;

    std::string m_settingsPath;
    GraphicsSettings m_settings;
    GraphicsSettings m_appliedSettings;
    bool m_muted = false;    // --mute
    bool m_autoplay = false; // --autoplay

    // Outlives the modes, which play sounds through it.
    AudioSystem m_audio;

    // Destruction order matters: the mode uses everything, SceneManager uses ModelManager, and
    // ModelManager allocates from the renderer's pools and reads the material library; all of them
    // need the Vulkan device.
    MaterialLibrary m_materials;
    VulkanContext m_vulkan;
    Renderer m_renderer;
    std::unique_ptr<ModelManager> m_models;
    std::unique_ptr<SceneManager> m_scenes;
    std::unique_ptr<AppMode> m_mode;
    // The editor while a scene it started is being played.
    std::unique_ptr<AppMode> m_suspendedMode;
};
