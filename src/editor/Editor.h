#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "imgui.h"
#include "app/AppMode.h"
#include "engine/Camera.h"
#include "engine/CameraAnimation.h"
#include "engine/Gizmo.h"
#include "engine/GraphicsSettings.h"
#include "engine/ModelManager.h"
#include "engine/Renderer.h"

// The scene editor: viewport interaction (picking, gizmo, camera) plus the docked ImGui panels.
// The implementation is split by panel across the Editor*.cpp files.
class Editor final : public AppMode {
public:
    explicit Editor(const EngineContext& engine);

    Editor(const Editor&) = delete;
    Editor& operator=(const Editor&) = delete;

    void onEvent(const SDL_Event& event) override;
    void update(float dt) override;
    void lateUpdate(float dt) override;
    bool uiVisible() const override { return !m_flyMode; }
    void drawUi() override;
    void fillFrame(FrameInput& frame) override;
    bool quitRequested() const override { return m_quitRequested; }
    ModeRequest takeModeRequest() override;
    void onResume() override;

    // Replaces the current scene and reports the outcome in the status bar.
    void openScene(const std::string& path);

private:
    // Hierarchy actions are applied after the tree is drawn so the instance list is stable while drawing.
    struct HierarchyActions {
        int deleteIndex = -1;
        int duplicateIndex = -1;
        int focusIndex = -1;
        bool createCube = false;
    };

    static constexpr const char* kScenesRoot = ".";
    static constexpr const char* kModelsRoot = "models";
    static constexpr const char* kCameraPathsRoot = ".";
    static constexpr ImVec2 kDialogSize{ 760.0f, 480.0f };
    static constexpr float kGizmoPickRadius = 12.0f;

    // One end of the scene orientation gizmo, in scene-view coordinates.
    struct OrientationHandle {
        int axis = 0; // 0..2 = +X +Y +Z, 3..5 = -X -Y -Z
        glm::vec3 direction{ 0.0f };
        glm::vec2 position{ 0.0f };
        float depth = 0.0f; // larger = closer to the viewer
    };

    // Smooth camera turn started from the orientation gizmo.
    struct ViewTurn {
        bool active = false;
        float elapsed = 0.0f;
        glm::vec3 pivot{ 0.0f };
        glm::vec3 startPosition{ 0.0f };
        float startYaw = 0.0f;
        float startPitch = 0.0f;
        float targetYaw = 0.0f;
        float targetPitch = 0.0f;
        float distance = 10.0f;
    };

    // ---- Editor.cpp: input ----
    void handleKeyDown(const SDL_KeyboardEvent& key);
    void dropStaleGizmoSelection();
    void handleViewportMouse(const SDL_Event& event);
    void handleViewportClick(float mouseX, float mouseY);
    bool tryBeginGizmoDrag(float mouseX, float mouseY, const glm::mat4& view, const glm::mat4& proj);
    void selectPickedSubmesh(const SubmeshHitResult& hit);
    void dragGizmo(float mouseX, float mouseY);
    bool gizmoVisible() const;
    GizmoShape currentGizmoShape() const;
    void handleCameraLook(const SDL_Event& event);
    void moveCamera(float dt);
    void setFlyMode(bool enabled);
    bool canPlay() const;
    void requestPlay();
    bool sceneViewContains(float x, float y) const;
    // Projection of the scene view; rendering, picking and gizmos all use this one.
    glm::mat4 sceneProjection() const;

    // ---- Editor.cpp: selection and actions ----
    bool validInstance(int index) const;
    bool hasSelection() const { return validInstance(m_gizmo.selectedInstance); }
    void validateSelection();
    void selectInstance(int index);
    void deselectAll();
    void focusOnInstance(int index);
    void addModelToScene(size_t modelIndex, bool atOrigin);
    void deleteInstance(int index);
    void duplicateInstance(int index);
    void unloadModel(size_t modelIndex);
    void newScene();
    void saveSceneTo(const std::string& path);
    void saveScene();
    void handleShortcuts();
    void updateWindowTitle();
    void setStatus(const std::string& message, bool isError = false);
    static std::string formatCount(size_t value);
    void fillHighlight(SelectionHighlight& highlight) const;
    glm::vec3 instanceWorldCenter(int index) const;
    void addCube();
    std::string uniqueInstanceName(const std::string& base) const;
    void beginRename(int index);

    // ---- EditorMenus.cpp ----
    void drawMainMenuBar();
    void drawFileMenu();
    void drawEditMenu();
    void drawViewMenu();
    void drawToolbar();
    void drawToolButton(const char* label, GizmoMode tool, const char* tooltip);
    void drawSnapPopup();
    void drawAddMenu();
    void drawStatusBar();
    void drawDockSpace();
    void buildDefaultLayout(ImGuiID dockspaceId);
    void drawViewportOverlay();
    void drawHelpPopups();
    void drawGraphicsSettingsWindow();
    void drawControlsPopup();
    void drawAboutPopup();
    void openFileDialog(const char* key, const char* title, const char* filters, const std::string& root,
        const char* defaultFileName, bool confirmOverwrite);
    void openSceneDialog();
    void saveSceneAsDialog();
    void importModelDialog();
    void drawFileDialogs();

    // ---- EditorGizmo.cpp ----
    void drawTransformGizmo();
    std::array<OrientationHandle, 6> orientationHandles() const;
    glm::vec2 orientationCenter() const;
    int pickOrientationHandle(float x, float y) const;
    void drawOrientationGizmo();
    void beginViewTurn(int axis);
    void updateViewTurn(float dt);

    // ---- EditorHierarchy.cpp ----
    void drawHierarchy();
    void drawModelList();
    void drawSceneTree();
    void drawInstanceNode(int instanceIndex, HierarchyActions& actions);
    void drawInstanceContextMenu(int instanceIndex, HierarchyActions& actions);
    bool drawRenameField(int instanceIndex);

    // ---- EditorInspector.cpp ----
    void drawInspector();
    void drawInspectorHeader(ModelInstance& instance, const GPUModel* model);
    void drawTransformSection(int instanceIndex);

    // ---- EditorAnimation.cpp ----
    void drawAnimationPanel();
    void drawAnimationTransport();
    void drawKeyframeList();
    bool drawKeyframeDetails();
    void drawAnimationFileRow();
    void drawAnimationFileDialog();
    bool loadCameraPath();
    void rebuildPathLines();

    // ---- EditorStatistics.cpp ----
    void drawStatisticsPanel();

    SDL_Window* m_window;
    const VulkanContext& m_vulkan;
    Renderer& m_renderer;
    ModelManager& m_models;
    SceneManager& m_scenes;
    GraphicsSettings& m_settings;

    // Camera and viewport
    Camera m_camera;
    bool m_flyMode = false; // UI hidden, mouse always looks around
    bool m_rightMouseHeld = false;
    float m_cameraSpeedMultiplier = 1.0f;
    // Scene viewport = central dock area in window coordinates. The 3D scene renders only here,
    // and mouse picking / gizmo math is relative to it.
    ViewRect m_sceneView;
    CameraAnimator m_cameraAnimator;
    bool m_showCameraPath = true;
    bool m_showGrid = true;

    // Selection and tools
    Gizmo m_gizmo;
    GizmoMode m_tool = GizmoMode::Translate;
    GizmoAxis m_hoveredAxis = GizmoAxis::None;
    // Ctrl while dragging: absolute grid step for moves, increments for rotate/scale deltas.
    float m_snapTranslate = 1.0f;
    float m_snapRotate = 15.0f;
    float m_snapScale = 0.1f;

    // Orientation gizmo
    int m_hoveredOrientation = -1;
    ViewTurn m_viewTurn;

    // Panels and popups
    bool m_showHierarchy = true;
    bool m_showInspector = true;
    bool m_showAnimationPanel = true;
    bool m_showStatisticsPanel = true;
    bool m_showGraphicsSettings = false;
    bool m_resetLayout = false;
    bool m_openControlsPopup = false;
    bool m_openAboutPopup = false;
    bool m_quitRequested = false;
    ModeRequest m_modeRequest = ModeRequest::None;
    std::string m_windowTitle;

    // Status bar
    std::string m_statusMessage = "Ready";
    bool m_statusIsError = false;
    double m_statusTime = 0.0;

    // Hierarchy
    ImGuiTextFilter m_hierarchyFilter;
    int m_renamingInstance = -1;
    bool m_renameFocusPending = false;
    char m_renameBuffer[256] = "";

    // Camera animation panel
    char m_pathName[128] = "CameraPath1";
    char m_cameraPathFile[256] = "camera_path.cmap";
    float m_newKeyframeTime = 0.0f;
    bool m_newKeyframeCurved = true;
    int m_selectedKeyframe = -1;

    // Statistics panel
    float m_frameTimes[240] = {};
    int m_frameTimeOffset = 0;
};
