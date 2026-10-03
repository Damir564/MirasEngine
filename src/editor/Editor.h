#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>
#include "imgui.h"
#include "app/AppMode.h"
#include "engine/Camera.h"
#include "engine/CameraAnimation.h"
#include "engine/Gizmo.h"
#include "engine/GraphicsSettings.h"
#include "engine/ModelManager.h"
#include "engine/PolyMesh.h"
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
    static constexpr const char* kPrefabsRoot = "prefabs";
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
    // Selects the object under the mouse, or deselects when there is none.
    void pickObject(float mouseX, float mouseY);
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

    // ---- EditorLevel.cpp ----
    void drawLevelPanel();
    void drawShapeMenuItems();
    void addLevelShape(PolyShape shape);
    // Uploads the mesh as a new model and returns its index, or nothing on failure.
    std::optional<size_t> createLevelModel(PolyMesh mesh, const std::string& name);
    // Editable geometry of the selected instance's model, or null when it is not a level model or is locked.
    PolyMesh* selectedLevelMesh();
    // The selected face, or null when face editing is off or the face belongs to another instance.
    PolyFace* selectedLevelFace();
    // Same for vertex editing; drops indices the mesh no longer has. Empty when there is none.
    const std::vector<uint32_t>& selectedLevelVertices();
    // Face of the selected level instance under the mouse (scene-view pixels), or -1.
    int pickLevelFace(float mouseX, float mouseY) const;
    // Vertex of the selected level instance drawn closest to the mouse, within a few pixels, or -1.
    int pickLevelVertex(float mouseX, float mouseY) const;
    // Face/vertex picking for a viewport click; false when nothing was hit and objects should be picked.
    bool handleLevelClick(float mouseX, float mouseY);
    void dragLevelVertex(float mouseX, float mouseY);
    // Selects the vertices inside the box, or picks an object when the mouse barely moved.
    void finishVertexMarquee(float mouseX, float mouseY);
    void drawLevelFaceOverlay();
    void drawFaceProperties(PolyMesh& mesh, PolyFace& face);
    void drawVertexProperties(PolyMesh& mesh);
    // Push/pull and extrude; by index, because extruding adds faces and moves the face storage.
    void drawFaceGeometry(PolyMesh& mesh, uint32_t faceIndex);
    void drawLevelMaterials(PolyMesh& mesh);
    void drawLevelTextureDialog();
    // Uploads the selected shape after an edit and marks the edit for the undo history.
    void rebuildSelectedLevelModel(const char* action);
    // Replaces the selected shape with an edited copy, unless the edit removed every face.
    bool commitLevelTopology(PolyMesh&& edited, const char* action);
    void deleteLevelFace();
    void deleteLevelVertices();
    void mergeLevelVertices();
    // Both act on exactly two selected vertices: split the edge between them, or cut a face along them.
    void splitLevelEdge();
    void connectLevelVertices();
    enum class ClipKeep { Back, Front, Both };
    void drawLevelClip(PolyMesh& mesh);
    // Two selected vertices: plane through them along the view direction. Three or more: through the first three.
    void setLevelClipFromVertices();
    void applyLevelClip(ClipKeep keep);
    enum class MirrorAction { InPlace, Copy, SymmetrizePositive, SymmetrizeNegative };
    void drawLevelMirror();
    float levelMirrorPivot(const PolyMesh& mesh) const;
    void applyLevelMirror(MirrorAction action);
    void drawLevelHollow();
    // Subtract cuts another level object (the cutter) out of the selected one.
    void drawLevelSubtract();
    int levelCutterInstance() const; // -1 if the chosen cutter is gone or is the selection
    void applyLevelSubtract();
    // Del in face or vertex mode deletes the selected face or vertices. False in object mode, where
    // Del deletes the object instead.
    bool deleteLevelSelection();

    // Level models belong to their instances, so they are unloaded once no instance uses them.
    void releaseUnusedLevelModels();
    glm::vec3 placementPoint() const;

    // ---- EditorPrefab.cpp ----
    // The instance's model when it is a level model linked to a prefab file, else null.
    GPUModel* prefabModel(int instanceIndex);
    size_t prefabInstanceCount(size_t modelIndex) const;
    void savePrefabDialog(int instanceIndex);
    void loadPrefabDialog();
    void drawPrefabDialogs();
    // Writes the instance's level geometry to a prefab file and links the instance to it, locked.
    void saveAsPrefab(int instanceIndex, const std::string& path);
    // Adds a locked instance of the prefab; it shares the model of instances already in the scene.
    void addPrefab(const std::string& path);
    // Prefab files in kPrefabsRoot plus a Browse item.
    void drawPrefabMenuItems();
    // Level panel: link, lock and file actions of the selected prefab instance.
    void drawPrefabSection();
    void setInstanceLocked(int instanceIndex, bool locked);
    // Gives the instance its own copy of the geometry, no longer tied to the prefab.
    void unlinkPrefab(int instanceIndex);
    // Overwrites the prefab file with the current shared geometry.
    void writePrefabFile(int instanceIndex);

    // ---- EditorHistory.cpp ----
    // Undo history. Models are named by source path, which stays valid while other models are added
    // or removed, and objects by ModelInstance::id, which stays valid while indices shift.
    // An object as one side of a change.
    struct ObjectRecord {
        ModelInstance instance; // modelIndex is resolved from modelPath on restore
        size_t index = 0;       // position in the scene list
        std::string modelPath;
        // Level models: rebuilds the model when it was released together with its last instance.
        std::string modelName;
        std::string prefabPath;
        std::shared_ptr<const PolyMesh> mesh;
    };
    struct ObjectChange {
        std::optional<ObjectRecord> before; // empty: the object was created
        std::optional<ObjectRecord> after;  // empty: the object was deleted
    };
    // One step: a level geometry edit, object changes (transform, color, visibility, name, lock, model,
    // existence), or both when one action did both (e.g. subtract deleting the cutter).
    struct HistoryEntry {
        std::string action;
        std::string modelPath; // empty when no geometry changed
        PolyMesh before;
        PolyMesh after;
        std::vector<ObjectChange> objects;
    };
    // Turns finished geometry edits into history entries; runs once per frame after the panels.
    void updateLevelHistory();
    // Actions that add, delete or change objects call this; only then is the scene compared with the
    // snapshot, once the action (e.g. a drag) has finished.
    void markSceneChanged() { m_sceneChanged = true; }
    // Turns a marked change into a history entry; runs right after updateLevelHistory().
    void updateObjectHistory();
    // Retakes the snapshot without recording, e.g. after a scene load, an undo, or unloading a model.
    void resetObjectBaseline();
    ObjectRecord makeObjectRecord(size_t index,
        std::unordered_map<std::string, std::shared_ptr<const PolyMesh>>& meshCopies) const;
    bool objectChangedSinceBaseline();
    // A gizmo, vertex or widget drag is still going; it becomes one entry when it ends.
    bool editInProgress() const;
    static bool sameObjectState(const ObjectRecord& a, const ObjectRecord& b);
    static std::string describeObjectChanges(const std::vector<ObjectChange>& changes);
    // Model index for a record, rebuilding a released level model under its old path if needed.
    std::optional<size_t> resolveRecordModel(const ObjectRecord& record);
    bool applyObjectChanges(const std::vector<ObjectChange>& changes, bool undo);
    bool applyLevelEdit(const HistoryEntry& entry, bool undo);
    bool applyHistoryEntry(const HistoryEntry& entry, bool undo);
    void pushHistory(HistoryEntry entry);
    void undo();
    void redo();
    void clearHistory();

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
    bool m_showLevelPanel = true;
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

    // Level tool
    PolyShapeParams m_newShape;
    // Face and vertex modes: clicks on the selected level shape pick its parts instead of objects.
    enum class LevelEditMode { Object, Face, Vertex };
    LevelEditMode m_levelMode = LevelEditMode::Object;
    int m_selectedFace = -1;
    std::vector<uint32_t> m_selectedVertices;
    int m_levelSelectionInstance = -1; // face and vertex indices are only meaningful for this instance
    // The clicked vertex is dragged in the plane facing the camera through its start position; the
    // rest of the selection follows by the same world offset.
    struct VertexDrag {
        bool active = false;
        bool moved = false; // positions changed since the last rebuild
        glm::vec3 planePoint{ 0.0f }; // world start of the clicked vertex
        glm::vec3 grabOffset{ 0.0f }; // vertex minus the plane point under the mouse, so it does not jump
        std::vector<glm::vec3> startPositions; // object space, parallel to m_selectedVertices
    };
    VertexDrag m_vertexDrag;
    // Box selection started on empty space in vertex mode; scene-view pixels.
    struct VertexMarquee {
        bool active = false;
        bool additive = false; // Shift: add to the selection instead of replacing it
        glm::vec2 start{ 0.0f };
        glm::vec2 end{ 0.0f };
    };
    VertexMarquee m_vertexMarquee;
    int m_textureDialogMaterial = -1; // material of the selected shape the open texture dialog is for
    float m_faceOpDistance = 1.0f;
    // Plane dot(normal, p) == offset in the selected shape's object space. "Front" is the side the
    // normal points to.
    struct LevelClip {
        glm::vec3 normal{ 0.0f, 1.0f, 0.0f };
        float offset = 1.0f;
        bool preview = false; // the Clip section was open last frame
    };
    LevelClip m_levelClip;
    // Mirror plane: coordinate `axis` of the selected shape's object space equals the pivot.
    enum class MirrorPivot { Origin, Center, Min, Max };
    struct LevelMirror {
        int axis = 0;
        MirrorPivot pivot = MirrorPivot::Center;
        bool preview = false; // the Mirror section was open last frame
    };
    LevelMirror m_levelMirror;
    float m_hollowThickness = 0.25f;
    // Kept by name: instance indices shift when objects are deleted.
    std::string m_levelCutterName;
    bool m_deleteCutter = false;
    // Instance the open "Save as Prefab" dialog is for; the name catches index shifts from deletes.
    int m_prefabSaveInstance = -1;
    std::string m_prefabSaveName;

    // Undo history
    std::vector<HistoryEntry> m_undoStack;
    std::vector<HistoryEntry> m_redoStack;
    // The selected shape's mesh as of the last history entry; a finished edit is stored against it.
    std::string m_levelBaselinePath;
    PolyMesh m_levelBaseline;
    // Set by edits and kept while a widget is still being dragged, so one drag is one entry.
    bool m_levelEditPending = false;
    std::string m_levelEditAction;
    // A geometry entry was pushed this frame; object changes of the same frame join it.
    bool m_levelEntryPushed = false;
    // Every object as of the last history entry, in scene order.
    std::vector<ObjectRecord> m_objectBaseline;
    bool m_objectBaselineValid = false;
    bool m_sceneChanged = false;
};
