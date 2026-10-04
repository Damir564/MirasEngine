#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>
#include <nlohmann/json_fwd.hpp>
#include "imgui.h"
#include "app/AppMode.h"
#include "engine/Camera.h"
#include "engine/CameraAnimation.h"
#include "engine/Gizmo.h"
#include "engine/GraphicsSettings.h"
#include "engine/MaterialLibrary.h"
#include "engine/ModelManager.h"
#include "engine/PolyMesh.h"
#include "engine/Renderer.h"

class McpServer;

// The scene editor: viewport interaction (picking, gizmo, camera) plus the docked ImGui panels.
// The implementation is split by panel across the Editor*.cpp files.
class Editor final : public AppMode {
public:
    explicit Editor(const EngineContext& engine);
    ~Editor() override;

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
    // Lets MCP clients (e.g. Claude Code) edit the scene through http://127.0.0.1:<port>/mcp.
    void startMcpServer(uint16_t port);
    // Runs a command script (one MCP command per line, see EditorCommands.cpp) as one undo step, once
    // no edit is in progress and the scene has loaded.
    void runScript(const std::string& path);

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
    // m_gridSnap, inverted while Ctrl is held.
    bool snapActive() const;
    // Face/vertex editing of a level shape: its own object-space grid replaces the ground grid.
    bool levelGridActive();
    // The grid in use: the edited shape's (PolyMesh::gridSize) or the world grid.
    float activeGridSize();
    void setActiveGridSize(float size);
    // Moves the active grid to the next larger (+1) or smaller (-1) preset.
    void stepGridSize(int direction);

    // ---- EditorMenus.cpp ----
    void drawMainMenuBar();
    void drawFileMenu();
    void drawEditMenu();
    void drawViewMenu();
    void drawToolbar();
    void drawToolButton(const char* label, GizmoMode tool, const char* tooltip);
    void drawSnapPopup();
    void drawGridControls();
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
    void runScriptDialog();
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
    // The active selected face, or null when face editing is off or the face belongs to another instance.
    PolyFace* selectedLevelFace();
    // Every selected face (active one first); drops indices the mesh no longer has. Empty when none.
    const std::vector<uint32_t>& selectedLevelFaces();
    void clearFaceSelection();
    // Plain click: just this face (kept as is when it is already selected, so the group stays);
    // toggle: Shift+click adds or removes it.
    void selectLevelFace(uint32_t face, bool toggle);
    void selectFacesWhere(const std::function<bool(const PolyMesh&, const PolyFace&)>& predicate);
    // Same for vertex editing; drops indices the mesh no longer has. Empty when there is none.
    const std::vector<uint32_t>& selectedLevelVertices();
    // Face of the level instance (default: the selected one) under the mouse (scene-view pixels), or -1.
    int pickLevelFace(float mouseX, float mouseY, int instanceIndex = -1) const;
    // Vertex of the selected level instance drawn closest to the mouse, within a few pixels, or -1.
    int pickLevelVertex(float mouseX, float mouseY) const;
    // Face/vertex picking for a viewport click; false when nothing was hit and objects should be picked.
    bool handleLevelClick(float mouseX, float mouseY);
    void dragLevelVertex(float mouseX, float mouseY);
    // The selected edge (edge mode, selected instance); false when there is none or undo removed it.
    bool selectedLevelEdge(uint32_t& a, uint32_t& b);
    // Edge of the selected level instance drawn closest to the mouse, within a few pixels.
    bool pickLevelEdge(float mouseX, float mouseY, uint32_t& a, uint32_t& b) const;
    // Position along the line point + t * axis (object space of the instance; t in object units)
    // closest to the mouse ray; false when the view looks straight along the line.
    bool lineDragParam(int instance, const glm::vec3& point, const glm::vec3& axis, float mouseX, float mouseY,
        float& param) const;
    void dragLevelFace(float mouseX, float mouseY);
    void dragLevelEdge(float mouseX, float mouseY);
    // Drawing on a face: false (and drawing ends) once the face or selection it was started on is gone.
    bool faceDrawValid();
    // Point on the drawn face's plane under the mouse: a face corner within a few pixels, else snapped to
    // the shape's grid (or, without snapping, to a nearby edge). False when the plane isn't under the mouse.
    bool faceDrawPoint(float mouseX, float mouseY, glm::vec3& out);
    // The shape's outline from the clicked points plus, if given, the point under the mouse.
    std::vector<glm::vec3> faceDrawOutline(const glm::vec3* hover);
    void handleFaceDrawClick(float mouseX, float mouseY);
    // closed: the last point joins the first; otherwise the points are a cut from border to border.
    void applyFaceDraw(bool closed);
    // Enter applies, Backspace drops the last point, Escape cancels. True while drawing.
    bool handleFaceDrawKeys();
    // Selects the vertices inside the box, or picks an object when the mouse barely moved.
    void finishVertexMarquee(float mouseX, float mouseY);
    void drawLevelFaceOverlay();
    void drawFaceProperties(PolyMesh& mesh, PolyFace& face);
    void drawVertexProperties(PolyMesh& mesh);
    // Push/pull and extrude; by index, because extruding adds faces and moves the face storage.
    void drawFaceGeometry(PolyMesh& mesh, uint32_t faceIndex);
    void drawEdgeGeometry(PolyMesh& mesh, uint32_t a, uint32_t b);
    void drawLevelMaterials(PolyMesh& mesh);
    // What the open texture dialog sets: an inline slot of the selected shape, or a shared material.
    struct TextureDialogTarget {
        int slot = -1;
        std::string sharedPath;
        bool normal = false; // shared materials only
    };
    void openTextureDialog(TextureDialogTarget target);
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

    // ---- EditorUv.cpp ----
    // Face mode keys over the viewport: arrows nudge the UV offset by one grid cell, Ctrl+Left/Right
    // rotate, Alt+Up/Down scale; Ctrl+Shift+C / V copy and paste UVs and material.
    void handleFaceKeys();
    void copyFaceAttributes();
    void pasteFaceAttributes();
    // Wraps the selected faces' textures around from the active face across shared edges.
    void wrapSelectedFaces();
    // 2D view of the selected faces' UVs over the texture: drag moves, Shift+drag rotates, wheel scales;
    // right-drag pans and Ctrl+wheel zooms the view.
    void drawUvEditor(PolyMesh& mesh);

    // ---- EditorMaterials.cpp ----
    // "Brick" for a slot linked to materials/Brick.mat, else "Material <n>"; "None (white)" past the end.
    std::string levelSlotName(const PolyMesh& mesh, uint32_t slot);
    // MaterialAsset::texelSize of the slot's shared material, 1 when it is not linked.
    float levelSlotTexelSize(const PolyMesh& mesh, uint32_t slot);
    // Combo listing the library plus "New material"; `path` empty = `noneLabel`. True when it changed.
    bool drawSharedMaterialCombo(const char* id, std::string& path, const char* noneLabel);
    // Editors for a shared material's values; true when one changed (then pass it to saveSharedMaterial()).
    bool drawSharedMaterialProperties(MaterialAsset& material);
    // Writes the .mat file and rebuilds every model using it.
    void saveSharedMaterial(const MaterialAsset& material);
    // Inspector: shared materials replacing the material slots of a model loaded from a file.
    void drawModelMaterialOverrides(size_t modelIndex);
    // The Materials panel: thumbnails of the library, the current material and its properties.
    void drawMaterialsPanel();
    void drawMaterialTile(const MaterialAsset& material, float size);
    void drawMaterialPopups();
    // Slot of `mesh` linked to the shared material, added when missing; kNoPolyMaterial when full.
    static uint32_t levelSlotFor(PolyMesh& mesh, const std::string& materialPath);
    // Drops linked slots no face uses any more, renumbering the faces.
    static void pruneLinkedSlots(PolyMesh& mesh);
    // The selected face (face mode) or every face of the selected level shape, or every material slot
    // of the selected file model.
    void applyMaterialToSelection(const std::string& materialPath);
    // `slot` of a file model (kNoSourceMaterial included); all slots when `allSlots`.
    void applyMaterialToModel(size_t modelIndex, const std::string& materialPath, uint32_t slot, bool allSlots);
    // Points every object using `from` at `to` (after a rename).
    void relinkMaterial(const std::string& from, const std::string& to);
    // While a material is dragged: a drop target over the scene view.
    void drawMaterialDropTarget();
    void dropMaterial(const std::string& materialPath, float mouseX, float mouseY, bool wholeObject);
    // Material drops on another object select it first and are applied the next frame, so the undo
    // history has its snapshot of that object.
    void applyPendingMaterialDrop();
    // Eyedropper (I over the viewport): the shared material of the face or model slot under the mouse.
    void pickMaterialUnderMouse();

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
    // The model of the prefab: the one instances already share, else loaded from the file. Empty (with
    // the status set) when the file can't be loaded.
    std::optional<size_t> loadPrefabModel(const std::string& path);
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
    // A level shape's geometry before and after a step.
    struct LevelEdit {
        std::string modelPath;
        PolyMesh before;
        PolyMesh after;
    };
    // One step: geometry edits (one shape from the editor, several from an MCP command or script), object
    // changes (transform, color, visibility, name, lock, model, existence), or both when one action did
    // both (e.g. subtract deleting the cutter).
    struct HistoryEntry {
        std::string action;
        std::vector<LevelEdit> levels; // empty when no geometry changed
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
    bool applyLevelEdits(const HistoryEntry& entry, bool undo);
    bool applyHistoryEntry(const HistoryEntry& entry, bool undo);
    void pushHistory(HistoryEntry entry);
    void undo();
    void redo();
    void clearHistory();

    // ---- EditorCommands.cpp ----
    // The commands MCP clients call: a table of handlers taking JSON arguments. A nested type, so they
    // can use the editor's internals.
    struct CommandApi;
    // Runs queued MCP calls once no edit is in progress; each call is one undo step. In lateUpdate().
    void pollMcpServer();
    // Runs the script queued by runScript(). In lateUpdate().
    void pollScript();
    // Commands between these become one undo step named `action`.
    void beginCommandBatch();
    void endCommandBatch(const std::string& action);
    // The mesh of a level model, for a command to edit in place: remembered for the undo step and
    // re-uploaded once at endCommandBatch().
    PolyMesh& batchEditMesh(size_t modelIndex);

    SDL_Window* m_window;
    const VulkanContext& m_vulkan;
    Renderer& m_renderer;
    ModelManager& m_models;
    SceneManager& m_scenes;
    GraphicsSettings& m_settings;
    MaterialLibrary& m_materials;

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
    // World grid cell size, also given to new level shapes as their own grid. Snapping (world and shape
    // grids) is on while m_gridSnap is set; Ctrl inverts it mid-drag.
    static constexpr float kGridSizes[] = { 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f };
    float m_gridSize = 1.0f;
    bool m_gridSnap = true;
    // Increments for rotate/scale gizmo deltas.
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
    bool m_showMaterialsPanel = true;
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
    // Face, edge and vertex modes: clicks on the selected level shape pick its parts instead of objects.
    enum class LevelEditMode { Object, Face, Vertex, Edge };
    LevelEditMode m_levelMode = LevelEditMode::Object;
    // Face mode: m_selectedFace is the active face (properties shown, dragged, source of wraps);
    // m_selectedFaces holds every selected face, the active one included.
    int m_selectedFace = -1;
    std::vector<uint32_t> m_selectedFaces;
    // Copied face UVs and material (Ctrl+Shift+C), pasted onto selected faces with Ctrl+Shift+V.
    struct FaceClipboard {
        bool valid = false;
        bool hasMaterial = false;
        PolyMaterial material;
        glm::vec2 uvScale{ 1.0f };
        glm::vec2 uvOffset{ 0.0f };
        float uvRotation = 0.0f;
    };
    FaceClipboard m_faceClipboard;
    // 2D UV editor view: UV coordinate at the canvas center and UV units across it.
    glm::vec2 m_uvViewCenter{ 0.5f };
    float m_uvViewSpan = 3.0f;
    // A UV drag in the 2D editor: faces' values at its start.
    struct UvDrag {
        bool active = false;
        bool rotate = false;
        glm::vec2 startMouse{ 0.0f };
        std::vector<glm::vec2> startOffsets;
        std::vector<float> startRotations;
    };
    UvDrag m_uvDrag;
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
    // Face mode: dragging a face moves it along its normal (push/pull), or extrudes it when Alt was held
    // at the start. Each step re-applies the edit to the mesh as it was when the drag began.
    struct FaceDrag {
        bool active = false;
        bool moved = false; // mesh changed since the last rebuild
        bool extrude = false;
        int instance = -1;
        uint32_t face = 0;
        glm::vec3 center{ 0.0f }; // object space
        glm::vec3 normal{ 0.0f }; // object space, unit length
        float startParam = 0.0f;  // position along the normal under the mouse when the drag began
        float distance = 0.0f;    // currently applied, object space
        PolyMesh startMesh;
    };
    FaceDrag m_faceDrag;
    // Edge mode: one selected edge, by its two vertices.
    bool m_edgeSelected = false;
    std::array<uint32_t, 2> m_selectedEdge{};
    // Dragging an edge moves it along the average normal of its faces, or extrudes it (Alt at the start)
    // along PolyMesh::edgeExtrudeDirection(); re-applied to the starting mesh like FaceDrag.
    struct EdgeDrag {
        bool active = false;
        bool moved = false; // mesh changed since the last rebuild
        bool extrude = false;
        int instance = -1;
        uint32_t a = 0, b = 0;
        glm::vec3 center{ 0.0f };    // object space
        glm::vec3 direction{ 0.0f }; // object space, unit length
        float startParam = 0.0f;
        glm::vec3 offset{ 0.0f };    // currently applied, object space
        PolyMesh startMesh;
    };
    EdgeDrag m_edgeDrag;
    // Face mode: a shape drawn on the active face, which then divides it (PolyMesh::divideFace).
    enum class FaceDrawShape { Polygon, Rectangle, Circle };
    struct FaceDraw {
        bool active = false;
        int instance = -1;
        uint32_t face = 0;
        std::vector<glm::vec3> points; // clicked so far, object space on the face plane
    };
    FaceDraw m_faceDraw;
    FaceDrawShape m_faceDrawShape = FaceDrawShape::Rectangle;
    int m_faceDrawSegments = 16; // circle
    // Box selection started on empty space in vertex mode; scene-view pixels.
    struct VertexMarquee {
        bool active = false;
        bool additive = false; // Shift: add to the selection instead of replacing it
        glm::vec2 start{ 0.0f };
        glm::vec2 end{ 0.0f };
    };
    VertexMarquee m_vertexMarquee;
    TextureDialogTarget m_textureDialogTarget;

    // Materials panel
    std::string m_currentMaterial; // path; new level shapes get it too
    ImGuiTextFilter m_materialFilter;
    float m_materialTileSize = 72.0f;
    char m_materialNameBuffer[128] = "";
    std::string m_materialPopupPath; // material the rename/delete popup is for
    bool m_openMaterialRename = false;
    bool m_openMaterialDelete = false;
    struct MaterialDrop {
        bool pending = false;
        uint64_t instanceId = 0;
        int face = -1;           // level shapes; -1 = whole object
        uint32_t slot = kNoSourceMaterial; // file models
        bool wholeObject = false;
        std::string materialPath;
    };
    MaterialDrop m_materialDrop;
    float m_faceOpDistance = 1.0f;
    float m_faceInsetDistance = 0.25f;
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

    // MCP server and the command batch being run
    std::unique_ptr<McpServer> m_mcp;
    struct CommandBatch {
        bool active = false;
        std::vector<std::string> existingModels; // level model paths when the batch began
        std::vector<LevelEdit> edits;            // `after` is filled in at the end
        std::vector<std::string> dirty;          // level models to re-upload
    };
    CommandBatch m_batch;
    std::string m_pendingScript;
};
