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
#include "EditorKeymap.h"
#include "EditorPrefs.h"
#include "engine/Camera.h"
#include "engine/CameraAnimation.h"
#include "engine/Gizmo.h"
#include "engine/GraphicsSettings.h"
#include "engine/MaterialLibrary.h"
#include "engine/ModelManager.h"
#include "engine/PolyMesh.h"
#include "engine/Renderer.h"

class McpServer;
struct EntityPreset;
struct EntityTypeInfo;

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
    // Applies style changes and a requested saved layout, and saves changed preferences.
    void beforeUiFrame() override;
    void drawUi() override;
    void fillFrame(FrameInput& frame) override;
    bool quitRequested() const override { return m_quitRequested; }
    // With unsaved changes: asks to save them first and returns false.
    bool confirmQuit() override;
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
        int separateIndex = -1;
        bool createCube = false;
        bool unite = false;
        bool uniteAndSave = false;
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
    // clicks: 2 for the second click of a double-click.
    void handleViewportClick(float mouseX, float mouseY, int clicks);
    // Selects the object under the mouse (see clickSelect()), or deselects when there is none.
    void pickObject(float mouseX, float mouseY);
    bool tryBeginGizmoDrag(float mouseX, float mouseY, const glm::mat4& view, const glm::mat4& proj);
    // The active object follows the mouse; other selected objects move with it (dragGroup()).
    void dragGizmo(float mouseX, float mouseY);
    bool gizmoVisible() const;
    GizmoShape currentGizmoShape();
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
    // An active object is selected (it is the gizmo's; see EditorSelection.cpp for the rest).
    bool hasSelection() const { return validInstance(m_gizmo.selectedInstance); }
    void focusOnInstance(int index);
    void addModelToScene(size_t modelIndex, bool atOrigin);
    void deleteInstance(int index);
    void duplicateInstance(int index);
    // A copy of the instance moved by `offset`, named after it; level geometry is copied unless it is a
    // prefab's. Returns its index, or nothing (status set) when the copy failed.
    std::optional<size_t> copyInstance(int index, const glm::vec3& offset);
    void unloadModel(size_t modelIndex);
    void newScene();
    void saveSceneTo(const std::string& path);
    void saveScene();
    void handleShortcuts();
    void updateWindowTitle();
    void setStatus(const std::string& message, bool isError = false);
    static std::string formatCount(size_t value);
    void fillHighlight(SelectionHighlight& highlight);
    glm::vec3 instanceWorldCenter(int index) const;
    void addCube();
    // A game entity (EntityTypes.h) as its marker model, at `at` (default: the placement point) and facing
    // the way the camera looks. Returns its index, or nothing (status set) when the model can't be loaded.
    std::optional<size_t> addEntity(const std::string& type, const glm::vec3* at = nullptr);
    void drawEntityMenuItems();
    // Inspector: the object's entity type and parameters.
    void drawEntitySection(int instanceIndex);
    // `name`, or when it is taken "name (n)" with the lowest free n; "Box (2)" counts as "Box".
    std::string uniqueInstanceName(const std::string& name) const;
    void beginRename(int index);
    // m_gridSnap, inverted while Ctrl is held.
    bool snapActive() const;
    // Held modifier keys: the view_mouse step's while a simulated mouse event runs, else the keyboard's.
    SDL_Keymod keyMods() const;
    // Face/vertex editing of a level shape: its own object-space grid replaces the ground grid.
    bool levelGridActive();
    // The grid in use: the edited shape's (PolyMesh::gridSize) or the world grid.
    float activeGridSize();
    void setActiveGridSize(float size);
    // Moves the active grid to the next larger (+1) or smaller (-1) preset.
    void stepGridSize(int direction);

    // ---- EditorEntities.cpp ----
    // Entity placement: pick a type (or enemy preset) in the Level panel or the Add menu, then every click
    // on a surface places one there and selects it, until Esc or the button again.
    void beginEntityPlacement(const std::string& type, int preset);
    void stopEntityPlacement();
    bool entityPlacementActive() const { return !m_entityPlace.type.empty(); }
    // "Enemy (Brute)".
    std::string entityPlaceLabel() const;
    // A left press in the scene view; false when the tool is off.
    bool handleEntityPlaceClick(float mouseX, float mouseY);
    // Where a click puts an entity: on the floor under the mouse; a wall hit stands it in front of the wall.
    bool entityPlacementPoint(float mouseX, float mouseY, glm::vec3& out) const;
    void drawEntityPlaceOverlay();
    // Range sphere or cone of the selected lights.
    void drawLightOverlay();
    // Level panel: a button per entity type and enemy preset.
    void drawEntityPalette();
    // The preset's values (others are kept) and tint.
    static void applyEntityPreset(ModelInstance& instance, const EntityPreset& preset);
    // The type's own defaults get a palette button next to its presets unless a built-in preset covers them.
    static bool entityDefaultPlaceable(const EntityTypeInfo& type);
    // Inspector: "Save as preset..." for the entity's current values, and its name popup.
    void drawSaveEntityPreset(const ModelInstance& instance);

    // ---- EditorGroups.cpp ----
    // Groups are objects sharing a ModelInstance::group name. A click in the viewport selects the whole
    // group (a double-click then just the object under the mouse); copies of a whole group form a new group.
    std::vector<int> groupMembers(const std::string& group) const;
    // `name`, or "name (n)" when a group already uses it.
    std::string uniqueGroupName(const std::string& name) const;
    // "Group n" with the lowest free n.
    std::string nextGroupName() const;
    // clickSelect() for the object's whole group.
    void selectWithGroup(int index, SDL_Keymod mods);
    // The object is in a group of several, all of them selected.
    bool wholeGroupSelected(int index) const;
    // Adds the rest of every group that has a selected object.
    void expandSelectionToGroups();
    // Puts the selected objects into one new group (taking them out of any other).
    void groupSelection();
    void ungroupSelection();
    void renameGroup(const std::string& from, const std::string& to);
    // Names for copies of the objects: a group copied whole gets a new name, so the copy is a group of its
    // own; groups copied in part are missing here, their copies join them. Old -> new name.
    std::unordered_map<std::string, std::string> groupNamesForCopies(const std::vector<int>& indices) const;
    // Inspector: the active object's group (rename, select, ungroup).
    void drawGroupSection(int instanceIndex);
    void drawGroupNode(const std::string& group, HierarchyActions& actions);
    // Object prefabs (Prefab.h ObjectPrefab): any selected objects, re-added later as a group.
    void saveObjectPrefabDialog();
    // Saves the objects (by id; gone ones are skipped); several become a group. Status says how it went.
    void saveObjectPrefab(const std::vector<uint64_t>& ids, const std::string& path);
    // Adds the prefab's objects around `at` (default: the placement point), as a group when there are several.
    void addObjectPrefab(const std::string& path, const glm::vec3* at = nullptr);

    // ---- EditorSelection.cpp ----
    // Several objects can be selected (m_selection, by id). The active one, m_gizmo.selectedInstance, carries
    // the gizmo, the inspector and face editing; group actions act on all of them.
    bool isSelected(int index) const;
    // Selected indices in scene order; with activeFirst, the active object comes first.
    std::vector<int> selectedIndices(bool activeFirst = false) const;
    size_t selectionCount() const { return m_selection.size(); }
    // Just this object, active.
    void selectInstance(int index);
    void deselectAll();
    // Adds the object and makes it active.
    void addToSelection(int index);
    void toggleSelection(int index);
    // Visible objects; the active one stays active when it is visible.
    void selectAll();
    // Exactly these objects; `active` (one of them) gets the gizmo, else the first.
    void selectIndices(const std::vector<int>& indices, int active);
    // Hierarchy Shift+click: the rows from the active object to `to` that pass the search filter.
    void selectRange(int to);
    // The gizmo's object, without changing the rest of the selection; -1 = none.
    void setActive(int index);
    // Click on an object: plain selects just it, Shift adds it, Ctrl toggles it.
    void clickSelect(int index, SDL_Keymod mods);
    // Drops objects that are gone; the active object stays selected, or another selected one takes over.
    void validateSelection();
    // Object under the mouse (scene-view pixels), or -1.
    int objectUnderMouse(float mouseX, float mouseY) const;
    // Box selection started on empty space: objects whose center is inside; a click without a drag
    // deselects (unless Shift/Ctrl was held).
    void finishObjectMarquee(float mouseX, float mouseY);
    void drawObjectMarquee();
    // World-space bounds; level shapes from their mesh. False when the model is missing.
    bool instanceWorldBounds(int index, glm::vec3& lo, glm::vec3& hi) const;
    bool selectionWorldBounds(glm::vec3& lo, glm::vec3& hi) const;
    void focusSelection();
    void deleteSelection();
    // One object: duplicateInstance(). Several: copied side by side with the group, the copies selected.
    void duplicateSelection();
    // The clipboard keeps the geometry of level shapes, so objects can be pasted into another scene.
    void copySelection();
    void cutSelection();
    // At the copied positions; the pasted objects become the selection.
    void pasteClipboard();
    // Each selected object, lowest first, falls straight down onto the first surface below (or y = 0).
    void dropSelectionToFloor();
    // Distance down from `origin` to the nearest visible object other than `ignore`; infinity when none.
    float distanceToSurfaceBelow(const glm::vec3& origin, int ignore) const;
    // Turns the selection about the vertical axis through the active object.
    void rotateSelection(float degrees);
    // Hides the selection, or shows it when all of it is hidden.
    void toggleSelectionVisibility();
    void unhideAll();
    // Remembers every selected object's transform as a gizmo drag starts.
    void beginGroupDrag();
    // Applies the active object's change since the drag began to the rest of the selection, about the
    // active object's starting position: same offset, same turn about the gizmo axis, same scale factor.
    void dragGroup(int axis, float degrees, float scaleFactor);

    // ---- EditorSceneFiles.cpp ----
    // Unsaved changes: the scene is not at the undo step it was saved or opened at.
    bool sceneDirty() const;
    uint64_t sceneStateId() const;
    // After a save to the scene's file: clean, and listed under recent scenes.
    void markSceneSaved(const std::string& path);
    // Once the edits of a script that saved mid-way are in the history, the scene counts as saved again.
    void settleSavedState();
    // What waits for the unsaved-changes prompt.
    enum class SceneAction { None, New, Open, OpenDialog, Quit };
    // Runs the action, first asking to save unsaved changes.
    void requestSceneAction(SceneAction action, const std::string& path = {});
    void runSceneAction(SceneAction action, const std::string& path);
    // After Save As from the prompt: the waiting action runs if the scene got saved, else is dropped.
    void finishPendingSceneAction();
    void drawUnsavedChangesPopup();
    void drawRecentScenesMenu();
    // Writes unsaved changes to kAutosavePath every m_prefs.autosaveMinutes; the scene's file is left alone.
    void updateAutosave();

    // ---- EditorMenus.cpp ----
    void drawMainMenuBar();
    void drawFileMenu();
    void drawEditMenu();
    void drawViewMenu();
    void drawToolbar();
    void drawToolButton(const char* label, GizmoMode tool, const char* tooltip, EditorAction shortcut);
    // "text (shortcut)", or just the text when the action is unbound.
    std::string withShortcut(const char* text, EditorAction action) const;
    void drawSnapPopup();
    void drawGridControls();
    void drawAddMenu();
    void drawStatusBar();
    void drawDockSpace();
    void drawViewportOverlay();
    void drawHelpPopups();
    void drawGraphicsSettingsWindow();
    // The open scene's sky, sun, fog and grading (SceneManager::settings()), saved with the scene.
    void drawSceneSettingsWindow();
    void drawControlsPopup();
    void drawAboutPopup();
    void openFileDialog(const char* key, const char* title, const char* filters, const std::string& root,
        const char* defaultFileName, bool confirmOverwrite);
    void openSceneDialog();
    void saveSceneAsDialog();
    void importModelDialog();
    void runScriptDialog();
    void drawFileDialogs();

    // ---- EditorShortcuts.cpp ----
    // The Keyboard Shortcuts window: every action with its two bindings; click one to record a new chord.
    void drawKeymapWindow();
    void drawKeymapRow(EditorAction action);
    // While recording: the next key (with the modifiers held) becomes the binding; a modifier key pressed
    // and released alone binds that key; Esc cancels.
    void updateKeyCapture();
    bool capturingKey() const { return m_keyCapture.action >= 0; }
    // Sends the chords queued by the press_keys MCP command as SDL events, one at a time with a free frame
    // between them, so they take the same path as typed keys. In lateUpdate().
    void pumpSimulatedKeys();
    // One step of the view_mouse MCP command per frame, given straight to the viewport handlers (the
    // panels never see it, and ImGui's idea of the mouse, which follows the real cursor, is ignored).
    void pumpSimulatedMouse();
    // ImGui has the mouse (over a panel), unless the event being handled is a simulated one.
    bool uiOwnsMouse() const;

    // ---- EditorPreferences.cpp ----
    // `pixels` at the current UI scale, for fixed widths in the panels.
    float px(float pixels) const { return pixels * m_prefs.uiScale; }
    void drawPreferencesWindow();
    void drawInterfacePrefs();
    void drawViewportPrefs();
    void drawLayoutPrefs();
    void drawFilePrefs();
    // Built-in dock layouts, built by drawDockSpace() when requested.
    enum class LayoutPreset { Default, LevelDesign, Compact };
    void buildLayout(ImGuiID dockspaceId, LayoutPreset preset);
    // Saved layouts are imgui.ini snapshots in layoutsRoot(): dock layout plus panel visibility.
    std::vector<std::string> savedLayouts() const;
    void saveLayout(const std::string& name);
    // Queued: the file is read in beforeUiFrame(), outside the ImGui frame.
    void loadLayout(const std::string& name);
    void deleteLayout(const std::string& name);
    void drawLayoutMenuItems();
    // Panel visibility is stored in imgui.ini under [MirasEditor][Panels]. The handler stays registered after
    // the editor is gone (ImGui saves once more on shutdown), writing the last known values.
    void registerIniHandler();
    void detachIniHandler();
    struct PanelFlag {
        const char* name; // window title
        bool* shown;
    };
    std::array<PanelFlag, 6> panelFlags();

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
    // What the game collides with (ModelInstance::collision), for every selected object.
    void drawCollisionSection(int instanceIndex);
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
    // Adds the whole flat surface of every selected face (PolyMesh::coplanarRegion); also a double-click.
    void selectSurface();
    // Joins the selected shape's coplanar neighbouring faces that look alike (PolyMesh::mergeCoplanarFaces).
    void mergeLevelFaces();
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
    // Where Create puts a new shape: on the surface under the middle of the view, snapped to the grid.
    glm::vec3 placementPoint() const;

    // ---- EditorLevelDraw.cpp ----
    // Shape drawing: drag a footprint on the ground or the top of any object (snapped to the world grid),
    // then move the mouse up or down to set the height and click. A click without dragging places the
    // New shape panel's size. Draws the panel's shape type and stays on until Esc or the toggle.
    void toggleShapeDraw();
    bool shapeDrawActive() const { return m_shapeDraw.phase != ShapeDraw::Phase::Off; }
    // A left press in the scene view; false when the tool is off.
    bool handleShapeDrawPress(float mouseX, float mouseY);
    void handleShapeDrawMotion(float mouseX, float mouseY);
    void handleShapeDrawRelease(float mouseX, float mouseY);
    // Cancel/finish keys; true when they were used.
    bool handleShapeDrawKeys();
    void drawShapeDrawOverlay();
    // The point on the plane y = height under the mouse, X/Z snapped to the world grid while snapping.
    bool shapeDrawPoint(float mouseX, float mouseY, float height, glm::vec3& out) const;
    // What the mouse points at to build on: a surface facing up (its hit point), else the ground (y = 0).
    bool surfacePoint(float mouseX, float mouseY, glm::vec3& out) const;
    // Adds the drawn shape (or, for a click, one of the panel's size). Keeps the tool ready, or with
    // m_editAfterDraw turns it off and starts editing the new shape.
    void finishShapeDraw(bool clicked);
    // Shape editing session on the selected level shape, like a brush in TrenchBroom: Tab enters the last
    // used face/edge/vertex/part mode and leaves it again; Esc clears the faces/vertices picked, then leaves.
    // The shape stays one object throughout.
    void toggleShapeEdit();
    void beginShapeEdit();
    void finishShapeEdit();
    bool shapeEditActive() { return m_levelMode != LevelEditMode::Object && selectedLevelMesh() != nullptr; }
    // Faces, an edge, vertices or parts are picked on the edited shape.
    bool shapeEditHasPicks();
    void clearShapeEditPicks();
    // Banner over the viewport naming the edited shape and the keys.
    void drawShapeEditOverlay();
    // Moves every selected face `distance` along its normal (each vertex once, so neighbouring selected
    // faces stay joined), or extrudes each of them.
    void moveSelectedFaces(float distance, bool extrude);
    // Rounds the selected vertices to the shape's grid.
    void snapSelectedVertices();

    // ---- EditorLevelUnite.cpp ----
    // levelSlotTexelSize() of every material slot, for PolyMesh::transformKeepingUVs().
    std::vector<float> levelSlotTexelSizes(const PolyMesh& mesh);
    // Selected objects that are level shapes (prefab instances included), the active one first.
    std::vector<int> selectedLevelShapes() const;
    // Merges the selected level shapes into one new shape and selects it. Geometry is baked into world
    // space around an origin at the bottom center (on the grid while snapping); textures stay in place,
    // equal materials are shared and differing tints are baked into the materials. The parts stay apart
    // (not welded), so separateSelectedShape() splits them again. Returns the new index, or -1 (status set).
    // solid: the shapes (all closed) are merged into one volume instead (polyMeshUnion): overlaps and the
    // faces inside are removed and flat neighbouring faces joined; they can't be separated again.
    int uniteSelectedShapes(bool solid = false);
    // The active shape's connected parts merged into one volume, as uniteSelectedShapes(true).
    void mergeShapeParts();
    // Splits the active shape into its connected parts, one object each with its origin at its bottom center.
    void separateSelectedShape();
    // One level shape: a shape prefab (savePrefabDialog). Anything else: an object prefab of the selection.
    void saveSelectionAsPrefab();
    // Unites the selection when it holds several shapes, then opens Save as Prefab for the result.
    void uniteAndSavePrefab();
    // Level panel: unite, separate and save buttons for the selected shapes.
    void drawUniteSection();
    // Part mode. Faces of the selected parts; drops parts the mesh no longer has. Empty outside part mode.
    std::vector<uint32_t> selectedPartFaces();
    // Plain click: just the part with this face; toggle (Shift): adds or removes it.
    void selectPart(uint32_t face, bool toggle);
    void selectAllParts();
    // Object-space center of the selected parts' bounds; false when none is selected.
    bool selectedPartsCenter(glm::vec3& center);
    // Where the transform gizmo sits: the selected parts' center in part mode, else the active object.
    glm::vec3 gizmoPivot();
    void beginPartDrag();
    // The gizmo drag as a world-space move, turn or scale about the parts' center, applied in the shape's space.
    void dragParts(float amount, int axis);
    // Transforms the selected parts by a world-space matrix (about nothing: callers build the pivot in).
    void transformSelectedParts(const glm::mat4& world, const char* action);
    void rotateSelectedParts(float degrees);
    // A copy beside the selected parts (whole grid cells along X), which becomes the selection.
    void duplicateSelectedParts();
    void deleteSelectedParts();
    // Moves the selected parts out into an object of their own.
    void detachSelectedParts();
    void drawPartControls(PolyMesh& mesh);
    // Double-click on a shape made of several parts: part mode with the part under the mouse selected.
    bool enterPartMode(float mouseX, float mouseY);

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
        uint64_t serial = 0; // unique, names the scene state after this step (sceneStateId())
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
    // Scene viewport = central dock area in window coordinates. The 3D scene renders only here,
    // and mouse picking / gizmo math is relative to it.
    ViewRect m_sceneView;
    CameraAnimator m_cameraAnimator;
    bool m_showCameraPath = true;
    bool m_showGrid = true;

    // Selection and tools
    Gizmo m_gizmo;
    std::vector<uint64_t> m_selection;   // ModelInstance ids, sorted; the active object's included
    uint64_t m_activeId = 0;             // id of m_gizmo.selectedInstance, to follow it when indices shift
    std::vector<int> m_highlightIndices; // what fillHighlight() gave the renderer this frame
    // Transforms of the selected objects as a gizmo drag began.
    struct GroupDragItem {
        int index = -1;
        glm::vec3 position{ 0.0f };
        glm::vec3 rotation{ 0.0f };
        glm::vec3 scale{ 1.0f };
    };
    std::vector<GroupDragItem> m_groupDrag;
    // Objects copied with Ctrl+C, with what it takes to rebuild them in this or another scene.
    struct ClipboardObject {
        ModelInstance instance;
        std::string modelPath;  // file models and built-ins
        std::string modelName;
        std::string prefabPath; // prefab instances
        std::shared_ptr<const PolyMesh> mesh; // level shapes that are not prefabs
    };
    std::vector<ClipboardObject> m_clipboard;
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
    bool m_showSceneSettings = false;
    SceneSettings m_savedSceneSettings; // as in the scene's file, for sceneDirty()
    bool m_showKeymap = false;
    bool m_showPreferences = false;
    std::optional<LayoutPreset> m_layoutRequest;
    bool m_openControlsPopup = false;
    bool m_openAboutPopup = false;
    bool m_quitRequested = false;
    ModeRequest m_modeRequest = ModeRequest::None;
    std::string m_windowTitle;

    // Preferences and layouts
    static std::string layoutsRoot() { return configPath("layouts"); }
    EditorPrefs m_prefs;
    EditorPrefs m_savedPrefs;   // as last written to kEditorPrefsPath
    EditorPrefs m_appliedStyle; // the style fields as last given to EditorStyle::apply()
    uint32_t m_panelBits = 0;   // panel visibility as last seen, to notice changes for imgui.ini
    std::string m_pendingLayout; // contents of a saved layout to load before the next frame
    char m_layoutName[64] = "";
    int m_preferencesTab = -1; // tab the Preferences window switches to the next time it is drawn

    // Keyboard shortcuts
    EditorKeymap m_keymap;
    struct KeyCapture {
        int action = -1; // EditorAction being rebound, -1 = not recording
        int slot = 0;
        ImGuiKey lonelyModifier = ImGuiKey_None; // modifier pressed with no other key since
        bool openPopup = false;
    };
    KeyCapture m_keyCapture;
    ImGuiTextFilter m_keymapFilter;
    struct SimulatedKeys {
        std::vector<ImGuiKeyChord> queue;
        size_t next = 0;
        int holdFrames = 2;
        int framesLeft = 0;
        ImGuiKeyChord down = ImGuiKey_None; // chord currently held
    };
    SimulatedKeys m_simulatedKeys;
    struct MouseStep {
        enum class Action { Move, Down, Up } action = Action::Move;
        glm::vec2 position{ 0.5f }; // 0..1 across the scene view
        uint8_t button = 1;         // SDL_BUTTON_LEFT / SDL_BUTTON_RIGHT
        uint8_t clicks = 1;         // 2: the second press of a double-click
        SDL_Keymod mods = SDL_KMOD_NONE; // modifier keys held for the step
    };
    std::vector<MouseStep> m_mouseSteps;
    size_t m_nextMouseStep = 0;
    bool m_mouseSimulated = false; // a simulated event is being handled
    SDL_Keymod m_simulatedMods = SDL_KMOD_NONE;

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
    // Part: the connected parts of a shape (e.g. of a united group) are picked and transformed whole.
    enum class LevelEditMode { Object, Face, Vertex, Edge, Part };
    LevelEditMode m_levelMode = LevelEditMode::Object;
    LevelEditMode m_lastShapeEditMode = LevelEditMode::Face; // what Tab goes back to
    bool m_editAfterDraw = true; // the Draw tool hands the new shape over to editing
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
        bool solid = false; // extrude through PolyMesh::extrudeFacesSolid (closed shapes)
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
    // Dragging an edge moves it, or extrudes it (Alt at the start), along PolyMesh::edgeExtrudeDirection()
    // (in the face's plane for a border edge); re-applied to the starting mesh like FaceDrag.
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
    // Part mode: the selected parts of the shape, each by one of its faces (moving parts keeps face indices).
    std::vector<uint32_t> m_selectedParts;
    // A gizmo drag of the selected parts, re-applied to the mesh as it was when the drag began.
    struct PartDrag {
        bool active = false;
        bool moved = false; // mesh changed since the last rebuild
        int instance = -1;
        std::vector<uint32_t> faces; // every face of the dragged parts
        glm::vec3 center{ 0.0f };    // object space: the gizmo's pivot
        glm::mat4 applied{ 1.0f };   // world-space transform applied so far
        std::vector<float> texelSizes;
        PolyMesh startMesh;
    };
    PartDrag m_partDrag;
    struct ShapeDraw {
        enum class Phase { Off, Ready, Footprint, Height };
        Phase phase = Phase::Off;
        float baseY = 0.0f;
        glm::vec3 start{ 0.0f }; // footprint corners in world space, at y = baseY
        glm::vec3 end{ 0.0f };
        float height = 0.0f;
        float startHeight = 0.0f; // when the height phase began
        float startParam = 0.0f;  // mouse position along the vertical line then
    };
    ShapeDraw m_shapeDraw;
    // Entity placement tool (EditorEntities.cpp); off while type is empty.
    struct EntityPlace {
        std::string type;
        int preset = -1; // into the type's presets, -1 = its defaults
    };
    EntityPlace m_entityPlace;
    char m_entityPresetName[64] = {}; // Save as preset popup
    FaceDrawShape m_faceDrawShape = FaceDrawShape::Rectangle;
    int m_faceDrawSegments = 16; // circle
    // Box selection started on empty space, of vertices (vertex mode) or objects; scene-view pixels.
    struct Marquee {
        bool active = false;
        bool additive = false; // Shift: add to the selection instead of replacing it
        glm::vec2 start{ 0.0f };
        glm::vec2 end{ 0.0f };
    };
    Marquee m_vertexMarquee;
    Marquee m_objectMarquee;
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
    // Objects an object-prefab save is for (by id); empty when the dialog saves a shape.
    std::vector<uint64_t> m_prefabSaveIds;
    char m_groupNameBuffer[128] = {};
    bool m_groupNameActive = false; // the inspector's group name field was being edited last frame

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

    // Unsaved changes and autosave (EditorSceneFiles.cpp); states are named by HistoryEntry::serial.
    uint64_t m_historySerial = 0; // last serial handed out
    uint64_t m_baseState = 0;     // the state with an empty undo stack
    uint64_t m_savedState = 0;
    uint64_t m_autosavedState = 0;
    uint64_t m_lastAutosaveTicks = 0; // SDL_GetTicks() of the last autosave check
    SceneAction m_pendingSceneAction = SceneAction::None;
    std::string m_pendingScenePath;
    bool m_openUnsavedPopup = false;
    bool m_pendingAfterSaveAs = false; // the prompt's Save opened Save As
    bool m_savedBeforeHistory = false; // see settleSavedState()

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
