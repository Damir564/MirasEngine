#include "Editor.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include "McpServer.h" // complete type for m_mcp
#include "engine/EntityTypes.h"
#include "engine/ModelLoader.h"
#include "engine/ModelManager.h"
#include "engine/SceneManager.h"
#include "engine/Log.h"

Editor::Editor(const EngineContext& engine)
    : m_window(engine.window)
    , m_vulkan(engine.vulkan)
    , m_renderer(engine.renderer)
    , m_models(engine.models)
    , m_scenes(engine.scenes)
    , m_settings(engine.settings)
    , m_materials(engine.materials)
{
    int width = 0, height = 0;
    SDL_GetWindowSize(m_window, &width, &height);
    m_sceneView = { 0.0f, 0.0f, static_cast<float>(std::max(width, 1)), static_cast<float>(std::max(height, 1)) };

    m_cameraAnimator.getPath().name = m_pathName;
    LOG_INFO("Camera animation system initialized\n");
    m_keymap.load(editorKeymapPath());
    loadEntityPresets();

    // Before the first ImGui frame, which reads imgui.ini (and with it the panel visibility).
    m_prefs = m_savedPrefs = loadEditorPrefs();
    m_camera.speed = m_prefs.cameraSpeed;
    m_camera.sensitivity = m_prefs.lookSensitivity;
    EditorStyle::apply(m_prefs.styleOptions());
    m_appliedStyle = m_prefs;
    registerIniHandler();

    SDL_SetWindowRelativeMouseMode(m_window, m_flyMode);
}

// ---------------------------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------------------------

void Editor::onEvent(const SDL_Event& event)
{
    m_keymap.processEvent(event);
    // WantTextInput, not WantCaptureKeyboard: with keyboard nav enabled the latter stays true after clicking any button.
    if ((m_flyMode || !ImGui::GetIO().WantTextInput) && event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat)
        handleKeyDown(event.key);
    if (!m_flyMode)
        handleViewportMouse(event);
    handleCameraLook(event);
}

// Shortcuts that also work in fly mode, where ImGui gets no input; the rest are in handleShortcuts().
void Editor::handleKeyDown(const SDL_KeyboardEvent& key)
{
    // Esc always leaves fly mode, whatever the bindings, so the hidden UI can't get stuck. Quitting is
    // File > Exit or closing the window.
    if (key.scancode == SDL_SCANCODE_ESCAPE && m_flyMode) {
        setFlyMode(false);
        return;
    }
    if (capturingKey())
        return;
    if (m_keymap.matches(EditorAction::ToggleFlyMode, key))
        setFlyMode(!m_flyMode);

    dropStaleGizmoSelection();

    if (m_keymap.matches(EditorAction::Play, key))
        requestPlay();
}

bool Editor::canPlay() const
{
    return !m_models.getInstances().empty() && !m_scenes.isLoading();
}

void Editor::requestPlay()
{
    if (!canPlay()) {
        setStatus("Nothing to play: add objects or wait for the scene to finish loading", true);
        return;
    }
    if (m_flyMode)
        setFlyMode(false);
    m_rightMouseHeld = false;
    m_modeRequest = ModeRequest::PlayScene;
}

ModeRequest Editor::takeModeRequest()
{
    return std::exchange(m_modeRequest, ModeRequest::None);
}

void Editor::onResume()
{
    // The game changed the title and mouse mode, and ImGui got no events while it ran.
    m_windowTitle.clear();
    m_rightMouseHeld = false;
    m_vertexDrag.active = false;
    m_faceDrag.active = false;
    m_edgeDrag.active = false;
    m_partDrag.active = false;
    m_vertexMarquee.active = false;
    m_objectMarquee.active = false;
    if (shapeDrawActive())
        m_shapeDraw.phase = ShapeDraw::Phase::Ready;
    SDL_SetWindowRelativeMouseMode(m_window, m_flyMode);
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    io.AddMouseButtonEvent(ImGuiMouseButton_Right, false);
    io.ClearInputKeys();
    // Whatever the game session did to the scene is not an editor action.
    resetObjectBaseline();
    setStatus("Play stopped");
}

void Editor::dropStaleGizmoSelection()
{
    if (m_gizmo.selectedInstance < 0)
        return;
    const auto& instances = m_models.getInstances();
    if (m_gizmo.selectedInstance >= static_cast<int>(instances.size())) {
        deselectAll();
        return;
    }
    GPUModel* model = m_models.getModel(instances[m_gizmo.selectedInstance].modelIndex);
    if (!model || !model->isValid())
        deselectAll();
}

void Editor::handleViewportMouse(const SDL_Event& event)
{
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT &&
        !uiOwnsMouse() && sceneViewContains(event.button.x, event.button.y))
        handleViewportClick(event.button.x - m_sceneView.x, event.button.y - m_sceneView.y, event.button.clicks);

    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_LEFT) {
        m_gizmo.isDragging = false;
        m_gizmo.activeAxis = GizmoAxis::None;
        m_partDrag.active = false;
        m_vertexDrag.active = false;
        m_faceDrag.active = false;
        m_edgeDrag.active = false;
        if (m_vertexMarquee.active)
            finishVertexMarquee(event.button.x - m_sceneView.x, event.button.y - m_sceneView.y);
        if (m_objectMarquee.active)
            finishObjectMarquee(event.button.x - m_sceneView.x, event.button.y - m_sceneView.y);
        handleShapeDrawRelease(event.button.x - m_sceneView.x, event.button.y - m_sceneView.y);
    }
    if (event.type == SDL_EVENT_MOUSE_MOTION && shapeDrawActive())
        handleShapeDrawMotion(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);

    if (event.type == SDL_EVENT_MOUSE_MOTION && m_gizmo.isDragging && validInstance(m_gizmo.selectedInstance))
        dragGizmo(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);
    if (event.type == SDL_EVENT_MOUSE_MOTION && m_vertexDrag.active)
        dragLevelVertex(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);
    if (event.type == SDL_EVENT_MOUSE_MOTION && m_faceDrag.active)
        dragLevelFace(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);
    if (event.type == SDL_EVENT_MOUSE_MOTION && m_edgeDrag.active)
        dragLevelEdge(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);
    if (event.type == SDL_EVENT_MOUSE_MOTION && m_vertexMarquee.active)
        m_vertexMarquee.end = glm::vec2(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);
    if (event.type == SDL_EVENT_MOUSE_MOTION && m_objectMarquee.active)
        m_objectMarquee.end = glm::vec2(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);
}

void Editor::handleViewportClick(float mouseX, float mouseY, int clicks)
{
    const int orientationAxis = pickOrientationHandle(mouseX, mouseY);
    if (orientationAxis >= 0) {
        beginViewTurn(orientationAxis);
        return;
    }

    // Drawing a shape or on a face, or placing entities, takes every click, so the gizmo or other objects
    // can't get in the way.
    if (handleEntityPlaceClick(mouseX, mouseY))
        return;
    if (handleShapeDrawPress(mouseX, mouseY))
        return;
    if (faceDrawValid()) {
        handleFaceDrawClick(mouseX, mouseY);
        return;
    }

    const glm::mat4 view = getView(m_camera);
    const glm::mat4 proj = sceneProjection();
    if (tryBeginGizmoDrag(mouseX, mouseY, view, proj))
        return;
    if (clicks >= 2 && m_levelMode == LevelEditMode::Object) {
        // The first click took the whole group; the second one picks the object out of it.
        const int picked = objectUnderMouse(mouseX, mouseY);
        if (picked >= 0 && wholeGroupSelected(picked)) {
            selectInstance(picked);
            setStatus("Selected " + m_models.getInstances()[picked].name + " in its group");
            return;
        }
        if (enterPartMode(mouseX, mouseY))
            return;
    }

    // In face/edge/vertex mode the selected level shape keeps the selection; clicks elsewhere pick objects as usual.
    if (m_levelMode != LevelEditMode::Object && handleLevelClick(mouseX, mouseY)) {
        if (clicks >= 2 && m_levelMode == LevelEditMode::Face && selectedLevelFace())
            selectSurface();
        return;
    }
    const SDL_Keymod mods = keyMods();
    const int picked = objectUnderMouse(mouseX, mouseY);
    if (picked >= 0) {
        selectWithGroup(picked, mods);
        return;
    }
    // Empty space starts a box selection; without a drag, the release deselects.
    const bool additive = (mods & (SDL_KMOD_SHIFT | SDL_KMOD_CTRL)) != 0;
    m_objectMarquee = { true, additive, glm::vec2(mouseX, mouseY), glm::vec2(mouseX, mouseY) };
}

void Editor::pickObject(float mouseX, float mouseY)
{
    const SDL_Keymod mods = keyMods();
    const int picked = objectUnderMouse(mouseX, mouseY);
    if (picked >= 0)
        selectWithGroup(picked, mods);
    else if (!(mods & (SDL_KMOD_SHIFT | SDL_KMOD_CTRL)))
        deselectAll();
}

bool Editor::gizmoVisible() const
{
    return !m_flyMode && hasSelection() && m_gizmo.mode != GizmoMode::None &&
        m_models.getInstances()[m_gizmo.selectedInstance].visible;
}

GizmoShape Editor::currentGizmoShape()
{
    if (!gizmoVisible())
        return {};
    return buildGizmoShape(m_gizmo.mode, gizmoPivot(), m_camera.position, getView(m_camera), sceneProjection(),
        m_sceneView.width, m_sceneView.height);
}

bool Editor::tryBeginGizmoDrag(float mouseX, float mouseY, const glm::mat4& view, const glm::mat4& proj)
{
    const GizmoShape shape = currentGizmoShape();
    const glm::vec2 mouse(mouseX, mouseY);
    const GizmoPick pick = pickGizmoShape(shape, mouse, kGizmoPickRadius);
    if (pick.axis == GizmoAxis::None)
        return false;

    auto& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    const glm::mat4 viewProj = proj * view;
    const glm::vec3 axisDir = gizmoAxisDirection(pick.axis);
    glm::vec2 screenDir(1.0f, 0.0f);
    float pixelsPerUnit = 1.0f;
    if (m_gizmo.mode == GizmoMode::Rotate) {
        // Dragging along the ring's tangent at the grabbed point turns the object the way the ring moves.
        const glm::vec3 tangent = glm::normalize(glm::cross(axisDir, pick.worldPoint - shape.center));
        const glm::vec2 a = worldToScreen(pick.worldPoint, viewProj, m_sceneView.width, m_sceneView.height);
        const glm::vec2 b = worldToScreen(pick.worldPoint + tangent * (0.1f * shape.scale), viewProj,
            m_sceneView.width, m_sceneView.height);
        if (glm::length(b - a) > 0.001f && a.x > -5000.0f && b.x > -5000.0f)
            screenDir = glm::normalize(b - a);
    }
    else {
        // Mouse motion counts only along the axis as it appears on screen.
        const glm::vec2 a = shape.screenCenter;
        const glm::vec2 b = worldToScreen(shape.center + axisDir * shape.scale, viewProj,
            m_sceneView.width, m_sceneView.height);
        const float length = glm::length(b - a);
        if (length >= 1.0f && b.x > -5000.0f) {
            screenDir = (b - a) / length;
            pixelsPerUnit = length / shape.scale;
        }
    }

    m_gizmo.activeAxis = pick.axis;
    m_gizmo.isDragging = true;
    m_gizmo.dragStart = mouse;
    m_gizmo.dragDirection = screenDir;
    m_gizmo.pixelsPerUnit = pixelsPerUnit;
    m_gizmo.originalPosition = instance.position;
    m_gizmo.originalRotation = instance.rotation;
    m_gizmo.originalScale = instance.scale;
    // In part mode with parts selected the gizmo is theirs; otherwise the selected objects follow it.
    beginPartDrag();
    if (!m_partDrag.active)
        beginGroupDrag();
    return true;
}

void Editor::dragGizmo(float mouseX, float mouseY)
{
    auto& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    const float amount = glm::dot(glm::vec2(mouseX, mouseY) - m_gizmo.dragStart, m_gizmo.dragDirection);
    const int axis = static_cast<int>(m_gizmo.activeAxis) - 1;
    if (axis < 0 || axis > 2)
        return;
    if (m_partDrag.active) {
        dragParts(amount, axis);
        return;
    }
    const glm::vec3 axisDir = gizmoAxisDirection(m_gizmo.activeAxis);
    const bool snap = snapActive();
    const auto snapTo = [](float value, float step) { return step > 0.0f ? std::round(value / step) * step : value; };
    markSceneChanged(); // recorded when the drag ends

    float degrees = 0.0f;
    float scaleFactor = 1.0f;
    switch (m_gizmo.mode) {
    case GizmoMode::Translate:
        instance.position = m_gizmo.originalPosition + axisDir * (amount / m_gizmo.pixelsPerUnit);
        if (snap)
            instance.position[axis] = snapTo(instance.position[axis], m_gridSize);
        break;
    case GizmoMode::Rotate: {
        const float degreesPerPixel = 0.5f;
        degrees = amount * degreesPerPixel;
        if (snap) degrees = snapTo(degrees, m_snapRotate);
        // About the world axis of the ring that is dragged.
        instance.rotation = rotateEulerAboutAxis(m_gizmo.originalRotation, axis, degrees);
        break;
    }
    case GizmoMode::Scale: {
        const float scalePerPixel = 0.01f;
        float delta = amount * scalePerPixel;
        if (snap) delta = snapTo(delta, m_snapScale);
        instance.scale = glm::max(m_gizmo.originalScale + axisDir * delta, glm::vec3(0.01f));
        scaleFactor = instance.scale[axis] / std::max(m_gizmo.originalScale[axis], 1e-4f);
        break;
    }
    default:
        break;
    }
    dragGroup(axis, degrees, scaleFactor);
}

void Editor::handleCameraLook(const SDL_Event& event)
{
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_RIGHT &&
        !m_flyMode && !uiOwnsMouse() && sceneViewContains(event.button.x, event.button.y))
        m_rightMouseHeld = true;
    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_RIGHT)
        m_rightMouseHeld = false;

    if (event.type == SDL_EVENT_MOUSE_MOTION && !m_cameraAnimator.isPlaying() && (m_flyMode || m_rightMouseHeld)) {
        m_camera.yaw += event.motion.xrel * m_camera.sensitivity;
        m_camera.pitch -= event.motion.yrel * m_camera.sensitivity * (m_prefs.invertLookY ? -1.0f : 1.0f);
        m_camera.pitch = glm::clamp(m_camera.pitch, -89.0f, 89.0f);
    }
}

void Editor::moveCamera(float dt)
{
    if ((!m_flyMode && ImGui::GetIO().WantTextInput) || m_cameraAnimator.isPlaying() || capturingKey())
        return;
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const glm::vec3 front = getFront(m_camera);
    const glm::vec3 right = glm::normalize(glm::cross(front, up));
    const float step = m_camera.speed * dt * (m_keymap.held(EditorAction::CameraFast) ? 4.0f : 1.0f);
    const auto held = [this](EditorAction action) { return m_keymap.held(action); };
    if (held(EditorAction::CameraForward)) m_camera.position += front * step;
    if (held(EditorAction::CameraBack)) m_camera.position -= front * step;
    if (held(EditorAction::CameraLeft)) m_camera.position -= right * step;
    if (held(EditorAction::CameraRight)) m_camera.position += right * step;
    if (held(EditorAction::CameraUp)) m_camera.position += up * step;
    if (held(EditorAction::CameraDown)) m_camera.position -= up * step;
}

void Editor::setFlyMode(bool enabled)
{
    m_flyMode = enabled;
    SDL_SetWindowRelativeMouseMode(m_window, enabled);
}

bool Editor::sceneViewContains(float x, float y) const
{
    return x >= m_sceneView.x && x < m_sceneView.x + m_sceneView.width &&
        y >= m_sceneView.y && y < m_sceneView.y + m_sceneView.height;
}

// ---------------------------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------------------------

void Editor::update(float dt)
{
    if (m_flyMode) {
        // Fly mode hides the editor UI, so the scene uses the whole window.
        int width = 0, height = 0;
        SDL_GetWindowSize(m_window, &width, &height);
        m_sceneView = { 0.0f, 0.0f, static_cast<float>(std::max(width, 1)), static_cast<float>(std::max(height, 1)) };
    }
    moveCamera(dt);
    updateViewTurn(dt);
}

void Editor::lateUpdate(float dt)
{
    if (m_cameraAnimator.update(dt)) {
        const CameraState state = m_cameraAnimator.getCurrentState();
        m_camera.position = state.position;
        m_camera.yaw = state.yaw;
        m_camera.pitch = state.pitch;
    }
    pollMcpServer();
    pollScript();
    pumpSimulatedKeys();
    pumpSimulatedMouse();
}

void Editor::drawUi()
{
    validateSelection();
    handleShortcuts();
    applyPendingMaterialDrop();
    // The gizmo shows the active tool on the selected object ("Select" tool = no gizmo).
    m_gizmo.mode = hasSelection() ? m_tool : GizmoMode::None;
    updateWindowTitle();

    drawMainMenuBar();
    drawToolbar();
    drawStatusBar();
    drawDockSpace();
    drawViewportOverlay();
    drawLevelFaceOverlay();
    drawObjectMarquee();
    drawShapeDrawOverlay();
    drawEntityPlaceOverlay();
    drawShapeEditOverlay();
    drawLightOverlay();
    drawTransformGizmo();
    drawOrientationGizmo();
    if (m_showHierarchy) drawHierarchy();
    if (m_showInspector) drawInspector();
    if (m_showStatisticsPanel) drawStatisticsPanel();
    if (m_showLevelPanel) drawLevelPanel();
    if (m_showMaterialsPanel) drawMaterialsPanel();
    drawMaterialDropTarget();
    if (m_showAnimationPanel) drawAnimationPanel();
    if (m_showGraphicsSettings) drawGraphicsSettingsWindow();
    if (m_showSceneSettings) drawSceneSettingsWindow();
    if (m_showKeymap) drawKeymapWindow();
    else m_keyCapture = {};
    if (m_showPreferences) drawPreferencesWindow();
    updateLevelHistory();
    updateObjectHistory();
    settleSavedState();
    updateAutosave();
    drawFileDialogs();
    drawHelpPopups();
}

void Editor::fillFrame(FrameInput& frame)
{
    frame.models = &m_models;
    frame.view = getView(m_camera);
    frame.proj = sceneProjection();
    frame.cameraPosition = m_camera.position;
    frame.viewport = m_sceneView;
    fillHighlight(frame.highlight);
    frame.showPath = !m_flyMode && m_showCameraPath;
    const bool levelGrid = !m_flyMode && m_showGrid && levelGridActive();
    frame.showGrid = !m_flyMode && m_showGrid && !levelGrid;
    frame.levelGrid = levelGrid;
    frame.gridCellSize = activeGridSize();
}

bool Editor::snapActive() const
{
    const bool ctrl = (keyMods() & SDL_KMOD_CTRL) != 0;
    return m_gridSnap != ctrl;
}

SDL_Keymod Editor::keyMods() const
{
    return m_mouseSimulated ? m_simulatedMods : SDL_GetModState();
}

bool Editor::levelGridActive()
{
    return m_levelMode != LevelEditMode::Object && selectedLevelMesh() != nullptr;
}

float Editor::activeGridSize()
{
    return levelGridActive() ? selectedLevelMesh()->gridSize : m_gridSize;
}

void Editor::setActiveGridSize(float size)
{
    size = std::clamp(size, kMinPolyGridSize, kMaxPolyGridSize);
    char text[64];
    if (levelGridActive()) {
        PolyMesh* mesh = selectedLevelMesh();
        if (mesh->gridSize == size)
            return;
        mesh->gridSize = size;
        // Saved with the shape, so it goes through the level history like other shape edits.
        rebuildSelectedLevelModel("shape grid size");
        snprintf(text, sizeof(text), "Shape grid size %g", size);
    }
    else {
        m_gridSize = size;
        snprintf(text, sizeof(text), "Grid size %g", size);
    }
    setStatus(text);
}

void Editor::stepGridSize(int direction)
{
    constexpr int count = static_cast<int>(std::size(kGridSizes));
    const float size = activeGridSize();
    // Nearest preset first, so a size set elsewhere still steps sensibly.
    int current = 0;
    for (int i = 1; i < count; ++i)
        if (std::abs(std::log2(kGridSizes[i] / size)) < std::abs(std::log2(kGridSizes[current] / size)))
            current = i;
    setActiveGridSize(kGridSizes[std::clamp(current + direction, 0, count - 1)]);
}

glm::mat4 Editor::sceneProjection() const
{
    return getProjection(m_sceneView.width, m_sceneView.height, kCameraNearPlane, m_settings.viewDistance,
        m_prefs.fieldOfView);
}

void Editor::fillHighlight(SelectionHighlight& highlight)
{
    highlight = {};
    m_highlightIndices.clear();
    if (m_flyMode || !hasSelection())
        return;
    // The shape grid is drawn on what is highlighted, so face/vertex editing outlines the edited shape only.
    if (levelGridActive())
        m_highlightIndices.push_back(m_gizmo.selectedInstance);
    else
        m_highlightIndices = selectedIndices();
    highlight.instances = m_highlightIndices;
}

void Editor::handleShortcuts()
{
    if (ImGui::GetIO().WantTextInput || capturingKey())
        return;
    using A = EditorAction;
    const auto pressed = [this](EditorAction action) { return m_keymap.pressed(action); };
    const bool selection = hasSelection();
    // Part mode on a shape: duplicate, delete, select all, turn and detach act on its parts.
    const bool partMode = m_levelMode == LevelEditMode::Part && selectedLevelMesh();
    const bool partsSelected = partMode && !selectedPartFaces().empty();
    if (pressed(A::NewScene)) requestSceneAction(SceneAction::New);
    if (pressed(A::OpenScene)) requestSceneAction(SceneAction::OpenDialog);
    if (pressed(A::SaveSceneAs)) saveSceneAsDialog();
    if (pressed(A::SaveScene)) saveScene();
    if (pressed(A::ImportModel)) importModelDialog();
    if (pressed(A::Undo)) undo();
    if (pressed(A::Redo)) redo();
    if (pressed(A::Duplicate) && selection) {
        if (partsSelected) duplicateSelectedParts();
        else duplicateSelection();
    }
    if (pressed(A::Delete) && selection && !deleteLevelSelection()) deleteSelection();
    if (pressed(A::Focus) && selection) focusSelection();
    if (pressed(A::SelectAll)) {
        if (m_levelMode == LevelEditMode::Face && selectedLevelMesh())
            selectFacesWhere([](const PolyMesh&, const PolyFace&) { return true; });
        else if (partMode)
            selectAllParts();
        else
            selectAll();
    }
    if (pressed(A::Copy) && selection) copySelection();
    if (pressed(A::Cut) && selection) cutSelection();
    if (pressed(A::Paste)) pasteClipboard();
    if (pressed(A::DropToFloor) && selection) dropSelectionToFloor();
    if (pressed(A::RotateClockwise) && selection) {
        if (partsSelected) rotateSelectedParts(-90.0f);
        else rotateSelection(-90.0f);
    }
    if (pressed(A::RotateCounterClockwise) && selection) {
        if (partsSelected) rotateSelectedParts(90.0f);
        else rotateSelection(90.0f);
    }
    if (pressed(A::Hide) && selection) toggleSelectionVisibility();
    if (pressed(A::UnhideAll)) unhideAll();
    // Their cancel key wins over deselecting.
    const bool drawingOnFace = handleFaceDrawKeys();
    const bool drawingShape = handleShapeDrawKeys();
    const bool placingEntities = entityPlacementActive();
    if (placingEntities && pressed(A::Deselect) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId))
        stopEntityPlacement();
    if (pressed(A::Deselect) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId) && !drawingOnFace &&
        !drawingShape && !placingEntities) {
        // While editing a shape: first drops the picked faces/vertices, then ends the edit keeping the shape
        // selected, like leaving a group.
        if (shapeEditActive()) {
            if (shapeEditHasPicks())
                clearShapeEditPicks();
            else
                finishShapeEdit();
        }
        else if (m_levelMode == LevelEditMode::Part) {
            m_levelMode = LevelEditMode::Object;
            m_selectedParts.clear();
        }
        else {
            deselectAll();
        }
    }
    if (pressed(A::EditShape) && !drawingOnFace && !drawingShape) toggleShapeEdit();
    if (pressed(A::DrawShape)) toggleShapeDraw();
    if (pressed(A::UniteShapes)) uniteSelectedShapes();
    if (pressed(A::SeparateShape)) {
        if (partsSelected) detachSelectedParts();
        else separateSelectedShape();
    }
    if (pressed(A::SaveAsPrefab)) saveSelectionAsPrefab();
    if (pressed(A::GroupObjects)) groupSelection();
    if (pressed(A::UngroupObjects) && selection) ungroupSelection();
    if (pressed(A::SelectTool)) m_tool = GizmoMode::None;
    if (pressed(A::MoveTool)) m_tool = GizmoMode::Translate;
    if (pressed(A::RotateTool)) m_tool = GizmoMode::Rotate;
    if (pressed(A::ScaleTool)) m_tool = GizmoMode::Scale;
    if (pressed(A::ToggleSnap)) {
        m_gridSnap = !m_gridSnap;
        setStatus(m_gridSnap ? "Grid snapping on" : "Grid snapping off");
    }
    if (pressed(A::Eyedropper)) pickMaterialUnderMouse();
    if (pressed(A::ObjectMode)) m_levelMode = LevelEditMode::Object;
    if (pressed(A::FaceMode)) m_levelMode = LevelEditMode::Face;
    if (pressed(A::EdgeMode)) m_levelMode = LevelEditMode::Edge;
    if (pressed(A::VertexMode)) m_levelMode = LevelEditMode::Vertex;
    if (pressed(A::PartMode)) m_levelMode = LevelEditMode::Part;
    if (m_levelMode != LevelEditMode::Object)
        m_lastShapeEditMode = m_levelMode;
    handleFaceKeys();
    if (pressed(A::GridSmaller)) stepGridSize(-1);
    if (pressed(A::GridLarger)) stepGridSize(1);
    if (pressed(A::ToggleGrid)) m_showGrid = !m_showGrid;
    if (pressed(A::ShowControls)) m_openControlsPopup = true;
    if (pressed(A::ShowKeymap)) m_showKeymap = true;
    if (pressed(A::Rename) && selection) beginRename(m_gizmo.selectedInstance);
}

void Editor::updateWindowTitle()
{
    const std::string& path = m_scenes.currentPath();
    std::string title = "MirasEngine - " + (path.empty() ? std::string("Untitled") : path) + (sceneDirty() ? "*" : "");
    if (title != m_windowTitle) {
        SDL_SetWindowTitle(m_window, title.c_str());
        m_windowTitle = std::move(title);
    }
}

void Editor::setStatus(const std::string& message, bool isError)
{
    m_statusMessage = message;
    m_statusIsError = isError;
    m_statusTime = ImGui::GetTime();
    if (isError)
        LOG_ERROR("[EDITOR] " << message << "\n");
    else
        LOG_INFO("[EDITOR] " << message << "\n");
}

std::string Editor::formatCount(size_t value)
{
    const std::string digits = std::to_string(value);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------------------------

bool Editor::validInstance(int index) const
{
    return index >= 0 && index < static_cast<int>(m_models.getInstances().size());
}

// ---------------------------------------------------------------------------------------------
// Object actions
// ---------------------------------------------------------------------------------------------

void Editor::focusOnInstance(int index)
{
    if (!validInstance(index))
        return;
    const auto& instance = m_models.getInstances()[index];
    GPUModel* model = m_models.getModel(instance.modelIndex);
    if (!model)
        return;
    const float maxScale = std::max({ instance.scale.x, instance.scale.y, instance.scale.z });
    glm::vec3 center = glm::vec3(instance.getTransformMatrix() * glm::vec4(model->boundsCenter, 1.0f));
    const float radius = model->boundsRadius * maxScale;
    // A sphere of radius r fits the vertical field of view at distance r / sin(fov / 2); 10% margin.
    const float fit = 1.1f / std::sin(glm::radians(m_prefs.fieldOfView * 0.5f));
    m_camera.position = center - getFront(m_camera) * std::max(radius * fit, 1.0f);
}

void Editor::addModelToScene(size_t modelIndex, bool atOrigin)
{
    GPUModel* model = m_models.getModel(modelIndex);
    if (!model)
        return;
    glm::vec3 position(0.0f);
    if (!atOrigin)
        position = m_camera.position + getFront(m_camera) * std::max(model->boundsRadius * 2.2f, 2.0f) - model->boundsCenter;
    const size_t newIndex = m_models.createInstance(modelIndex, position);
    markSceneChanged();
    selectInstance(static_cast<int>(newIndex));
    setStatus("Added " + model->name + " to the scene");
}

void Editor::deleteInstance(int index)
{
    if (!validInstance(index))
        return;
    const std::string name = m_models.getInstances()[index].name;
    deselectAll();
    m_models.removeInstance(static_cast<size_t>(index));
    releaseUnusedLevelModels();
    markSceneChanged();
    setStatus("Deleted " + name);
}

void Editor::duplicateInstance(int index)
{
    if (!validInstance(index))
        return;
    const ModelInstance& source = m_models.getInstances()[index];
    const GPUModel* model = m_models.getModel(source.modelIndex);
    const float offset = model ? model->boundsRadius * std::max({ source.scale.x, source.scale.y, source.scale.z }) : 1.0f;
    const std::string name = source.name;
    const auto copy = copyInstance(index, glm::vec3(offset, 0.0f, 0.0f));
    if (!copy)
        return;
    selectInstance(static_cast<int>(*copy));
    setStatus("Duplicated " + name);
}

std::optional<size_t> Editor::copyInstance(int index, const glm::vec3& offset)
{
    const ModelInstance source = m_models.getInstances()[index];
    GPUModel* model = m_models.getModel(source.modelIndex);
    // Level geometry is copied so the duplicate can be edited on its own; prefab instances keep sharing it.
    size_t modelIndex = source.modelIndex;
    if (model && model->polyMesh && model->prefabPath.empty()) {
        const auto copy = createLevelModel(*model->polyMesh, model->name);
        if (!copy)
            return std::nullopt;
        modelIndex = *copy;
    }
    const size_t newIndex = m_models.createInstance(modelIndex, source.position + offset, source.rotation, source.scale);
    ModelInstance& created = m_models.getInstances()[newIndex];
    created.name = uniqueInstanceName(source.name);
    created.color = source.color;
    created.visible = source.visible;
    created.locked = source.locked;
    created.entity = source.entity;
    created.entityParams = source.entityParams;
    created.group = source.group;
    created.collision = source.collision;
    markSceneChanged();
    return newIndex;
}

glm::vec3 Editor::instanceWorldCenter(int index) const
{
    const ModelInstance& instance = m_models.getInstances()[index];
    const auto& models = m_models.getModels();
    if (instance.modelIndex >= models.size() || !models[instance.modelIndex])
        return instance.position;
    return glm::vec3(instance.getTransformMatrix() * glm::vec4(models[instance.modelIndex]->boundsCenter, 1.0f));
}

std::string Editor::uniqueInstanceName(const std::string& name) const
{
    // "Box (2)" -> "Box", so copies of copies count on from the original name.
    std::string base = name;
    if (base.size() > 4 && base.back() == ')') {
        const size_t open = base.rfind(" (");
        if (open != std::string::npos && open + 3 < base.size() &&
            std::all_of(base.begin() + open + 2, base.end() - 1, [](char c) { return c >= '0' && c <= '9'; }))
            base.erase(open);
    }
    const auto& instances = m_models.getInstances();
    const auto taken = [&](const std::string& name) {
        return std::any_of(instances.begin(), instances.end(), [&](const ModelInstance& i) { return i.name == name; });
    };
    if (!taken(name))
        return name;
    for (int n = 1;; ++n) {
        std::string candidate = base + " (" + std::to_string(n) + ")";
        if (!taken(candidate))
            return candidate;
    }
}

void Editor::addCube()
{
    size_t modelIndex = 0;
    if (const auto found = m_models.findModelByPath(kBuiltinCubePath)) {
        modelIndex = *found;
    }
    else {
        try {
            modelIndex = m_models.loadModelSync(kBuiltinCubePath, "Cube");
        }
        catch (const std::exception& e) {
            setStatus(std::string("Failed to create cube: ") + e.what(), true);
            return;
        }
    }
    const glm::vec3 position = glm::round((m_camera.position + getFront(m_camera) * 5.0f) / m_gridSize) * m_gridSize;
    const size_t newIndex = m_models.createInstance(modelIndex, position);
    const std::string name = uniqueInstanceName("Cube");
    m_models.getInstances()[newIndex].name = name;
    markSceneChanged();
    selectInstance(static_cast<int>(newIndex));
    setStatus("Added " + name);
}

std::optional<size_t> Editor::addEntity(const std::string& type, const glm::vec3* at)
{
    const EntityTypeInfo* info = findEntityType(type);
    if (!info) {
        setStatus("Unknown entity type: " + type, true);
        return std::nullopt;
    }
    size_t modelIndex = 0;
    if (const auto found = m_models.findModelByPath(info->model)) {
        modelIndex = *found;
    }
    else {
        try {
            modelIndex = m_models.loadModelSync(info->model, info->label);
        }
        catch (const std::exception& e) {
            setStatus(std::string("Can't load ") + info->model + ": " + e.what(), true);
            return std::nullopt;
        }
    }
    const size_t index = m_models.createInstance(modelIndex, at ? *at : placementPoint());
    ModelInstance& instance = m_models.getInstances()[index];
    instance.name = uniqueInstanceName(info->label);
    instance.entity = info->id;
    instance.entityParams = info->defaultParams;
    // Entities face +Z; turn it the way the camera looks, in 45 degree steps.
    const glm::vec3 front = getFront(m_camera);
    instance.rotation.y = std::round(glm::degrees(std::atan2(front.x, front.z)) / 45.0f) * 45.0f;
    markSceneChanged();
    selectInstance(static_cast<int>(index));
    setStatus("Added " + instance.name);
    return index;
}

void Editor::drawEntityMenuItems()
{
    for (const EntityTypeInfo& type : entityTypes()) {
        if (type.presets.empty()) {
            if (ImGui::MenuItem(type.label, nullptr, m_entityPlace.type == type.id))
                beginEntityPlacement(type.id, -1);
            ImGui::SetItemTooltip("%s\nClick surfaces in the viewport to place it; Esc stops.", type.description);
            continue;
        }
        if (ImGui::BeginMenu(type.label)) {
            if (entityDefaultPlaceable(type)) {
                if (ImGui::MenuItem("Default", nullptr, m_entityPlace.type == type.id && m_entityPlace.preset < 0))
                    beginEntityPlacement(type.id, -1);
                ImGui::SetItemTooltip("%s\nClick surfaces in the viewport to place it; Esc stops.", type.description);
            }
            for (size_t i = 0; i < type.presets.size(); ++i) {
                const bool active = m_entityPlace.type == type.id && m_entityPlace.preset == static_cast<int>(i);
                if (ImGui::MenuItem(type.presets[i].label.c_str(), nullptr, active))
                    beginEntityPlacement(type.id, static_cast<int>(i));
                ImGui::SetItemTooltip("%s\nClick surfaces in the viewport to place it; Esc stops.", type.presets[i].params.c_str());
            }
            ImGui::EndMenu();
        }
    }
}

void Editor::beginRename(int index)
{
    if (!validInstance(index))
        return;
    m_showHierarchy = true;
    m_renamingInstance = index;
    m_renameFocusPending = true;
    const std::string& name = m_models.getInstances()[index].name;
    strncpy(m_renameBuffer, name.c_str(), sizeof(m_renameBuffer) - 1);
    m_renameBuffer[sizeof(m_renameBuffer) - 1] = '\0';
}

void Editor::unloadModel(size_t modelIndex)
{
    GPUModel* model = m_models.getModel(modelIndex);
    if (!model)
        return;
    const std::string name = model->name;
    deselectAll();
    m_models.unloadModel(modelIndex);
    // Its instances can't be brought back without the model, so their removal is not an undo step.
    resetObjectBaseline();
    setStatus("Unloaded " + name);
}

// ---------------------------------------------------------------------------------------------
// Scene actions
// ---------------------------------------------------------------------------------------------

void Editor::newScene()
{
    deselectAll();
    m_scenes.clear();
    m_scenes.settings() = {};
    m_savedSceneSettings = {};
    clearHistory();
    m_scenes.setCurrentPath({});
    m_savedState = sceneStateId();
    setStatus("New scene");
}

void Editor::openScene(const std::string& path)
{
    const SceneManager::OpenResult opened = m_scenes.open(path);
    if (!opened.ok) {
        // A recent scene that was moved or deleted leaves the list.
        if (std::erase(m_prefs.recentScenes, path) > 0)
            setStatus("Failed to open scene (removed from recent scenes): " + path, true);
        else
            setStatus("Failed to open scene: " + path, true);
        return;
    }
    // The previous scene is gone, so the selection would point at a stale instance.
    deselectAll();
    clearHistory();
    markSceneSaved(path);
    for (const auto& missing : opened.missingFiles)
        setStatus("Model file not found: " + missing, true);
    setStatus("Opening " + path + " (" + std::to_string(opened.queuedModels) + " models)...");
}

void Editor::saveSceneTo(const std::string& path)
{
    if (m_scenes.save(path)) {
        markSceneSaved(path);
        setStatus("Scene saved: " + path);
    }
    else {
        setStatus("Failed to save scene: " + path, true);
    }
}

void Editor::saveScene()
{
    if (m_scenes.currentPath().empty())
        saveSceneAsDialog();
    else
        saveSceneTo(m_scenes.currentPath());
}
