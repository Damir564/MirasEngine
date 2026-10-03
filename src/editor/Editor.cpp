#include "Editor.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <utility>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
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
{
    int width = 0, height = 0;
    SDL_GetWindowSize(m_window, &width, &height);
    m_sceneView = { 0.0f, 0.0f, static_cast<float>(std::max(width, 1)), static_cast<float>(std::max(height, 1)) };

    m_cameraAnimator.getPath().name = m_pathName;
    LOG_INFO("Camera animation system initialized\n");

    SDL_SetWindowRelativeMouseMode(m_window, m_flyMode);
}

// ---------------------------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------------------------

void Editor::onEvent(const SDL_Event& event)
{
    // WantTextInput, not WantCaptureKeyboard: with keyboard nav enabled the latter stays true after clicking any button.
    if (m_flyMode || !ImGui::GetIO().WantTextInput) {
        if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat)
            handleKeyDown(event.key);
        if (event.type == SDL_EVENT_KEY_UP && !event.key.repeat && !(event.key.mod & SDL_KMOD_SHIFT))
            m_cameraSpeedMultiplier = 1.0f;
    }
    if (!m_flyMode)
        handleViewportMouse(event);
    handleCameraLook(event);
}

void Editor::handleKeyDown(const SDL_KeyboardEvent& key)
{
    // Escape leaves fly mode (and deselects via the UI shortcut); quitting is File > Exit or closing the window.
    if (key.scancode == SDL_SCANCODE_ESCAPE && m_flyMode)
        setFlyMode(false);

    if (key.mod & SDL_KMOD_SHIFT) {
        m_cameraSpeedMultiplier = 4.0f;
        if (key.scancode == SDL_SCANCODE_GRAVE)
            setFlyMode(!m_flyMode);
    }

    dropStaleGizmoSelection();

    if (key.scancode == SDL_SCANCODE_F5)
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
    m_vertexMarquee.active = false;
    m_cameraSpeedMultiplier = 1.0f;
    SDL_SetWindowRelativeMouseMode(m_window, m_flyMode);
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    io.AddMouseButtonEvent(ImGuiMouseButton_Right, false);
    io.ClearInputKeys();
    setStatus("Play stopped");
}

void Editor::dropStaleGizmoSelection()
{
    if (m_gizmo.selectedInstance < 0)
        return;
    const auto& instances = m_models.getInstances();
    if (m_gizmo.selectedInstance >= static_cast<int>(instances.size())) {
        m_gizmo.deselect();
        return;
    }
    GPUModel* model = m_models.getModel(instances[m_gizmo.selectedInstance].modelIndex);
    if (!model || !model->isValid())
        m_gizmo.deselect();
}

void Editor::handleViewportMouse(const SDL_Event& event)
{
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT &&
        !ImGui::GetIO().WantCaptureMouse && sceneViewContains(event.button.x, event.button.y))
        handleViewportClick(event.button.x - m_sceneView.x, event.button.y - m_sceneView.y);

    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_LEFT) {
        m_gizmo.isDragging = false;
        m_gizmo.activeAxis = GizmoAxis::None;
        m_vertexDrag.active = false;
        if (m_vertexMarquee.active)
            finishVertexMarquee(event.button.x - m_sceneView.x, event.button.y - m_sceneView.y);
    }

    if (event.type == SDL_EVENT_MOUSE_MOTION && m_gizmo.isDragging && validInstance(m_gizmo.selectedInstance))
        dragGizmo(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);
    if (event.type == SDL_EVENT_MOUSE_MOTION && m_vertexDrag.active)
        dragLevelVertex(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);
    if (event.type == SDL_EVENT_MOUSE_MOTION && m_vertexMarquee.active)
        m_vertexMarquee.end = glm::vec2(event.motion.x - m_sceneView.x, event.motion.y - m_sceneView.y);
}

void Editor::handleViewportClick(float mouseX, float mouseY)
{
    const int orientationAxis = pickOrientationHandle(mouseX, mouseY);
    if (orientationAxis >= 0) {
        beginViewTurn(orientationAxis);
        return;
    }

    const glm::mat4 view = getView(m_camera);
    const glm::mat4 proj = sceneProjection();
    if (tryBeginGizmoDrag(mouseX, mouseY, view, proj))
        return;

    // In face/vertex mode the selected level shape keeps the selection; clicks elsewhere pick objects as usual.
    if (m_levelMode != LevelEditMode::Object && handleLevelClick(mouseX, mouseY))
        return;
    pickObject(mouseX, mouseY);
}

void Editor::pickObject(float mouseX, float mouseY)
{
    const glm::mat4 view = getView(m_camera);
    const glm::mat4 proj = sceneProjection();
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height, view, proj);
    const SubmeshHitResult hit = pickSubmesh(ray, m_models.getInstances(),
        [&](size_t index) { return m_models.getModel(index); });
    if (hit.hit()) {
        selectPickedSubmesh(hit);
    }
    else {
        m_gizmo.deselect();
    }
}

bool Editor::gizmoVisible() const
{
    return !m_flyMode && hasSelection() && m_gizmo.mode != GizmoMode::None &&
        m_models.getInstances()[m_gizmo.selectedInstance].visible;
}

GizmoShape Editor::currentGizmoShape() const
{
    if (!gizmoVisible())
        return {};
    return buildGizmoShape(m_gizmo.mode, m_models.getInstances()[m_gizmo.selectedInstance].position,
        m_camera.position, getView(m_camera), sceneProjection(), m_sceneView.width, m_sceneView.height);
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
    return true;
}

void Editor::selectPickedSubmesh(const SubmeshHitResult& hit)
{
    m_gizmo.select(hit.instanceIndex);
    LOG_INFO("[PICK] Instance " << hit.instanceIndex << " | Submesh " << hit.submeshIndex
        << " | t: " << hit.t << "\n");
}

void Editor::dragGizmo(float mouseX, float mouseY)
{
    auto& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    const float amount = glm::dot(glm::vec2(mouseX, mouseY) - m_gizmo.dragStart, m_gizmo.dragDirection);
    const int axis = static_cast<int>(m_gizmo.activeAxis) - 1;
    if (axis < 0 || axis > 2)
        return;
    const glm::vec3 axisDir = gizmoAxisDirection(m_gizmo.activeAxis);
    const bool snap = (SDL_GetModState() & SDL_KMOD_CTRL) != 0;
    const auto snapTo = [](float value, float step) { return step > 0.0f ? std::round(value / step) * step : value; };

    switch (m_gizmo.mode) {
    case GizmoMode::Translate:
        instance.position = m_gizmo.originalPosition + axisDir * (amount / m_gizmo.pixelsPerUnit);
        if (snap)
            instance.position[axis] = snapTo(instance.position[axis], m_snapTranslate);
        break;
    case GizmoMode::Rotate: {
        const float degreesPerPixel = 0.5f;
        float degrees = amount * degreesPerPixel;
        if (snap) degrees = snapTo(degrees, m_snapRotate);
        instance.rotation = m_gizmo.originalRotation + axisDir * degrees;
        break;
    }
    case GizmoMode::Scale: {
        const float scalePerPixel = 0.01f;
        float delta = amount * scalePerPixel;
        if (snap) delta = snapTo(delta, m_snapScale);
        instance.scale = glm::max(m_gizmo.originalScale + axisDir * delta, glm::vec3(0.01f));
        break;
    }
    default:
        break;
    }
}

void Editor::handleCameraLook(const SDL_Event& event)
{
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_RIGHT &&
        !m_flyMode && !ImGui::GetIO().WantCaptureMouse && sceneViewContains(event.button.x, event.button.y))
        m_rightMouseHeld = true;
    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_RIGHT)
        m_rightMouseHeld = false;

    if (event.type == SDL_EVENT_MOUSE_MOTION && !m_cameraAnimator.isPlaying() && (m_flyMode || m_rightMouseHeld)) {
        m_camera.yaw += event.motion.xrel * m_camera.sensitivity;
        m_camera.pitch -= event.motion.yrel * m_camera.sensitivity;
        m_camera.pitch = glm::clamp(m_camera.pitch, -89.0f, 89.0f);
    }
}

void Editor::moveCamera(float dt)
{
    if ((!m_flyMode && ImGui::GetIO().WantTextInput) || m_cameraAnimator.isPlaying())
        return;
    const bool* keys = SDL_GetKeyboardState(nullptr);
    const glm::vec3 front = getFront(m_camera);
    const glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0, 1, 0)));
    const float step = m_camera.speed * dt * m_cameraSpeedMultiplier;
    if (keys[SDL_SCANCODE_W]) m_camera.position += front * step;
    if (keys[SDL_SCANCODE_A]) m_camera.position -= right * step;
    if (keys[SDL_SCANCODE_D]) m_camera.position += right * step;
    if (keys[SDL_SCANCODE_S]) m_camera.position -= front * step;
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
}

void Editor::drawUi()
{
    validateSelection();
    handleShortcuts();
    // The gizmo shows the active tool on the selected object ("Select" tool = no gizmo).
    m_gizmo.mode = hasSelection() ? m_tool : GizmoMode::None;
    updateWindowTitle();

    drawMainMenuBar();
    drawToolbar();
    drawStatusBar();
    drawDockSpace();
    drawViewportOverlay();
    drawLevelFaceOverlay();
    drawTransformGizmo();
    drawOrientationGizmo();
    if (m_showHierarchy) drawHierarchy();
    if (m_showInspector) drawInspector();
    if (m_showStatisticsPanel) drawStatisticsPanel();
    if (m_showLevelPanel) drawLevelPanel();
    if (m_showAnimationPanel) drawAnimationPanel();
    if (m_showGraphicsSettings) drawGraphicsSettingsWindow();
    updateLevelHistory();
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
    frame.showGrid = !m_flyMode && m_showGrid;
}

glm::mat4 Editor::sceneProjection() const
{
    return getProjection(m_sceneView.width, m_sceneView.height, kCameraNearPlane, m_settings.viewDistance);
}

void Editor::fillHighlight(SelectionHighlight& highlight) const
{
    highlight = {};
    if (m_flyMode || !hasSelection())
        return;
    highlight.instance = m_gizmo.selectedInstance;
}

void Editor::handleShortcuts()
{
    if (ImGui::GetIO().WantTextInput)
        return;
    const bool selection = hasSelection();
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O)) openSceneDialog();
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S)) saveSceneAsDialog();
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S)) saveScene();
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_I)) importModelDialog();
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Z)) undoLevelEdit();
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Y) ||
        ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z)) redoLevelEdit();
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_D) && selection) duplicateInstance(m_gizmo.selectedInstance);
    if (ImGui::IsKeyChordPressed(ImGuiKey_Delete) && selection && !deleteLevelSelection()) deleteInstance(m_gizmo.selectedInstance);
    if (ImGui::IsKeyChordPressed(ImGuiKey_F) && selection) focusOnInstance(m_gizmo.selectedInstance);
    if (ImGui::IsKeyChordPressed(ImGuiKey_Escape) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId)) deselectAll();
    if (ImGui::IsKeyChordPressed(ImGuiKey_Q)) m_tool = GizmoMode::None;
    if (ImGui::IsKeyChordPressed(ImGuiKey_1)) m_tool = GizmoMode::Translate;
    if (ImGui::IsKeyChordPressed(ImGuiKey_2)) m_tool = GizmoMode::Rotate;
    if (ImGui::IsKeyChordPressed(ImGuiKey_3)) m_tool = GizmoMode::Scale;
    if (ImGui::IsKeyChordPressed(ImGuiKey_F1)) m_openControlsPopup = true;
    if (ImGui::IsKeyChordPressed(ImGuiKey_F2) && selection) beginRename(m_gizmo.selectedInstance);
}

void Editor::updateWindowTitle()
{
    const std::string& path = m_scenes.currentPath();
    std::string title = "MirasEngine - " + (path.empty() ? std::string("Untitled") : path);
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

void Editor::validateSelection()
{
    if (!validInstance(m_gizmo.selectedInstance) && m_gizmo.selectedInstance != -1)
        m_gizmo.deselect();
}

void Editor::selectInstance(int index)
{
    m_gizmo.select(index);
}

void Editor::deselectAll()
{
    m_gizmo.deselect();
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
    // 60 degree vertical FOV: a sphere of radius r fits at distance r / sin(30deg) = 2r.
    m_camera.position = center - getFront(m_camera) * std::max(radius * 2.2f, 1.0f);
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
    setStatus("Deleted " + name);
}

void Editor::duplicateInstance(int index)
{
    if (!validInstance(index))
        return;
    const ModelInstance source = m_models.getInstances()[index];
    GPUModel* model = m_models.getModel(source.modelIndex);
    const float offset = model ? model->boundsRadius * std::max({ source.scale.x, source.scale.y, source.scale.z }) : 1.0f;
    // Level geometry is copied so the duplicate can be edited on its own; prefab instances keep sharing it.
    size_t modelIndex = source.modelIndex;
    if (model && model->polyMesh && model->prefabPath.empty()) {
        const auto copy = createLevelModel(*model->polyMesh, model->name);
        if (!copy)
            return;
        modelIndex = *copy;
    }
    const size_t newIndex = m_models.createInstance(modelIndex, source.position + glm::vec3(offset, 0.0f, 0.0f),
        source.rotation, source.scale);
    m_models.getInstances()[newIndex].color = source.color;
    m_models.getInstances()[newIndex].locked = source.locked;
    selectInstance(static_cast<int>(newIndex));
    setStatus("Duplicated " + source.name);
}

glm::vec3 Editor::instanceWorldCenter(int index) const
{
    const ModelInstance& instance = m_models.getInstances()[index];
    const auto& models = m_models.getModels();
    if (instance.modelIndex >= models.size() || !models[instance.modelIndex])
        return instance.position;
    return glm::vec3(instance.getTransformMatrix() * glm::vec4(models[instance.modelIndex]->boundsCenter, 1.0f));
}

std::string Editor::uniqueInstanceName(const std::string& base) const
{
    const auto& instances = m_models.getInstances();
    const auto taken = [&](const std::string& name) {
        return std::any_of(instances.begin(), instances.end(), [&](const ModelInstance& i) { return i.name == name; });
    };
    if (!taken(base))
        return base;
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
    const float step = m_snapTranslate > 0.0f ? m_snapTranslate : 1.0f;
    const glm::vec3 position = glm::round((m_camera.position + getFront(m_camera) * 5.0f) / step) * step;
    const size_t newIndex = m_models.createInstance(modelIndex, position);
    const std::string name = uniqueInstanceName("Cube");
    m_models.getInstances()[newIndex].name = name;
    selectInstance(static_cast<int>(newIndex));
    setStatus("Added " + name);
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
    setStatus("Unloaded " + name);
}

// ---------------------------------------------------------------------------------------------
// Scene actions
// ---------------------------------------------------------------------------------------------

void Editor::newScene()
{
    deselectAll();
    m_scenes.clear();
    clearLevelHistory();
    m_scenes.setCurrentPath({});
    setStatus("New scene");
}

void Editor::openScene(const std::string& path)
{
    const SceneManager::OpenResult opened = m_scenes.open(path);
    if (!opened.ok) {
        setStatus("Failed to open scene: " + path, true);
        return;
    }
    // The previous scene is gone, so the selection would point at a stale instance.
    deselectAll();
    clearLevelHistory();
    for (const auto& missing : opened.missingFiles)
        setStatus("Model file not found: " + missing, true);
    setStatus("Opening " + path + " (" + std::to_string(opened.queuedModels) + " models)...");
}

void Editor::saveSceneTo(const std::string& path)
{
    if (m_scenes.save(path))
        setStatus("Scene saved: " + path);
    else
        setStatus("Failed to save scene: " + path, true);
}

void Editor::saveScene()
{
    if (m_scenes.currentPath().empty())
        saveSceneAsDialog();
    else
        saveSceneTo(m_scenes.currentPath());
}
