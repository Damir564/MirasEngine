#include "Editor.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iterator>
#include "imgui_internal.h"
#include "EditorStyle.h"
#include "FileDialog.h"
#include "app/SettingsUi.h"
#include "engine/ModelManager.h"
#include "engine/SceneManager.h"

// ---------------------------------------------------------------------------------------------
// Menu bar
// ---------------------------------------------------------------------------------------------

void Editor::drawMainMenuBar()
{
    if (!ImGui::BeginMainMenuBar())
        return;
    drawFileMenu();
    drawEditMenu();
    drawAddMenu();
    drawViewMenu();
    if (ImGui::BeginMenu("Settings")) {
        ImGui::MenuItem("Preferences...", nullptr, &m_showPreferences);
        ImGui::MenuItem("Graphics...", nullptr, &m_showGraphicsSettings);
        ImGui::MenuItem("Keyboard Shortcuts...", m_keymap.shortcutLabel(EditorAction::ShowKeymap).c_str(), &m_showKeymap);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("Controls", m_keymap.shortcutLabel(EditorAction::ShowControls).c_str())) m_openControlsPopup = true;
        if (ImGui::MenuItem("About MirasEngine")) m_openAboutPopup = true;
        ImGui::EndMenu();
    }

    const std::string& path = m_scenes.currentPath();
    const bool dirty = sceneDirty();
    const std::string sceneLabel = (path.empty() ? std::string("Untitled scene") : path) + (dirty ? " *" : "");
    const float labelWidth = ImGui::CalcTextSize(sceneLabel.c_str()).x;
    ImGui::SameLine(ImGui::GetWindowWidth() - labelWidth - ImGui::GetStyle().WindowPadding.x * 2);
    ImGui::TextDisabled("%s", sceneLabel.c_str());
    if (dirty)
        ImGui::SetItemTooltip("Unsaved changes");
    ImGui::EndMainMenuBar();
}

void Editor::drawFileMenu()
{
    if (!ImGui::BeginMenu("File"))
        return;
    const auto key = [this](EditorAction action) { return m_keymap.shortcutLabel(action); };
    if (ImGui::MenuItem("New Scene", key(EditorAction::NewScene).c_str())) requestSceneAction(SceneAction::New);
    if (ImGui::MenuItem("Open Scene...", key(EditorAction::OpenScene).c_str())) requestSceneAction(SceneAction::OpenDialog);
    drawRecentScenesMenu();
    if (ImGui::MenuItem("Save Scene", key(EditorAction::SaveScene).c_str())) saveScene();
    if (ImGui::MenuItem("Save Scene As...", key(EditorAction::SaveSceneAs).c_str())) saveSceneAsDialog();
    ImGui::Separator();
    if (ImGui::MenuItem("Import Model...", key(EditorAction::ImportModel).c_str())) importModelDialog();
    if (ImGui::MenuItem("Run Script...")) runScriptDialog();
    ImGui::Separator();
    if (ImGui::MenuItem("Exit", "Alt+F4")) requestSceneAction(SceneAction::Quit);
    ImGui::EndMenu();
}

void Editor::drawEditMenu()
{
    if (!ImGui::BeginMenu("Edit"))
        return;
    const bool selection = hasSelection();
    const std::string undoLabel = m_undoStack.empty() ? "Undo" : "Undo " + m_undoStack.back().action;
    const std::string redoLabel = m_redoStack.empty() ? "Redo" : "Redo " + m_redoStack.back().action;
    const auto key = [this](EditorAction action) { return m_keymap.shortcutLabel(action); };
    if (ImGui::MenuItem(undoLabel.c_str(), key(EditorAction::Undo).c_str(), false, !m_undoStack.empty())) undo();
    if (ImGui::MenuItem(redoLabel.c_str(), key(EditorAction::Redo).c_str(), false, !m_redoStack.empty())) redo();
    ImGui::Separator();
    if (ImGui::MenuItem("Cut", key(EditorAction::Cut).c_str(), false, selection)) cutSelection();
    if (ImGui::MenuItem("Copy", key(EditorAction::Copy).c_str(), false, selection)) copySelection();
    if (ImGui::MenuItem("Paste", key(EditorAction::Paste).c_str(), false, !m_clipboard.empty())) pasteClipboard();
    if (ImGui::MenuItem("Duplicate", key(EditorAction::Duplicate).c_str(), false, selection)) duplicateSelection();
    if (ImGui::MenuItem("Delete", key(EditorAction::Delete).c_str(), false, selection)) deleteSelection();
    if (ImGui::MenuItem("Rename", key(EditorAction::Rename).c_str(), false, selection)) beginRename(m_gizmo.selectedInstance);
    ImGui::Separator();
    if (ImGui::MenuItem("Select All", key(EditorAction::SelectAll).c_str())) selectAll();
    if (ImGui::MenuItem("Deselect", key(EditorAction::Deselect).c_str(), false, selection)) deselectAll();
    if (ImGui::MenuItem("Focus Selected", key(EditorAction::Focus).c_str(), false, selection)) focusSelection();
    ImGui::Separator();
    if (ImGui::MenuItem("Drop to Floor", key(EditorAction::DropToFloor).c_str(), false, selection)) dropSelectionToFloor();
    if (ImGui::MenuItem("Rotate 90 Clockwise", key(EditorAction::RotateClockwise).c_str(), false, selection))
        rotateSelection(-90.0f);
    if (ImGui::MenuItem("Rotate 90 Counter-clockwise", key(EditorAction::RotateCounterClockwise).c_str(), false, selection))
        rotateSelection(90.0f);
    if (ImGui::MenuItem("Hide / Show Selected", key(EditorAction::Hide).c_str(), false, selection)) toggleSelectionVisibility();
    if (ImGui::MenuItem("Show All", key(EditorAction::UnhideAll).c_str())) unhideAll();
    ImGui::Separator();
    if (ImGui::MenuItem("Group", key(EditorAction::GroupObjects).c_str(), false, selectionCount() >= 2)) groupSelection();
    const bool grouped = std::any_of(m_selection.begin(), m_selection.end(), [this](uint64_t id) {
        const auto& instances = m_models.getInstances();
        return std::any_of(instances.begin(), instances.end(),
            [id](const ModelInstance& instance) { return instance.id == id && !instance.group.empty(); });
    });
    if (ImGui::MenuItem("Ungroup", key(EditorAction::UngroupObjects).c_str(), false, grouped)) ungroupSelection();
    ImGui::Separator();
    const size_t shapes = selectedLevelShapes().size();
    if (ImGui::MenuItem("Unite Shapes", key(EditorAction::UniteShapes).c_str(), false, shapes >= 2)) uniteSelectedShapes();
    if (ImGui::MenuItem("Merge Shapes into One Solid", nullptr, false, shapes >= 2)) uniteSelectedShapes(true);
    if (ImGui::MenuItem("Separate Shape", key(EditorAction::SeparateShape).c_str(), false, shapes == 1)) separateSelectedShape();
    if (ImGui::MenuItem("Unite and Save as Prefab...", nullptr, false, shapes >= 2)) uniteAndSavePrefab();
    if (ImGui::MenuItem("Save Selection as Prefab...", key(EditorAction::SaveAsPrefab).c_str(), false, selection))
        saveSelectionAsPrefab();
    ImGui::SetItemTooltip("One level shape: its geometry. Anything else (models, entities, several objects): "
        "the objects, placed again as a group");
    ImGui::Separator();
    if (ImGui::MenuItem("Select Tool", key(EditorAction::SelectTool).c_str(), m_tool == GizmoMode::None)) m_tool = GizmoMode::None;
    if (ImGui::MenuItem("Move Tool", key(EditorAction::MoveTool).c_str(), m_tool == GizmoMode::Translate)) m_tool = GizmoMode::Translate;
    if (ImGui::MenuItem("Rotate Tool", key(EditorAction::RotateTool).c_str(), m_tool == GizmoMode::Rotate)) m_tool = GizmoMode::Rotate;
    if (ImGui::MenuItem("Scale Tool", key(EditorAction::ScaleTool).c_str(), m_tool == GizmoMode::Scale)) m_tool = GizmoMode::Scale;
    if (ImGui::MenuItem("Grid Snapping", key(EditorAction::ToggleSnap).c_str(), m_gridSnap)) m_gridSnap = !m_gridSnap;
    ImGui::EndMenu();
}

void Editor::drawAddMenu()
{
    if (!ImGui::BeginMenu("Add"))
        return;
    if (ImGui::MenuItem("Cube")) addCube();
    ImGui::SeparatorText("Level shapes");
    drawShapeMenuItems();
    ImGui::SeparatorText("Game entities");
    drawEntityMenuItems();
    ImGui::SeparatorText("Prefabs");
    drawPrefabMenuItems();
    ImGui::EndMenu();
}

void Editor::drawViewMenu()
{
    if (!ImGui::BeginMenu("View"))
        return;
    ImGui::MenuItem("Hierarchy", nullptr, &m_showHierarchy);
    ImGui::MenuItem("Inspector", nullptr, &m_showInspector);
    ImGui::MenuItem("Camera Animation", nullptr, &m_showAnimationPanel);
    ImGui::MenuItem("Statistics", nullptr, &m_showStatisticsPanel);
    ImGui::MenuItem("Level", nullptr, &m_showLevelPanel);
    ImGui::MenuItem("Materials", nullptr, &m_showMaterialsPanel);
    ImGui::Separator();
    ImGui::MenuItem("Show Camera Path", nullptr, &m_showCameraPath);
    ImGui::MenuItem("Grid", m_keymap.shortcutLabel(EditorAction::ToggleGrid).c_str(), &m_showGrid);
    ImGui::Separator();
    if (ImGui::MenuItem("Fly Mode (hide UI)", m_keymap.shortcutLabel(EditorAction::ToggleFlyMode).c_str())) setFlyMode(true);
    if (ImGui::BeginMenu("Layout")) {
        drawLayoutMenuItems();
        ImGui::EndMenu();
    }
    ImGui::EndMenu();
}

// ---------------------------------------------------------------------------------------------
// Toolbar and status bar
// ---------------------------------------------------------------------------------------------

namespace {
constexpr ImGuiWindowFlags kBarFlags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;
}

std::string Editor::withShortcut(const char* text, EditorAction action) const
{
    const std::string shortcut = m_keymap.shortcutLabel(action);
    return shortcut.empty() ? std::string(text) : std::string(text) + " (" + shortcut + ")";
}

void Editor::drawToolButton(const char* label, GizmoMode tool, const char* tooltip, EditorAction shortcut)
{
    const bool active = m_tool == tool;
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button, EditorStyle::kAccent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, EditorStyle::kAccentHover);
    }
    if (ImGui::Button(label, ImVec2(px(64.0f), 0))) m_tool = tool;
    if (active) ImGui::PopStyleColor(2);
    ImGui::SetItemTooltip("%s", withShortcut(tooltip, shortcut).c_str());
    ImGui::SameLine(0, 2);
}

void Editor::drawToolbar()
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 5));
    if (ImGui::BeginViewportSideBar("##Toolbar", ImGui::GetMainViewport(), ImGuiDir_Up, ImGui::GetFrameHeight() + 10.0f, kBarFlags)) {
        drawToolButton("Select", GizmoMode::None, "Select objects without a transform gizmo", EditorAction::SelectTool);
        drawToolButton("Move", GizmoMode::Translate, "Move the selected object", EditorAction::MoveTool);
        drawToolButton("Rotate", GizmoMode::Rotate, "Rotate the selected object", EditorAction::RotateTool);
        drawToolButton("Scale", GizmoMode::Scale, "Scale the selected object", EditorAction::ScaleTool);

        ImGui::SameLine(0, 16);
        ImGui::BeginDisabled(!hasSelection());
        if (ImGui::Button("Focus")) focusSelection();
        ImGui::SetItemTooltip("%s", withShortcut("Move the camera to the selection", EditorAction::Focus).c_str());
        ImGui::EndDisabled();

        ImGui::SameLine(0, 16);
        drawGridControls();

        ImGui::SameLine(0, 16);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Camera speed");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(px(140.0f));
        ImGui::SliderFloat("##cameraSpeed", &m_camera.speed, 0.5f, 200.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Keyboard movement speed (hold %s for 4x)",
            m_keymap.shortcutLabel(EditorAction::CameraFast).c_str());

        ImGui::SameLine(0, 16);
        ImGui::Checkbox("Grid", &m_showGrid);
        ImGui::SetItemTooltip("Show the ground grid (View > Grid)");

        ImGui::SameLine(0, 16);
        ImGui::Checkbox("Camera path", &m_showCameraPath);

        const ImGuiStyle& style = ImGui::GetStyle();
        const float flyWidth = ImGui::CalcTextSize("Fly Mode").x + style.FramePadding.x * 2;
        const float playWidth = px(80.0f);
        ImGui::SameLine(ImGui::GetWindowWidth() - flyWidth - playWidth - style.ItemSpacing.x - 8);
        ImGui::BeginDisabled(!canPlay());
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.45f, 0.22f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.56f, 0.28f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.13f, 0.38f, 0.18f, 1.0f));
        if (ImGui::Button("Play", ImVec2(playWidth, 0))) requestPlay();
        ImGui::PopStyleColor(3);
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("%s. While playing, Esc pauses and F5 stops.",
            withShortcut("Play this scene with the player controller", EditorAction::Play).c_str());
        ImGui::SameLine();
        if (ImGui::Button("Fly Mode")) setFlyMode(true);
        const std::string flyKey = m_keymap.shortcutLabel(EditorAction::ToggleFlyMode);
        ImGui::SetItemTooltip("Hide the UI and look around with the mouse (%s%sEsc to exit)", flyKey.c_str(),
            flyKey.empty() ? "" : " or ");
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void Editor::drawGridControls()
{
    const bool shapeGrid = levelGridActive();
    const float current = activeGridSize();
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled(shapeGrid ? "Shape grid" : "Grid");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(px(70.0f));
    char preview[16];
    snprintf(preview, sizeof(preview), "%g", current);
    if (ImGui::BeginCombo("##gridSize", preview)) {
        for (const float size : kGridSizes) {
            char label[16];
            snprintf(label, sizeof(label), "%g", size);
            if (ImGui::Selectable(label, size == current))
                setActiveGridSize(size);
        }
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip(shapeGrid
        ? "Grid of the selected shape ([ smaller, ] larger); saved with the shape and moves with it"
        : "World grid cell size ([ smaller, ] larger); new level shapes start with this grid");
    ImGui::SameLine();
    ImGui::Checkbox("Snap", &m_gridSnap);
    ImGui::SetItemTooltip("Snap moves, vertices and new shapes to the grid; hold Ctrl while dragging to invert");
    ImGui::SameLine();
    if (ImGui::Button("Steps")) ImGui::OpenPopup("SnapSettings");
    ImGui::SetItemTooltip("Rotate and scale snap increments");
    drawSnapPopup();
}

void Editor::drawSnapPopup()
{
    if (!ImGui::BeginPopup("SnapSettings"))
        return;
    ImGui::TextDisabled("Used by the rotate and scale gizmos while snapping");
    ImGui::PushItemWidth(120);
    if (ImGui::DragFloat("Rotate (degrees)", &m_snapRotate, 0.5f, 0.1f, 180.0f, "%.1f"))
        m_snapRotate = std::max(m_snapRotate, 0.1f);
    if (ImGui::DragFloat("Scale", &m_snapScale, 0.01f, 0.01f, 10.0f, "%.2f"))
        m_snapScale = std::max(m_snapScale, 0.01f);
    ImGui::PopItemWidth();
    if (ImGui::Button("Reset")) {
        m_snapRotate = 15.0f;
        m_snapScale = 0.1f;
    }
    ImGui::EndPopup();
}

void Editor::drawStatusBar()
{
    if (ImGui::BeginViewportSideBar("##StatusBar", ImGui::GetMainViewport(), ImGuiDir_Down, ImGui::GetFrameHeight(),
            kBarFlags | ImGuiWindowFlags_MenuBar) &&
        ImGui::BeginMenuBar()) {
        const auto& tasks = m_models.getLoadingTasks();
        if (!tasks.empty()) {
            static constexpr char kSpinner[] = "|/-\\";
            const char spin = kSpinner[static_cast<int>(ImGui::GetTime() * 8.0) % 4];
            const std::string more = tasks.size() > 1 ? " (+" + std::to_string(tasks.size() - 1) + " more)" : "";
            ImGui::TextColored(EditorStyle::kHighlight, "%c  Loading %s%s", spin, tasks.front().name.c_str(), more.c_str());
        }
        else {
            const bool recent = ImGui::GetTime() - m_statusTime < 5.0;
            const ImVec4 color = m_statusIsError ? EditorStyle::kError
                : (recent ? ImGui::GetStyleColorVec4(ImGuiCol_Text) : EditorStyle::kTextDim);
            ImGui::TextColored(color, "%s", m_statusMessage.c_str());
        }

        const auto& instances = m_models.getInstances();
        size_t triangles = 0;
        for (const auto& instance : instances)
            if (instance.visible)
                if (GPUModel* model = m_models.getModel(instance.modelIndex)) triangles += model->indexCount / 3;
        const ImGuiIO& io = ImGui::GetIO();
        char selected[32] = "";
        if (selectionCount() > 1)
            snprintf(selected, sizeof(selected), " (%zu selected)", selectionCount());
        char right[256];
        snprintf(right, sizeof(right), "Grid: %g%s   |   Objects: %zu%s   Models: %zu   Triangles: %s   |   %.0f FPS (%.2f ms)",
            activeGridSize(), m_gridSnap ? "" : " (snap off)", instances.size(), selected, m_models.getModels().size(),
            formatCount(triangles).c_str(), io.Framerate, 1000.0f / std::max(io.Framerate, 0.001f));
        const float rightWidth = ImGui::CalcTextSize(right).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - rightWidth - 12);
        ImGui::TextDisabled("%s", right);
        ImGui::EndMenuBar();
    }
    ImGui::End();
}

// ---------------------------------------------------------------------------------------------
// Dock space and viewport
// ---------------------------------------------------------------------------------------------

void Editor::drawDockSpace()
{
    ImGuiViewport* mainViewport = ImGui::GetMainViewport();
    const ImGuiID dockspaceId = ImGui::GetID("EditorDockSpace");
    if (m_layoutRequest || ImGui::DockBuilderGetNode(dockspaceId) == nullptr) {
        buildLayout(dockspaceId, m_layoutRequest.value_or(LayoutPreset::Default));
        m_layoutRequest.reset();
    }
    ImGui::DockSpaceOverViewport(dockspaceId, mainViewport, ImGuiDockNodeFlags_PassthruCentralNode);

    // The 3D scene is rendered into the central (empty) dock area.
    if (ImGuiDockNode* centralNode = ImGui::DockBuilderGetCentralNode(dockspaceId)) {
        m_sceneView = { centralNode->Pos.x, centralNode->Pos.y,
            std::max(centralNode->Size.x, 1.0f), std::max(centralNode->Size.y, 1.0f) };
    }
    else {
        m_sceneView = { mainViewport->WorkPos.x, mainViewport->WorkPos.y,
            std::max(mainViewport->WorkSize.x, 1.0f), std::max(mainViewport->WorkSize.y, 1.0f) };
    }
}

void Editor::drawViewportOverlay()
{
    if (!m_prefs.showViewportHelp)
        return;
    ImGui::SetNextWindowPos(ImVec2(m_sceneView.x + 10, m_sceneView.y + 10));
    ImGui::SetNextWindowBgAlpha(0.55f);
    const ImGuiWindowFlags overlayFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoMouseInputs;
    if (ImGui::Begin("##ViewportOverlay", nullptr, overlayFlags)) {
        const char* toolName = m_tool == GizmoMode::None ? "Select"
            : m_tool == GizmoMode::Translate ? "Move"
            : m_tool == GizmoMode::Rotate ? "Rotate" : "Scale";
        ImGui::Text("Perspective  |  %s tool", toolName);
        // "WASD", or "Up/Left/Down/Right" once a binding is longer than one character.
        const std::string keys[] = { m_keymap.shortcutLabel(EditorAction::CameraForward),
            m_keymap.shortcutLabel(EditorAction::CameraLeft), m_keymap.shortcutLabel(EditorAction::CameraBack),
            m_keymap.shortcutLabel(EditorAction::CameraRight) };
        const bool single = std::all_of(std::begin(keys), std::end(keys), [](const std::string& k) { return k.size() <= 1; });
        std::string moveKeys;
        for (const std::string& key : keys)
            moveKeys += (moveKeys.empty() || single ? "" : "/") + key;
        ImGui::TextDisabled("LMB select   RMB+drag look   %s move   %s focus", moveKeys.c_str(),
            m_keymap.shortcutLabel(EditorAction::Focus).c_str());
        ImGui::TextDisabled("%s grid %g   snap %s, Ctrl inverts", levelGridActive() ? "Shape" : "World",
            activeGridSize(), m_gridSnap ? "on" : "off");
    }
    ImGui::End();
}

// ---------------------------------------------------------------------------------------------
// Help popups
// ---------------------------------------------------------------------------------------------

void Editor::drawHelpPopups()
{
    if (m_openControlsPopup) ImGui::OpenPopup("Controls");
    if (m_openAboutPopup) ImGui::OpenPopup("About MirasEngine");
    m_openControlsPopup = false;
    m_openAboutPopup = false;
    drawControlsPopup();
    drawAboutPopup();
    drawUnsavedChangesPopup();
}

void Editor::drawControlsPopup()
{
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (!ImGui::BeginPopupModal("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        return;
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("##controls", 2, flags, ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 24.0f))) {
        ImGui::TableSetupColumn("Input", ImGuiTableColumnFlags_WidthFixed, px(190.0f));
        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, px(330.0f));
        const auto row = [](const char* input, const char* action) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(EditorStyle::kHighlight, "%s", input);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(action);
        };
        const auto heading = [](const char* text) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", text);
        };
        heading("Mouse");
        static constexpr const char* kMouseRows[][2] = {
            { "Left click", "Select object" },
            { "Shift / Ctrl+click", "Add to / toggle in the selection (also in the Hierarchy)" },
            { "Drag on empty space", "Box-select objects (vertices in vertex mode)" },
            { "Double-click a united shape", "Part mode: pick and move its parts (Esc leaves)" },
            { "Right mouse + drag", "Look around" },
            { "Drag face (face mode)", "Push/pull along its normal; Alt+drag extrudes" },
            { "Shift+click face", "Add/remove a face from the selection (face mode)" },
            { "Alt+Shift+click face", "Wrap the active face's texture onto a neighbour" },
            { "Drag a material tile", "Drop on a face to apply; Shift+drop: whole object" },
            { "Double-click a material", "Apply to the selected face or object" },
            { "Ctrl (while dragging)", "Invert grid snapping for moves, vertices, rotate and scale" },
            { "Click view gizmo axis", "Look along that axis (top-right of viewport)" },
        };
        for (const auto& mouseRow : kMouseRows)
            row(mouseRow[0], mouseRow[1]);

        // Keyboard rows come from the keymap, so they show the user's own bindings.
        const char* category = nullptr;
        for (int a = 0; a < EditorKeymap::kActionCount; ++a) {
            const EditorAction action = static_cast<EditorAction>(a);
            const std::string keys = m_keymap.allShortcutsLabel(action);
            if (keys.empty())
                continue;
            const EditorActionInfo& info = EditorKeymap::info(action);
            if (!category || std::strcmp(category, info.category) != 0) {
                category = info.category;
                heading(category);
            }
            row(keys.c_str(), info.label);
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    if (ImGui::Button("Close", ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape))
        ImGui::CloseCurrentPopup();
    ImGui::SameLine();
    if (ImGui::Button("Customize...", ImVec2(120, 0))) {
        m_showKeymap = true;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void Editor::drawAboutPopup()
{
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (!ImGui::BeginPopupModal("About MirasEngine", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        return;
    ImGui::Text("MirasEngine");
    ImGui::TextDisabled("Vulkan 1.3 renderer and glTF viewer");
    ImGui::Separator();
    ImGui::Text("Dear ImGui %s", ImGui::GetVersion());
    ImGui::Spacing();
    if (ImGui::Button("Close", ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape))
        ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

// ---------------------------------------------------------------------------------------------
// File dialogs
// ---------------------------------------------------------------------------------------------

void Editor::openFileDialog(const char* key, const char* title, const char* filters, const std::string& root,
    const char* defaultFileName, bool confirmOverwrite)
{
    namespace fs = std::filesystem;
    fs::create_directories(root);
    IGFD::FileDialogConfig config;
    config.path = fs::weakly_canonical(fs::absolute(root)).string();
    config.countSelectionMax = 1;
    config.flags = ImGuiFileDialogFlags_Modal;
    if (confirmOverwrite) config.flags |= ImGuiFileDialogFlags_ConfirmOverwrite;
    if (defaultFileName) config.fileName = defaultFileName;
    ImGuiFileDialog::Instance()->OpenDialog(key, title, filters, config);
}

void Editor::openSceneDialog()
{
    openFileDialog("BrowseSceneDlg", "Open Scene", ".scn", kScenesRoot, nullptr, false);
}

void Editor::saveSceneAsDialog()
{
    openFileDialog("SaveAsSceneDlg", "Save Scene As", ".scn", kScenesRoot, "untitled.scn", true);
}

void Editor::importModelDialog()
{
    openFileDialog("BrowseModelDlg", "Import 3D Model",
        "3D Models{.gltf,.glb},.gltf,.glb", kModelsRoot, nullptr, false);
}

void Editor::runScriptDialog()
{
    openFileDialog("RunScriptDlg", "Run Script", "Scripts{.mscript,.txt},.mscript,.txt", "scripts", nullptr, false);
}

void Editor::drawFileDialogs()
{
    ImGuiFileDialog* dialog = ImGuiFileDialog::Instance();
    if (dialog->Display("RunScriptDlg", ImGuiWindowFlags_NoCollapse, kDialogSize)) {
        if (dialog->IsOk())
            runScript(toStoredPath(dialog->GetFilePathName()));
        dialog->Close();
    }
    if (dialog->Display("BrowseSceneDlg", ImGuiWindowFlags_NoCollapse, kDialogSize)) {
        if (dialog->IsOk()) {
            openScene(toStoredPath(dialog->GetFilePathName()));
        }
        dialog->Close();
    }
    if (dialog->Display("SaveAsSceneDlg", ImGuiWindowFlags_NoCollapse, kDialogSize)) {
        if (dialog->IsOk()) {
            std::filesystem::path path(dialog->GetFilePathName());
            if (!path.has_extension() || path.extension() != ".scn") path.replace_extension(".scn");
            m_scenes.setCurrentPath(toStoredPath(path.string()));
            saveSceneTo(m_scenes.currentPath());
        }
        dialog->Close();
        finishPendingSceneAction();
    }
    if (dialog->Display("BrowseModelDlg", ImGuiWindowFlags_NoCollapse, kDialogSize)) {
        if (dialog->IsOk()) {
            const std::string path = toStoredPath(dialog->GetFilePathName());
            const std::string name = std::filesystem::path(path).stem().string();
            m_models.loadModelAsync(path, name);
            setStatus("Importing " + name + "...");
        }
        dialog->Close();
    }
    drawPrefabDialogs();
    // Also used by the Materials panel, so it is not tied to the Level panel being open.
    drawLevelTextureDialog();
}

// ---------------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------------

void Editor::drawGraphicsSettingsWindow()
{
    ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Graphics Settings", &m_showGraphicsSettings))
        drawGraphicsSettings(m_settings, m_renderer.capabilities());
    ImGui::End();
}
