#include "Editor.h"
#include <cfloat>
#include <unordered_set>
#include "EditorStyle.h"
#include "engine/ModelManager.h"

void Editor::drawHierarchy()
{
    if (ImGui::Begin("Hierarchy", &m_showHierarchy)) {
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputTextWithHint("##hierarchySearch", "Search objects...",
                m_hierarchyFilter.InputBuf, IM_ARRAYSIZE(m_hierarchyFilter.InputBuf)))
            m_hierarchyFilter.Build();
        if (ImGui::CollapsingHeader("Models", ImGuiTreeNodeFlags_DefaultOpen))
            drawModelList();
        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
            drawSceneTree();

        if (ImGui::BeginPopupContextWindow("HierarchyEmptyMenu",
                ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
            if (ImGui::BeginMenu("Create")) {
                if (ImGui::MenuItem("Cube")) addCube();
                ImGui::SeparatorText("Level shapes");
                drawShapeMenuItems();
                ImGui::SeparatorText("Game entities");
                drawEntityMenuItems();
                ImGui::SeparatorText("Prefabs");
                drawPrefabMenuItems();
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("Import Model...")) importModelDialog();
            ImGui::EndPopup();
        }
    }
    ImGui::End();
}

void Editor::drawModelList()
{
    const auto& models = m_models.getModels();
    const auto& tasks = m_models.getLoadingTasks();
    int deferredUnload = -1;
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& model = models[i];
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Selectable(model->name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick) &&
            ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
            addModelToScene(i, false);
        if (ImGui::BeginItemTooltip()) {
            ImGui::TextUnformatted(model->sourcePath.c_str());
            ImGui::TextDisabled("%s vertices, %s triangles, %zu textures",
                formatCount(model->vertexCount).c_str(), formatCount(model->indexCount / 3).c_str(),
                model->textures.size());
            ImGui::TextDisabled("Double-click to add to the scene");
            ImGui::EndTooltip();
        }
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Add to Scene")) addModelToScene(i, false);
            if (ImGui::MenuItem("Add at Origin")) addModelToScene(i, true);
            ImGui::Separator();
            if (ImGui::MenuItem("Unload")) deferredUnload = static_cast<int>(i);
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }
    for (const auto& task : tasks) {
        const char* state = task.state == LoadingState::LoadingCPU ? "parsing"
            : task.state == LoadingState::UploadingGPU ? "uploading"
            : task.state == LoadingState::Failed ? "failed" : "queued";
        ImGui::TextDisabled("%s  (%s...)", task.name.c_str(), state);
    }
    if (models.empty() && tasks.empty())
        ImGui::TextDisabled("No models loaded.");
    if (ImGui::Button("Import Model...", ImVec2(-FLT_MIN, 0))) importModelDialog();
    if (deferredUnload >= 0) unloadModel(static_cast<size_t>(deferredUnload));
}

void Editor::drawSceneTree()
{
    auto& instances = m_models.getInstances();
    // Compact rows: small visibility checkboxes and tight spacing, like editor hierarchies.
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 3));
    if (instances.empty())
        ImGui::TextDisabled("The scene is empty.\nDouble-click a model above to add it.");

    if (!validInstance(m_renamingInstance))
        m_renamingInstance = -1;

    HierarchyActions actions;
    // A group is listed where its first object is, with its objects under it.
    std::unordered_set<std::string> listedGroups;
    for (int i = 0; i < static_cast<int>(instances.size()); ++i) {
        const std::string group = instances[i].group;
        if (group.empty())
            drawInstanceNode(i, actions);
        else if (listedGroups.insert(group).second)
            drawGroupNode(group, actions);
    }

    if (actions.focusIndex >= 0) {
        if (m_gizmo.selectedInstance != actions.focusIndex) selectInstance(actions.focusIndex);
        focusOnInstance(actions.focusIndex);
    }
    // Menu actions on a selected row act on the whole selection.
    if (actions.duplicateIndex >= 0) {
        if (isSelected(actions.duplicateIndex)) duplicateSelection();
        else duplicateInstance(actions.duplicateIndex);
    }
    if (actions.createCube) addCube();
    if (actions.deleteIndex >= 0) {
        if (isSelected(actions.deleteIndex)) deleteSelection();
        else deleteInstance(actions.deleteIndex);
    }
    if (actions.unite) uniteSelectedShapes();
    if (actions.uniteAndSave) uniteAndSavePrefab();
    if (actions.separateIndex >= 0) {
        selectInstance(actions.separateIndex);
        separateSelectedShape();
    }
    ImGui::PopStyleVar(2);
}

void Editor::drawInstanceNode(int instanceIndex, HierarchyActions& actions)
{
    auto& instance = m_models.getInstances()[instanceIndex];
    if (m_hierarchyFilter.IsActive() && !m_hierarchyFilter.PassFilter(instance.name.c_str()))
        return;

    ImGui::PushID(instanceIndex);
    if (ImGui::Checkbox("##visible", &instance.visible))
        markSceneChanged();
    ImGui::SetItemTooltip(instance.visible ? "Hide object" : "Show object");
    ImGui::SameLine();

    if (m_renamingInstance == instanceIndex) {
        drawRenameField(instanceIndex);
        ImGui::PopID();
        return;
    }

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
        ImGuiTreeNodeFlags_SpanAvailWidth;
    if (isSelected(instanceIndex))
        flags |= ImGuiTreeNodeFlags_Selected;
    std::string tag = !prefabModel(instanceIndex) ? "" : instance.locked ? "  [prefab, locked]" : "  [prefab]";
    if (!instance.entity.empty())
        tag += "  [" + instance.entity + "]";
    const char* prefabTag = tag.c_str();
    const bool active = m_gizmo.selectedInstance == instanceIndex && selectionCount() > 1;
    if (active)
        ImGui::PushStyleColor(ImGuiCol_Text, EditorStyle::kHighlight);
    ImGui::TreeNodeEx("##instance", flags, "%s%s", instance.name.c_str(), prefabTag);
    if (active)
        ImGui::PopStyleColor();
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        const ImGuiIO& io = ImGui::GetIO();
        if (io.KeyCtrl)
            toggleSelection(instanceIndex);
        else if (io.KeyShift)
            selectRange(instanceIndex);
        else
            selectInstance(instanceIndex);
    }
    // Right-clicking outside the selection selects the row, so the menu acts on what is shown selected.
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && !isSelected(instanceIndex))
        selectInstance(instanceIndex);
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        actions.focusIndex = instanceIndex;
    drawInstanceContextMenu(instanceIndex, actions);
    ImGui::PopID();
}

// Inline rename: Enter or clicking elsewhere commits, Esc cancels. Returns true when editing ended.
bool Editor::drawRenameField(int instanceIndex)
{
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (m_renameFocusPending) {
        ImGui::SetKeyboardFocusHere();
        m_renameFocusPending = false;
    }
    const bool entered = ImGui::InputText("##rename", m_renameBuffer, sizeof(m_renameBuffer),
        ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
    bool commit = entered;
    bool finished = entered;
    if (!entered && ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        finished = true;
    }
    else if (!entered && ImGui::IsItemDeactivated()) {
        commit = finished = true;
    }
    else if (!ImGui::IsItemActive() && !m_renameFocusPending && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
             !ImGui::IsItemHovered()) {
        // Focus never arrived (e.g. the row was scrolled away); treat an outside click as commit.
        commit = finished = true;
    }
    if (commit) {
        std::string name = m_renameBuffer;
        const size_t first = name.find_first_not_of(" \t");
        const size_t last = name.find_last_not_of(" \t");
        name = first == std::string::npos ? std::string() : name.substr(first, last - first + 1);
        if (!name.empty() && validInstance(instanceIndex)) {
            m_models.getInstances()[instanceIndex].name = name;
            markSceneChanged();
            setStatus("Renamed to " + name);
        }
    }
    if (finished)
        m_renamingInstance = -1;
    return finished;
}

void Editor::drawInstanceContextMenu(int instanceIndex, HierarchyActions& actions)
{
    if (!ImGui::BeginPopupContextItem("InstanceContextMenu"))
        return;
    const auto shortcut = [this](EditorAction action) { return m_keymap.shortcutLabel(action); };
    if (ImGui::MenuItem("Focus", shortcut(EditorAction::Focus).c_str())) actions.focusIndex = instanceIndex;
    if (ImGui::MenuItem("Rename", shortcut(EditorAction::Rename).c_str())) {
        selectInstance(instanceIndex);
        beginRename(instanceIndex);
    }
    const bool group = isSelected(instanceIndex) && selectionCount() > 1;
    const std::string count = group ? " " + std::to_string(selectionCount()) + " Objects" : "";
    if (ImGui::MenuItem(("Duplicate" + count).c_str(), shortcut(EditorAction::Duplicate).c_str()))
        actions.duplicateIndex = instanceIndex;
    if (ImGui::MenuItem(("Delete" + count).c_str(), shortcut(EditorAction::Delete).c_str()))
        actions.deleteIndex = instanceIndex;
    if (ImGui::MenuItem("Copy", shortcut(EditorAction::Copy).c_str())) copySelection();
    if (ImGui::MenuItem("Drop to Floor", shortcut(EditorAction::DropToFloor).c_str())) dropSelectionToFloor();
    ImGui::Separator();
    const GPUModel* model = m_models.getModel(m_models.getInstances()[instanceIndex].modelIndex);
    const size_t shapes = isSelected(instanceIndex) ? selectedLevelShapes().size() : 0;
    if (shapes >= 2) {
        const std::string unite = "Unite " + std::to_string(shapes) + " Shapes";
        if (ImGui::MenuItem(unite.c_str(), shortcut(EditorAction::UniteShapes).c_str())) actions.unite = true;
        if (ImGui::MenuItem("Unite and Save as Prefab...")) actions.uniteAndSave = true;
    }
    else {
        if (ImGui::MenuItem("Save as Prefab...", shortcut(EditorAction::SaveAsPrefab).c_str()))
            saveSelectionAsPrefab();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", selectionCount() <= 1 && model && model->polyMesh
                ? "Save the shape's geometry and materials (not its transform)"
                : "Save the selected objects (models, shapes, entities) to place again from Add > Prefabs");
        if (model && model->polyMesh && model->polyMesh->partCount() > 1 &&
            ImGui::MenuItem("Separate Shape", shortcut(EditorAction::SeparateShape).c_str()))
            actions.separateIndex = instanceIndex;
    }
    if (isSelected(instanceIndex) && selectionCount() > 1 &&
        ImGui::MenuItem("Group Selection", shortcut(EditorAction::GroupObjects).c_str()))
        groupSelection();
    if (!m_models.getInstances()[instanceIndex].group.empty() &&
        ImGui::MenuItem("Ungroup", shortcut(EditorAction::UngroupObjects).c_str()))
        ungroupSelection();
    if (prefabModel(instanceIndex)) {
        const bool locked = m_models.getInstances()[instanceIndex].locked;
        if (ImGui::MenuItem(locked ? "Unlock Prefab" : "Lock Prefab"))
            setInstanceLocked(instanceIndex, !locked);
    }
    ImGui::Separator();
    if (ImGui::BeginMenu("Create")) {
        if (ImGui::MenuItem("Cube")) actions.createCube = true;
        ImGui::EndMenu();
    }
    ImGui::EndPopup();
}
