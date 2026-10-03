#include "Editor.h"
#include <cfloat>
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
    for (int i = 0; i < static_cast<int>(instances.size()); ++i)
        drawInstanceNode(i, actions);

    if (actions.focusIndex >= 0) {
        if (m_gizmo.selectedInstance != actions.focusIndex) selectInstance(actions.focusIndex);
        focusOnInstance(actions.focusIndex);
    }
    if (actions.duplicateIndex >= 0) duplicateInstance(actions.duplicateIndex);
    if (actions.createCube) addCube();
    if (actions.deleteIndex >= 0) deleteInstance(actions.deleteIndex);
    ImGui::PopStyleVar(2);
}

void Editor::drawInstanceNode(int instanceIndex, HierarchyActions& actions)
{
    auto& instance = m_models.getInstances()[instanceIndex];
    if (m_hierarchyFilter.IsActive() && !m_hierarchyFilter.PassFilter(instance.name.c_str()))
        return;

    ImGui::PushID(instanceIndex);
    ImGui::Checkbox("##visible", &instance.visible);
    ImGui::SetItemTooltip(instance.visible ? "Hide object" : "Show object");
    ImGui::SameLine();

    if (m_renamingInstance == instanceIndex) {
        drawRenameField(instanceIndex);
        ImGui::PopID();
        return;
    }

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
        ImGuiTreeNodeFlags_SpanAvailWidth;
    if (m_gizmo.selectedInstance == instanceIndex)
        flags |= ImGuiTreeNodeFlags_Selected;
    const char* prefabTag = !prefabModel(instanceIndex) ? "" : instance.locked ? "  [prefab, locked]" : "  [prefab]";
    ImGui::TreeNodeEx("##instance", flags, "%s%s", instance.name.c_str(), prefabTag);
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
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
    if (ImGui::MenuItem("Focus", "F")) actions.focusIndex = instanceIndex;
    if (ImGui::MenuItem("Rename", "F2")) {
        selectInstance(instanceIndex);
        beginRename(instanceIndex);
    }
    if (ImGui::MenuItem("Duplicate", "Ctrl+D")) actions.duplicateIndex = instanceIndex;
    if (ImGui::MenuItem("Delete", "Del")) actions.deleteIndex = instanceIndex;
    ImGui::Separator();
    const GPUModel* model = m_models.getModel(m_models.getInstances()[instanceIndex].modelIndex);
    if (ImGui::MenuItem("Save as Prefab...", nullptr, false, model && model->polyMesh))
        savePrefabDialog(instanceIndex);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", model && model->polyMesh ? "Save the shape's geometry and materials (not its transform)"
                                                         : "Only level shapes can be saved as prefabs");
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
