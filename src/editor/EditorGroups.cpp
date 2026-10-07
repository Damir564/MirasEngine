#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <unordered_set>
#include <SDL3/SDL_keycode.h>
#include "EditorStyle.h"
#include "FileDialog.h"
#include "engine/ModelManager.h"
#include "engine/Prefab.h"

namespace {

std::string trimmed(const std::string& text)
{
    const size_t first = text.find_first_not_of(" \t");
    const size_t last = text.find_last_not_of(" \t");
    return first == std::string::npos ? std::string() : text.substr(first, last - first + 1);
}

} // namespace

std::vector<int> Editor::groupMembers(const std::string& group) const
{
    std::vector<int> members;
    if (group.empty())
        return members;
    const auto& instances = m_models.getInstances();
    for (int i = 0; i < static_cast<int>(instances.size()); ++i)
        if (instances[i].group == group)
            members.push_back(i);
    return members;
}

std::string Editor::uniqueGroupName(const std::string& name) const
{
    const auto taken = [this](const std::string& candidate) {
        const auto& instances = m_models.getInstances();
        return std::any_of(instances.begin(), instances.end(),
            [&](const ModelInstance& instance) { return instance.group == candidate; });
    };
    const std::string base = name.empty() ? std::string("Group") : name;
    if (!taken(base))
        return base;
    for (int n = 2;; ++n) {
        const std::string candidate = base + " (" + std::to_string(n) + ")";
        if (!taken(candidate))
            return candidate;
    }
}

std::string Editor::nextGroupName() const
{
    const auto& instances = m_models.getInstances();
    for (int n = 1;; ++n) {
        const std::string candidate = "Group " + std::to_string(n);
        if (std::none_of(instances.begin(), instances.end(),
                [&](const ModelInstance& instance) { return instance.group == candidate; }))
            return candidate;
    }
}

void Editor::selectWithGroup(int index, SDL_Keymod mods)
{
    if (!validInstance(index))
        return;
    const std::vector<int> members = groupMembers(m_models.getInstances()[index].group);
    if (members.size() <= 1) {
        clickSelect(index, mods);
        return;
    }
    if (mods & SDL_KMOD_CTRL) {
        // Toggles the whole group, by the clicked object's state.
        const bool remove = isSelected(index);
        for (int member : members)
            if (isSelected(member) == remove)
                toggleSelection(member);
        if (!remove)
            setActive(index);
        return;
    }
    if (mods & SDL_KMOD_SHIFT) {
        for (int member : members)
            addToSelection(member);
        setActive(index);
        return;
    }
    selectIndices(members, index);
}

bool Editor::wholeGroupSelected(int index) const
{
    if (!validInstance(index))
        return false;
    const std::vector<int> members = groupMembers(m_models.getInstances()[index].group);
    return members.size() > 1 && std::all_of(members.begin(), members.end(), [this](int m) { return isSelected(m); });
}

void Editor::expandSelectionToGroups()
{
    const auto& instances = m_models.getInstances();
    std::unordered_set<std::string> groups;
    for (int index : selectedIndices())
        if (!instances[index].group.empty())
            groups.insert(instances[index].group);
    if (groups.empty())
        return;
    const int active = m_gizmo.selectedInstance;
    std::vector<int> indices;
    for (int i = 0; i < static_cast<int>(instances.size()); ++i)
        if (isSelected(i) || groups.contains(instances[i].group))
            indices.push_back(i);
    selectIndices(indices, active);
}

void Editor::groupSelection()
{
    const std::vector<int> indices = selectedIndices();
    if (indices.size() < 2) {
        setStatus("Select two or more objects to group (Shift+click adds to the selection)", true);
        return;
    }
    auto& instances = m_models.getInstances();
    const std::string name = nextGroupName();
    for (int index : indices)
        instances[index].group = name;
    markSceneChanged();
    setStatus("Grouped " + std::to_string(indices.size()) + " objects as " + name +
        ". A click selects the whole group, a double-click one object");
}

void Editor::ungroupSelection()
{
    auto& instances = m_models.getInstances();
    size_t count = 0;
    for (int index : selectedIndices())
        if (!instances[index].group.empty()) {
            instances[index].group.clear();
            ++count;
        }
    if (count == 0) {
        setStatus("Nothing selected is in a group");
        return;
    }
    markSceneChanged();
    setStatus("Ungrouped " + std::to_string(count) + (count == 1 ? " object" : " objects"));
}

void Editor::renameGroup(const std::string& from, const std::string& to)
{
    const std::string name = trimmed(to);
    if (name.empty() || name == from)
        return;
    const std::vector<int> members = groupMembers(from);
    const std::string unique = uniqueGroupName(name);
    for (int member : members)
        m_models.getInstances()[member].group = unique;
    markSceneChanged();
    setStatus("Renamed group to " + unique);
}

std::unordered_map<std::string, std::string> Editor::groupNamesForCopies(const std::vector<int>& indices) const
{
    const auto& instances = m_models.getInstances();
    std::unordered_map<std::string, size_t> copied;
    for (int index : indices)
        if (!instances[index].group.empty())
            ++copied[instances[index].group];
    std::unordered_map<std::string, std::string> names;
    std::unordered_set<std::string> used;
    for (const auto& [group, count] : copied) {
        if (count < groupMembers(group).size())
            continue; // part of a group: the copies join it
        // Not yet in the scene, nor given to another copied group.
        std::string name = uniqueGroupName(group);
        for (int n = 2; used.contains(name); ++n)
            name = uniqueGroupName(group + " (" + std::to_string(n) + ")");
        used.insert(name);
        names[group] = name;
    }
    return names;
}

void Editor::drawGroupSection(int instanceIndex)
{
    const std::string group = m_models.getInstances()[instanceIndex].group;
    const std::vector<int> members = groupMembers(group);
    EditorStyle::propertyLabel("Group");
    const float buttonsWidth = ImGui::CalcTextSize("Select").x + ImGui::CalcTextSize("Ungroup").x +
        ImGui::GetStyle().FramePadding.x * 4.0f + ImGui::GetStyle().ItemSpacing.x * 2.0f;
    ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x - buttonsWidth, 60.0f));
    // The buffer follows the group until the field is being edited.
    if (!m_groupNameActive) {
        strncpy(m_groupNameBuffer, group.c_str(), sizeof(m_groupNameBuffer) - 1);
        m_groupNameBuffer[sizeof(m_groupNameBuffer) - 1] = '\0';
    }
    ImGui::InputText("##groupName", m_groupNameBuffer, sizeof(m_groupNameBuffer));
    m_groupNameActive = ImGui::IsItemActive();
    if (ImGui::IsItemDeactivatedAfterEdit())
        renameGroup(group, m_groupNameBuffer);
    ImGui::SetItemTooltip("%zu objects; rename the group here", members.size());
    ImGui::SameLine();
    if (ImGui::Button("Select"))
        selectIndices(members, instanceIndex);
    ImGui::SetItemTooltip("Select the whole group");
    ImGui::SameLine();
    if (ImGui::Button("Ungroup")) {
        for (int member : members)
            m_models.getInstances()[member].group.clear();
        markSceneChanged();
        setStatus("Ungrouped " + group);
    }
    ImGui::SetItemTooltip("%s", withShortcut("Dissolve the group; its objects stay where they are",
        EditorAction::UngroupObjects).c_str());
}

void Editor::drawGroupNode(const std::string& group, HierarchyActions& actions)
{
    const std::vector<int> members = groupMembers(group);
    if (members.empty())
        return;
    const auto& instances = m_models.getInstances();
    if (m_hierarchyFilter.IsActive() && !m_hierarchyFilter.PassFilter(group.c_str()) &&
        std::none_of(members.begin(), members.end(),
            [&](int m) { return m_hierarchyFilter.PassFilter(instances[m].name.c_str()); }))
        return;

    ImGui::PushID(("group:" + group).c_str());
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
        ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen;
    const bool allSelected = std::all_of(members.begin(), members.end(), [this](int m) { return isSelected(m); });
    if (allSelected)
        flags |= ImGuiTreeNodeFlags_Selected;
    ImGui::PushStyleColor(ImGuiCol_Text, EditorStyle::kAccent);
    const bool open = ImGui::TreeNodeEx("##group", flags, "%s  (%zu)", group.c_str(), members.size());
    ImGui::PopStyleColor();
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && !ImGui::IsItemToggledOpen()) {
        const ImGuiIO& io = ImGui::GetIO();
        selectWithGroup(members.front(), io.KeyCtrl ? SDL_KMOD_CTRL : io.KeyShift ? SDL_KMOD_SHIFT : SDL_KMOD_NONE);
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && !allSelected)
        selectIndices(members, members.front());
    ImGui::SetItemTooltip("Group: click selects all of it; the rows below select single objects");
    if (ImGui::BeginPopupContextItem("GroupContextMenu")) {
        const auto shortcut = [this](EditorAction action) { return m_keymap.shortcutLabel(action); };
        if (ImGui::MenuItem("Focus", shortcut(EditorAction::Focus).c_str())) {
            selectIndices(members, members.front());
            focusSelection();
        }
        if (ImGui::MenuItem("Duplicate Group", shortcut(EditorAction::Duplicate).c_str())) {
            selectIndices(members, members.front());
            actions.duplicateIndex = members.front();
        }
        if (ImGui::MenuItem("Save Group as Prefab...", shortcut(EditorAction::SaveAsPrefab).c_str())) {
            selectIndices(members, members.front());
            saveObjectPrefabDialog();
        }
        if (ImGui::MenuItem("Ungroup", shortcut(EditorAction::UngroupObjects).c_str())) {
            selectIndices(members, members.front());
            ungroupSelection();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Delete Group", shortcut(EditorAction::Delete).c_str())) {
            selectIndices(members, members.front());
            actions.deleteIndex = members.front();
        }
        ImGui::EndPopup();
    }
    if (open) {
        for (int member : members)
            drawInstanceNode(member, actions);
        ImGui::TreePop();
    }
    ImGui::PopID();
}

// ---------------------------------------------------------------------------------------------
// Object prefabs
// ---------------------------------------------------------------------------------------------

void Editor::saveObjectPrefabDialog()
{
    const std::vector<int> indices = selectedIndices(true);
    if (indices.empty())
        return;
    const ModelInstance& active = m_models.getInstances()[indices.front()];
    m_prefabSaveIds.clear();
    for (int index : indices)
        m_prefabSaveIds.push_back(m_models.getInstances()[index].id);
    m_prefabSaveInstance = -1;
    m_prefabSaveName = active.group.empty() ? active.name : active.group;
    const std::string fileName = m_prefabSaveName + kPrefabExtension;
    openFileDialog("SavePrefabDlg", "Save Objects as Prefab", kPrefabExtension, kPrefabsRoot, fileName.c_str(), true);
}

void Editor::saveObjectPrefab(const std::vector<uint64_t>& ids, const std::string& path)
{
    auto& instances = m_models.getInstances();
    std::vector<int> indices;
    for (uint64_t id : ids) {
        const auto it = std::find_if(instances.begin(), instances.end(), [&](const ModelInstance& i) { return i.id == id; });
        if (it != instances.end())
            indices.push_back(static_cast<int>(it - instances.begin()));
    }
    if (indices.empty()) {
        setStatus("Cannot save prefab: the objects were removed", true);
        return;
    }
    // The origin is the bottom center of the objects, on the grid while snapping, so placing the prefab
    // on a floor stands it there.
    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    for (int index : indices) {
        glm::vec3 objectMin, objectMax;
        if (instanceWorldBounds(index, objectMin, objectMax)) {
            lo = glm::min(lo, objectMin);
            hi = glm::max(hi, objectMax);
        }
    }
    glm::vec3 origin = lo.x <= hi.x ? glm::vec3((lo.x + hi.x) * 0.5f, lo.y, (lo.z + hi.z) * 0.5f) : glm::vec3(0.0f);
    if (snapActive())
        origin = glm::round(origin / m_gridSize) * m_gridSize;

    ObjectPrefab prefab;
    prefab.name = std::filesystem::path(path).stem().string();
    for (int index : indices) {
        const ModelInstance& instance = instances[index];
        const GPUModel* model = m_models.getModel(instance.modelIndex);
        if (!model)
            continue;
        ObjectPrefab::Object& object = prefab.objects.emplace_back();
        object.name = instance.name;
        object.modelName = model->name;
        if (model->polyMesh && !model->prefabPath.empty())
            object.prefabPath = model->prefabPath;
        else if (model->polyMesh)
            object.mesh = *model->polyMesh;
        else
            object.modelPath = model->sourcePath;
        object.position = instance.position - origin;
        object.rotation = instance.rotation;
        object.scale = instance.scale;
        object.color = instance.color;
        object.visible = instance.visible;
        object.locked = instance.locked;
        object.entity = instance.entity;
        object.entityParams = instance.entityParams;
    }
    if (!::saveObjectPrefab(path, prefab)) {
        setStatus("Failed to save prefab: " + path, true);
        return;
    }
    // Saved together, they stay together.
    if (indices.size() > 1) {
        const std::string& current = instances[indices.front()].group;
        const bool oneGroup = !current.empty() && groupMembers(current).size() == indices.size() &&
            std::all_of(indices.begin(), indices.end(), [&](int i) { return instances[i].group == current; });
        if (!oneGroup) {
            const std::string group = uniqueGroupName(prefab.name);
            for (int index : indices)
                instances[index].group = group;
            markSceneChanged();
        }
    }
    setStatus("Saved " + std::to_string(prefab.objects.size()) + " objects as prefab " + path +
        "; place it from Add > Prefabs");
}

void Editor::addObjectPrefab(const std::string& path, const glm::vec3* at)
{
    std::optional<ObjectPrefab> prefab = loadObjectPrefab(path);
    if (!prefab) {
        setStatus("Failed to load prefab: " + path, true);
        return;
    }
    if (prefab->name.empty())
        prefab->name = std::filesystem::path(path).stem().string();
    const glm::vec3 origin = at ? *at : placementPoint();
    const std::string group = prefab->objects.size() > 1 ? uniqueGroupName(prefab->name) : std::string();
    std::vector<int> added;
    std::string failed;
    for (const ObjectPrefab::Object& object : prefab->objects) {
        std::optional<size_t> modelIndex;
        if (!object.prefabPath.empty()) {
            modelIndex = loadPrefabModel(object.prefabPath);
        }
        else if (object.mesh) {
            modelIndex = createLevelModel(*object.mesh, object.modelName.empty() ? object.name : object.modelName);
        }
        else if (const auto found = m_models.findModelByPath(object.modelPath)) {
            modelIndex = found;
        }
        else {
            try {
                modelIndex = m_models.loadModelSync(object.modelPath, object.modelName);
            }
            catch (const std::exception& e) {
                failed = object.modelPath + ": " + e.what();
            }
        }
        if (!modelIndex)
            continue;
        const size_t index = m_models.createInstance(*modelIndex, origin + object.position, object.rotation, object.scale);
        ModelInstance& created = m_models.getInstances()[index];
        created.name = uniqueInstanceName(object.name);
        created.color = object.color;
        created.visible = object.visible;
        created.locked = object.locked && !object.prefabPath.empty();
        created.entity = object.entity;
        created.entityParams = object.entityParams;
        created.group = group;
        added.push_back(static_cast<int>(index));
    }
    if (added.empty()) {
        setStatus("Nothing of prefab " + prefab->name + " could be added" + (failed.empty() ? "" : " (" + failed + ")"), true);
        return;
    }
    markSceneChanged();
    selectIndices(added, added.front());
    if (!failed.empty())
        setStatus("Added prefab " + prefab->name + " without some objects: " + failed, true);
    else
        setStatus("Added prefab " + prefab->name + (group.empty() ? "" : " as group " + group));
}
