#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <filesystem>
#include "EditorStyle.h"
#include "FileDialog.h"
#include "engine/ModelManager.h"
#include "engine/Prefab.h"

GPUModel* Editor::prefabModel(int instanceIndex)
{
    if (!validInstance(instanceIndex))
        return nullptr;
    GPUModel* model = m_models.getModel(m_models.getInstances()[instanceIndex].modelIndex);
    return model && model->polyMesh && !model->prefabPath.empty() ? model : nullptr;
}

size_t Editor::prefabInstanceCount(size_t modelIndex) const
{
    const auto& instances = m_models.getInstances();
    return static_cast<size_t>(std::count_if(instances.begin(), instances.end(),
        [modelIndex](const ModelInstance& instance) { return instance.modelIndex == modelIndex; }));
}

void Editor::savePrefabDialog(int instanceIndex)
{
    if (!validInstance(instanceIndex))
        return;
    m_prefabSaveInstance = instanceIndex;
    m_prefabSaveName = m_models.getInstances()[instanceIndex].name;
    const std::string fileName = m_prefabSaveName + kPrefabExtension;
    openFileDialog("SavePrefabDlg", "Save as Prefab", kPrefabExtension, kPrefabsRoot, fileName.c_str(), true);
}

void Editor::loadPrefabDialog()
{
    openFileDialog("LoadPrefabDlg", "Load Prefab", kPrefabExtension, kPrefabsRoot, nullptr, false);
}

void Editor::drawPrefabDialogs()
{
    ImGuiFileDialog* dialog = ImGuiFileDialog::Instance();
    if (dialog->Display("SavePrefabDlg", ImGuiWindowFlags_NoCollapse, kDialogSize)) {
        if (dialog->IsOk()) {
            std::filesystem::path path(dialog->GetFilePathName());
            if (path.extension() != kPrefabExtension)
                path.replace_extension(kPrefabExtension);
            if (validInstance(m_prefabSaveInstance) && m_models.getInstances()[m_prefabSaveInstance].name == m_prefabSaveName)
                saveAsPrefab(m_prefabSaveInstance, toStoredPath(path.string()));
            else
                setStatus("Cannot save prefab: " + m_prefabSaveName + " was removed", true);
        }
        m_prefabSaveInstance = -1;
        dialog->Close();
    }
    if (dialog->Display("LoadPrefabDlg", ImGuiWindowFlags_NoCollapse, kDialogSize)) {
        if (dialog->IsOk())
            addPrefab(toStoredPath(dialog->GetFilePathName()));
        dialog->Close();
    }
}

void Editor::saveAsPrefab(int instanceIndex, const std::string& path)
{
    if (!validInstance(instanceIndex))
        return;
    auto& instances = m_models.getInstances();
    GPUModel* model = m_models.getModel(instances[instanceIndex].modelIndex);
    if (!model || !model->polyMesh) {
        setStatus("Only level shapes can be saved as prefabs", true);
        return;
    }
    // Saving one instance of a shared prefab under a new file must not relink the others.
    if (!model->prefabPath.empty() && model->prefabPath != path &&
        prefabInstanceCount(instances[instanceIndex].modelIndex) > 1) {
        const auto copy = createLevelModel(*model->polyMesh, model->name);
        if (!copy)
            return;
        instances[instanceIndex].modelIndex = *copy;
        model = m_models.getModel(*copy);
    }

    const Prefab prefab{ std::filesystem::path(path).stem().string(), *model->polyMesh };
    if (!savePrefab(path, prefab)) {
        setStatus("Failed to save prefab: " + path, true);
        return;
    }
    model->name = prefab.name;
    model->prefabPath = path;
    // Instances of whatever was saved to this file before now follow the new geometry.
    const size_t modelIndex = instances[instanceIndex].modelIndex;
    for (ModelInstance& other : instances) {
        const GPUModel* otherModel = m_models.getModel(other.modelIndex);
        if (other.modelIndex != modelIndex && otherModel && otherModel->prefabPath == path)
            other.modelIndex = modelIndex;
    }
    instances[instanceIndex].locked = true;
    releaseUnusedLevelModels();
    markSceneChanged();
    setStatus("Saved prefab " + path);
}

std::optional<size_t> Editor::loadPrefabModel(const std::string& path)
{
    const auto& models = m_models.getModels();
    for (size_t i = 0; i < models.size(); ++i) {
        if (models[i] && models[i]->polyMesh && models[i]->prefabPath == path)
            return i;
    }
    std::optional<Prefab> prefab = loadPrefab(path);
    if (!prefab) {
        setStatus("Failed to load prefab: " + path, true);
        return std::nullopt;
    }
    if (prefab->name.empty())
        prefab->name = std::filesystem::path(path).stem().string();
    const auto modelIndex = createLevelModel(std::move(prefab->mesh), prefab->name);
    if (modelIndex)
        m_models.getModel(*modelIndex)->prefabPath = path;
    return modelIndex;
}

void Editor::addPrefab(const std::string& path)
{
    const auto modelIndex = loadPrefabModel(path);
    if (!modelIndex)
        return;

    const std::string name = uniqueInstanceName(m_models.getModel(*modelIndex)->name);
    const size_t newIndex = m_models.createInstance(*modelIndex, placementPoint());
    ModelInstance& instance = m_models.getInstances()[newIndex];
    instance.name = name;
    instance.locked = true;
    markSceneChanged();
    selectInstance(static_cast<int>(newIndex));
    setStatus("Added prefab " + name);
}

void Editor::drawPrefabMenuItems()
{
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    std::error_code ec;
    for (auto it = fs::directory_iterator(kPrefabsRoot, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().extension() == kPrefabExtension)
            files.push_back(it->path());
    }
    std::sort(files.begin(), files.end());
    for (const fs::path& file : files) {
        ImGui::PushID(file.string().c_str());
        if (ImGui::MenuItem(file.stem().string().c_str()))
            addPrefab(toStoredPath(file.string()));
        ImGui::PopID();
    }
    if (files.empty())
        ImGui::TextDisabled("No prefabs in %s/", kPrefabsRoot);
    if (ImGui::MenuItem("Load Prefab..."))
        loadPrefabDialog();
}

void Editor::drawPrefabSection()
{
    const int index = m_gizmo.selectedInstance;
    const GPUModel* model = prefabModel(index);
    if (!model)
        return;
    const std::string prefabPath = model->prefabPath;
    const bool locked = m_models.getInstances()[index].locked;
    const size_t count = prefabInstanceCount(m_models.getInstances()[index].modelIndex);

    ImGui::Text("Prefab: %s", std::filesystem::path(prefabPath).stem().string().c_str());
    ImGui::SetItemTooltip("%s", prefabPath.c_str());
    if (count == 1)
        ImGui::TextDisabled("1 instance in the scene");
    else
        ImGui::TextDisabled("%zu instances in the scene share it", count);

    const float width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    if (ImGui::Button(locked ? "Unlock" : "Lock", ImVec2(width, 0.0f)))
        setInstanceLocked(index, !locked);
    ImGui::SetItemTooltip("%s", locked ? "Allow editing; changes apply to every instance of this prefab"
                                       : "Protect the shared geometry from edits");
    ImGui::SameLine();
    if (ImGui::Button("Save to file", ImVec2(width, 0.0f)))
        writePrefabFile(index);
    ImGui::SetItemTooltip("Overwrite %s with the current geometry", prefabPath.c_str());
    if (ImGui::Button("Unlink from prefab", ImVec2(-FLT_MIN, 0.0f)))
        unlinkPrefab(index);
    ImGui::SetItemTooltip("Give this object its own copy of the geometry");
    if (locked)
        ImGui::TextDisabled("Locked. Unlock to edit the shape.");
    ImGui::Spacing();
}

void Editor::setInstanceLocked(int instanceIndex, bool locked)
{
    if (!prefabModel(instanceIndex))
        return;
    ModelInstance& instance = m_models.getInstances()[instanceIndex];
    instance.locked = locked;
    markSceneChanged();
    if (locked) {
        setStatus("Locked " + instance.name);
        return;
    }
    const size_t count = prefabInstanceCount(instance.modelIndex);
    setStatus("Unlocked " + instance.name + (count > 1
        ? ": edits change all " + std::to_string(count) + " instances of the prefab" : std::string()));
}

void Editor::unlinkPrefab(int instanceIndex)
{
    const GPUModel* model = prefabModel(instanceIndex);
    if (!model)
        return;
    const auto copy = createLevelModel(*model->polyMesh, model->name);
    if (!copy)
        return;
    ModelInstance& instance = m_models.getInstances()[instanceIndex];
    instance.modelIndex = *copy;
    instance.locked = false;
    const std::string name = instance.name;
    releaseUnusedLevelModels();
    markSceneChanged();
    setStatus("Unlinked " + name + " from its prefab");
}

void Editor::writePrefabFile(int instanceIndex)
{
    const GPUModel* model = prefabModel(instanceIndex);
    if (!model)
        return;
    if (savePrefab(model->prefabPath, { model->name, *model->polyMesh }))
        setStatus("Saved prefab " + model->prefabPath);
    else
        setStatus("Failed to save prefab: " + model->prefabPath, true);
}
