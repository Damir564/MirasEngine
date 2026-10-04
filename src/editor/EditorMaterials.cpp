#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <set>
#include "EditorStyle.h"
#include "engine/ModelManager.h"

namespace {

constexpr const char* kMaterialPayload = "MIRAS_MATERIAL";

ImTextureID toImTexture(vk::DescriptorSet set)
{
    return static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(static_cast<VkDescriptorSet>(set)));
}

std::string textureLabel(const std::string& path)
{
    return path.empty() ? "None"
        : path == kCheckerTexturePath ? "Checker"
        : std::filesystem::path(path).filename().string();
}

} // namespace

std::string Editor::levelSlotName(const PolyMesh& mesh, uint32_t slot)
{
    if (slot >= mesh.materials.size())
        return "None (white)";
    const std::string& path = mesh.materials[slot].materialPath;
    if (path.empty())
        return "Material " + std::to_string(slot);
    const MaterialAsset* shared = m_materials.find(path);
    return shared ? shared->name : std::filesystem::path(path).stem().string() + " (missing)";
}

float Editor::levelSlotTexelSize(const PolyMesh& mesh, uint32_t slot)
{
    if (slot >= mesh.materials.size())
        return 1.0f;
    const MaterialAsset* shared = m_materials.find(mesh.materials[slot].materialPath);
    return shared ? shared->texelSize : 1.0f;
}

bool Editor::drawSharedMaterialCombo(const char* id, std::string& path, const char* noneLabel)
{
    const MaterialAsset* current = m_materials.find(path);
    const std::string preview = path.empty() ? std::string(noneLabel)
        : current ? current->name : std::filesystem::path(path).stem().string() + " (missing)";
    bool changed = false;
    if (ImGui::BeginCombo(id, preview.c_str(), ImGuiComboFlags_HeightLarge)) {
        if (ImGui::Selectable(noneLabel, path.empty())) {
            changed = !path.empty();
            path.clear();
        }
        for (const auto& [materialPath, material] : m_materials.materials()) {
            if (ImGui::Selectable(material.name.c_str(), materialPath == path) && materialPath != path) {
                path = materialPath;
                changed = true;
            }
            ImGui::SetItemTooltip("%s", materialPath.c_str());
        }
        ImGui::Separator();
        if (ImGui::Selectable("+ New material")) {
            if (auto created = m_materials.create("Material")) {
                path = *created;
                changed = true;
                setStatus("Created " + *created);
            }
            else {
                setStatus("Failed to create a material in " + m_materials.root(), true);
            }
        }
        if (ImGui::Selectable("Rescan folder")) {
            m_materials.scan();
            setStatus("Found " + std::to_string(m_materials.materials().size()) + " materials in " + m_materials.root());
        }
        ImGui::EndCombo();
    }
    return changed;
}

bool Editor::drawSharedMaterialProperties(MaterialAsset& material)
{
    bool changed = false;
    EditorStyle::propertyLabel("Color");
    ImGui::SetNextItemWidth(-FLT_MIN);
    changed |= ImGui::ColorEdit3("##sharedColor", &material.color.x);
    EditorStyle::propertyLabel("Roughness");
    ImGui::SetNextItemWidth(-FLT_MIN);
    changed |= ImGui::SliderFloat("##sharedRoughness", &material.roughness, 0.0f, 1.0f, "%.2f");
    EditorStyle::propertyLabel("Metallic");
    ImGui::SetNextItemWidth(-FLT_MIN);
    changed |= ImGui::SliderFloat("##sharedMetallic", &material.metallic, 0.0f, 1.0f, "%.2f");
    EditorStyle::propertyLabel("Texel size");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::DragFloat("##sharedTexel", &material.texelSize, 0.01f, 0.01f, 1000.0f, "%.2f")) {
        material.texelSize = std::max(material.texelSize, 0.01f);
        changed = true;
    }
    ImGui::SetItemTooltip("World units one texture repeat covers on level shapes");

    // Texture rows: name, then Browse / (Checker) / Clear.
    const auto textureRow = [&](const char* label, std::string& texture, bool normal) {
        ImGui::PushID(label);
        EditorStyle::propertyLabel(label);
        ImGui::TextUnformatted(textureLabel(texture).c_str());
        if (!texture.empty() && ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", texture.c_str());
        if (ImGui::Button("Browse..."))
            openTextureDialog({ -1, material.path, normal });
        if (!normal) {
            ImGui::SameLine();
            if (ImGui::Button("Checker")) {
                texture = kCheckerTexturePath;
                changed = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear") && !texture.empty()) {
            texture.clear();
            changed = true;
        }
        ImGui::PopID();
    };
    textureRow("Texture", material.baseColorTexture, false);
    textureRow("Normal map", material.normalTexture, true);
    return changed;
}

void Editor::saveSharedMaterial(const MaterialAsset& material)
{
    if (!m_materials.save(material)) {
        setStatus("Failed to save " + material.path, true);
        return;
    }
    // Every object using it changes, but the .mat file is not part of the scene's undo history.
    m_models.refreshMaterial(material.path);
}

void Editor::drawModelMaterialOverrides(size_t modelIndex)
{
    const GPUModel* model = m_models.getModel(modelIndex);
    if (!model || model->polyMesh)
        return;
    std::set<uint32_t> slots;
    for (const SubmeshInfo& sub : model->submeshes)
        slots.insert(sub.sourceMaterial);

    for (const uint32_t slot : slots) {
        std::string label = slot == kNoSourceMaterial ? std::string("Default")
            : slot < model->materialNames.size() && !model->materialNames[slot].empty() ? model->materialNames[slot]
            : "Material " + std::to_string(slot);
        const auto it = model->materialOverrides.find(slot);
        std::string path = it != model->materialOverrides.end() ? it->second : std::string();
        ImGui::PushID(static_cast<int>(slot));
        EditorStyle::propertyLabel(label.c_str());
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (drawSharedMaterialCombo("##override", path, "(model's own)")) {
            applyMaterialToModel(modelIndex, path, slot, false);
            ImGui::PopID();
            return; // the model was replaced; draw the rest next frame
        }
        ImGui::PopID();
    }
    ImGui::TextDisabled("Shared by every object using this model.");
}

// ---------------------------------------------------------------------------------------------
// Applying materials
// ---------------------------------------------------------------------------------------------

uint32_t Editor::levelSlotFor(PolyMesh& mesh, const std::string& materialPath)
{
    for (size_t i = 0; i < mesh.materials.size(); ++i)
        if (mesh.materials[i].materialPath == materialPath)
            return static_cast<uint32_t>(i);
    if (mesh.materials.size() >= kMaxPolyMaterialSlots)
        return kNoPolyMaterial;
    PolyMaterial slot;
    slot.materialPath = materialPath;
    mesh.materials.push_back(std::move(slot));
    return static_cast<uint32_t>(mesh.materials.size() - 1);
}

void Editor::pruneLinkedSlots(PolyMesh& mesh)
{
    std::vector<bool> used(mesh.materials.size(), false);
    for (const PolyFace& face : mesh.faces)
        if (face.material < used.size())
            used[face.material] = true;
    // Unlinked slots stay: they may be set up by hand before any face uses them.
    std::vector<uint32_t> remap(mesh.materials.size(), kNoPolyMaterial);
    std::vector<PolyMaterial> kept;
    for (size_t i = 0; i < mesh.materials.size(); ++i) {
        if (!used[i] && !mesh.materials[i].materialPath.empty())
            continue;
        remap[i] = static_cast<uint32_t>(kept.size());
        kept.push_back(std::move(mesh.materials[i]));
    }
    if (kept.size() == mesh.materials.size()) {
        mesh.materials = std::move(kept);
        return;
    }
    mesh.materials = std::move(kept);
    for (PolyFace& face : mesh.faces)
        if (face.material < remap.size())
            face.material = remap[face.material];
}

void Editor::applyMaterialToSelection(const std::string& materialPath)
{
    if (!hasSelection()) {
        setStatus("Select an object (or a face in face mode) to apply the material to", true);
        return;
    }
    const ModelInstance& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    const GPUModel* model = m_models.getModel(instance.modelIndex);
    if (!model)
        return;
    if (model->polyMesh) {
        PolyMesh* mesh = selectedLevelMesh();
        if (!mesh) {
            setStatus(instance.name + " is a locked prefab; unlock it in the Level panel first", true);
            return;
        }
        const uint32_t slot = levelSlotFor(*mesh, materialPath);
        if (slot == kNoPolyMaterial) {
            setStatus("Too many materials on " + instance.name, true);
            return;
        }
        if (selectedLevelFace())
            for (uint32_t f : selectedLevelFaces())
                mesh->faces[f].material = slot;
        else
            for (PolyFace& face : mesh->faces)
                face.material = slot;
        pruneLinkedSlots(*mesh);
        rebuildSelectedLevelModel("apply material");
        setStatus("Applied " + std::filesystem::path(materialPath).stem().string() + " to " + instance.name);
        return;
    }
    applyMaterialToModel(instance.modelIndex, materialPath, kNoSourceMaterial, true);
}

void Editor::applyMaterialToModel(size_t modelIndex, const std::string& materialPath, uint32_t slot, bool allSlots)
{
    GPUModel* model = m_models.getModel(modelIndex);
    if (!model || model->polyMesh)
        return;
    std::set<uint32_t> slots;
    if (allSlots)
        for (const SubmeshInfo& sub : model->submeshes)
            slots.insert(sub.sourceMaterial);
    else
        slots.insert(slot);
    for (const uint32_t s : slots) {
        if (materialPath.empty())
            model->materialOverrides.erase(s);
        else
            model->materialOverrides[s] = materialPath;
    }
    const std::string name = model->name; // reloading replaces the model object
    try {
        m_models.reloadModel(modelIndex);
        setStatus("Updated materials of " + name + " (every object using this model)");
    }
    catch (const std::exception& e) {
        setStatus(std::string("Failed to reload model: ") + e.what(), true);
    }
}

void Editor::relinkMaterial(const std::string& from, const std::string& to)
{
    for (size_t i = 0; i < m_models.getModels().size(); ++i) {
        GPUModel* model = m_models.getModel(i);
        if (!model || !ModelManager::usesMaterial(*model, from))
            continue;
        if (model->polyMesh)
            for (PolyMaterial& slot : model->polyMesh->materials) {
                if (slot.materialPath == from)
                    slot.materialPath = to;
            }
        else
            for (auto& entry : model->materialOverrides)
                if (entry.second == from)
                    entry.second = to;
        try {
            if (model->polyMesh)
                m_models.rebuildPolyMesh(i);
            else
                m_models.reloadModel(i);
        }
        catch (const std::exception& e) {
            setStatus(std::string("Failed to update a model: ") + e.what(), true);
        }
    }
    // The selected shape changed outside the undo history; snapshot it again.
    m_levelBaselinePath.clear();
}

// ---------------------------------------------------------------------------------------------
// Viewport: drag and drop, eyedropper
// ---------------------------------------------------------------------------------------------

void Editor::drawMaterialDropTarget()
{
    const ImGuiPayload* payload = ImGui::GetDragDropPayload();
    if (m_flyMode || !payload || !payload->IsDataType(kMaterialPayload))
        return;
    // Only exists during the drag, so it never gets in the way of viewport clicks.
    ImGui::SetNextWindowPos(ImVec2(m_sceneView.x, m_sceneView.y));
    ImGui::SetNextWindowSize(ImVec2(m_sceneView.width, m_sceneView.height));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    if (ImGui::Begin("##MaterialDrop", nullptr, flags)) {
        ImGui::InvisibleButton("##dropArea", ImVec2(std::max(m_sceneView.width, 1.0f), std::max(m_sceneView.height, 1.0f)));
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* accepted = ImGui::AcceptDragDropPayload(kMaterialPayload,
                    ImGuiDragDropFlags_AcceptNoDrawDefaultRect)) {
                const std::string path(static_cast<const char*>(accepted->Data));
                const ImVec2 mouse = ImGui::GetIO().MousePos;
                dropMaterial(path, mouse.x - m_sceneView.x, mouse.y - m_sceneView.y, ImGui::GetIO().KeyShift);
            }
            ImGui::EndDragDropTarget();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void Editor::dropMaterial(const std::string& materialPath, float mouseX, float mouseY, bool wholeObject)
{
    m_currentMaterial = materialPath;
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height,
        getView(m_camera), sceneProjection());
    const SubmeshHitResult hit = pickSubmesh(ray, m_models.getInstances(),
        [&](size_t index) { return m_models.getModel(index); });
    if (!hit.hit()) {
        setStatus("Drop the material onto an object", true);
        return;
    }
    const ModelInstance& instance = m_models.getInstances()[hit.instanceIndex];
    const size_t modelIndex = instance.modelIndex;
    const GPUModel* model = m_models.getModel(modelIndex);
    if (!model)
        return;
    if (!model->polyMesh) {
        const uint32_t slot = hit.submeshIndex < model->submeshes.size()
            ? model->submeshes[hit.submeshIndex].sourceMaterial : kNoSourceMaterial;
        selectInstance(hit.instanceIndex);
        applyMaterialToModel(modelIndex, materialPath, slot, wholeObject);
        return;
    }
    if (instance.locked) {
        setStatus(instance.name + " is a locked prefab; unlock it in the Level panel first", true);
        return;
    }
    const int face = wholeObject ? -1 : pickLevelFace(mouseX, mouseY, hit.instanceIndex);
    m_materialDrop = { true, instance.id, face, kNoSourceMaterial, wholeObject || face < 0, materialPath };
    if (m_gizmo.selectedInstance == hit.instanceIndex) {
        applyPendingMaterialDrop();
    }
    else {
        selectInstance(hit.instanceIndex);
    }
}

void Editor::applyPendingMaterialDrop()
{
    if (!m_materialDrop.pending)
        return;
    MaterialDrop drop = std::move(m_materialDrop);
    m_materialDrop = {};
    if (!hasSelection() || m_models.getInstances()[m_gizmo.selectedInstance].id != drop.instanceId)
        return;
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh)
        return;
    const uint32_t slot = levelSlotFor(*mesh, drop.materialPath);
    if (slot == kNoPolyMaterial) {
        setStatus("Too many materials on this shape", true);
        return;
    }
    if (!drop.wholeObject && drop.face >= 0 && drop.face < static_cast<int>(mesh->faces.size())) {
        mesh->faces[drop.face].material = slot;
        // Show which face got it.
        if (m_levelMode == LevelEditMode::Face)
            selectLevelFace(static_cast<uint32_t>(drop.face), false);
    }
    else {
        for (PolyFace& face : mesh->faces)
            face.material = slot;
    }
    pruneLinkedSlots(*mesh);
    rebuildSelectedLevelModel("apply material");
    setStatus("Applied " + std::filesystem::path(drop.materialPath).stem().string() +
        (drop.wholeObject ? " to the whole shape" : " to a face"));
}

void Editor::pickMaterialUnderMouse()
{
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    if (m_flyMode || ImGui::GetIO().WantCaptureMouse || !sceneViewContains(mouse.x, mouse.y))
        return;
    const float x = mouse.x - m_sceneView.x, y = mouse.y - m_sceneView.y;
    const Ray ray = screenToWorldRay(x, y, m_sceneView.width, m_sceneView.height, getView(m_camera), sceneProjection());
    const SubmeshHitResult hit = pickSubmesh(ray, m_models.getInstances(),
        [&](size_t index) { return m_models.getModel(index); });
    if (!hit.hit())
        return;
    const GPUModel* model = m_models.getModel(m_models.getInstances()[hit.instanceIndex].modelIndex);
    if (!model)
        return;
    std::string path;
    if (model->polyMesh) {
        const int face = pickLevelFace(x, y, hit.instanceIndex);
        const PolyMesh& mesh = *model->polyMesh;
        if (face >= 0 && mesh.faces[face].material < mesh.materials.size())
            path = mesh.materials[mesh.faces[face].material].materialPath;
    }
    else if (hit.submeshIndex < model->submeshes.size()) {
        const auto it = model->materialOverrides.find(model->submeshes[hit.submeshIndex].sourceMaterial);
        if (it != model->materialOverrides.end())
            path = it->second;
    }
    if (path.empty()) {
        setStatus("That surface has no shared material", true);
        return;
    }
    m_currentMaterial = path;
    m_showMaterialsPanel = true;
    setStatus("Picked " + std::filesystem::path(path).stem().string());
}

// ---------------------------------------------------------------------------------------------
// Materials panel
// ---------------------------------------------------------------------------------------------

void Editor::drawMaterialTile(const MaterialAsset& material, float size)
{
    ImGui::PushID(material.path.c_str());
    ImGui::BeginGroup();
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1(p0.x + size, p0.y + size);
    ImGui::InvisibleButton("##tile", ImVec2(size, size));
    const bool hovered = ImGui::IsItemHovered();
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left) || ImGui::IsItemClicked(ImGuiMouseButton_Right))
        m_currentMaterial = material.path;
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        applyMaterialToSelection(material.path);
    if (ImGui::BeginDragDropSource()) {
        ImGui::SetDragDropPayload(kMaterialPayload, material.path.c_str(), material.path.size() + 1);
        ImGui::Text("%s", material.name.c_str());
        ImGui::TextDisabled("Drop on a face; Shift: whole object");
        ImGui::EndDragDropSource();
    }
    else if (hovered) {
        ImGui::SetTooltip("%s\n%s\nDouble-click: apply to the selection\nDrag onto a face in the viewport",
            material.name.c_str(), material.path.c_str());
    }
    if (ImGui::BeginPopupContextItem("##tileMenu")) {
        if (ImGui::MenuItem("Apply to selection", nullptr, false, hasSelection()))
            applyMaterialToSelection(material.path);
        if (ImGui::MenuItem("Duplicate")) {
            if (auto created = m_materials.create(material.name, material))
                m_currentMaterial = *created;
        }
        if (ImGui::MenuItem("Rename...")) {
            m_materialPopupPath = material.path;
            m_openMaterialRename = true;
        }
        if (ImGui::MenuItem("Delete...")) {
            m_materialPopupPath = material.path;
            m_openMaterialDelete = true;
        }
        ImGui::EndPopup();
    }

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const ImU32 tint = ImGui::ColorConvertFloat4ToU32(ImVec4(material.color.r, material.color.g, material.color.b, 1.0f));
    if (const vk::DescriptorSet set = m_models.previewTexture(material.baseColorTexture))
        drawList->AddImage(ImTextureRef(toImTexture(set)), p0, p1, ImVec2(0, 0), ImVec2(1, 1), tint);
    else
        drawList->AddRectFilled(p0, p1, tint);
    const bool selected = material.path == m_currentMaterial;
    if (selected || hovered)
        drawList->AddRect(p0, p1, ImGui::GetColorU32(selected ? EditorStyle::kAccent : EditorStyle::kTextDim),
            0.0f, 0, selected ? 3.0f : 1.0f);

    // Name under the tile, cut to its width.
    const float lineHeight = ImGui::GetTextLineHeight();
    const ImVec2 textPos(p0.x, p1.y + 2.0f);
    drawList->PushClipRect(textPos, ImVec2(p1.x, textPos.y + lineHeight), true);
    drawList->AddText(textPos, ImGui::GetColorU32(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled), material.name.c_str());
    drawList->PopClipRect();
    ImGui::Dummy(ImVec2(size, lineHeight + 2.0f));
    ImGui::EndGroup();
    ImGui::PopID();
}

void Editor::drawMaterialPopups()
{
    if (m_openMaterialRename) {
        m_openMaterialRename = false;
        const MaterialAsset* material = m_materials.find(m_materialPopupPath);
        snprintf(m_materialNameBuffer, sizeof(m_materialNameBuffer), "%s", material ? material->name.c_str() : "");
        ImGui::OpenPopup("Rename Material");
    }
    if (m_openMaterialDelete) {
        m_openMaterialDelete = false;
        ImGui::OpenPopup("Delete Material");
    }

    if (ImGui::BeginPopupModal("Rename Material", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (ImGui::IsWindowAppearing())
            ImGui::SetKeyboardFocusHere();
        const bool enter = ImGui::InputText("##materialName", m_materialNameBuffer, sizeof(m_materialNameBuffer),
            ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::TextDisabled("Objects using it are updated.");
        if (ImGui::Button("Rename", ImVec2(100, 0)) || enter) {
            if (auto renamed = m_materials.rename(m_materialPopupPath, m_materialNameBuffer)) {
                relinkMaterial(m_materialPopupPath, *renamed);
                if (m_currentMaterial == m_materialPopupPath)
                    m_currentMaterial = *renamed;
                setStatus("Renamed to " + *renamed);
            }
            else {
                setStatus("Failed to rename " + m_materialPopupPath, true);
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(100, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopupModal("Delete Material", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Delete %s?", m_materialPopupPath.c_str());
        ImGui::TextDisabled("Objects using it fall back to their own material values.");
        ImGui::PushStyleColor(ImGuiCol_Button, EditorStyle::kDanger);
        const bool confirmed = ImGui::Button("Delete", ImVec2(100, 0));
        ImGui::PopStyleColor();
        if (confirmed) {
            if (m_materials.remove(m_materialPopupPath)) {
                m_models.refreshMaterial(m_materialPopupPath);
                if (m_currentMaterial == m_materialPopupPath)
                    m_currentMaterial.clear();
                setStatus("Deleted " + m_materialPopupPath);
            }
            else {
                setStatus("Failed to delete " + m_materialPopupPath, true);
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(100, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void Editor::drawMaterialsPanel()
{
    if (!ImGui::Begin("Materials", &m_showMaterialsPanel)) {
        ImGui::End();
        return;
    }
    const MaterialAsset* current = m_materials.find(m_currentMaterial);

    if (ImGui::Button("New")) {
        if (auto created = m_materials.create("Material")) {
            m_currentMaterial = *created;
            setStatus("Created " + *created);
        }
        else {
            setStatus("Failed to create a material in " + m_materials.root(), true);
        }
        current = m_materials.find(m_currentMaterial);
    }
    ImGui::SetItemTooltip("New material file in %s/", m_materials.root().c_str());
    ImGui::SameLine();
    ImGui::BeginDisabled(!current);
    if (ImGui::Button("Duplicate") && current) {
        if (auto created = m_materials.create(current->name, *current))
            m_currentMaterial = *created;
        current = m_materials.find(m_currentMaterial);
    }
    ImGui::SameLine();
    if (ImGui::Button("Rename") && current) {
        m_materialPopupPath = current->path;
        m_openMaterialRename = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete") && current) {
        m_materialPopupPath = current->path;
        m_openMaterialDelete = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Rescan")) {
        m_materials.scan();
        current = m_materials.find(m_currentMaterial);
    }
    ImGui::SetItemTooltip("Read the %s/ folder again (e.g. after adding files outside the editor)", m_materials.root().c_str());
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x - 110.0f, 80.0f));
    if (ImGui::InputTextWithHint("##materialFilter", "Search", m_materialFilter.InputBuf, IM_ARRAYSIZE(m_materialFilter.InputBuf)))
        m_materialFilter.Build();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::SliderFloat("##tileSize", &m_materialTileSize, 40.0f, 160.0f, "%.0f px");

    // Tiles on the left, the current material's properties on the right (below when narrow).
    const float available = ImGui::GetContentRegionAvail().x;
    const bool sideBySide = available > 520.0f;
    const float propertiesWidth = sideBySide ? std::min(320.0f, available * 0.4f) : 0.0f;
    const float gridHeight = sideBySide ? 0.0f : std::max(ImGui::GetContentRegionAvail().y * 0.5f, 120.0f);
    if (ImGui::BeginChild("##materialGrid", ImVec2(sideBySide ? available - propertiesWidth - ImGui::GetStyle().ItemSpacing.x : 0.0f, gridHeight),
            ImGuiChildFlags_Borders)) {
        const ImGuiStyle& style = ImGui::GetStyle();
        const float cell = m_materialTileSize + style.ItemSpacing.x;
        const int columns = std::max(1, static_cast<int>((ImGui::GetContentRegionAvail().x + style.ItemSpacing.x) / cell));
        int column = 0;
        for (const auto& [path, material] : m_materials.materials()) {
            if (!m_materialFilter.PassFilter(material.name.c_str()))
                continue;
            if (column > 0)
                ImGui::SameLine();
            drawMaterialTile(material, m_materialTileSize);
            column = (column + 1) % columns;
        }
        if (m_materials.materials().empty()) {
            ImGui::TextDisabled("No materials in %s/ yet.", m_materials.root().c_str());
            ImGui::TextDisabled("Click New, or Make shared on a level shape's material.");
        }
    }
    ImGui::EndChild();
    if (sideBySide)
        ImGui::SameLine();

    if (ImGui::BeginChild("##materialProperties", ImVec2(0.0f, 0.0f))) {
        // The tiles may have changed the current material or the library this frame.
        current = m_materials.find(m_currentMaterial);
        if (current) {
            ImGui::TextUnformatted(current->name.c_str());
            ImGui::TextDisabled("%s", current->path.c_str());
            ImGui::BeginDisabled(!hasSelection());
            if (ImGui::Button("Apply to selection", ImVec2(-FLT_MIN, 0.0f)))
                applyMaterialToSelection(current->path);
            ImGui::EndDisabled();
            ImGui::SetItemTooltip("The selected face in face mode, otherwise the whole selected object");
            ImGui::Spacing();
            MaterialAsset edited = *current;
            if (drawSharedMaterialProperties(edited))
                saveSharedMaterial(edited);
        }
        else {
            ImGui::TextDisabled("Click a material to edit it.");
            ImGui::TextDisabled("New level shapes use the selected one.");
        }
    }
    ImGui::EndChild();

    drawMaterialPopups();
    ImGui::End();
}
