#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include "engine/ModelManager.h"

namespace {

// Multiplies an object's tint into the mesh's own material colors, for a united shape whose parts had
// different tints. False when some slot is a shared material, whose color can't change per object.
bool bakeTint(PolyMesh& mesh, const glm::vec3& tint)
{
    // Faces past the slots render plain white; they get a slot to carry the tint.
    uint32_t plainSlot = kNoPolyMaterial;
    for (PolyFace& face : mesh.faces) {
        if (face.material < mesh.materials.size())
            continue;
        if (plainSlot == kNoPolyMaterial) {
            plainSlot = static_cast<uint32_t>(mesh.materials.size());
            mesh.materials.emplace_back();
        }
        face.material = plainSlot;
    }
    bool kept = true;
    for (PolyMaterial& material : mesh.materials) {
        if (!material.materialPath.empty()) {
            kept = false;
            continue;
        }
        material.color = glm::vec4(glm::vec3(material.color) * tint, material.color.a);
    }
    return kept;
}

// A turn about an axis through the origin; quarter turns are exact, so grid positions stay on the grid.
glm::mat4 turnAbout(const glm::vec3& axis, float degrees)
{
    glm::mat4 turn = glm::rotate(glm::mat4(1.0f), glm::radians(degrees), axis);
    if (std::fmod(degrees, 90.0f) == 0.0f && (axis == glm::vec3(1, 0, 0) || axis == glm::vec3(0, 1, 0) || axis == glm::vec3(0, 0, 1)))
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r)
                turn[c][r] = std::round(turn[c][r]);
    return turn;
}

// Object-space bounds of the faces' vertices.
void faceBounds(const PolyMesh& mesh, const std::vector<uint32_t>& faces, glm::vec3& lo, glm::vec3& hi)
{
    lo = glm::vec3(FLT_MAX);
    hi = glm::vec3(-FLT_MAX);
    for (uint32_t f : faces)
        for (uint32_t v : mesh.faces[f].verts) {
            lo = glm::min(lo, mesh.positions[v]);
            hi = glm::max(hi, mesh.positions[v]);
        }
}

} // namespace

std::vector<float> Editor::levelSlotTexelSizes(const PolyMesh& mesh)
{
    std::vector<float> sizes;
    for (uint32_t slot = 0; slot < mesh.materials.size(); ++slot)
        sizes.push_back(levelSlotTexelSize(mesh, slot));
    return sizes;
}

std::vector<int> Editor::selectedLevelShapes() const
{
    std::vector<int> shapes;
    for (int index : selectedIndices(true)) {
        const GPUModel* model = m_models.getModel(m_models.getInstances()[index].modelIndex);
        if (model && model->polyMesh)
            shapes.push_back(index);
    }
    return shapes;
}

int Editor::uniteSelectedShapes(bool solid)
{
    // A pending geometry edit or drag refers to shapes this would delete.
    if (m_levelEditPending || m_vertexDrag.active || m_faceDrag.active || m_edgeDrag.active || m_gizmo.isDragging)
        return -1;
    const std::vector<int> shapes = selectedLevelShapes();
    if (shapes.size() < 2) {
        setStatus("Select two or more level shapes to unite (Shift+click adds to the selection)", true);
        return -1;
    }
    auto& instances = m_models.getInstances();
    if (solid)
        for (int index : shapes)
            if (!m_models.getModel(instances[index].modelIndex)->polyMesh->isClosed()) {
                setStatus(instances[index].name + " is not closed (e.g. a plane), so it has no volume to merge; "
                    "use Unite instead", true);
                return -1;
            }

    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    for (int index : shapes) {
        glm::vec3 shapeMin, shapeMax;
        if (instanceWorldBounds(index, shapeMin, shapeMax)) {
            lo = glm::min(lo, shapeMin);
            hi = glm::max(hi, shapeMax);
        }
    }
    glm::vec3 origin((lo.x + hi.x) * 0.5f, lo.y, (lo.z + hi.z) * 0.5f);
    if (snapActive())
        origin = glm::round(origin / m_gridSize) * m_gridSize;

    const glm::vec3 tint = instances[shapes.front()].color;
    const bool sameTint = std::all_of(shapes.begin(), shapes.end(), [&](int i) { return instances[i].color == tint; });
    PolyMesh united;
    united.gridSize = m_models.getModel(instances[shapes.front()].modelIndex)->polyMesh->gridSize;
    bool tintsKept = true;
    for (int index : shapes) {
        const ModelInstance& instance = instances[index];
        PolyMesh part = *m_models.getModel(instance.modelIndex)->polyMesh;
        part.transformKeepingUVs(glm::translate(glm::mat4(1.0f), -origin) * instance.getTransformMatrix(),
            levelSlotTexelSizes(part));
        if (!sameTint && instance.color != glm::vec3(1.0f))
            tintsKept = bakeTint(part, instance.color) && tintsKept;
        if (solid && !united.faces.empty()) {
            PolyMesh merged = polyMeshUnion(united, part);
            if (merged.materials.size() > kMaxPolyMaterialSlots) {
                setStatus("Cannot merge: together the shapes use more than " + std::to_string(kMaxPolyMaterialSlots) +
                    " materials", true);
                return -1;
            }
            united = std::move(merged);
            continue;
        }
        if (!united.append(part)) {
            setStatus("Cannot unite: together the shapes use more than " + std::to_string(kMaxPolyMaterialSlots) +
                " materials", true);
            return -1;
        }
    }

    if (solid) {
        united.mergeCoplanarFaces();
        if (united.faces.empty()) {
            setStatus("Merging left no faces", true);
            return -1;
        }
    }
    const std::string name = instances[shapes.front()].name;
    const std::string modelName = m_models.getModel(instances[shapes.front()].modelIndex)->name;
    const auto modelIndex = createLevelModel(std::move(united), modelName);
    if (!modelIndex)
        return -1;
    // The instance first: releasing the parts' models below shifts model indices, which instances follow.
    const uint64_t unitedId = instances[m_models.createInstance(*modelIndex, origin)].id;
    std::vector<int> removed = shapes;
    std::sort(removed.begin(), removed.end(), std::greater<>());
    deselectAll();
    for (int index : removed)
        m_models.removeInstance(static_cast<size_t>(index));
    releaseUnusedLevelModels();

    const auto found = std::find_if(instances.begin(), instances.end(),
        [&](const ModelInstance& instance) { return instance.id == unitedId; });
    const int index = static_cast<int>(found - instances.begin());
    // The first part's name is free again, so the result takes it.
    found->name = uniqueInstanceName(name);
    found->color = sameTint ? tint : glm::vec3(1.0f);
    markSceneChanged();
    selectInstance(index);
    setStatus(std::string(solid ? "Merged " : "United ") + std::to_string(shapes.size()) + " shapes into " + found->name +
        (tintsKept ? "" : " (tints of shared materials were dropped)"));
    return index;
}

void Editor::mergeShapeParts()
{
    if (m_levelEditPending || m_vertexDrag.active || m_faceDrag.active || m_edgeDrag.active || m_gizmo.isDragging)
        return;
    const PolyMesh* mesh = selectedLevelMesh();
    if (!mesh) {
        setStatus("Select an unlocked level shape", true);
        return;
    }
    std::vector<PolyMesh> parts = mesh->splitParts();
    if (parts.size() < 2) {
        setStatus("The shape is one piece: nothing to merge");
        return;
    }
    for (const PolyMesh& part : parts)
        if (!part.isClosed()) {
            setStatus("Some part is not closed (e.g. a plane), so it has no volume to merge", true);
            return;
        }
    PolyMesh merged = std::move(parts.front());
    for (size_t i = 1; i < parts.size(); ++i)
        merged = polyMeshUnion(merged, parts[i]);
    merged.mergeCoplanarFaces();
    if (merged.materials.size() > kMaxPolyMaterialSlots) {
        setStatus("Cannot merge: the parts use more than " + std::to_string(kMaxPolyMaterialSlots) + " materials", true);
        return;
    }
    const size_t partCount = parts.size();
    if (commitLevelTopology(std::move(merged), "merge parts")) {
        clearFaceSelection();
        m_selectedVertices.clear();
        m_edgeSelected = false;
        m_selectedParts.clear();
        setStatus("Merged " + std::to_string(partCount) + " parts into " + std::to_string(selectedLevelMesh()->partCount()));
    }
}

void Editor::separateSelectedShape()
{
    if (m_levelEditPending || m_vertexDrag.active || m_faceDrag.active || m_edgeDrag.active || m_gizmo.isDragging)
        return;
    auto& instances = m_models.getInstances();
    const int index = m_gizmo.selectedInstance;
    const GPUModel* model = hasSelection() ? m_models.getModel(instances[index].modelIndex) : nullptr;
    if (!model || !model->polyMesh) {
        setStatus("Select a level shape to separate", true);
        return;
    }
    std::vector<PolyMesh> parts = model->polyMesh->splitParts();
    const ModelInstance source = instances[index];
    if (parts.size() < 2) {
        setStatus(source.name + " is one piece: nothing to separate");
        return;
    }
    const std::string modelName = model->name;
    const glm::mat4 transform = source.getTransformMatrix();

    std::vector<uint64_t> createdIds;
    for (PolyMesh& part : parts) {
        // Each part's origin moves to its bottom center, on the shape's grid, so its gizmo sits on it.
        glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
        for (const glm::vec3& p : part.positions) {
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
        }
        const glm::vec3 center = glm::round(glm::vec3((lo.x + hi.x) * 0.5f, lo.y, (lo.z + hi.z) * 0.5f) /
            part.gridSize) * part.gridSize;
        part.transformKeepingUVs(glm::translate(glm::mat4(1.0f), -center), levelSlotTexelSizes(part));
        const auto partModel = createLevelModel(std::move(part), modelName);
        if (!partModel)
            break;
        const glm::vec3 position(transform * glm::vec4(center, 1.0f));
        ModelInstance& created = instances[m_models.createInstance(*partModel, position, source.rotation, source.scale)];
        created.color = source.color;
        created.visible = source.visible;
        createdIds.push_back(created.id);
    }
    if (createdIds.size() < parts.size()) {
        // Out of memory or similar: leave the shape whole.
        for (uint64_t id : createdIds) {
            const auto it = std::find_if(instances.begin(), instances.end(), [&](const ModelInstance& i) { return i.id == id; });
            if (it != instances.end())
                m_models.removeInstance(static_cast<size_t>(it - instances.begin()));
        }
        releaseUnusedLevelModels();
        return;
    }

    deselectAll();
    m_models.removeInstance(static_cast<size_t>(index)); // the parts were added after it
    releaseUnusedLevelModels();
    std::vector<int> selected;
    for (uint64_t id : createdIds) {
        const auto it = std::find_if(instances.begin(), instances.end(), [&](const ModelInstance& i) { return i.id == id; });
        it->name = uniqueInstanceName(source.name);
        selected.push_back(static_cast<int>(it - instances.begin()));
    }
    markSceneChanged();
    selectIndices(selected, selected.front());
    setStatus("Separated " + source.name + " into " + std::to_string(selected.size()) + " objects");
}

void Editor::saveSelectionAsPrefab()
{
    if (!hasSelection())
        return;
    const GPUModel* model = m_models.getModel(m_models.getInstances()[m_gizmo.selectedInstance].modelIndex);
    if (selectionCount() == 1 && model && model->polyMesh) {
        savePrefabDialog(m_gizmo.selectedInstance);
        return;
    }
    saveObjectPrefabDialog();
}

void Editor::uniteAndSavePrefab()
{
    int index = m_gizmo.selectedInstance;
    if (selectedLevelShapes().size() >= 2)
        index = uniteSelectedShapes();
    const GPUModel* model = validInstance(index) ? m_models.getModel(m_models.getInstances()[index].modelIndex) : nullptr;
    if (!model || !model->polyMesh) {
        if (validInstance(index))
            setStatus("Only level shapes can be saved as prefabs", true);
        return;
    }
    savePrefabDialog(index);
}

void Editor::drawUniteSection()
{
    const std::vector<int> shapes = selectedLevelShapes();
    const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    if (shapes.size() >= 2) {
        ImGui::Text("%zu level shapes selected", shapes.size());
        if (ImGui::Button("Unite", ImVec2(half, 0.0f)))
            uniteSelectedShapes();
        ImGui::SetItemTooltip("%s", withShortcut("Merge them into one object. Separate splits it again",
            EditorAction::UniteShapes).c_str());
        ImGui::SameLine();
        if (ImGui::Button("Unite and save...", ImVec2(half, 0.0f)))
            uniteAndSavePrefab();
        ImGui::SetItemTooltip("Merge them into one object and save it as a prefab, to place again from "
            "Add > Prefabs");
        if (ImGui::Button("Merge into one solid", ImVec2(-FLT_MIN, 0.0f)))
            uniteSelectedShapes(true);
        ImGui::SetItemTooltip("Weld them into a single volume: overlapping parts and the faces between them are "
            "removed, and flat neighbouring faces become one. Closed shapes only; can't be separated again");
        ImGui::Spacing();
        return;
    }
    if (shapes.size() != 1 || shapes.front() != m_gizmo.selectedInstance)
        return;
    const GPUModel* model = m_models.getModel(m_models.getInstances()[shapes.front()].modelIndex);
    const size_t parts = model->polyMesh->partCount();
    ImGui::BeginDisabled(parts < 2);
    const std::string separate = parts < 2 ? std::string("Separate") : "Separate (" + std::to_string(parts) + " parts)";
    if (ImGui::Button(separate.c_str(), ImVec2(half, 0.0f)))
        separateSelectedShape();
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("%s", withShortcut(parts < 2 ? "The shape is one piece" : "One object per connected part",
        EditorAction::SeparateShape).c_str());
    ImGui::SameLine();
    if (ImGui::Button("Save as prefab...", ImVec2(half, 0.0f)))
        savePrefabDialog(shapes.front());
    ImGui::SetItemTooltip("Save the shape's geometry and materials to a prefab file, to place again from Add > Prefabs");
    if (!selectedLevelMesh())
        return; // locked prefab instance
    ImGui::BeginDisabled(parts < 2);
    if (ImGui::Button("Merge parts", ImVec2(half, 0.0f)))
        mergeShapeParts();
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Weld the shape's parts into one volume, removing overlaps and the faces inside");
    ImGui::SameLine();
    if (ImGui::Button("Join flat faces", ImVec2(half, 0.0f)))
        mergeLevelFaces();
    ImGui::SetItemTooltip("Join neighbouring faces that lie in one plane and share material and UVs into single faces");
}

// ---------------------------------------------------------------------------------------------
// Part mode
// ---------------------------------------------------------------------------------------------

std::vector<uint32_t> Editor::selectedPartFaces()
{
    std::vector<uint32_t> faces;
    const PolyMesh* mesh = selectedLevelMesh();
    if (m_levelMode != LevelEditMode::Part || !mesh || m_levelSelectionInstance != m_gizmo.selectedInstance) {
        m_selectedParts.clear();
        return faces;
    }
    if (m_selectedParts.empty())
        return faces;
    std::vector<uint32_t> partOfFace;
    mesh->partIds(partOfFace);
    // One face per part; parts the mesh no longer has (e.g. after an undo) drop out.
    std::vector<uint32_t> parts;
    std::vector<uint32_t> kept;
    for (uint32_t f : m_selectedParts) {
        if (f >= partOfFace.size() || partOfFace[f] == UINT32_MAX ||
            std::find(parts.begin(), parts.end(), partOfFace[f]) != parts.end())
            continue;
        parts.push_back(partOfFace[f]);
        kept.push_back(f);
    }
    m_selectedParts = std::move(kept);
    for (uint32_t f = 0; f < partOfFace.size(); ++f)
        if (std::find(parts.begin(), parts.end(), partOfFace[f]) != parts.end())
            faces.push_back(f);
    return faces;
}

void Editor::selectPart(uint32_t face, bool toggle)
{
    if (m_levelSelectionInstance != m_gizmo.selectedInstance)
        m_selectedParts.clear();
    m_levelSelectionInstance = m_gizmo.selectedInstance;
    const PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || face >= mesh->faces.size())
        return;
    selectedPartFaces(); // drops parts left over from an undo
    std::vector<uint32_t> partOfFace;
    mesh->partIds(partOfFace);
    const auto same = std::find_if(m_selectedParts.begin(), m_selectedParts.end(),
        [&](uint32_t f) { return partOfFace[f] == partOfFace[face]; });
    if (!toggle)
        m_selectedParts.assign(1, face);
    else if (same != m_selectedParts.end())
        m_selectedParts.erase(same);
    else
        m_selectedParts.push_back(face);
}

void Editor::selectAllParts()
{
    const PolyMesh* mesh = selectedLevelMesh();
    if (!mesh)
        return;
    m_levelSelectionInstance = m_gizmo.selectedInstance;
    std::vector<uint32_t> partOfFace;
    const size_t count = mesh->partIds(partOfFace);
    std::vector<bool> seen(count, false);
    m_selectedParts.clear();
    for (uint32_t f = 0; f < partOfFace.size(); ++f) {
        if (partOfFace[f] == UINT32_MAX || seen[partOfFace[f]])
            continue;
        seen[partOfFace[f]] = true;
        m_selectedParts.push_back(f);
    }
}

bool Editor::selectedPartsCenter(glm::vec3& center)
{
    const std::vector<uint32_t> faces = selectedPartFaces();
    if (faces.empty())
        return false;
    glm::vec3 lo, hi;
    faceBounds(*selectedLevelMesh(), faces, lo, hi);
    center = (lo + hi) * 0.5f;
    return true;
}

glm::vec3 Editor::gizmoPivot()
{
    const ModelInstance& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    glm::vec3 center;
    if (selectedPartsCenter(center))
        return glm::vec3(instance.getTransformMatrix() * glm::vec4(center, 1.0f));
    return instance.position;
}

void Editor::beginPartDrag()
{
    m_partDrag.active = false;
    PolyMesh* mesh = selectedLevelMesh();
    std::vector<uint32_t> faces = selectedPartFaces();
    glm::vec3 center;
    if (!mesh || faces.empty() || !selectedPartsCenter(center))
        return;
    m_partDrag.active = true;
    m_partDrag.moved = false;
    m_partDrag.instance = m_gizmo.selectedInstance;
    m_partDrag.faces = std::move(faces);
    m_partDrag.center = center;
    m_partDrag.applied = glm::mat4(1.0f);
    m_partDrag.texelSizes = levelSlotTexelSizes(*mesh);
    m_partDrag.startMesh = *mesh;
}

void Editor::dragParts(float amount, int axis)
{
    PolyMesh* mesh = selectedLevelMesh();
    // Anything changing the selection mid-drag (undo, another object, mode) ends the drag.
    if (!mesh || m_gizmo.selectedInstance != m_partDrag.instance || m_levelMode != LevelEditMode::Part) {
        m_partDrag.active = false;
        return;
    }
    const glm::vec3 axisDir = gizmoAxisDirection(m_gizmo.activeAxis);
    const bool snap = snapActive();
    const auto snapTo = [](float value, float step) { return step > 0.0f ? std::round(value / step) * step : value; };
    const glm::mat4 model = m_models.getInstances()[m_partDrag.instance].getTransformMatrix();
    const glm::vec3 pivot(model * glm::vec4(m_partDrag.center, 1.0f));
    const glm::mat4 toPivot = glm::translate(glm::mat4(1.0f), pivot);
    const glm::mat4 fromPivot = glm::translate(glm::mat4(1.0f), -pivot);
    glm::mat4 world(1.0f);
    switch (m_gizmo.mode) {
    case GizmoMode::Translate: {
        // By whole cells of the shape's grid, so parts that were on it stay on it.
        float distance = amount / m_gizmo.pixelsPerUnit;
        if (snap) distance = snapTo(distance, m_partDrag.startMesh.gridSize);
        world = glm::translate(glm::mat4(1.0f), axisDir * distance);
        break;
    }
    case GizmoMode::Rotate: {
        float degrees = amount * 0.5f;
        if (snap) degrees = snapTo(degrees, m_snapRotate);
        world = toPivot * turnAbout(axisDir, degrees) * fromPivot;
        break;
    }
    case GizmoMode::Scale: {
        float delta = amount * 0.01f;
        if (snap) delta = snapTo(delta, m_snapScale);
        glm::vec3 factor(1.0f);
        factor[axis] = std::max(1.0f + delta, 0.01f);
        world = toPivot * glm::scale(glm::mat4(1.0f), factor) * fromPivot;
        break;
    }
    default:
        return;
    }
    if (world == m_partDrag.applied)
        return;
    m_partDrag.applied = world;
    *mesh = m_partDrag.startMesh;
    mesh->transformFacesKeepingUVs(m_partDrag.faces, glm::inverse(model) * world * model, m_partDrag.texelSizes);
    // Rebuilt once per frame in updateLevelHistory(); several motion events can arrive per frame.
    m_partDrag.moved = true;
}

void Editor::transformSelectedParts(const glm::mat4& world, const char* action)
{
    PolyMesh* mesh = selectedLevelMesh();
    const std::vector<uint32_t> faces = selectedPartFaces();
    if (!mesh || faces.empty() || m_partDrag.active)
        return;
    const glm::mat4 model = m_models.getInstances()[m_gizmo.selectedInstance].getTransformMatrix();
    mesh->transformFacesKeepingUVs(faces, glm::inverse(model) * world * model, levelSlotTexelSizes(*mesh));
    rebuildSelectedLevelModel(action);
}

void Editor::rotateSelectedParts(float degrees)
{
    glm::vec3 center;
    if (!selectedPartsCenter(center))
        return;
    const glm::vec3 pivot(m_models.getInstances()[m_gizmo.selectedInstance].getTransformMatrix() * glm::vec4(center, 1.0f));
    transformSelectedParts(glm::translate(glm::mat4(1.0f), pivot) * turnAbout(glm::vec3(0.0f, 1.0f, 0.0f), degrees) *
        glm::translate(glm::mat4(1.0f), -pivot), "rotate part");
    setStatus(std::string("Rotated ") + (m_selectedParts.size() == 1 ? "the part" : "the parts") +
        (degrees < 0.0f ? " clockwise" : " counter-clockwise"));
}

void Editor::duplicateSelectedParts()
{
    PolyMesh* mesh = selectedLevelMesh();
    const std::vector<uint32_t> faces = selectedPartFaces();
    if (!mesh || faces.empty() || m_partDrag.active)
        return;
    // Beside the originals along X, a whole number of grid cells away so the copies stay on the grid.
    glm::vec3 lo, hi;
    faceBounds(*mesh, faces, lo, hi);
    const float offset = std::max(std::ceil((hi.x - lo.x) / mesh->gridSize - 1e-4f), 1.0f) * mesh->gridSize;
    const std::vector<uint32_t> copies = mesh->copyFaces(faces);
    mesh->transformFacesKeepingUVs(copies, glm::translate(glm::mat4(1.0f), glm::vec3(offset, 0.0f, 0.0f)),
        levelSlotTexelSizes(*mesh));
    // The copies keep the order of `faces`, so each selected part's face maps to its copy.
    std::vector<uint32_t> selected;
    for (uint32_t face : m_selectedParts) {
        const auto at = std::find(faces.begin(), faces.end(), face);
        if (at != faces.end())
            selected.push_back(copies[static_cast<size_t>(at - faces.begin())]);
    }
    m_selectedParts = std::move(selected);
    rebuildSelectedLevelModel("duplicate part");
    setStatus("Duplicated " + std::to_string(m_selectedParts.size()) + (m_selectedParts.size() == 1 ? " part" : " parts"));
}

void Editor::deleteSelectedParts()
{
    PolyMesh* mesh = selectedLevelMesh();
    const std::vector<uint32_t> faces = selectedPartFaces();
    if (!mesh || faces.empty())
        return;
    std::vector<bool> remove(mesh->faces.size(), false);
    for (uint32_t f : faces)
        remove[f] = true;
    PolyMesh edited = *mesh;
    edited.faces.clear();
    for (uint32_t f = 0; f < mesh->faces.size(); ++f)
        if (!remove[f])
            edited.faces.push_back(mesh->faces[f]);
    edited.removeUnusedVertices();
    const size_t count = m_selectedParts.size();
    if (!commitLevelTopology(std::move(edited), "delete part"))
        return;
    m_selectedParts.clear();
    setStatus("Deleted " + std::to_string(count) + (count == 1 ? " part" : " parts"));
}

void Editor::detachSelectedParts()
{
    PolyMesh* mesh = selectedLevelMesh();
    const std::vector<uint32_t> faces = selectedPartFaces();
    if (!mesh || faces.empty() || m_partDrag.active)
        return;
    if (faces.size() == mesh->faces.size()) {
        setStatus("Every part is selected: use Separate, or move the whole object", true);
        return;
    }
    auto& instances = m_models.getInstances();
    const ModelInstance source = instances[m_gizmo.selectedInstance];
    const std::string modelName = m_models.getModel(source.modelIndex)->name;

    // Split the faces between what stays and what leaves; the leaving parts keep only the slots they use.
    std::vector<bool> leaves(mesh->faces.size(), false);
    for (uint32_t f : faces)
        leaves[f] = true;
    PolyMesh kept = *mesh;
    kept.faces.clear();
    PolyMesh leaving = kept;
    for (uint32_t f = 0; f < mesh->faces.size(); ++f)
        (leaves[f] ? leaving : kept).faces.push_back(mesh->faces[f]);
    kept.removeUnusedVertices();
    PolyMesh detached;
    detached.gridSize = mesh->gridSize;
    for (const PolyMesh& part : leaving.splitParts())
        detached.append(part);

    // Its origin at its bottom center, on the grid, like a new shape.
    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    for (const glm::vec3& p : detached.positions) {
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }
    const glm::vec3 center = glm::round(glm::vec3((lo.x + hi.x) * 0.5f, lo.y, (lo.z + hi.z) * 0.5f) /
        detached.gridSize) * detached.gridSize;
    detached.transformKeepingUVs(glm::translate(glm::mat4(1.0f), -center), levelSlotTexelSizes(detached));
    const auto modelIndex = createLevelModel(std::move(detached), modelName);
    if (!modelIndex)
        return;
    const size_t count = m_selectedParts.size();
    commitLevelTopology(std::move(kept), "detach part");
    m_selectedParts.clear();

    // The shape stays selected in part mode; the new object goes into the same undo step.
    const glm::vec3 position(source.getTransformMatrix() * glm::vec4(center, 1.0f));
    ModelInstance& created = instances[m_models.createInstance(*modelIndex, position, source.rotation, source.scale)];
    created.name = uniqueInstanceName(source.name);
    created.color = source.color;
    created.visible = source.visible;
    markSceneChanged();
    setStatus("Detached " + std::to_string(count) + (count == 1 ? " part" : " parts") + " into " + created.name);
}

bool Editor::enterPartMode(float mouseX, float mouseY)
{
    const int picked = objectUnderMouse(mouseX, mouseY);
    if (picked < 0)
        return false;
    const ModelInstance& instance = m_models.getInstances()[picked];
    const GPUModel* model = m_models.getModel(instance.modelIndex);
    if (!model || !model->polyMesh || instance.locked || model->polyMesh->partCount() < 2)
        return false;
    selectInstance(picked);
    m_levelMode = LevelEditMode::Part;
    const int face = pickLevelFace(mouseX, mouseY);
    if (face >= 0)
        selectPart(static_cast<uint32_t>(face), false);
    setStatus("Editing the parts of " + instance.name + ": click a part, drag the gizmo; " +
        m_keymap.shortcutLabel(EditorAction::Deselect) + " goes back");
    return true;
}

void Editor::drawPartControls(PolyMesh& mesh)
{
    const size_t parts = mesh.partCount();
    const bool selected = !selectedPartFaces().empty();
    ImGui::Text("%zu parts, %zu selected", parts, m_selectedParts.size());
    if (!selected) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("Click a part in the viewport to select it, Shift+click for more. The gizmo then moves, "
            "turns or scales it. Double-click a united shape to get here.");
        ImGui::PopStyleColor();
    }
    const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    if (ImGui::Button("Select all", ImVec2(half, 0.0f)))
        selectAllParts();
    ImGui::SameLine();
    ImGui::BeginDisabled(!selected);
    if (ImGui::Button("Duplicate", ImVec2(half, 0.0f)))
        duplicateSelectedParts();
    ImGui::SetItemTooltip("%s", withShortcut("A copy beside it, inside this shape", EditorAction::Duplicate).c_str());
    if (ImGui::Button("Delete", ImVec2(half, 0.0f)))
        deleteSelectedParts();
    ImGui::SetItemTooltip("%s", withShortcut("Remove the selected parts from the shape", EditorAction::Delete).c_str());
    ImGui::SameLine();
    if (ImGui::Button("Detach", ImVec2(half, 0.0f)))
        detachSelectedParts();
    ImGui::SetItemTooltip("%s", withShortcut("Move the selected parts out into an object of their own",
        EditorAction::SeparateShape).c_str());
    ImGui::EndDisabled();
}
