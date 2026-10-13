#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <SDL3/SDL_keyboard.h>
#include <glm/gtc/constants.hpp>
#include "EditorStyle.h"
#include "FileDialog.h"
#include "engine/ModelManager.h"
#include "EditorLevelInternal.h"

void Editor::rebuildSelectedLevelModel(const char* action)
{
    // A drag keeps the action it started with.
    if (!m_levelEditPending)
        m_levelEditAction = action;
    m_levelEditPending = true;
    const ModelInstance& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    try {
        m_models.rebuildPolyMesh(instance.modelIndex);
    }
    catch (const std::exception& e) {
        setStatus("Failed to rebuild " + instance.name + ": " + e.what(), true);
    }
}

void Editor::drawFaceProperties(PolyMesh& mesh, PolyFace& face)
{
    // The widgets show the active face; whatever they change is copied to the rest of the selection.
    const std::vector<uint32_t> selection = selectedLevelFaces();
    const PolyFace before = face;
    bool changed = false;
    if (selection.size() > 1)
        ImGui::TextDisabled("Editing %zu faces (values of the active one shown)", selection.size());
    const int current = face.material < mesh.materials.size() ? static_cast<int>(face.material) : -1;
    EditorStyle::propertyLabel("Material");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##faceMaterial", levelSlotName(mesh, face.material).c_str())) {
        if (ImGui::Selectable(levelSlotName(mesh, kNoPolyMaterial).c_str(), current < 0)) {
            face.material = kNoPolyMaterial;
            changed = true;
        }
        for (int i = 0; i < static_cast<int>(mesh.materials.size()); ++i) {
            ImGui::PushID(i);
            const bool picked = ImGui::Selectable(levelSlotName(mesh, static_cast<uint32_t>(i)).c_str(), current == i);
            ImGui::PopID();
            if (picked) {
                face.material = static_cast<uint32_t>(i);
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::Button("Use on all faces", ImVec2(-FLT_MIN, 0.0f))) {
        for (PolyFace& other : mesh.faces)
            other.material = face.material;
        changed = true;
    }

    EditorStyle::propertyLabel("UV scale");
    ImGui::SetNextItemWidth(-FLT_MIN);
    changed |= ImGui::DragFloat2("##uvScale", &face.uvScale.x, 0.01f, 0.01f, 100.0f, "%.2f");
    EditorStyle::propertyLabel("UV offset");
    ImGui::SetNextItemWidth(-FLT_MIN);
    changed |= ImGui::DragFloat2("##uvOffset", &face.uvOffset.x, 0.01f, -100.0f, 100.0f, "%.2f");
    EditorStyle::propertyLabel("UV rotation");
    ImGui::SetNextItemWidth(-FLT_MIN);
    changed |= ImGui::DragFloat("##uvRotation", &face.uvRotation, 1.0f, -360.0f, 360.0f, "%.0f deg");
    if (ImGui::Button("Reset UVs", ImVec2(-FLT_MIN, 0.0f))) {
        face.uvScale = glm::vec2(1.0f);
        face.uvOffset = glm::vec2(0.0f);
        face.uvRotation = 0.0f;
        changed = true;
    }
    ImGui::SetItemTooltip("Scale 1: one texture repeat per material texel size, the same on every face");
    if (changed) {
        for (uint32_t f : selection) {
            PolyFace& other = mesh.faces[f];
            if (&other == &face)
                continue;
            if (face.material != before.material) other.material = face.material;
            if (face.uvScale != before.uvScale) other.uvScale = face.uvScale;
            if (face.uvOffset != before.uvOffset) other.uvOffset = face.uvOffset;
            if (face.uvRotation != before.uvRotation) other.uvRotation = face.uvRotation;
        }
    }
    // Each row: label, then three buttons; returns which one was pressed or -1.
    const auto buttonRow = [](const char* label, const char* const (&names)[3]) {
        EditorStyle::propertyLabel(label);
        const float third = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2.0f) / 3.0f;
        int pressed = -1;
        ImGui::PushID(label);
        for (int i = 0; i < 3; ++i) {
            if (i > 0)
                ImGui::SameLine();
            if (ImGui::Button(names[i], ImVec2(third, 0.0f)))
                pressed = i;
        }
        ImGui::PopID();
        return pressed;
    };
    // Fit, align and rotate work on each selected face by itself.
    if (const int fit = buttonRow("Fit", { "Both", "U", "V" }); fit >= 0) {
        for (uint32_t f : selection)
            mesh.fitFaceUVs(f, fit != 2, fit != 1, levelSlotTexelSize(mesh, mesh.faces[f].material));
        changed = true;
    }
    if (const int align = buttonRow("Align U", { "Left", "Center", "Right" }); align >= 0) {
        for (uint32_t f : selection)
            mesh.alignFaceUVs(f, glm::vec2(align * 0.5f, -1.0f), levelSlotTexelSize(mesh, mesh.faces[f].material));
        changed = true;
    }
    if (const int align = buttonRow("Align V", { "Top", "Center", "Bottom" }); align >= 0) {
        for (uint32_t f : selection)
            mesh.alignFaceUVs(f, glm::vec2(-1.0f, align * 0.5f), levelSlotTexelSize(mesh, mesh.faces[f].material));
        changed = true;
    }
    if (const int rotate = buttonRow("Rotate", { "-90", "+90", "180" }); rotate >= 0) {
        for (uint32_t f : selection) {
            float& rotation = mesh.faces[f].uvRotation;
            rotation += rotate == 0 ? -90.0f : rotate == 1 ? 90.0f : 180.0f;
            rotation = std::fmod(rotation + 540.0f, 360.0f) - 180.0f; // keep within [-180, 180)
        }
        changed = true;
    }
    ImGui::BeginDisabled(selection.size() < 2);
    if (ImGui::Button("Wrap from active face", ImVec2(-FLT_MIN, 0.0f)))
        wrapSelectedFaces();
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Continue the active face's texture across shared edges onto the other selected faces\n"
            "(Alt+Shift+click a neighbouring face does one step)");
    if (ImGui::Button("Use UVs on all faces", ImVec2(-FLT_MIN, 0.0f))) {
        // UVs are projected in object space, so faces on one plane still line up after this.
        for (PolyFace& other : mesh.faces) {
            other.uvScale = face.uvScale;
            other.uvOffset = face.uvOffset;
            other.uvRotation = face.uvRotation;
        }
        changed = true;
    }
    ImGui::SetItemTooltip("Copies scale, offset and rotation to every face of this object");

    if (changed)
        rebuildSelectedLevelModel("face edit");
}

void Editor::drawVertexProperties(PolyMesh& mesh)
{
    const std::vector<uint32_t>& vertices = m_selectedVertices;
    bool changed = false;
    if (vertices.size() == 1) {
        const uint32_t vertex = vertices[0];
        ImGui::Text("Vertex %u", vertex);
        EditorStyle::propertyLabel("Position");
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= ImGui::DragFloat3("##vertexPosition", &mesh.positions[vertex].x, 0.01f, -10000.0f, 10000.0f, "%.3f");
        ImGui::SetItemTooltip("Object space, before the object's scale and rotation");
        const auto faceCount = std::count_if(mesh.faces.begin(), mesh.faces.end(), [vertex](const PolyFace& face) {
            return std::find(face.verts.begin(), face.verts.end(), vertex) != face.verts.end();
        });
        ImGui::TextDisabled("Shared by %d faces", static_cast<int>(faceCount));
    }
    else {
        ImGui::Text("%zu vertices", vertices.size());
        glm::vec3 center(0.0f);
        for (uint32_t v : vertices)
            center += mesh.positions[v];
        center /= static_cast<float>(vertices.size());
        glm::vec3 edited = center;
        EditorStyle::propertyLabel("Center");
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::DragFloat3("##vertexCenter", &edited.x, 0.01f, -10000.0f, 10000.0f, "%.3f")) {
            const glm::vec3 delta = edited - center;
            for (uint32_t v : vertices)
                mesh.positions[v] += delta;
            changed = true;
        }
        ImGui::SetItemTooltip("Average position in object space; editing it moves all selected vertices");
    }

    if (ImGui::Button("Snap to grid", ImVec2(-FLT_MIN, 0.0f))) {
        for (uint32_t v : vertices)
            mesh.positions[v] = glm::round(mesh.positions[v] / mesh.gridSize) * mesh.gridSize;
        changed = true;
    }
    ImGui::SetItemTooltip("Round the positions to the shape's grid (object space)");
    if (changed)
        rebuildSelectedLevelModel("move vertex");
}

bool Editor::commitLevelTopology(PolyMesh&& edited, const char* action)
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh)
        return false;
    // An empty mesh has nothing to upload or pick; removing the object is the way to get rid of it.
    if (edited.faces.empty()) {
        setStatus("That would remove every face; delete the object instead", true);
        return false;
    }
    *mesh = std::move(edited);
    rebuildSelectedLevelModel(action);
    return true;
}

void Editor::deleteLevelFace()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || !selectedLevelFace())
        return;
    PolyMesh edited = *mesh;
    // Highest index first, so the indices still to delete stay valid.
    std::vector<uint32_t> faces = selectedLevelFaces();
    std::sort(faces.begin(), faces.end(), std::greater<>());
    for (uint32_t f : faces)
        edited.deleteFace(f);
    if (commitLevelTopology(std::move(edited), faces.size() > 1 ? "delete faces" : "delete face"))
        clearFaceSelection();
}

void Editor::deleteLevelVertices()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || selectedLevelVertices().empty())
        return;
    PolyMesh edited = *mesh;
    edited.deleteVertices(m_selectedVertices);
    if (commitLevelTopology(std::move(edited), "delete vertices"))
        m_selectedVertices.clear();
}

void Editor::mergeLevelVertices()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || selectedLevelVertices().size() < 2)
        return;
    PolyMesh edited = *mesh;
    const uint32_t merged = edited.mergeVertices(m_selectedVertices);
    if (!commitLevelTopology(std::move(edited), "merge vertices"))
        return;
    m_selectedVertices.clear();
    if (merged != UINT32_MAX)
        m_selectedVertices.push_back(merged);
}

void Editor::splitLevelEdge()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || selectedLevelVertices().size() != 2)
        return;
    const uint32_t mid = mesh->splitEdge(m_selectedVertices[0], m_selectedVertices[1]);
    if (mid == UINT32_MAX)
        return;
    rebuildSelectedLevelModel("split edge");
    // Selecting just the new vertex lets it be dragged straight away.
    m_selectedVertices = { mid };
}

void Editor::connectLevelVertices()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || selectedLevelVertices().size() != 2)
        return;
    if (mesh->connectVertices(m_selectedVertices[0], m_selectedVertices[1]) != UINT32_MAX)
        rebuildSelectedLevelModel("connect vertices");
}

void Editor::drawLevelClip(PolyMesh& mesh)
{
    LevelClip& clip = m_levelClip;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    EditorStyle::propertyLabel("Axis");
    const float third = (ImGui::GetContentRegionAvail().x - spacing * 2.0f) / 3.0f;
    constexpr const char* kAxes[] = { "X", "Y", "Z" };
    for (int i = 0; i < 3; ++i) {
        if (i > 0)
            ImGui::SameLine();
        if (ImGui::Button(kAxes[i], ImVec2(third, 0.0f))) {
            clip.normal = glm::vec3(0.0f);
            clip.normal[i] = 1.0f;
            const glm::vec2 range = meshExtent(mesh, clip.normal);
            clip.offset = (range.x + range.y) * 0.5f;
        }
    }

    // Worked out from the normal each frame, so a plane taken from vertices shows up here too.
    float angles[2] = { glm::degrees(std::atan2(clip.normal.x, clip.normal.z)),
        glm::degrees(std::asin(glm::clamp(clip.normal.y, -1.0f, 1.0f))) };
    EditorStyle::propertyLabel("Yaw, pitch");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::DragFloat2("##clipAngles", angles, 0.5f, -180.0f, 180.0f, "%.1f")) {
        const float yaw = glm::radians(angles[0]);
        const float pitch = glm::radians(glm::clamp(angles[1], -90.0f, 90.0f));
        clip.normal = { std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw) };
    }

    const glm::vec2 range = meshExtent(mesh, clip.normal);
    EditorStyle::propertyLabel("Position");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::DragFloat("##clipOffset", &clip.offset, 0.01f, range.x, range.y, "%.3f");

    const float half = (ImGui::GetContentRegionAvail().x - spacing) * 0.5f;
    if (ImGui::Button("Flip side", ImVec2(half, 0.0f))) {
        clip.normal = -clip.normal;
        clip.offset = -clip.offset;
    }
    ImGui::SetItemTooltip("Swap front and back");
    ImGui::SameLine();
    ImGui::BeginDisabled(selectedLevelVertices().size() < 2);
    if (ImGui::Button("From vertices", ImVec2(half, 0.0f)))
        setLevelClipFromVertices();
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("In vertex mode: two selected vertices give a plane through them along the view\n"
                          "direction, three or more a plane through the first three");

    const bool cuts = clip.offset > range.x + 1e-4f && clip.offset < range.y - 1e-4f;
    if (!cuts)
        ImGui::TextDisabled("The plane misses the shape.");
    ImGui::BeginDisabled(!cuts);
    const float width = (ImGui::GetContentRegionAvail().x - spacing * 2.0f) / 3.0f;
    if (ImGui::Button("Keep back", ImVec2(width, 0.0f)))
        applyLevelClip(ClipKeep::Back);
    ImGui::SetItemTooltip("Remove the part the arrow points into");
    ImGui::SameLine();
    if (ImGui::Button("Keep front", ImVec2(width, 0.0f)))
        applyLevelClip(ClipKeep::Front);
    ImGui::SetItemTooltip("Remove the part behind the arrow");
    ImGui::SameLine();
    if (ImGui::Button("Split", ImVec2(width, 0.0f)))
        applyLevelClip(ClipKeep::Both);
    ImGui::SetItemTooltip("Cut in two; the front part becomes a new object");
    ImGui::EndDisabled();
}

void Editor::setLevelClipFromVertices()
{
    const PolyMesh* mesh = selectedLevelMesh();
    const std::vector<uint32_t>& verts = selectedLevelVertices();
    if (!mesh || verts.size() < 2)
        return;
    const glm::vec3 a = mesh->positions[verts[0]];
    const glm::vec3 b = mesh->positions[verts[1]];
    glm::vec3 normal(0.0f);
    if (verts.size() >= 3)
        normal = glm::cross(b - a, mesh->positions[verts[2]] - a);
    if (glm::length(normal) < 1e-6f) {
        // Edge-on from the camera, so the cut follows the line between the two vertices on screen.
        const glm::mat4 toObject = glm::inverse(m_models.getInstances()[m_gizmo.selectedInstance].getTransformMatrix());
        const glm::vec3 view = glm::vec3(toObject * glm::vec4(getFront(m_camera), 0.0f));
        normal = glm::cross(b - a, view);
    }
    if (glm::length(normal) < 1e-6f) {
        setStatus("Those vertices don't define a plane", true);
        return;
    }
    m_levelClip.normal = glm::normalize(normal);
    m_levelClip.offset = glm::dot(m_levelClip.normal, a);
}

void Editor::applyLevelClip(ClipKeep keep)
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh)
        return;
    PolyMesh back = *mesh;
    back.clip(m_levelClip.normal, m_levelClip.offset);
    PolyMesh front = *mesh;
    front.clip(-m_levelClip.normal, -m_levelClip.offset);
    if (back.faces.empty() || front.faces.empty()) {
        setStatus("The plane doesn't cut the shape", true);
        return;
    }
    const ModelInstance source = m_models.getInstances()[m_gizmo.selectedInstance];
    if (!commitLevelTopology(std::move(keep == ClipKeep::Front ? front : back), "clip"))
        return;
    clearFaceSelection();
    m_selectedVertices.clear();
    if (keep != ClipKeep::Both)
        return;

    // The front part becomes its own shape with the same transform, so it stays where it was.
    const std::string name = uniqueInstanceName(source.name);
    const auto modelIndex = createLevelModel(std::move(front), name);
    if (!modelIndex)
        return;
    const size_t newIndex = m_models.createInstance(*modelIndex, source.position, source.rotation, source.scale);
    m_models.getInstances()[newIndex].name = name;
    m_models.getInstances()[newIndex].color = source.color;
    markSceneChanged();
    setStatus("Split " + source.name + "; the front part is " + name);
}

float Editor::levelMirrorPivot(const PolyMesh& mesh) const
{
    glm::vec3 dir(0.0f);
    dir[m_levelMirror.axis] = 1.0f;
    const glm::vec2 range = meshExtent(mesh, dir);
    switch (m_levelMirror.pivot) {
    case MirrorPivot::Origin: return 0.0f;
    case MirrorPivot::Min: return range.x;
    case MirrorPivot::Max: return range.y;
    default: return (range.x + range.y) * 0.5f;
    }
}

void Editor::drawLevelMirror()
{
    LevelMirror& mirror = m_levelMirror;
    constexpr const char* kAxes[] = { "X", "Y", "Z" };
    EditorStyle::propertyLabel("Axis");
    for (int i = 0; i < 3; ++i) {
        if (i > 0)
            ImGui::SameLine();
        ImGui::PushID(i);
        ImGui::RadioButton(kAxes[i], &mirror.axis, i);
        ImGui::PopID();
    }

    constexpr const char* kPivots[] = { "Origin", "Center", "Min side", "Max side" };
    int pivot = static_cast<int>(mirror.pivot);
    EditorStyle::propertyLabel("Plane at");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::Combo("##mirrorPivot", &pivot, kPivots, IM_ARRAYSIZE(kPivots)))
        mirror.pivot = static_cast<MirrorPivot>(pivot);
    ImGui::SetItemTooltip("Object origin, middle of the shape, or one of its two ends along the axis");

    const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    // "##inPlace": the section header is also called "Mirror" and shares this ID scope.
    if (ImGui::Button("Mirror##inPlace", ImVec2(half, 0.0f)))
        applyLevelMirror(MirrorAction::InPlace);
    ImGui::SetItemTooltip("Flip the shape across the plane");
    ImGui::SameLine();
    if (ImGui::Button("Mirror copy", ImVec2(half, 0.0f)))
        applyLevelMirror(MirrorAction::Copy);
    ImGui::SetItemTooltip("Add a flipped copy as a new object; with Min/Max side it ends up next to this one");

    // Symmetrize keeps one half and replaces the other with its reflection.
    const char* axis = kAxes[mirror.axis];
    const std::string toNegative = std::string("+") + axis + " to -" + axis;
    const std::string toPositive = std::string("-") + axis + " to +" + axis;
    if (ImGui::Button(toNegative.c_str(), ImVec2(half, 0.0f)))
        applyLevelMirror(MirrorAction::SymmetrizePositive);
    ImGui::SetItemTooltip("Symmetrize: replace the -%s half with a mirror image of the +%s half", axis, axis);
    ImGui::SameLine();
    if (ImGui::Button(toPositive.c_str(), ImVec2(half, 0.0f)))
        applyLevelMirror(MirrorAction::SymmetrizeNegative);
    ImGui::SetItemTooltip("Symmetrize: replace the +%s half with a mirror image of the -%s half", axis, axis);
}

void Editor::applyLevelMirror(MirrorAction action)
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh)
        return;
    const int axis = m_levelMirror.axis;
    const float pivot = levelMirrorPivot(*mesh);
    if (action == MirrorAction::InPlace) {
        // Vertex and face indices stay the same, so the selection survives.
        mesh->mirror(axis, pivot);
        rebuildSelectedLevelModel("mirror");
        return;
    }

    PolyMesh edited = *mesh;
    if (action == MirrorAction::Copy) {
        edited.mirror(axis, pivot);
        const ModelInstance source = m_models.getInstances()[m_gizmo.selectedInstance];
        const std::string name = uniqueInstanceName(source.name);
        const auto modelIndex = createLevelModel(std::move(edited), name);
        if (!modelIndex)
            return;
        const size_t newIndex = m_models.createInstance(*modelIndex, source.position, source.rotation, source.scale);
        m_models.getInstances()[newIndex].name = name;
        m_models.getInstances()[newIndex].color = source.color;
        markSceneChanged();
        selectInstance(static_cast<int>(newIndex));
        setStatus("Added mirrored copy " + name);
        return;
    }

    if (!edited.symmetrize(axis, pivot, action == MirrorAction::SymmetrizeNegative)) {
        setStatus("Nothing on that side of the plane to mirror", true);
        return;
    }
    if (commitLevelTopology(std::move(edited), "symmetrize")) {
        clearFaceSelection();
        m_selectedVertices.clear();
    }
}

void Editor::drawLevelHollow()
{
    EditorStyle::propertyLabel("Thickness");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::DragFloat("##hollowThickness", &m_hollowThickness, 0.01f, 0.01f, 100.0f, "%.2f");
    ImGui::SetItemTooltip("Wall thickness, measured inward from the current surface");
    // "##apply": the section header is also called "Hollow".
    if (ImGui::Button("Hollow##apply", ImVec2(-FLT_MIN, 0.0f))) {
        PolyMesh* mesh = selectedLevelMesh();
        if (!mesh)
            return;
        PolyMesh edited = *mesh;
        edited.hollow(m_hollowThickness);
        if (commitLevelTopology(std::move(edited), "hollow")) {
            clearFaceSelection();
            m_selectedVertices.clear();
        }
    }
    ImGui::SetItemTooltip("Turn the shape into walls facing inward, to make a room; cut doors with Subtract");
}

int Editor::levelCutterInstance() const
{
    const auto& instances = m_models.getInstances();
    const auto& models = m_models.getModels();
    for (int i = 0; i < static_cast<int>(instances.size()); ++i) {
        if (instances[i].name != m_levelCutterName)
            continue;
        const bool isLevel = instances[i].modelIndex < models.size() && models[instances[i].modelIndex] &&
            models[instances[i].modelIndex]->polyMesh;
        return isLevel && i != m_gizmo.selectedInstance ? i : -1;
    }
    return -1;
}

void Editor::drawLevelSubtract()
{
    const auto& instances = m_models.getInstances();
    const auto& models = m_models.getModels();
    const int cutter = levelCutterInstance();
    EditorStyle::propertyLabel("Cutter");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##cutter", cutter >= 0 ? m_levelCutterName.c_str() : "Choose an object")) {
        for (int i = 0; i < static_cast<int>(instances.size()); ++i) {
            const size_t model = instances[i].modelIndex;
            if (i == m_gizmo.selectedInstance || model >= models.size() || !models[model] || !models[model]->polyMesh)
                continue;
            ImGui::PushID(i);
            if (ImGui::Selectable(instances[i].name.c_str(), i == cutter))
                m_levelCutterName = instances[i].name;
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("Level object whose volume is cut away; place it overlapping the selected one");
    ImGui::Checkbox("Delete cutter afterwards", &m_deleteCutter);
    ImGui::SetItemTooltip("Undo restores the cut object but not a deleted cutter");

    ImGui::BeginDisabled(cutter < 0);
    // "##apply": the section header is also called "Subtract".
    if (ImGui::Button("Subtract##apply", ImVec2(-FLT_MIN, 0.0f)))
        applyLevelSubtract();
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Cut the cutter's volume out of the selected object; both should be closed shapes");
}

void Editor::applyLevelSubtract()
{
    PolyMesh* mesh = selectedLevelMesh();
    const int cutter = levelCutterInstance();
    if (!mesh || cutter < 0)
        return;
    const int target = m_gizmo.selectedInstance;
    const auto& instances = m_models.getInstances();
    const std::string targetName = instances[target].name;
    const std::string cutterName = instances[cutter].name;

    // Into the selected object's space, which is where its mesh lives.
    PolyMesh cut = *m_models.getModels()[instances[cutter].modelIndex]->polyMesh;
    cut.transform(glm::inverse(instances[target].getTransformMatrix()) * instances[cutter].getTransformMatrix());

    // Separate shapes would come back unchanged apart from needless splits.
    const auto bounds = [](const PolyMesh& m, glm::vec3& lo, glm::vec3& hi) {
        lo = glm::vec3(FLT_MAX);
        hi = glm::vec3(-FLT_MAX);
        for (const glm::vec3& p : m.positions) {
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
        }
    };
    glm::vec3 targetLo, targetHi, cutLo, cutHi;
    bounds(*mesh, targetLo, targetHi);
    bounds(cut, cutLo, cutHi);
    if (glm::any(glm::lessThan(cutHi, targetLo)) || glm::any(glm::greaterThan(cutLo, targetHi))) {
        setStatus(cutterName + " doesn't overlap " + targetName, true);
        return;
    }

    if (!commitLevelTopology(polyMeshSubtract(*mesh, cut), "subtract"))
        return;
    clearFaceSelection();
    m_selectedVertices.clear();
    if (m_deleteCutter) {
        m_models.removeInstance(static_cast<size_t>(cutter));
        releaseUnusedLevelModels();
        markSceneChanged();
        selectInstance(cutter < target ? target - 1 : target);
        m_levelCutterName.clear();
    }
    setStatus("Subtracted " + cutterName + " from " + targetName);
}

bool Editor::deleteLevelSelection()
{
    // In face/edge/vertex mode Del never removes the whole object, even with nothing picked.
    if (m_levelMode == LevelEditMode::Object || !selectedLevelMesh())
        return false;
    if (m_vertexDrag.active || m_faceDrag.active || m_edgeDrag.active || m_partDrag.active || m_levelEditPending)
        return true;
    if (m_levelMode == LevelEditMode::Edge)
        return true; // edges have no delete of their own
    if (m_levelMode == LevelEditMode::Part)
        deleteSelectedParts();
    else if (m_levelMode == LevelEditMode::Face)
        deleteLevelFace();
    else
        deleteLevelVertices();
    return true;
}

std::optional<size_t> Editor::createLevelModel(PolyMesh mesh, const std::string& name)
{
    try {
        return m_models.addPolyMesh(std::move(mesh), name);
    }
    catch (const std::exception& e) {
        setStatus(std::string("Failed to create ") + name + ": " + e.what(), true);
        return std::nullopt;
    }
}

void Editor::releaseUnusedLevelModels()
{
    const auto& instances = m_models.getInstances();
    // Backwards, because unloading shifts the indices of later models.
    for (size_t i = m_models.getModels().size(); i-- > 0;) {
        const GPUModel* model = m_models.getModel(i);
        if (!model || !model->polyMesh)
            continue;
        const bool used = std::any_of(instances.begin(), instances.end(),
            [i](const ModelInstance& instance) { return instance.modelIndex == i; });
        if (!used)
            m_models.unloadModel(i);
    }
}
