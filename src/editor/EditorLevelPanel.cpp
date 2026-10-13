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

glm::vec3 Editor::placementPoint() const
{
    // On the floor or ground under the middle of the view when it is near, else a few units in front of
    // the camera.
    glm::vec3 surface;
    if (surfacePoint(m_sceneView.width * 0.5f, m_sceneView.height * 0.5f, surface) &&
        glm::distance(surface, m_camera.position) < 50.0f) {
        surface.x = std::round(surface.x / m_gridSize) * m_gridSize;
        surface.z = std::round(surface.z / m_gridSize) * m_gridSize;
        return surface;
    }
    glm::vec3 point = glm::round((m_camera.position + getFront(m_camera) * 5.0f) / m_gridSize) * m_gridSize;
    point.y = std::max(point.y, 0.0f);
    return point;
}

void Editor::addLevelShape(PolyShape shape)
{
    PolyShapeParams params = m_newShape;
    params.shape = shape;
    const std::string name = uniqueInstanceName(kPolyShapeNames[static_cast<int>(shape)]);
    // Placed on a world grid point with its sides on its own grid, so they lie on world grid lines too.
    PolyMesh mesh = makePolyShapeOnGrid(params, m_gridSize, m_gridSnap);
    // New shapes are painted with the Materials panel's current material.
    if (m_materials.find(m_currentMaterial))
        levelSlotFor(mesh, m_currentMaterial); // slot 0, which every new face uses
    const auto modelIndex = createLevelModel(std::move(mesh), name);
    if (!modelIndex)
        return;
    const size_t newIndex = m_models.createInstance(*modelIndex, placementPoint());
    m_models.getInstances()[newIndex].name = name;
    markSceneChanged();
    selectInstance(static_cast<int>(newIndex));
    setStatus("Added " + name);
}

void Editor::drawShapeMenuItems()
{
    for (int i = 0; i < IM_ARRAYSIZE(kPolyShapeNames); ++i) {
        if (ImGui::MenuItem(kPolyShapeNames[i]))
            addLevelShape(static_cast<PolyShape>(i));
    }
}

void Editor::drawFaceGeometry(PolyMesh& mesh, uint32_t faceIndex)
{
    EditorStyle::propertyLabel("Distance");
    ImGui::SetNextItemWidth(-FLT_MIN);
    // Steps by one grid cell; typed values are rounded to whole cells while snapping is on.
    const float grid = mesh.gridSize;
    ImGui::InputFloat("##faceDistance", &m_faceOpDistance, grid, grid * 4.0f, "%.3g");
    if (ImGui::IsItemDeactivatedAfterEdit() && m_gridSnap)
        m_faceOpDistance = std::round(m_faceOpDistance / grid) * grid;
    m_faceOpDistance = std::clamp(m_faceOpDistance, -1000.0f, 1000.0f);
    ImGui::SetItemTooltip("Distance along the face normal, in object space");
    const float width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    const bool push = ImGui::Button("Push/Pull", ImVec2(width, 0.0f));
    ImGui::SetItemTooltip("Move the face along its normal; neighbouring faces stretch");
    ImGui::SameLine();
    const bool extrude = ImGui::Button("Extrude", ImVec2(width, 0.0f));
    ImGui::SetItemTooltip("Build new side faces out to the moved face. On a closed shape the new part merges with "
        "anything it overlaps, and a negative distance cuts into the shape instead");

    EditorStyle::propertyLabel("Inset");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    // Insets are usually a fraction of a cell, so this one steps by quarter cells and doesn't snap.
    ImGui::InputFloat("##faceInset", &m_faceInsetDistance, grid * 0.25f, grid, "%.3g");
    m_faceInsetDistance = std::clamp(m_faceInsetDistance, 0.0f, 1000.0f);
    ImGui::SetItemTooltip("How far the new edges sit inside the face's border, in object space");
    ImGui::SameLine();
    if (ImGui::Button("Inset", ImVec2(-FLT_MIN, 0.0f))) {
        if (mesh.insetFace(faceIndex, m_faceInsetDistance))
            rebuildSelectedLevelModel("inset");
        else
            setStatus("Inset too large for this face", true);
    }
    ImGui::SetItemTooltip("Add edges inside the face along its border; the inner part stays selected, ready to extrude");

    EditorStyle::propertyLabel("Draw");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    int drawShape = static_cast<int>(m_faceDrawShape);
    if (ImGui::Combo("##faceDrawShape", &drawShape, "Polygon\0Rectangle\0Circle\0")) {
        m_faceDrawShape = static_cast<FaceDrawShape>(drawShape);
        m_faceDraw.points.clear();
    }
    ImGui::SetItemTooltip("Shape to draw on the face; applying it divides the face along its outline");
    ImGui::SameLine();
    const bool drawing = faceDrawValid();
    if (ImGui::Button(drawing ? "Cancel##faceDraw" : "Draw##faceDraw", ImVec2(-FLT_MIN, 0.0f))) {
        m_faceDraw = {};
        if (!drawing) {
            m_faceDraw.active = true;
            m_faceDraw.instance = m_gizmo.selectedInstance;
            m_faceDraw.face = faceIndex;
        }
    }
    ImGui::SetItemTooltip("Draw on this face in the viewport; points snap to the shape's grid (Ctrl: no snap)");
    if (m_faceDrawShape == FaceDrawShape::Circle) {
        EditorStyle::propertyLabel("Sides");
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::SliderInt("##faceDrawSides", &m_faceDrawSegments, 3, 64);
    }
    if (drawing) {
        static constexpr const char* kHints[] = {
            "Click points on the face; click the first one to close. Border to border cuts the face. "
            "Enter: finish, Backspace: remove point, Esc: cancel",
            "Click two opposite corners. Esc: cancel",
            "Click the centre, then a point on the rim. Esc: cancel",
        };
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", kHints[static_cast<int>(m_faceDrawShape)]);
        ImGui::PopTextWrapPos();
    }

    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Keys: %s / %s push or pull the selected faces one cell, %s extrudes them, %s extrudes inwards.",
        m_keymap.shortcutLabel(EditorAction::FacePushOut).c_str(), m_keymap.shortcutLabel(EditorAction::FacePullIn).c_str(),
        m_keymap.shortcutLabel(EditorAction::FaceExtrude).c_str(),
        m_keymap.shortcutLabel(EditorAction::FaceExtrudeIn).c_str());
    ImGui::PopTextWrapPos();

    if (!push && !extrude)
        return;
    if (std::abs(m_faceOpDistance) < 1e-4f) {
        setStatus("Distance is zero", true);
        return;
    }
    // Every selected face, not just the active one.
    moveSelectedFaces(m_faceOpDistance, extrude);
}

void Editor::drawEdgeGeometry(PolyMesh& mesh, uint32_t a, uint32_t b)
{
    EditorStyle::propertyLabel("Distance");
    ImGui::SetNextItemWidth(-FLT_MIN);
    // Shared with the face tools; steps by one grid cell.
    const float grid = mesh.gridSize;
    ImGui::InputFloat("##edgeDistance", &m_faceOpDistance, grid, grid * 4.0f, "%.3g");
    if (ImGui::IsItemDeactivatedAfterEdit() && m_gridSnap)
        m_faceOpDistance = std::round(m_faceOpDistance / grid) * grid;
    m_faceOpDistance = std::clamp(m_faceOpDistance, -1000.0f, 1000.0f);
    ImGui::SetItemTooltip("Object space; with snapping on, the offset is rounded to the grid per axis");
    const float width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    const bool push = ImGui::Button("Push/Pull", ImVec2(width, 0.0f));
    ImGui::SetItemTooltip("Move the edge along the average normal of its faces, or outward within the face for an "
        "open border edge (resizing a plane); the faces stretch along");
    ImGui::SameLine();
    const bool extrude = ImGui::Button("Extrude", ImVec2(width, 0.0f));
    ImGui::SetItemTooltip("Build a new face out from the edge: on an open border it continues the surface, "
        "elsewhere it sticks out along the faces' normal as a two-sided fin");

    if (ImGui::Button("Split", ImVec2(width, 0.0f))) {
        const uint32_t mid = mesh.splitEdge(a, b);
        if (mid != UINT32_MAX) {
            m_selectedEdge = { a, mid };
            rebuildSelectedLevelModel("split edge");
            return;
        }
    }
    ImGui::SetItemTooltip("Add a vertex at the edge's middle; the first half stays selected");
    ImGui::SameLine();
    if (ImGui::Button("Select vertices", ImVec2(width, 0.0f))) {
        m_selectedVertices = { a, b };
        m_levelMode = LevelEditMode::Vertex;
        return;
    }
    ImGui::SetItemTooltip("Switch to vertex mode with the edge's two ends selected");

    if (!push && !extrude)
        return;
    const glm::vec3 direction = mesh.edgeExtrudeDirection(a, b);
    const glm::vec3 offset = edgeOffset(direction, m_faceOpDistance, grid, m_gridSnap);
    if (glm::length(offset) < 1e-4f) {
        setStatus("Distance is zero", true);
        return;
    }
    m_selectedEdge = editEdge(mesh, a, b, offset, extrude);
    rebuildSelectedLevelModel(push ? "push/pull edge" : "extrude edge");
}

void Editor::drawLevelMaterials(PolyMesh& mesh)
{
    bool changed = false;
    int removed = -1;
    for (int i = 0; i < static_cast<int>(mesh.materials.size()); ++i) {
        PolyMaterial& material = mesh.materials[i];
        ImGui::PushID(i);
        const std::string slotName = levelSlotName(mesh, static_cast<uint32_t>(i));
        if (ImGui::TreeNodeEx("##material", ImGuiTreeNodeFlags_DefaultOpen, "%d: %s", i, slotName.c_str())) {
            if (ImGui::SmallButton("Select faces")) {
                clearFaceSelection();
                for (uint32_t f = 0; f < mesh.faces.size(); ++f)
                    if (mesh.faces[f].material == static_cast<uint32_t>(i))
                        m_selectedFaces.push_back(f);
                if (!m_selectedFaces.empty()) {
                    m_selectedFace = static_cast<int>(m_selectedFaces.front());
                    m_levelSelectionInstance = m_gizmo.selectedInstance;
                    m_levelMode = LevelEditMode::Face;
                }
                setStatus(std::to_string(m_selectedFaces.size()) + " faces use " + slotName);
            }
            ImGui::SetItemTooltip("Switch to face mode with every face using this material selected");
            EditorStyle::propertyLabel("Shared");
            ImGui::SetNextItemWidth(-FLT_MIN);
            changed |= drawSharedMaterialCombo("##shared", material.materialPath, "(this shape only)");
            ImGui::SetItemTooltip("A shared material (.mat file) is the same on every object using it");
            if (!material.materialPath.empty()) {
                if (const MaterialAsset* shared = m_materials.find(material.materialPath)) {
                    MaterialAsset edited = *shared;
                    if (drawSharedMaterialProperties(edited))
                        saveSharedMaterial(edited);
                    ImGui::TextDisabled("%s", shared->path.c_str());
                }
                else {
                    ImGui::TextColored(EditorStyle::kError, "Missing: %s", material.materialPath.c_str());
                }
                if (ImGui::Button("Remove material"))
                    removed = i;
                ImGui::TreePop();
                ImGui::PopID();
                continue;
            }

            EditorStyle::propertyLabel("Color");
            ImGui::SetNextItemWidth(-FLT_MIN);
            changed |= ImGui::ColorEdit3("##color", &material.color.x);
            EditorStyle::propertyLabel("Roughness");
            ImGui::SetNextItemWidth(-FLT_MIN);
            changed |= ImGui::SliderFloat("##roughness", &material.roughness, 0.0f, 1.0f, "%.2f");
            EditorStyle::propertyLabel("Metallic");
            ImGui::SetNextItemWidth(-FLT_MIN);
            changed |= ImGui::SliderFloat("##metallic", &material.metallic, 0.0f, 1.0f, "%.2f");

            EditorStyle::propertyLabel("Texture");
            const std::string textureName = material.texturePath.empty() ? "None"
                : material.texturePath == kCheckerTexturePath ? "Checker"
                : std::filesystem::path(material.texturePath).filename().string();
            ImGui::TextUnformatted(textureName.c_str());
            if (!material.texturePath.empty() && ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", material.texturePath.c_str());
            if (ImGui::Button("Browse..."))
                openTextureDialog({ i, std::string(), false });
            ImGui::SameLine();
            if (ImGui::Button("Checker")) {
                material.texturePath = kCheckerTexturePath;
                changed = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear")) {
                material.texturePath.clear();
                changed = true;
            }
            if (ImGui::Button("Make shared")) {
                MaterialAsset initial;
                initial.color = material.color;
                initial.roughness = material.roughness;
                initial.metallic = material.metallic;
                initial.baseColorTexture = material.texturePath;
                const std::string name = m_models.getInstances()[m_gizmo.selectedInstance].name;
                if (auto created = m_materials.create(name, initial)) {
                    material.materialPath = *created;
                    changed = true;
                    setStatus("Saved " + *created);
                }
                else {
                    setStatus("Failed to create a material in " + m_materials.root(), true);
                }
            }
            ImGui::SetItemTooltip("Save this material as a .mat file other objects can use too");
            ImGui::SameLine();
            if (ImGui::Button("Remove material"))
                removed = i;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    if (removed >= 0) {
        mesh.materials.erase(mesh.materials.begin() + removed);
        // Later materials move down one slot; faces that used the removed one fall back to white.
        for (PolyFace& face : mesh.faces) {
            if (face.material == static_cast<uint32_t>(removed))
                face.material = kNoPolyMaterial;
            else if (face.material != kNoPolyMaterial && face.material > static_cast<uint32_t>(removed))
                --face.material;
        }
        changed = true;
    }
    if (mesh.materials.size() < kMaxPolyMaterialSlots &&ImGui::Button("Add material", ImVec2(-FLT_MIN, 0.0f))) {
        mesh.materials.push_back({});
        // A shape's first material goes on every face, which is nearly always what is wanted.
        if (mesh.materials.size() == 1)
            for (PolyFace& face : mesh.faces)
                face.material = 0;
        changed = true;
    }
    if (changed)
        rebuildSelectedLevelModel("material edit");
}

void Editor::openTextureDialog(TextureDialogTarget target)
{
    m_textureDialogTarget = std::move(target);
    openFileDialog("BrowseLevelTextureDlg", m_textureDialogTarget.normal ? "Choose Normal Map" : "Choose Texture",
        "Images", "png;jpg;jpeg;tga;bmp", kTexturesRoot, nullptr, false);
}

void Editor::drawLevelTextureDialog()
{
    std::string chosen;
    if (!m_fileDialog.take("BrowseLevelTextureDlg", chosen))
        return;
    const TextureDialogTarget& target = m_textureDialogTarget;
    if (!chosen.empty()) {
        const std::string texture = toStoredPath(chosen);
        if (!target.sharedPath.empty()) {
            if (const MaterialAsset* shared = m_materials.find(target.sharedPath)) {
                MaterialAsset edited = *shared;
                (target.normal ? edited.normalTexture : edited.baseColorTexture) = texture;
                saveSharedMaterial(edited);
            }
        }
        else if (PolyMesh* mesh = selectedLevelMesh();
                 mesh && target.slot >= 0 && target.slot < static_cast<int>(mesh->materials.size())) {
            mesh->materials[target.slot].texturePath = texture;
            rebuildSelectedLevelModel("set texture");
        }
    }
    m_textureDialogTarget = {};
}

void Editor::drawLevelPanel()
{
    m_levelClip.preview = false;
    m_levelMirror.preview = false;
    if (ImGui::Begin("Level", &m_showLevelPanel)) {
        ImGui::SeparatorText("New shape");
        int shape = static_cast<int>(m_newShape.shape);
        EditorStyle::propertyLabel("Shape");
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::Combo("##shape", &shape, kPolyShapeNames, IM_ARRAYSIZE(kPolyShapeNames)))
            m_newShape.shape = static_cast<PolyShape>(shape);

        EditorStyle::propertyLabel("Size");
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::DragFloat3("##size", &m_newShape.size.x, m_gridSize * 0.05f, 0.01f, 1000.0f, "%.3g");
        if (ImGui::IsItemDeactivatedAfterEdit() && m_gridSnap)
            m_newShape = snapPolyShapeSize(m_newShape, m_gridSize);
        ImGui::SetItemTooltip("Rounded to whole grid cells while snapping is on");
        switch (m_newShape.shape) {
        case PolyShape::Plane:
            EditorStyle::propertyLabel("Subdivisions");
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::SliderInt("##subdivisions", &m_newShape.subdivisions, 1, 64);
            ImGui::SetItemTooltip("Grid cells per side; 1 makes the plane a single face");
            break;
        case PolyShape::Cylinder:
        case PolyShape::Arch:
            EditorStyle::propertyLabel("Segments");
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::SliderInt("##segments", &m_newShape.segments, 3, 64);
            break;
        case PolyShape::Stairs:
            EditorStyle::propertyLabel("Steps");
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::SliderInt("##steps", &m_newShape.steps, 1, 32);
            break;
        default:
            break;
        }
        if (m_newShape.shape == PolyShape::Arch) {
            EditorStyle::propertyLabel("Thickness");
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::DragFloat("##thickness", &m_newShape.thickness, 0.01f, 0.01f, 100.0f, "%.2f");
        }

        const float createWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        if (ImGui::Button("Create", ImVec2(createWidth, 0.0f)))
            addLevelShape(m_newShape.shape);
        ImGui::SetItemTooltip("Add the shape on the floor in the middle of the view");
        ImGui::SameLine();
        const bool drawing = shapeDrawActive();
        if (drawing) {
            ImGui::PushStyleColor(ImGuiCol_Button, EditorStyle::kAccent);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, EditorStyle::kAccentHover);
        }
        if (ImGui::Button(drawing ? "Drawing..." : "Draw", ImVec2(-FLT_MIN, 0.0f)))
            toggleShapeDraw();
        if (drawing)
            ImGui::PopStyleColor(2);
        ImGui::SetItemTooltip("%s", withShortcut("Draw shapes in the viewport: drag a footprint on the ground or a "
            "floor, move the mouse to set the height, click. A click places this size. Esc stops",
            EditorAction::DrawShape).c_str());
        ImGui::Checkbox("Edit after drawing", &m_editAfterDraw);
        ImGui::SetItemTooltip("A drawn shape goes straight into editing (faces, edges, vertices) as one object; "
            "%s or Esc finishes. Off: the Draw tool stays on for the next shape.",
            m_keymap.shortcutLabel(EditorAction::EditShape).c_str());

        ImGui::SeparatorText("Place entities");
        drawEntityPalette();

        ImGui::SeparatorText("Selected");
        drawPrefabSection();
        drawUniteSection();
        if (PolyMesh* mesh = selectedLevelMesh()) {
            ImGui::Text("%zu vertices, %zu faces", mesh->positions.size(), mesh->faces.size());
            EditorStyle::propertyLabel("Shape grid");
            ImGui::SetNextItemWidth(-FLT_MIN);
            char gridLabel[16];
            snprintf(gridLabel, sizeof(gridLabel), "%g", mesh->gridSize);
            if (ImGui::BeginCombo("##shapeGrid", gridLabel)) {
                for (const float size : kGridSizes) {
                    snprintf(gridLabel, sizeof(gridLabel), "%g", size);
                    if (ImGui::Selectable(gridLabel, size == mesh->gridSize) && size != mesh->gridSize) {
                        mesh->gridSize = size;
                        rebuildSelectedLevelModel("shape grid size");
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SetItemTooltip("This shape's own grid (object space): face and vertex edits snap to it");
            // Rebuilding replaces any manual edits, so it only happens on request.
            if (ImGui::Button("Rebuild from shape settings", ImVec2(-FLT_MIN, 0.0f))) {
                // Materials survive; the new faces all start on material 0.
                PolyMesh rebuilt = makePolyShapeOnGrid(m_newShape, mesh->gridSize, m_gridSnap);
                rebuilt.materials = std::move(mesh->materials);
                *mesh = std::move(rebuilt);
                clearFaceSelection();
                m_selectedVertices.clear();
                m_edgeSelected = false;
                rebuildSelectedLevelModel("rebuild shape");
                setStatus("Rebuilt " + m_models.getInstances()[m_gizmo.selectedInstance].name);
            }

            ImGui::SeparatorText("Materials");
            drawLevelMaterials(*mesh);

            ImGui::SeparatorText("Edit");
            const bool editing = m_levelMode != LevelEditMode::Object;
            if (editing) {
                ImGui::PushStyleColor(ImGuiCol_Button, EditorStyle::kAccent);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, EditorStyle::kAccentHover);
            }
            if (ImGui::Button(editing ? "Done editing" : "Edit shape", ImVec2(-FLT_MIN, 0.0f)))
                toggleShapeEdit();
            if (editing)
                ImGui::PopStyleColor(2);
            ImGui::SetItemTooltip("%s", withShortcut(editing ? "Back to moving whole objects (Esc also does this once "
                "nothing is picked)" : "Edit this shape's faces, edges and vertices; it stays one object",
                EditorAction::EditShape).c_str());
            int mode = static_cast<int>(m_levelMode);
            // One row while it fits the panel, wrapping otherwise.
            const float rowRight = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
            const auto modeButton = [&](const char* label, LevelEditMode value, EditorAction shortcut, bool first) {
                const float width = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x +
                    ImGui::CalcTextSize(label).x;
                if (!first && ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + width <= rowRight)
                    ImGui::SameLine();
                ImGui::RadioButton(label, &mode, static_cast<int>(value));
                ImGui::SetItemTooltip("%s", m_keymap.shortcutLabel(shortcut).c_str());
            };
            modeButton("Object", LevelEditMode::Object, EditorAction::ObjectMode, true);
            modeButton("Faces", LevelEditMode::Face, EditorAction::FaceMode, false);
            modeButton("Edges", LevelEditMode::Edge, EditorAction::EdgeMode, false);
            modeButton("Vertices", LevelEditMode::Vertex, EditorAction::VertexMode, false);
            modeButton("Parts", LevelEditMode::Part, EditorAction::PartMode, false);
            if (mode != static_cast<int>(m_levelMode) && mode == static_cast<int>(LevelEditMode::Object))
                finishShapeEdit();
            m_levelMode = static_cast<LevelEditMode>(mode);
            if (m_levelMode != LevelEditMode::Object)
                m_lastShapeEditMode = m_levelMode;

            if (m_levelMode == LevelEditMode::Part)
                drawPartControls(*mesh);

            if (m_levelMode == LevelEditMode::Face) {
                if (PolyFace* face = selectedLevelFace()) {
                    const size_t selectedCount = selectedLevelFaces().size();
                    if (selectedCount > 1)
                        ImGui::Text("%zu faces (active: face %d)", selectedCount, m_selectedFace);
                    else
                        ImGui::Text("Face %d (%zu vertices)", m_selectedFace, face->verts.size());
                    const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
                    if (ImGui::Button("Select coplanar", ImVec2(half, 0.0f))) {
                        const glm::vec3 normal = mesh->faceNormal(*face);
                        const float offset = glm::dot(normal, mesh->faceCenter(*face));
                        selectFacesWhere([&](const PolyMesh& m, const PolyFace& other) {
                            return glm::dot(m.faceNormal(other), normal) > 0.999f &&
                                std::abs(glm::dot(normal, m.faceCenter(other)) - offset) < 1e-3f;
                        });
                    }
                    ImGui::SetItemTooltip("Add every face lying in the active face's plane");
                    ImGui::SameLine();
                    if (ImGui::Button("Same material", ImVec2(half, 0.0f))) {
                        const uint32_t material = face->material;
                        selectFacesWhere([material](const PolyMesh&, const PolyFace& other) { return other.material == material; });
                    }
                    ImGui::SetItemTooltip("Add every face with the active face's material");
                    if (ImGui::Button("Select surface", ImVec2(half, 0.0f)))
                        selectSurface();
                    ImGui::SetItemTooltip("%s", withShortcut("Add the whole flat surface around the selected faces "
                        "(faces joined by edges in the same plane). Also: double-click a face", EditorAction::SelectSurface).c_str());
                    ImGui::SameLine();
                    // Invalidates `face`; nothing below uses it.
                    if (ImGui::Button("Join flat faces", ImVec2(half, 0.0f)))
                        mergeLevelFaces();
                    ImGui::SetItemTooltip("Join neighbouring faces that lie in one plane and share material and UVs "
                        "into single faces, dropping the extra edges");
                    if (ImGui::Button("Copy UVs", ImVec2(half, 0.0f)))
                        copyFaceAttributes();
                    ImGui::SetItemTooltip("Copy the active face's material and UVs (%s)",
                        m_keymap.shortcutLabel(EditorAction::CopyFaceAttributes).c_str());
                    ImGui::SameLine();
                    ImGui::BeginDisabled(!m_faceClipboard.valid);
                    if (ImGui::Button("Paste UVs", ImVec2(half, 0.0f)))
                        pasteFaceAttributes();
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("Paste material and UVs onto the selected faces (%s)",
                            m_keymap.shortcutLabel(EditorAction::PasteFaceAttributes).c_str());

                    if (PolyFace* active = selectedLevelFace())
                        drawFaceProperties(*mesh, *active);
                    if (ImGui::CollapsingHeader("UV editor", ImGuiTreeNodeFlags_DefaultOpen))
                        drawUvEditor(*mesh);
                    ImGui::SeparatorText("Geometry");
                    drawFaceGeometry(*mesh, static_cast<uint32_t>(m_selectedFace));

                    const float width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
                    if (ImGui::Button("Flip", ImVec2(width, 0.0f))) {
                        for (uint32_t f : selectedLevelFaces())
                            mesh->flipFace(f);
                        rebuildSelectedLevelModel("flip face");
                    }
                    ImGui::SetItemTooltip("Turn the selected faces to point the other way");
                    ImGui::SameLine();
                    if (ImGui::Button("Delete", ImVec2(width, 0.0f)))
                        deleteLevelFace();
                    ImGui::SetItemTooltip("Remove the selected faces, leaving holes (Del)");
                    if (selectedLevelFace() && ImGui::Button("Select their vertices", ImVec2(-FLT_MIN, 0.0f))) {
                        m_selectedVertices.clear();
                        for (uint32_t f : selectedLevelFaces())
                            for (uint32_t v : mesh->faces[f].verts)
                                if (std::find(m_selectedVertices.begin(), m_selectedVertices.end(), v) == m_selectedVertices.end())
                                    m_selectedVertices.push_back(v);
                        m_levelMode = LevelEditMode::Vertex;
                    }
                }
                else {
                    ImGui::TextDisabled("Click a face of the shape in the viewport;");
                    ImGui::TextDisabled("Shift+click adds, double-click takes the whole surface.");
                    ImGui::TextDisabled("Drag to push/pull, Alt+drag to extrude (inwards cuts in).");
                }
            }
            else if (m_levelMode == LevelEditMode::Edge) {
                uint32_t a, b;
                if (selectedLevelEdge(a, b)) {
                    ImGui::Text("Edge %u-%u", a, b);
                    ImGui::SeparatorText("Geometry");
                    drawEdgeGeometry(*mesh, a, b);
                }
                else {
                    ImGui::TextDisabled("Click an edge of the shape in the viewport;");
                    ImGui::TextDisabled("drag to push/pull, Alt+drag to extrude.");
                }
            }
            else if (m_levelMode == LevelEditMode::Vertex) {
                const float width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
                if (ImGui::Button("Select all", ImVec2(width, 0.0f))) {
                    m_selectedVertices.resize(mesh->positions.size());
                    std::iota(m_selectedVertices.begin(), m_selectedVertices.end(), 0u);
                    m_levelSelectionInstance = m_gizmo.selectedInstance;
                }
                ImGui::SameLine();
                if (ImGui::Button("Select none", ImVec2(width, 0.0f)))
                    m_selectedVertices.clear();

                if (!selectedLevelVertices().empty()) {
                    drawVertexProperties(*mesh);

                    if (ImGui::Button("Snap to grid", ImVec2(-FLT_MIN, 0.0f)))
                        snapSelectedVertices();
                    ImGui::SetItemTooltip("Round the selected vertices to the shape's grid");

                    ImGui::BeginDisabled(m_selectedVertices.size() < 2);
                    if (ImGui::Button("Merge", ImVec2(width, 0.0f)))
                        mergeLevelVertices();
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("Collapse the selected vertices into one at their center");
                    ImGui::SameLine();
                    if (ImGui::Button("Delete", ImVec2(width, 0.0f)))
                        deleteLevelVertices();
                    ImGui::SetItemTooltip("Remove the vertices and every face using them (Del)");

                    // Re-checked per button: Split edge changes the selection to the new vertex.
                    auto pair = [this] { return m_selectedVertices.size() == 2; };
                    ImGui::BeginDisabled(!pair() || !mesh->isEdge(m_selectedVertices[0], m_selectedVertices[1]));
                    if (ImGui::Button("Split edge", ImVec2(width, 0.0f)))
                        splitLevelEdge();
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("Select the two ends of an edge to add a vertex at its middle");
                    ImGui::SameLine();
                    ImGui::BeginDisabled(!pair() || mesh->faceToConnect(m_selectedVertices[0], m_selectedVertices[1]) == UINT32_MAX);
                    if (ImGui::Button("Connect", ImVec2(width, 0.0f)))
                        connectLevelVertices();
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("Select two corners of one face, not next to each other, to cut it in two");
                }
                else {
                    ImGui::TextDisabled("Click a vertex or drag a box around some;");
                    ImGui::TextDisabled("Shift adds. Drag a vertex to move; Ctrl inverts snapping.");
                }
            }

            ImGui::Spacing();
            if (ImGui::CollapsingHeader("Clip")) {
                m_levelClip.preview = true;
                drawLevelClip(*mesh);
            }
            if (ImGui::CollapsingHeader("Mirror")) {
                m_levelMirror.preview = true;
                drawLevelMirror();
            }
            if (ImGui::CollapsingHeader("Hollow"))
                drawLevelHollow();
            if (ImGui::CollapsingHeader("Subtract"))
                drawLevelSubtract();
        }
        else if (!hasSelection() || !m_models.getInstances()[m_gizmo.selectedInstance].locked) {
            ImGui::TextDisabled("Select a level shape to edit it.");
        }
    }
    ImGui::End();
}
