#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <SDL3/SDL_keyboard.h>
#include "EditorStyle.h"
#include "FileDialog.h"
#include "engine/ModelManager.h"

namespace {
constexpr const char* kTexturesRoot = "textures";
constexpr size_t kMaxPolyMaterials = 255; // below kNoPolyMaterial and the scene reader's cap
constexpr size_t kMaxUndoSteps = 100;
constexpr ImU32 kFaceOutline = IM_COL32(255, 160, 40, 255);
constexpr ImU32 kFaceEdges = IM_COL32(255, 255, 255, 70);
constexpr ImU32 kVertexDot = IM_COL32(255, 255, 255, 200);
constexpr ImU32 kClipOutline = IM_COL32(80, 200, 255, 255);
constexpr ImU32 kMirrorOutline = IM_COL32(230, 110, 255, 255);
constexpr float kVertexPickRadius = 10.0f;

bool rayHitsPlane(const Ray& ray, const glm::vec3& point, const glm::vec3& normal, glm::vec3& hit)
{
    const float denom = glm::dot(ray.direction, normal);
    if (std::abs(denom) < 1e-6f)
        return false;
    const float t = glm::dot(point - ray.origin, normal) / denom;
    if (t <= 0.0f)
        return false;
    hit = ray.origin + ray.direction * t;
    return true;
}

// Smallest and largest dot(normal, p) over the mesh's vertices.
glm::vec2 meshExtent(const PolyMesh& mesh, const glm::vec3& normal)
{
    glm::vec2 range(FLT_MAX, -FLT_MAX);
    for (const glm::vec3& p : mesh.positions) {
        const float d = glm::dot(normal, p);
        range = { std::min(range.x, d), std::max(range.y, d) };
    }
    return mesh.positions.empty() ? glm::vec2(0.0f) : range;
}
}

PolyMesh* Editor::selectedLevelMesh()
{
    if (!hasSelection())
        return nullptr;
    const ModelInstance& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    if (instance.locked)
        return nullptr;
    GPUModel* model = m_models.getModel(instance.modelIndex);
    return model ? model->polyMesh.get() : nullptr;
}

PolyFace* Editor::selectedLevelFace()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (m_levelMode != LevelEditMode::Face || !mesh || m_levelSelectionInstance != m_gizmo.selectedInstance ||
        m_selectedFace < 0 || m_selectedFace >= static_cast<int>(mesh->faces.size()))
        return nullptr;
    return &mesh->faces[m_selectedFace];
}

const std::vector<uint32_t>& Editor::selectedLevelVertices()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (m_levelMode != LevelEditMode::Vertex || !mesh || m_levelSelectionInstance != m_gizmo.selectedInstance) {
        m_selectedVertices.clear();
        return m_selectedVertices;
    }
    // Undo can shrink the mesh under the selection.
    const size_t count = mesh->positions.size();
    std::erase_if(m_selectedVertices, [count](uint32_t v) { return v >= count; });
    return m_selectedVertices;
}

int Editor::pickLevelFace(float mouseX, float mouseY) const
{
    if (!hasSelection())
        return -1;
    const ModelInstance& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    const GPUModel* model = m_models.getModel(instance.modelIndex);
    if (!model || !model->polyMesh || !instance.visible || instance.locked)
        return -1;
    const PolyMesh& mesh = *model->polyMesh;

    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height,
        getView(m_camera), sceneProjection());
    // Unnormalized local direction, as in pickSubmesh().
    const glm::mat4 invTransform = glm::inverse(instance.getTransformMatrix());
    Ray localRay;
    localRay.origin = glm::vec3(invTransform * glm::vec4(ray.origin, 1.0f));
    localRay.direction = glm::vec3(invTransform * glm::vec4(ray.direction, 0.0f));

    int best = -1;
    float bestT = FLT_MAX;
    std::vector<std::array<uint32_t, 3>> tris;
    for (size_t f = 0; f < mesh.faces.size(); ++f) {
        const PolyFace& face = mesh.faces[f];
        tris.clear();
        mesh.triangulate(face, tris);
        for (const auto& tri : tris) {
            float t;
            if (rayIntersectsTriangle(localRay, mesh.positions[face.verts[tri[0]]], mesh.positions[face.verts[tri[1]]],
                    mesh.positions[face.verts[tri[2]]], t) && t < bestT) {
                bestT = t;
                best = static_cast<int>(f);
            }
        }
    }
    return best;
}

int Editor::pickLevelVertex(float mouseX, float mouseY) const
{
    if (!hasSelection())
        return -1;
    const ModelInstance& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    const GPUModel* model = m_models.getModel(instance.modelIndex);
    if (!model || !model->polyMesh || !instance.visible || instance.locked)
        return -1;
    const PolyMesh& mesh = *model->polyMesh;

    const glm::mat4 mvp = sceneProjection() * getView(m_camera) * instance.getTransformMatrix();
    const glm::vec2 mouse(mouseX, mouseY);
    int best = -1;
    float bestDistance = kVertexPickRadius;
    float bestDepth = FLT_MAX;
    for (size_t v = 0; v < mesh.positions.size(); ++v) {
        const glm::vec4 clip = mvp * glm::vec4(mesh.positions[v], 1.0f);
        if (clip.w <= 1e-4f)
            continue;
        const glm::vec2 screen = worldToScreen(mesh.positions[v], mvp, m_sceneView.width, m_sceneView.height);
        const float distance = glm::length(screen - mouse);
        // Vertices drawn on top of each other (front and back corner of a box) go to the nearer one.
        const bool sameSpot = best >= 0 && std::abs(distance - bestDistance) < 1.0f;
        if (sameSpot ? clip.w < bestDepth : distance < bestDistance) {
            best = static_cast<int>(v);
            bestDistance = distance;
            bestDepth = clip.w;
        }
    }
    return best;
}

bool Editor::handleLevelClick(float mouseX, float mouseY)
{
    if (m_levelMode == LevelEditMode::Face) {
        m_selectedFace = pickLevelFace(mouseX, mouseY);
        if (m_selectedFace < 0)
            return false;
        m_levelSelectionInstance = m_gizmo.selectedInstance;
        return true;
    }

    const bool shift = (SDL_GetModState() & SDL_KMOD_SHIFT) != 0;
    selectedLevelVertices(); // drops a selection left over from another instance or mode
    const int picked = pickLevelVertex(mouseX, mouseY);
    if (picked < 0) {
        if (!selectedLevelMesh()) {
            m_selectedVertices.clear();
            return false;
        }
        // Empty space starts a box selection; without a drag, the release still picks objects.
        m_vertexMarquee = { true, shift, glm::vec2(mouseX, mouseY), glm::vec2(mouseX, mouseY) };
        return true;
    }
    const uint32_t vertex = static_cast<uint32_t>(picked);
    m_levelSelectionInstance = m_gizmo.selectedInstance;
    const auto found = std::find(m_selectedVertices.begin(), m_selectedVertices.end(), vertex);
    if (shift && found != m_selectedVertices.end()) {
        m_selectedVertices.erase(found);
        return true;
    }
    if (shift)
        m_selectedVertices.push_back(vertex);
    else if (found == m_selectedVertices.end())
        m_selectedVertices.assign(1, vertex);
    // A plain click on an already selected vertex keeps the selection, so the group can be dragged.

    const ModelInstance& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    const PolyMesh& mesh = *m_models.getModel(instance.modelIndex)->polyMesh;
    const glm::vec3 world = glm::vec3(instance.getTransformMatrix() * glm::vec4(mesh.positions[vertex], 1.0f));
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height,
        getView(m_camera), sceneProjection());
    glm::vec3 hit;
    if (rayHitsPlane(ray, world, getFront(m_camera), hit)) {
        m_vertexDrag.active = true;
        m_vertexDrag.moved = false;
        m_vertexDrag.planePoint = world;
        m_vertexDrag.grabOffset = world - hit;
        m_vertexDrag.startPositions.clear();
        for (uint32_t v : m_selectedVertices)
            m_vertexDrag.startPositions.push_back(mesh.positions[v]);
    }
    return true;
}

void Editor::dragLevelVertex(float mouseX, float mouseY)
{
    PolyMesh* mesh = selectedLevelMesh();
    const std::vector<uint32_t>& vertices = selectedLevelVertices();
    // Anything changing the selection mid-drag (deletion, undo) ends the drag.
    if (!mesh || vertices.empty() || vertices.size() != m_vertexDrag.startPositions.size()) {
        m_vertexDrag.active = false;
        return;
    }
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height,
        getView(m_camera), sceneProjection());
    glm::vec3 world;
    if (!rayHitsPlane(ray, m_vertexDrag.planePoint, getFront(m_camera), world))
        return;
    world += m_vertexDrag.grabOffset;
    // Ctrl snaps to the world grid, like the move gizmo, so vertices line up with other objects.
    if ((SDL_GetModState() & SDL_KMOD_CTRL) != 0 && m_snapTranslate > 0.0f)
        world = glm::round(world / m_snapTranslate) * m_snapTranslate;

    const glm::vec3 delta = world - m_vertexDrag.planePoint;
    const glm::mat4 transform = m_models.getInstances()[m_gizmo.selectedInstance].getTransformMatrix();
    const glm::mat4 inverse = glm::inverse(transform);
    for (size_t i = 0; i < vertices.size(); ++i) {
        const glm::vec3 start = glm::vec3(transform * glm::vec4(m_vertexDrag.startPositions[i], 1.0f));
        const glm::vec3 local = glm::vec3(inverse * glm::vec4(start + delta, 1.0f));
        if (local != mesh->positions[vertices[i]]) {
            mesh->positions[vertices[i]] = local;
            // Rebuilt once per frame in updateLevelHistory(); several motion events can arrive per frame.
            m_vertexDrag.moved = true;
        }
    }
}

void Editor::finishVertexMarquee(float mouseX, float mouseY)
{
    m_vertexMarquee.active = false;
    const glm::vec2 lo = glm::min(m_vertexMarquee.start, glm::vec2(mouseX, mouseY));
    const glm::vec2 hi = glm::max(m_vertexMarquee.start, glm::vec2(mouseX, mouseY));
    selectedLevelVertices(); // drops a selection left over from another instance or mode
    if (hi.x - lo.x < 4.0f && hi.y - lo.y < 4.0f) {
        // A click: Shift keeps the selection, otherwise it is cleared and objects are picked as usual.
        if (!m_vertexMarquee.additive) {
            m_selectedVertices.clear();
            pickObject(m_vertexMarquee.start.x, m_vertexMarquee.start.y);
        }
        return;
    }

    const PolyMesh* mesh = selectedLevelMesh();
    if (!mesh)
        return;
    if (!m_vertexMarquee.additive)
        m_selectedVertices.clear();
    m_levelSelectionInstance = m_gizmo.selectedInstance;
    // Hidden vertices are included, like the dots that show through.
    const glm::mat4 mvp = sceneProjection() * getView(m_camera) *
        m_models.getInstances()[m_gizmo.selectedInstance].getTransformMatrix();
    for (size_t v = 0; v < mesh->positions.size(); ++v) {
        const glm::vec4 clip = mvp * glm::vec4(mesh->positions[v], 1.0f);
        if (clip.w <= 1e-4f)
            continue;
        const glm::vec2 p = worldToScreen(mesh->positions[v], mvp, m_sceneView.width, m_sceneView.height);
        const uint32_t vertex = static_cast<uint32_t>(v);
        if (p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y &&
            std::find(m_selectedVertices.begin(), m_selectedVertices.end(), vertex) == m_selectedVertices.end())
            m_selectedVertices.push_back(vertex);
    }
}

void Editor::drawLevelFaceOverlay()
{
    const bool clipPreview = m_showLevelPanel && m_levelClip.preview;
    const bool mirrorPreview = m_showLevelPanel && m_levelMirror.preview;
    if ((m_levelMode == LevelEditMode::Object && !clipPreview && !mirrorPreview) || m_flyMode)
        return;
    const PolyMesh* mesh = selectedLevelMesh();
    if (!mesh)
        return;
    const PolyFace* selected = selectedLevelFace();
    const std::vector<uint32_t>& selectedVertices = selectedLevelVertices();

    const glm::mat4 mvp = sceneProjection() * getView(m_camera) *
        m_models.getInstances()[m_gizmo.selectedInstance].getTransformMatrix();
    const ImVec2 origin(m_sceneView.x, m_sceneView.y);
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->PushClipRect(origin, ImVec2(origin.x + m_sceneView.width, origin.y + m_sceneView.height), false);

    const auto drawLine = [&](const glm::vec3& from, const glm::vec3& to, ImU32 color, float thickness) {
        const glm::vec2 a = worldToScreen(from, mvp, m_sceneView.width, m_sceneView.height);
        const glm::vec2 b = worldToScreen(to, mvp, m_sceneView.width, m_sceneView.height);
        // worldToScreen() reports points behind the camera far off-screen; skip those edges.
        if (a.x < -5000.0f || b.x < -5000.0f)
            return;
        drawList->AddLine(ImVec2(origin.x + a.x, origin.y + a.y), ImVec2(origin.x + b.x, origin.y + b.y), color, thickness);
    };
    const auto drawFace = [&](const PolyMesh& owner, const PolyFace& face, ImU32 color, float thickness) {
        const size_t count = face.verts.size();
        for (size_t i = 0; i < count; ++i)
            drawLine(owner.positions[face.verts[i]], owner.positions[face.verts[(i + 1) % count]], color, thickness);
    };
    // Edges are drawn without depth, so hidden ones show through; that also helps aim at back faces.
    for (const PolyFace& face : mesh->faces)
        drawFace(*mesh, face, kFaceEdges, 1.0f);
    if (selected)
        drawFace(*mesh, *selected, kFaceOutline, 3.0f);
    if (clipPreview) {
        // Outline of the cut, plus an arrow from its middle to the front side.
        PolyMesh cut = *mesh;
        const size_t caps = cut.clip(m_levelClip.normal, m_levelClip.offset);
        glm::vec3 center(0.0f);
        size_t corners = 0;
        for (size_t i = cut.faces.size() - caps; i < cut.faces.size(); ++i) {
            drawFace(cut, cut.faces[i], kClipOutline, 2.5f);
            for (uint32_t v : cut.faces[i].verts)
                center += cut.positions[v];
            corners += cut.faces[i].verts.size();
        }
        if (corners > 0) {
            center /= static_cast<float>(corners);
            const glm::vec2 range = meshExtent(*mesh, m_levelClip.normal);
            const float length = std::max(0.25f * (range.y - range.x), 0.1f);
            const glm::vec3 tip = center + m_levelClip.normal * length;
            drawLine(center, tip, kClipOutline, 2.5f);
            const glm::vec2 p = worldToScreen(tip, mvp, m_sceneView.width, m_sceneView.height);
            if (p.x > -5000.0f)
                drawList->AddCircleFilled(ImVec2(origin.x + p.x, origin.y + p.y), 5.0f, kClipOutline);
        }
    }
    if (mirrorPreview) {
        // A rectangle on the mirror plane, a little larger than the shape.
        const int axis = m_levelMirror.axis;
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;
        glm::vec3 lo(0.0f), hi(0.0f);
        for (int i : { u, v }) {
            glm::vec3 dir(0.0f);
            dir[i] = 1.0f;
            const glm::vec2 range = meshExtent(*mesh, dir);
            const float pad = std::max(0.1f * (range.y - range.x), 0.05f);
            lo[i] = range.x - pad;
            hi[i] = range.y + pad;
        }
        lo[axis] = hi[axis] = levelMirrorPivot(*mesh);
        glm::vec3 corners[4] = { lo, lo, hi, hi };
        corners[1][u] = hi[u];
        corners[3][u] = lo[u];
        for (int i = 0; i < 4; ++i)
            drawLine(corners[i], corners[(i + 1) % 4], kMirrorOutline, 2.5f);
    }
    if (m_levelMode == LevelEditMode::Vertex) {
        const auto drawVertex = [&](uint32_t v, bool isSelected) {
            const glm::vec2 p = worldToScreen(mesh->positions[v], mvp, m_sceneView.width, m_sceneView.height);
            if (p.x < -5000.0f)
                return;
            drawList->AddCircleFilled(ImVec2(origin.x + p.x, origin.y + p.y), isSelected ? 6.0f : 3.5f,
                isSelected ? kFaceOutline : kVertexDot);
        };
        for (size_t v = 0; v < mesh->positions.size(); ++v)
            drawVertex(static_cast<uint32_t>(v), false);
        // Selected dots go on top.
        for (uint32_t v : selectedVertices)
            drawVertex(v, true);
    }
    if (m_vertexMarquee.active) {
        const glm::vec2 lo = glm::min(m_vertexMarquee.start, m_vertexMarquee.end);
        const glm::vec2 hi = glm::max(m_vertexMarquee.start, m_vertexMarquee.end);
        const ImVec2 a(origin.x + lo.x, origin.y + lo.y);
        const ImVec2 b(origin.x + hi.x, origin.y + hi.y);
        drawList->AddRectFilled(a, b, IM_COL32(255, 160, 40, 40));
        drawList->AddRect(a, b, kFaceOutline);
    }
    drawList->PopClipRect();
}

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

void Editor::updateLevelHistory()
{
    if (m_vertexDrag.moved) {
        m_vertexDrag.moved = false;
        if (selectedLevelMesh())
            rebuildSelectedLevelModel("move vertex");
    }
    if (m_levelEditPending) {
        if (ImGui::IsAnyItemActive() || m_vertexDrag.active)
            return;
        const auto found = m_models.findModelByPath(m_levelBaselinePath);
        const GPUModel* model = found ? m_models.getModel(*found) : nullptr;
        if (model && model->polyMesh) {
            m_undoStack.push_back({ m_levelBaselinePath, m_levelEditAction, std::move(m_levelBaseline), *model->polyMesh });
            if (m_undoStack.size() > kMaxUndoSteps)
                m_undoStack.erase(m_undoStack.begin());
            m_redoStack.clear();
        }
        m_levelEditPending = false;
        m_levelBaselinePath.clear(); // re-snapshot below
    }

    const PolyMesh* mesh = selectedLevelMesh();
    const std::string path = mesh
        ? m_models.getModel(m_models.getInstances()[m_gizmo.selectedInstance].modelIndex)->sourcePath
        : std::string();
    if (path != m_levelBaselinePath) {
        m_levelBaselinePath = path;
        m_levelBaseline = mesh ? *mesh : PolyMesh{};
    }
}

bool Editor::applyLevelEdit(const LevelEdit& edit, bool undo)
{
    const auto found = m_models.findModelByPath(edit.modelPath);
    GPUModel* model = found ? m_models.getModel(*found) : nullptr;
    if (!model || !model->polyMesh) {
        setStatus("Cannot " + std::string(undo ? "undo " : "redo ") + edit.action + ": the object was deleted", true);
        return false;
    }
    *model->polyMesh = undo ? edit.before : edit.after;
    if (edit.modelPath == m_levelBaselinePath)
        m_levelBaseline = *model->polyMesh;
    try {
        m_models.rebuildPolyMesh(*found);
    }
    catch (const std::exception& e) {
        setStatus("Failed to rebuild " + model->name + ": " + e.what(), true);
    }
    setStatus((undo ? "Undo " : "Redo ") + edit.action);
    return true;
}

void Editor::undoLevelEdit()
{
    // Mid-drag the edit has no history entry yet.
    if (m_levelEditPending || m_vertexDrag.active)
        return;
    if (m_undoStack.empty()) {
        setStatus("Nothing to undo");
        return;
    }
    LevelEdit edit = std::move(m_undoStack.back());
    m_undoStack.pop_back();
    if (applyLevelEdit(edit, true))
        m_redoStack.push_back(std::move(edit));
}

void Editor::redoLevelEdit()
{
    if (m_levelEditPending || m_vertexDrag.active)
        return;
    if (m_redoStack.empty()) {
        setStatus("Nothing to redo");
        return;
    }
    LevelEdit edit = std::move(m_redoStack.back());
    m_redoStack.pop_back();
    if (applyLevelEdit(edit, false))
        m_undoStack.push_back(std::move(edit));
}

void Editor::clearLevelHistory()
{
    m_undoStack.clear();
    m_redoStack.clear();
    m_levelEditPending = false;
    m_levelBaselinePath.clear();
}

void Editor::drawFaceProperties(PolyMesh& mesh, PolyFace& face)
{
    bool changed = false;
    const int current = face.material < mesh.materials.size() ? static_cast<int>(face.material) : -1;
    const auto materialName = [](int index) {
        return index >= 0 ? "Material " + std::to_string(index) : std::string("None (white)");
    };
    EditorStyle::propertyLabel("Material");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##faceMaterial", materialName(current).c_str())) {
        if (ImGui::Selectable(materialName(-1).c_str(), current < 0)) {
            face.material = kNoPolyMaterial;
            changed = true;
        }
        for (int i = 0; i < static_cast<int>(mesh.materials.size()); ++i) {
            if (ImGui::Selectable(materialName(i).c_str(), current == i)) {
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

    const uint32_t faceIndex = static_cast<uint32_t>(&face - mesh.faces.data());
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
    if (const int fit = buttonRow("Fit", { "Both", "U", "V" }); fit >= 0) {
        mesh.fitFaceUVs(faceIndex, fit != 2, fit != 1);
        changed = true;
    }
    if (const int align = buttonRow("Align U", { "Left", "Center", "Right" }); align >= 0) {
        mesh.alignFaceUVs(faceIndex, glm::vec2(align * 0.5f, -1.0f));
        changed = true;
    }
    if (const int align = buttonRow("Align V", { "Top", "Center", "Bottom" }); align >= 0) {
        mesh.alignFaceUVs(faceIndex, glm::vec2(-1.0f, align * 0.5f));
        changed = true;
    }
    if (const int rotate = buttonRow("Rotate", { "-90", "+90", "180" }); rotate >= 0) {
        face.uvRotation += rotate == 0 ? -90.0f : rotate == 1 ? 90.0f : 180.0f;
        face.uvRotation = std::fmod(face.uvRotation + 540.0f, 360.0f) - 180.0f; // keep within [-180, 180)
        changed = true;
    }
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
        const float step = m_snapTranslate > 0.0f ? m_snapTranslate : 1.0f;
        for (uint32_t v : vertices)
            mesh.positions[v] = glm::round(mesh.positions[v] / step) * step;
        changed = true;
    }
    ImGui::SetItemTooltip("Round the object-space positions to the move snap step");
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
    edited.deleteFace(static_cast<uint32_t>(m_selectedFace));
    if (commitLevelTopology(std::move(edited), "delete face"))
        m_selectedFace = -1;
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
    m_selectedFace = -1;
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
        selectInstance(static_cast<int>(newIndex));
        setStatus("Added mirrored copy " + name);
        return;
    }

    if (!edited.symmetrize(axis, pivot, action == MirrorAction::SymmetrizeNegative)) {
        setStatus("Nothing on that side of the plane to mirror", true);
        return;
    }
    if (commitLevelTopology(std::move(edited), "symmetrize")) {
        m_selectedFace = -1;
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
            m_selectedFace = -1;
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
    m_selectedFace = -1;
    m_selectedVertices.clear();
    if (m_deleteCutter) {
        m_models.removeInstance(static_cast<size_t>(cutter));
        releaseUnusedLevelModels();
        selectInstance(cutter < target ? target - 1 : target);
        m_levelCutterName.clear();
    }
    setStatus("Subtracted " + cutterName + " from " + targetName);
}

bool Editor::deleteLevelSelection()
{
    // In face/vertex mode Del never removes the whole object, even with nothing picked.
    if (m_levelMode == LevelEditMode::Object || !selectedLevelMesh())
        return false;
    if (m_vertexDrag.active || m_levelEditPending)
        return true;
    if (m_levelMode == LevelEditMode::Face)
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

glm::vec3 Editor::placementPoint() const
{
    // Where the view ray meets the ground (y = 0), else a few units in front of the camera.
    const glm::vec3 front = getFront(m_camera);
    glm::vec3 point = m_camera.position + front * 5.0f;
    if (front.y < -1e-3f) {
        const float t = -m_camera.position.y / front.y;
        if (t < 50.0f)
            point = m_camera.position + front * t;
    }
    const float step = m_snapTranslate > 0.0f ? m_snapTranslate : 1.0f;
    point = glm::round(point / step) * step;
    point.y = std::max(point.y, 0.0f);
    return point;
}

void Editor::addLevelShape(PolyShape shape)
{
    PolyShapeParams params = m_newShape;
    params.shape = shape;
    const std::string name = uniqueInstanceName(kPolyShapeNames[static_cast<int>(shape)]);
    const auto modelIndex = createLevelModel(makePolyShape(params), name);
    if (!modelIndex)
        return;
    const size_t newIndex = m_models.createInstance(*modelIndex, placementPoint());
    m_models.getInstances()[newIndex].name = name;
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
    ImGui::DragFloat("##faceDistance", &m_faceOpDistance, 0.05f, -100.0f, 100.0f, "%.2f");
    const float width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    const bool push = ImGui::Button("Push/Pull", ImVec2(width, 0.0f));
    ImGui::SetItemTooltip("Move the face along its normal; neighbouring faces stretch");
    ImGui::SameLine();
    const bool extrude = ImGui::Button("Extrude", ImVec2(width, 0.0f));
    ImGui::SetItemTooltip("Build new side faces out to the moved face");
    if (!push && !extrude)
        return;
    if (std::abs(m_faceOpDistance) < 1e-4f) {
        setStatus("Distance is zero", true);
        return;
    }
    if (push)
        mesh.moveFace(faceIndex, m_faceOpDistance);
    else
        mesh.extrudeFace(faceIndex, m_faceOpDistance);
    rebuildSelectedLevelModel(push ? "push/pull" : "extrude");
}

void Editor::drawLevelMaterials(PolyMesh& mesh)
{
    bool changed = false;
    int removed = -1;
    for (int i = 0; i < static_cast<int>(mesh.materials.size()); ++i) {
        PolyMaterial& material = mesh.materials[i];
        ImGui::PushID(i);
        if (ImGui::TreeNodeEx("##material", ImGuiTreeNodeFlags_DefaultOpen, "Material %d", i)) {
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
            if (ImGui::Button("Browse...")) {
                m_textureDialogMaterial = i;
                openFileDialog("BrowseLevelTextureDlg", "Choose Texture",
                    "Images{.png,.jpg,.jpeg,.tga,.bmp},.png,.jpg,.jpeg,.tga,.bmp", kTexturesRoot, nullptr, false);
            }
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
    if (mesh.materials.size() < kMaxPolyMaterials && ImGui::Button("Add material", ImVec2(-FLT_MIN, 0.0f))) {
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

void Editor::drawLevelTextureDialog()
{
    ImGuiFileDialog* dialog = ImGuiFileDialog::Instance();
    if (!dialog->Display("BrowseLevelTextureDlg", ImGuiWindowFlags_NoCollapse, kDialogSize))
        return;
    PolyMesh* mesh = selectedLevelMesh();
    if (dialog->IsOk() && mesh && m_textureDialogMaterial >= 0 &&
        m_textureDialogMaterial < static_cast<int>(mesh->materials.size())) {
        mesh->materials[m_textureDialogMaterial].texturePath = toStoredPath(dialog->GetFilePathName());
        rebuildSelectedLevelModel("set texture");
    }
    m_textureDialogMaterial = -1;
    dialog->Close();
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
        ImGui::DragFloat3("##size", &m_newShape.size.x, 0.05f, 0.01f, 1000.0f, "%.2f");
        switch (m_newShape.shape) {
        case PolyShape::Plane:
        case PolyShape::Cylinder:
        case PolyShape::Arch:
            EditorStyle::propertyLabel(m_newShape.shape == PolyShape::Plane ? "Subdivisions" : "Segments");
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::SliderInt("##segments", &m_newShape.segments, m_newShape.shape == PolyShape::Plane ? 1 : 3, 64);
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

        if (ImGui::Button("Create", ImVec2(-FLT_MIN, 0.0f)))
            addLevelShape(m_newShape.shape);

        ImGui::SeparatorText("Selected");
        drawPrefabSection();
        if (PolyMesh* mesh = selectedLevelMesh()) {
            ImGui::Text("%zu vertices, %zu faces", mesh->positions.size(), mesh->faces.size());
            // Rebuilding replaces any manual edits, so it only happens on request.
            if (ImGui::Button("Rebuild from shape settings", ImVec2(-FLT_MIN, 0.0f))) {
                // Materials survive; the new faces all start on material 0.
                PolyMesh rebuilt = makePolyShape(m_newShape);
                rebuilt.materials = std::move(mesh->materials);
                *mesh = std::move(rebuilt);
                m_selectedFace = -1;
                m_selectedVertices.clear();
                rebuildSelectedLevelModel("rebuild shape");
                setStatus("Rebuilt " + m_models.getInstances()[m_gizmo.selectedInstance].name);
            }

            ImGui::SeparatorText("Materials");
            drawLevelMaterials(*mesh);

            ImGui::SeparatorText("Edit");
            int mode = static_cast<int>(m_levelMode);
            ImGui::RadioButton("Object", &mode, static_cast<int>(LevelEditMode::Object));
            ImGui::SameLine();
            ImGui::RadioButton("Faces", &mode, static_cast<int>(LevelEditMode::Face));
            ImGui::SameLine();
            ImGui::RadioButton("Vertices", &mode, static_cast<int>(LevelEditMode::Vertex));
            m_levelMode = static_cast<LevelEditMode>(mode);

            if (m_levelMode == LevelEditMode::Face) {
                if (PolyFace* face = selectedLevelFace()) {
                    ImGui::Text("Face %d (%zu vertices)", m_selectedFace, face->verts.size());
                    drawFaceProperties(*mesh, *face);
                    ImGui::SeparatorText("Geometry");
                    drawFaceGeometry(*mesh, static_cast<uint32_t>(m_selectedFace));

                    const float width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
                    if (ImGui::Button("Flip", ImVec2(width, 0.0f))) {
                        mesh->flipFace(static_cast<uint32_t>(m_selectedFace));
                        rebuildSelectedLevelModel("flip face");
                    }
                    ImGui::SetItemTooltip("Turn the face to point the other way");
                    ImGui::SameLine();
                    if (ImGui::Button("Delete face", ImVec2(width, 0.0f)))
                        deleteLevelFace();
                    ImGui::SetItemTooltip("Remove the face, leaving a hole (Del)");
                    if (PolyFace* current = selectedLevelFace(); current && ImGui::Button("Select its vertices", ImVec2(-FLT_MIN, 0.0f))) {
                        m_selectedVertices = current->verts;
                        m_levelMode = LevelEditMode::Vertex;
                    }
                }
                else {
                    ImGui::TextDisabled("Click a face of the shape in the viewport.");
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
                    ImGui::TextDisabled("Shift adds. Drag a vertex to move, Ctrl snaps.");
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
    drawLevelTextureDialog();
}
