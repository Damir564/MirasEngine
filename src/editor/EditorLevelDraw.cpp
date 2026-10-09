#include "Editor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include "EditorStyle.h"
#include "engine/ModelManager.h"

namespace {

constexpr ImU32 kDrawColor = IM_COL32(110, 255, 140, 255);
constexpr ImU32 kDrawFill = IM_COL32(110, 255, 140, 40);
// Surfaces facing up more than this (normal.y) can be built on.
constexpr float kUpwardFacing = 0.7f;

bool rayHitsHorizontalPlane(const Ray& ray, float height, glm::vec3& hit)
{
    if (std::abs(ray.direction.y) < 1e-6f)
        return false;
    const float t = (height - ray.origin.y) / ray.direction.y;
    if (t <= 0.0f)
        return false;
    hit = ray.origin + ray.direction * t;
    hit.y = height;
    return true;
}

// Position along the line origin + t * axis (unit axis) closest to the ray; false when they are parallel.
bool closestLineParam(const Ray& ray, const glm::vec3& origin, const glm::vec3& axis, float& param)
{
    const glm::vec3 w0 = origin - ray.origin;
    const float b = glm::dot(axis, ray.direction);
    const float c = glm::dot(ray.direction, ray.direction);
    const float d = glm::dot(axis, w0);
    const float e = glm::dot(ray.direction, w0);
    const float denom = c - b * b;
    if (denom < 1e-6f * c)
        return false;
    param = (b * e - c * d) / denom;
    return std::isfinite(param);
}

} // namespace

void Editor::toggleShapeDraw()
{
    if (shapeDrawActive()) {
        m_shapeDraw = {};
        setStatus("Draw shape tool off");
        return;
    }
    m_shapeDraw = {};
    m_shapeDraw.phase = ShapeDraw::Phase::Ready;
    m_faceDraw = {};
    m_entityPlace = {};
    // Picking faces or vertices would compete with the tool for clicks.
    m_levelMode = LevelEditMode::Object;
    setStatus(std::string("Draw ") + kPolyShapeNames[static_cast<int>(m_newShape.shape)] +
        ": drag a footprint, then set the height and click. A click places the panel's size. Esc stops.");
}

bool Editor::surfacePoint(float mouseX, float mouseY, glm::vec3& out) const
{
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height, getView(m_camera),
        sceneProjection());
    const SubmeshHitResult hit = pickSubmesh(ray, m_models.getInstances(),
        [&](size_t index) { return m_models.getModel(index); });
    if (hit.hit()) {
        const glm::vec3 point = ray.origin + ray.direction * hit.t;
        // Level shapes know which face was hit: build on top of floors, not halfway up walls. Other models
        // are taken as they are.
        const ModelInstance& instance = m_models.getInstances()[hit.instanceIndex];
        const GPUModel* model = m_models.getModel(instance.modelIndex);
        bool upward = true;
        if (model && model->polyMesh) {
            const int face = pickLevelFace(mouseX, mouseY, hit.instanceIndex);
            if (face >= 0) {
                const glm::vec3 local = model->polyMesh->faceNormal(model->polyMesh->faces[face]);
                const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(instance.getTransformMatrix())));
                upward = glm::normalize(normalMatrix * local).y > kUpwardFacing;
            }
        }
        if (upward) {
            out = point;
            // Hits come back a hair off the surface; floors built on the grid stay on it.
            out.y = std::round(out.y * 1000.0f) / 1000.0f;
            return true;
        }
    }
    return rayHitsHorizontalPlane(ray, 0.0f, out);
}

bool Editor::shapeDrawPoint(float mouseX, float mouseY, float height, glm::vec3& out) const
{
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height, getView(m_camera),
        sceneProjection());
    if (!rayHitsHorizontalPlane(ray, height, out))
        return false;
    if (snapActive()) {
        out.x = std::round(out.x / m_gridSize) * m_gridSize;
        out.z = std::round(out.z / m_gridSize) * m_gridSize;
    }
    return true;
}

bool Editor::handleShapeDrawPress(float mouseX, float mouseY)
{
    switch (m_shapeDraw.phase) {
    case ShapeDraw::Phase::Off:
        return false;
    case ShapeDraw::Phase::Ready: {
        glm::vec3 surface, corner;
        if (surfacePoint(mouseX, mouseY, surface) && shapeDrawPoint(mouseX, mouseY, surface.y, corner)) {
            m_shapeDraw.baseY = surface.y;
            m_shapeDraw.start = m_shapeDraw.end = corner;
            m_shapeDraw.phase = ShapeDraw::Phase::Footprint;
        }
        return true;
    }
    case ShapeDraw::Phase::Footprint:
        return true;
    case ShapeDraw::Phase::Height:
        finishShapeDraw(false);
        return true;
    }
    return false;
}

void Editor::handleShapeDrawMotion(float mouseX, float mouseY)
{
    if (m_shapeDraw.phase == ShapeDraw::Phase::Footprint) {
        glm::vec3 corner;
        if (shapeDrawPoint(mouseX, mouseY, m_shapeDraw.baseY, corner))
            m_shapeDraw.end = corner;
        return;
    }
    if (m_shapeDraw.phase != ShapeDraw::Phase::Height)
        return;
    const glm::vec3 center = (m_shapeDraw.start + m_shapeDraw.end) * 0.5f;
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height, getView(m_camera),
        sceneProjection());
    float param;
    if (!closestLineParam(ray, center, glm::vec3(0.0f, 1.0f, 0.0f), param))
        return;
    float height = m_shapeDraw.startHeight + (param - m_shapeDraw.startParam);
    if (snapActive())
        height = std::round(height / m_gridSize) * m_gridSize;
    m_shapeDraw.height = std::max(height, snapActive() ? m_gridSize : 0.01f);
}

void Editor::handleShapeDrawRelease(float mouseX, float mouseY)
{
    if (m_shapeDraw.phase != ShapeDraw::Phase::Footprint)
        return;
    handleShapeDrawMotion(mouseX, mouseY);
    const glm::vec3 size = glm::abs(m_shapeDraw.end - m_shapeDraw.start);
    // A click (or a line with no area) places a shape of the panel's size there.
    if (size.x < 1e-3f || size.z < 1e-3f) {
        finishShapeDraw(true);
        return;
    }
    if (m_newShape.shape == PolyShape::Plane) {
        finishShapeDraw(false);
        return;
    }
    const float grid = m_gridSize;
    const float height = snapActive() ? std::max(std::round(m_newShape.size.y / grid), 1.0f) * grid : m_newShape.size.y;
    const glm::vec3 center = (m_shapeDraw.start + m_shapeDraw.end) * 0.5f;
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height, getView(m_camera),
        sceneProjection());
    m_shapeDraw.height = m_shapeDraw.startHeight = height;
    m_shapeDraw.startParam = 0.0f;
    closestLineParam(ray, center, glm::vec3(0.0f, 1.0f, 0.0f), m_shapeDraw.startParam);
    m_shapeDraw.phase = ShapeDraw::Phase::Height;
}

bool Editor::handleShapeDrawKeys()
{
    if (!shapeDrawActive())
        return false;
    if (m_keymap.pressed(EditorAction::FaceDrawCancel)) {
        if (m_shapeDraw.phase == ShapeDraw::Phase::Ready) {
            m_shapeDraw = {};
            setStatus("Draw shape tool off");
        }
        else {
            m_shapeDraw.phase = ShapeDraw::Phase::Ready;
            setStatus("Shape cancelled");
        }
        return true;
    }
    if (m_shapeDraw.phase == ShapeDraw::Phase::Height && m_keymap.pressed(EditorAction::FaceDrawApply))
        finishShapeDraw(false);
    return true;
}

void Editor::finishShapeDraw(bool clicked)
{
    PolyShapeParams params = m_newShape;
    PolyMesh mesh;
    glm::vec3 position;
    if (clicked) {
        // As Create does: sides on grid lines with the shape centered on the clicked grid point.
        mesh = makePolyShapeOnGrid(params, m_gridSize, snapActive());
        position = m_shapeDraw.start;
    }
    else {
        const glm::vec3 lo = glm::min(m_shapeDraw.start, m_shapeDraw.end);
        const glm::vec3 hi = glm::max(m_shapeDraw.start, m_shapeDraw.end);
        params.size = glm::vec3(hi.x - lo.x, params.shape == PolyShape::Plane ? params.size.y : m_shapeDraw.height,
            hi.z - lo.z);
        mesh = makePolyShape(params);
        mesh.gridSize = m_gridSize;
        position = glm::vec3((lo.x + hi.x) * 0.5f, m_shapeDraw.baseY, (lo.z + hi.z) * 0.5f);
        // The next click-placed shape (and Create) use the size just drawn.
        m_newShape.size = params.size;
    }
    if (m_materials.find(m_currentMaterial))
        levelSlotFor(mesh, m_currentMaterial);
    const std::string name = uniqueInstanceName(kPolyShapeNames[static_cast<int>(params.shape)]);
    m_shapeDraw.phase = ShapeDraw::Phase::Ready;
    const auto modelIndex = createLevelModel(std::move(mesh), name);
    if (!modelIndex)
        return;
    const size_t newIndex = m_models.createInstance(*modelIndex, position);
    m_models.getInstances()[newIndex].name = name;
    markSceneChanged();
    selectInstance(static_cast<int>(newIndex));
    if (m_editAfterDraw) {
        m_shapeDraw = {};
        beginShapeEdit();
        return;
    }
    char text[160];
    snprintf(text, sizeof(text), "Added %s (%g x %g x %g)", name.c_str(), params.size.x, params.size.y, params.size.z);
    setStatus(text);
}

void Editor::toggleShapeEdit()
{
    if (shapeEditActive())
        finishShapeEdit();
    else
        beginShapeEdit();
}

void Editor::beginShapeEdit()
{
    if (!selectedLevelMesh()) {
        setStatus("Select a level shape (unlocked) to edit it", true);
        return;
    }
    m_shapeDraw = {};
    m_entityPlace = {};
    clearShapeEditPicks();
    m_levelMode = m_lastShapeEditMode;
    setStatus("Editing " + m_models.getInstances()[m_gizmo.selectedInstance].name + ": " +
        m_keymap.shortcutLabel(EditorAction::EditShape) + " or Esc finishes");
}

void Editor::finishShapeEdit()
{
    if (m_levelMode == LevelEditMode::Object)
        return;
    m_lastShapeEditMode = m_levelMode;
    m_faceDraw = {};
    clearShapeEditPicks();
    m_levelMode = LevelEditMode::Object;
    if (selectedLevelMesh())
        setStatus("Finished editing " + m_models.getInstances()[m_gizmo.selectedInstance].name);
}

bool Editor::shapeEditHasPicks()
{
    switch (m_levelMode) {
    case LevelEditMode::Face: return !selectedLevelFaces().empty();
    case LevelEditMode::Vertex: return !selectedLevelVertices().empty();
    case LevelEditMode::Edge: return m_edgeSelected;
    case LevelEditMode::Part: return !m_selectedParts.empty();
    default: return false;
    }
}

void Editor::clearShapeEditPicks()
{
    clearFaceSelection();
    m_selectedVertices.clear();
    m_edgeSelected = false;
    m_selectedParts.clear();
}

void Editor::drawShapeEditOverlay()
{
    if (m_flyMode || shapeDrawActive() || entityPlacementActive() || !shapeEditActive())
        return;
    static constexpr const char* kModeNames[] = { "Object", "Faces", "Vertices", "Edges", "Parts" };
    const auto key = [&](EditorAction action) {
        const std::string label = m_keymap.shortcutLabel(action);
        return label.empty() ? std::string() : " (" + label + ")";
    };
    char title[256];
    snprintf(title, sizeof(title), "Editing %s: %s   |   Faces%s  Edges%s  Vertices%s  Parts%s   |   Done: %s / Esc",
        m_models.getInstances()[m_gizmo.selectedInstance].name.c_str(), kModeNames[static_cast<int>(m_levelMode)],
        key(EditorAction::FaceMode).c_str(), key(EditorAction::EdgeMode).c_str(), key(EditorAction::VertexMode).c_str(),
        key(EditorAction::PartMode).c_str(), m_keymap.shortcutLabel(EditorAction::EditShape).c_str());
    ImDrawList* foreground = ImGui::GetForegroundDrawList();
    const ImVec2 size = ImGui::CalcTextSize(title);
    const ImVec2 at(m_sceneView.x + (m_sceneView.width - size.x) * 0.5f, m_sceneView.y + 12.0f);
    foreground->AddRectFilled(ImVec2(at.x - 8.0f, at.y - 4.0f), ImVec2(at.x + size.x + 8.0f, at.y + size.y + 4.0f),
        IM_COL32(0, 0, 0, 150), 4.0f);
    foreground->AddText(at, kDrawColor, title);
}

void Editor::drawShapeDrawOverlay()
{
    if (!shapeDrawActive() || m_flyMode)
        return;
    const glm::mat4 viewProj = sceneProjection() * getView(m_camera);
    const ImVec2 origin(m_sceneView.x, m_sceneView.y);
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->PushClipRect(origin, ImVec2(origin.x + m_sceneView.width, origin.y + m_sceneView.height), false);
    // worldToScreen() reports points behind the camera far off-screen.
    const auto toScreen = [&](const glm::vec3& p, ImVec2& out) {
        const glm::vec2 s = worldToScreen(p, viewProj, m_sceneView.width, m_sceneView.height);
        out = ImVec2(origin.x + s.x, origin.y + s.y);
        return s.x > -5000.0f;
    };
    const auto line = [&](const glm::vec3& a, const glm::vec3& b) {
        ImVec2 sa, sb;
        if (toScreen(a, sa) && toScreen(b, sb))
            drawList->AddLine(sa, sb, kDrawColor, 2.0f);
    };
    const auto label = [&](const glm::vec3& at, const char* text) {
        ImVec2 s;
        if (toScreen(at, s))
            drawList->AddText(ImVec2(s.x + 10.0f, s.y - 10.0f), kDrawColor, text);
    };

    const ImVec2 mouse = ImGui::GetMousePos();
    const bool overScene = sceneViewContains(mouse.x, mouse.y) && !ImGui::GetIO().WantCaptureMouse;
    if (m_shapeDraw.phase == ShapeDraw::Phase::Ready && overScene) {
        // Where a press would start: a cross on the snapped point.
        glm::vec3 surface, corner;
        if (surfacePoint(mouse.x - origin.x, mouse.y - origin.y, surface) &&
            shapeDrawPoint(mouse.x - origin.x, mouse.y - origin.y, surface.y, corner)) {
            const float arm = m_gridSize * 0.5f;
            line(corner - glm::vec3(arm, 0.0f, 0.0f), corner + glm::vec3(arm, 0.0f, 0.0f));
            line(corner - glm::vec3(0.0f, 0.0f, arm), corner + glm::vec3(0.0f, 0.0f, arm));
        }
    }
    else if (m_shapeDraw.phase != ShapeDraw::Phase::Ready) {
        const glm::vec3 lo = glm::min(m_shapeDraw.start, m_shapeDraw.end);
        const glm::vec3 hi = glm::max(m_shapeDraw.start, m_shapeDraw.end);
        const float y0 = m_shapeDraw.baseY;
        const glm::vec3 base[4] = { { lo.x, y0, lo.z }, { hi.x, y0, lo.z }, { hi.x, y0, hi.z }, { lo.x, y0, hi.z } };
        ImVec2 screen[4];
        bool visible = true;
        for (int i = 0; i < 4; ++i)
            visible = toScreen(base[i], screen[i]) && visible;
        if (visible)
            drawList->AddQuadFilled(screen[0], screen[1], screen[2], screen[3], kDrawFill);
        for (int i = 0; i < 4; ++i)
            line(base[i], base[(i + 1) % 4]);
        char text[96];
        if (m_shapeDraw.phase == ShapeDraw::Phase::Height) {
            const glm::vec3 up(0.0f, m_shapeDraw.height, 0.0f);
            for (int i = 0; i < 4; ++i) {
                line(base[i] + up, base[(i + 1) % 4] + up);
                line(base[i], base[i] + up);
            }
            snprintf(text, sizeof(text), "%g x %g x %g", hi.x - lo.x, m_shapeDraw.height, hi.z - lo.z);
            label(hi + up, text);
        }
        else {
            snprintf(text, sizeof(text), "%g x %g", hi.x - lo.x, hi.z - lo.z);
            label(m_shapeDraw.end, text);
        }
    }
    drawList->PopClipRect();

    // What to do next, in the top middle of the view.
    const char* hint = m_shapeDraw.phase == ShapeDraw::Phase::Height
        ? "Move the mouse to set the height, click to finish (Esc: cancel)"
        : m_shapeDraw.phase == ShapeDraw::Phase::Footprint ? "Release to set the footprint"
        : "Drag a footprint on the ground or a floor; click to place the panel's size (Esc: stop)";
    char title[160];
    snprintf(title, sizeof(title), "Draw %s: %s", kPolyShapeNames[static_cast<int>(m_newShape.shape)], hint);
    const ImVec2 size = ImGui::CalcTextSize(title);
    const ImVec2 at(origin.x + (m_sceneView.width - size.x) * 0.5f, origin.y + 12.0f);
    ImDrawList* foreground = ImGui::GetForegroundDrawList();
    foreground->AddRectFilled(ImVec2(at.x - 8.0f, at.y - 4.0f), ImVec2(at.x + size.x + 8.0f, at.y + size.y + 4.0f),
        IM_COL32(0, 0, 0, 150), 4.0f);
    foreground->AddText(at, kDrawColor, title);
}

void Editor::moveSelectedFaces(float distance, bool extrude)
{
    PolyMesh* mesh = selectedLevelMesh();
    const std::vector<uint32_t>& faces = selectedLevelFaces();
    if (!mesh || faces.empty())
        return;
    if (extrude && mesh->isClosed()) {
        // A closed shape grows without overlapping itself, or shrinks when extruded inwards.
        const std::vector<uint32_t> extruded = faces;
        std::vector<uint32_t> caps;
        if (!mesh->extrudeFacesSolid(extruded, distance, &caps)) {
            setStatus("Nothing of the shape would be left", true);
            return;
        }
        m_selectedFaces = caps;
        m_selectedFace = caps.empty() ? -1 : static_cast<int>(caps.front());
        rebuildSelectedLevelModel(distance < 0.0f ? "extrude inwards" : "extrude");
        return;
    }
    if (extrude) {
        // Extruding keeps each face's index (it becomes the cap), so the selection stays valid.
        for (uint32_t f : faces)
            mesh->extrudeFace(f, distance);
        rebuildSelectedLevelModel("extrude");
        return;
    }
    // Each vertex moves once, so that every selected face it touches ends up `distance` further out:
    // the offset o solves dot(o, n) = distance for each distinct normal n around the vertex.
    std::unordered_map<uint32_t, std::vector<glm::vec3>> normals;
    for (uint32_t f : faces) {
        const glm::vec3 n = mesh->faceNormal(mesh->faces[f]);
        for (uint32_t v : mesh->faces[f].verts) {
            std::vector<glm::vec3>& list = normals[v];
            const bool seen = std::any_of(list.begin(), list.end(), [&](const glm::vec3& m) { return glm::dot(m, n) > 0.999f; });
            if (!seen && list.size() < 3)
                list.push_back(n);
        }
    }
    for (const auto& [v, list] : normals) {
        glm::vec3 offset = list[0] * distance;
        if (list.size() == 2) {
            const float c = glm::dot(list[0], list[1]);
            if (c > -0.999f)
                offset = (list[0] + list[1]) * (distance / (1.0f + c));
        }
        else if (list.size() == 3) {
            const glm::mat3 rows = glm::transpose(glm::mat3(list[0], list[1], list[2]));
            if (std::abs(glm::determinant(rows)) > 1e-4f)
                offset = glm::inverse(rows) * glm::vec3(distance);
        }
        mesh->positions[v] += offset;
    }
    rebuildSelectedLevelModel("push/pull");
}

void Editor::snapSelectedVertices()
{
    PolyMesh* mesh = selectedLevelMesh();
    const std::vector<uint32_t>& vertices = selectedLevelVertices();
    if (!mesh || vertices.empty())
        return;
    const float grid = mesh->gridSize;
    bool changed = false;
    for (uint32_t v : vertices) {
        const glm::vec3 snapped = glm::round(mesh->positions[v] / grid) * grid;
        changed = changed || snapped != mesh->positions[v];
        mesh->positions[v] = snapped;
    }
    if (changed)
        rebuildSelectedLevelModel("snap vertices");
    else
        setStatus("Vertices already on the grid");
}
