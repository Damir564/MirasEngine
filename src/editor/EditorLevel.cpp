#include "Editor.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <SDL3/SDL_keyboard.h>
#include "EditorStyle.h"
#include "FileDialog.h"
#include "engine/ModelManager.h"

namespace {
constexpr const char* kTexturesRoot = "textures";
constexpr ImU32 kFaceOutline = IM_COL32(255, 160, 40, 255);
constexpr ImU32 kFaceGroupOutline = IM_COL32(255, 200, 120, 190);
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

// Shape sizes in whole grid cells (at least one), so the faces of a new shape lie on its grid lines.
PolyShapeParams snapShapeSize(PolyShapeParams params, float grid)
{
    params.size = glm::max(glm::round(params.size / grid), glm::vec3(1.0f)) * grid;
    return params;
}

// Shapes are built centered on X/Z; with an odd number of cells that puts their sides between grid
// lines, so shift them half a cell to put the sides on the object-space grid.
PolyMesh makeShapeOnGrid(const PolyShapeParams& params, float grid, bool snap)
{
    if (!snap) {
        PolyMesh mesh = makePolyShape(params);
        mesh.gridSize = grid;
        return mesh;
    }
    const PolyShapeParams snapped = snapShapeSize(params, grid);
    PolyMesh mesh = makePolyShape(snapped);
    mesh.gridSize = grid;
    const glm::vec2 half(snapped.size.x * 0.5f, snapped.size.z * 0.5f);
    const glm::vec2 shift = glm::round(half / grid) * grid - half;
    if (shift != glm::vec2(0.0f))
        for (glm::vec3& p : mesh.positions)
            p += glm::vec3(shift.x, 0.0f, shift.y);
    return mesh;
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

const std::vector<uint32_t>& Editor::selectedLevelFaces()
{
    if (!selectedLevelFace()) {
        m_selectedFaces.clear();
        return m_selectedFaces;
    }
    // Undo can shrink the mesh under the selection; the active face always leads.
    const size_t count = selectedLevelMesh()->faces.size();
    const uint32_t active = static_cast<uint32_t>(m_selectedFace);
    std::erase_if(m_selectedFaces, [&](uint32_t f) { return f >= count || f == active; });
    m_selectedFaces.insert(m_selectedFaces.begin(), active);
    return m_selectedFaces;
}

void Editor::clearFaceSelection()
{
    m_selectedFace = -1;
    m_selectedFaces.clear();
}

void Editor::selectLevelFace(uint32_t face, bool toggle)
{
    if (m_levelSelectionInstance != m_gizmo.selectedInstance)
        clearFaceSelection();
    m_levelSelectionInstance = m_gizmo.selectedInstance;
    selectedLevelFaces(); // drops faces left over from an undo
    const auto found = std::find(m_selectedFaces.begin(), m_selectedFaces.end(), face);
    if (!toggle) {
        // Clicking a face already in the group keeps the group, so it can be dragged or nudged together.
        if (found == m_selectedFaces.end())
            m_selectedFaces.assign(1, face);
        m_selectedFace = static_cast<int>(face);
        return;
    }
    if (found != m_selectedFaces.end()) {
        m_selectedFaces.erase(found);
        m_selectedFace = m_selectedFaces.empty() ? -1 : static_cast<int>(m_selectedFaces.front());
        return;
    }
    m_selectedFaces.push_back(face);
    m_selectedFace = static_cast<int>(face);
}

void Editor::selectFacesWhere(const std::function<bool(const PolyMesh&, const PolyFace&)>& predicate)
{
    const PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || !selectedLevelFace())
        return;
    const uint32_t active = static_cast<uint32_t>(m_selectedFace);
    m_selectedFaces.assign(1, active);
    for (uint32_t f = 0; f < mesh->faces.size(); ++f)
        if (f != active && predicate(*mesh, mesh->faces[f]))
            m_selectedFaces.push_back(f);
    setStatus(std::to_string(m_selectedFaces.size()) + " faces selected");
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

int Editor::pickLevelFace(float mouseX, float mouseY, int instanceIndex) const
{
    if (instanceIndex < 0)
        instanceIndex = m_gizmo.selectedInstance;
    if (!validInstance(instanceIndex))
        return -1;
    const ModelInstance& instance = m_models.getInstances()[instanceIndex];
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
        const int picked = pickLevelFace(mouseX, mouseY);
        if (picked < 0)
            return false;
        const SDL_Keymod mods = SDL_GetModState();
        const bool shift = (mods & SDL_KMOD_SHIFT) != 0;
        const bool alt = (mods & SDL_KMOD_ALT) != 0;
        if (shift && alt) {
            // Wrap: the clicked face continues the active face's texture, then becomes the active face,
            // so clicking around a pillar wraps all the way.
            PolyMesh* mesh = selectedLevelMesh();
            const PolyFace* active = selectedLevelFace();
            if (mesh && active && picked != m_selectedFace) {
                const uint32_t from = static_cast<uint32_t>(m_selectedFace);
                if (mesh->wrapFaceUVs(from, static_cast<uint32_t>(picked),
                        levelSlotTexelSize(*mesh, mesh->faces[from].material),
                        levelSlotTexelSize(*mesh, mesh->faces[picked].material))) {
                    rebuildSelectedLevelModel("wrap texture");
                    if (std::find(m_selectedFaces.begin(), m_selectedFaces.end(), uint32_t(picked)) == m_selectedFaces.end())
                        m_selectedFaces.push_back(static_cast<uint32_t>(picked));
                    m_selectedFace = picked;
                }
                else {
                    setStatus("Wrap needs a face sharing an edge with the active face", true);
                }
                return true;
            }
        }
        if (shift) {
            selectLevelFace(static_cast<uint32_t>(picked), true);
            return true;
        }
        selectLevelFace(static_cast<uint32_t>(picked), false);
        // Pressing on a face also grabs it, so it can be pushed/pulled (or extruded with Alt) right away.
        if (const PolyMesh* mesh = selectedLevelMesh()) {
            const PolyFace& face = mesh->faces[m_selectedFace];
            m_faceDrag.instance = m_gizmo.selectedInstance;
            m_faceDrag.face = static_cast<uint32_t>(m_selectedFace);
            m_faceDrag.center = mesh->faceCenter(face);
            m_faceDrag.normal = mesh->faceNormal(face);
            if (faceDragParam(mouseX, mouseY, m_faceDrag.startParam)) {
                m_faceDrag.active = true;
                m_faceDrag.moved = false;
                m_faceDrag.extrude = (SDL_GetModState() & SDL_KMOD_ALT) != 0;
                m_faceDrag.distance = 0.0f;
                m_faceDrag.startMesh = *mesh;
            }
        }
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

    // The shape's grid lives in its object space, so it moves and rotates with the shape.
    const glm::mat4 inverse = glm::inverse(m_models.getInstances()[m_gizmo.selectedInstance].getTransformMatrix());
    glm::vec3 grabbed = glm::vec3(inverse * glm::vec4(world, 1.0f));
    if (snapActive())
        grabbed = glm::round(grabbed / mesh->gridSize) * mesh->gridSize;
    const glm::vec3 delta = grabbed - glm::vec3(inverse * glm::vec4(m_vertexDrag.planePoint, 1.0f));
    for (size_t i = 0; i < vertices.size(); ++i) {
        const glm::vec3 local = m_vertexDrag.startPositions[i] + delta;
        if (local != mesh->positions[vertices[i]]) {
            mesh->positions[vertices[i]] = local;
            // Rebuilt once per frame in updateLevelHistory(); several motion events can arrive per frame.
            m_vertexDrag.moved = true;
        }
    }
}

bool Editor::faceDragParam(float mouseX, float mouseY, float& param) const
{
    if (!validInstance(m_faceDrag.instance))
        return false;
    // The normal line in world space, parameterized in object units so the result needs no conversion.
    const glm::mat4 transform = m_models.getInstances()[m_faceDrag.instance].getTransformMatrix();
    const glm::vec3 origin = glm::vec3(transform * glm::vec4(m_faceDrag.center, 1.0f));
    const glm::vec3 axis = glm::vec3(transform * glm::vec4(m_faceDrag.normal, 0.0f));
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height,
        getView(m_camera), sceneProjection());
    // Closest points of two lines.
    const glm::vec3 w0 = origin - ray.origin;
    const float a = glm::dot(axis, axis);
    const float b = glm::dot(axis, ray.direction);
    const float c = glm::dot(ray.direction, ray.direction);
    const float d = glm::dot(axis, w0);
    const float e = glm::dot(ray.direction, w0);
    const float denom = a * c - b * b;
    if (denom < 1e-6f * a * c)
        return false;
    param = (b * e - c * d) / denom;
    return std::isfinite(param);
}

void Editor::dragLevelFace(float mouseX, float mouseY)
{
    PolyMesh* mesh = selectedLevelMesh();
    // Anything changing the selection mid-drag (undo, deletion, another object) ends the drag.
    if (!mesh || m_gizmo.selectedInstance != m_faceDrag.instance || m_levelMode != LevelEditMode::Face ||
        m_faceDrag.face >= m_faceDrag.startMesh.faces.size()) {
        m_faceDrag.active = false;
        return;
    }
    float param;
    if (!faceDragParam(mouseX, mouseY, param))
        return;
    float distance = param - m_faceDrag.startParam;
    if (snapActive())
        distance = std::round(distance / m_faceDrag.startMesh.gridSize) * m_faceDrag.startMesh.gridSize;
    if (distance == m_faceDrag.distance)
        return;
    m_faceDrag.distance = distance;
    *mesh = m_faceDrag.startMesh;
    if (std::abs(distance) > 1e-6f) {
        if (m_faceDrag.extrude)
            mesh->extrudeFace(m_faceDrag.face, distance);
        else
            mesh->moveFace(m_faceDrag.face, distance);
    }
    // Rebuilt once per frame in updateLevelHistory(); several motion events can arrive per frame.
    m_faceDrag.moved = true;
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
    if (selected) {
        // The rest of the group thinner, the active face on top.
        for (uint32_t f : selectedLevelFaces())
            if (static_cast<int>(f) != m_selectedFace)
                drawFace(*mesh, mesh->faces[f], kFaceGroupOutline, 2.0f);
        drawFace(*mesh, *selected, kFaceOutline, 3.0f);
    }
    if (selected && m_faceDrag.active && m_faceDrag.distance != 0.0f) {
        const glm::vec2 p = worldToScreen(mesh->faceCenter(*selected), mvp, m_sceneView.width, m_sceneView.height);
        if (p.x > -5000.0f) {
            char label[48];
            snprintf(label, sizeof(label), "%s %+g", m_faceDrag.extrude ? "Extrude" : "Push/pull", m_faceDrag.distance);
            drawList->AddText(ImVec2(origin.x + p.x + 8.0f, origin.y + p.y - 8.0f), kFaceOutline, label);
        }
    }
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
    // In face/vertex mode Del never removes the whole object, even with nothing picked.
    if (m_levelMode == LevelEditMode::Object || !selectedLevelMesh())
        return false;
    if (m_vertexDrag.active || m_faceDrag.active || m_levelEditPending)
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
    point = glm::round(point / m_gridSize) * m_gridSize;
    point.y = std::max(point.y, 0.0f);
    return point;
}

void Editor::addLevelShape(PolyShape shape)
{
    PolyShapeParams params = m_newShape;
    params.shape = shape;
    const std::string name = uniqueInstanceName(kPolyShapeNames[static_cast<int>(shape)]);
    // Placed on a world grid point with its sides on its own grid, so they lie on world grid lines too.
    PolyMesh mesh = makeShapeOnGrid(params, m_gridSize, m_gridSnap);
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
        "Images{.png,.jpg,.jpeg,.tga,.bmp},.png,.jpg,.jpeg,.tga,.bmp", kTexturesRoot, nullptr, false);
}

void Editor::drawLevelTextureDialog()
{
    ImGuiFileDialog* dialog = ImGuiFileDialog::Instance();
    if (!dialog->Display("BrowseLevelTextureDlg", ImGuiWindowFlags_NoCollapse, kDialogSize))
        return;
    const TextureDialogTarget& target = m_textureDialogTarget;
    if (dialog->IsOk()) {
        const std::string texture = toStoredPath(dialog->GetFilePathName());
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
        ImGui::DragFloat3("##size", &m_newShape.size.x, m_gridSize * 0.05f, 0.01f, 1000.0f, "%.3g");
        if (ImGui::IsItemDeactivatedAfterEdit() && m_gridSnap)
            m_newShape = snapShapeSize(m_newShape, m_gridSize);
        ImGui::SetItemTooltip("Rounded to whole grid cells while snapping is on");
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
                PolyMesh rebuilt = makeShapeOnGrid(m_newShape, mesh->gridSize, m_gridSnap);
                rebuilt.materials = std::move(mesh->materials);
                *mesh = std::move(rebuilt);
                clearFaceSelection();
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
                    if (ImGui::Button("Copy UVs", ImVec2(half, 0.0f)))
                        copyFaceAttributes();
                    ImGui::SetItemTooltip("Copy the active face's material and UVs (Ctrl+Shift+C)");
                    ImGui::SameLine();
                    ImGui::BeginDisabled(!m_faceClipboard.valid);
                    if (ImGui::Button("Paste UVs", ImVec2(half, 0.0f)))
                        pasteFaceAttributes();
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("Paste material and UVs onto the selected faces (Ctrl+Shift+V)");

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
                    ImGui::TextDisabled("Shift+click adds. Drag to push/pull, Alt+drag to extrude.");
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
