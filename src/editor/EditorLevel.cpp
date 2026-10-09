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

namespace {
constexpr const char* kTexturesRoot = "textures";
constexpr ImU32 kFaceOutline = IM_COL32(255, 160, 40, 255);
constexpr ImU32 kFaceGroupOutline = IM_COL32(255, 200, 120, 190);
constexpr ImU32 kFaceEdges = IM_COL32(255, 255, 255, 70);
constexpr ImU32 kVertexDot = IM_COL32(255, 255, 255, 200);
constexpr ImU32 kClipOutline = IM_COL32(80, 200, 255, 255);
constexpr ImU32 kMirrorOutline = IM_COL32(230, 110, 255, 255);
constexpr ImU32 kDrawOutline = IM_COL32(110, 255, 140, 255);
constexpr float kVertexPickRadius = 10.0f;

int dominantAxis(const glm::vec3& n)
{
    const glm::vec3 a = glm::abs(n);
    return a.x >= a.y && a.x >= a.z ? 0 : (a.y >= a.z ? 1 : 2);
}

// Recomputes coordinate `axis` of p so that p lies on the plane through `point` with normal n.
glm::vec3 ontoPlaneAlong(glm::vec3 p, int axis, const glm::vec3& point, const glm::vec3& n)
{
    p[axis] = 0.0f;
    p[axis] = (glm::dot(n, point) - glm::dot(n, p)) / n[axis];
    return p;
}

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

// Edge edits move along directions that are often diagonal (the edge between two box faces), so the
// offset is snapped per axis: ends that started on the grid stay on it.
glm::vec3 edgeOffset(const glm::vec3& direction, float distance, float grid, bool snap)
{
    const glm::vec3 offset = direction * distance;
    return snap ? glm::round(offset / grid) * grid : offset;
}

// Moves edge a-b by offset, or extrudes it; returns the edge to select afterwards (the new one).
std::array<uint32_t, 2> editEdge(PolyMesh& mesh, uint32_t a, uint32_t b, const glm::vec3& offset, bool extrude)
{
    if (extrude) {
        uint32_t newA, newB;
        const size_t oldFaces = mesh.faces.size();
        if (!mesh.extrudeEdge(a, b, offset, newA, newB))
            return { a, b };
        // Continuing a flat surface keeps it one face: the new quad joins the face it extends.
        std::vector<int64_t> keys(mesh.faces.size(), -1);
        std::iota(keys.begin(), keys.begin() + oldFaces, int64_t(0));
        std::vector<uint32_t> remap;
        mesh.mergeCoplanarFaces(keys, &remap);
        return { remap[newA], remap[newB] };
    }
    mesh.positions[a] += offset;
    mesh.positions[b] += offset;
    return { a, b };
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

void Editor::selectSurface()
{
    const PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || !selectedLevelFace())
        return;
    std::vector<uint32_t> selection = selectedLevelFaces();
    for (uint32_t face : std::vector<uint32_t>(selection))
        for (uint32_t f : mesh->coplanarRegion(face))
            if (std::find(selection.begin(), selection.end(), f) == selection.end())
                selection.push_back(f);
    m_selectedFaces = std::move(selection);
    setStatus(m_selectedFaces.size() == 1 ? "The surface is one face"
        : "Surface: " + std::to_string(m_selectedFaces.size()) + " faces selected");
}

void Editor::mergeLevelFaces()
{
    PolyMesh* mesh = selectedLevelMesh();
    if (!mesh)
        return;
    PolyMesh edited = *mesh;
    const size_t before = edited.faces.size();
    const std::vector<uint32_t> remap = edited.mergeCoplanarFaces();
    if (edited.faces.size() == before) {
        setStatus("No neighbouring faces to join (they must be flat together and share material and UVs)");
        return;
    }
    // The selection follows its faces into the joined ones.
    std::vector<uint32_t> selection;
    for (uint32_t f : selectedLevelFaces())
        if (f < remap.size() && std::find(selection.begin(), selection.end(), remap[f]) == selection.end())
            selection.push_back(remap[f]);
    if (commitLevelTopology(std::move(edited), "join faces")) {
        // Vertices were renumbered.
        m_selectedVertices.clear();
        m_edgeSelected = false;
        m_selectedFaces = selection;
        m_selectedFace = selection.empty() ? -1 : static_cast<int>(selection.front());
        setStatus("Joined " + std::to_string(before) + " faces into " + std::to_string(mesh->faces.size()));
    }
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

bool Editor::selectedLevelEdge(uint32_t& a, uint32_t& b)
{
    const PolyMesh* mesh = selectedLevelMesh();
    // Undo can remove the edge under the selection.
    if (m_levelMode != LevelEditMode::Edge || !m_edgeSelected || !mesh ||
        m_levelSelectionInstance != m_gizmo.selectedInstance || !mesh->isEdge(m_selectedEdge[0], m_selectedEdge[1])) {
        m_edgeSelected = false;
        return false;
    }
    a = m_selectedEdge[0];
    b = m_selectedEdge[1];
    return true;
}

bool Editor::pickLevelEdge(float mouseX, float mouseY, uint32_t& outA, uint32_t& outB) const
{
    if (!hasSelection())
        return false;
    const ModelInstance& instance = m_models.getInstances()[m_gizmo.selectedInstance];
    const GPUModel* model = m_models.getModel(instance.modelIndex);
    if (!model || !model->polyMesh || !instance.visible || instance.locked)
        return false;
    const PolyMesh& mesh = *model->polyMesh;

    const glm::mat4 mvp = sceneProjection() * getView(m_camera) * instance.getTransformMatrix();
    std::vector<glm::vec2> screen(mesh.positions.size());
    std::vector<float> depth(mesh.positions.size());
    for (size_t v = 0; v < mesh.positions.size(); ++v) {
        depth[v] = (mvp * glm::vec4(mesh.positions[v], 1.0f)).w;
        if (depth[v] > 1e-4f)
            screen[v] = worldToScreen(mesh.positions[v], mvp, m_sceneView.width, m_sceneView.height);
    }
    const glm::vec2 mouse(mouseX, mouseY);
    bool found = false;
    float bestDistance = kVertexPickRadius;
    float bestDepth = FLT_MAX;
    for (const PolyFace& face : mesh.faces) {
        const size_t count = face.verts.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t u = face.verts[i], w = face.verts[(i + 1) % count];
            if (depth[u] <= 1e-4f || depth[w] <= 1e-4f)
                continue;
            const glm::vec2 along = screen[w] - screen[u];
            const float length2 = glm::dot(along, along);
            const float t = length2 > 1e-6f ? std::clamp(glm::dot(mouse - screen[u], along) / length2, 0.0f, 1.0f) : 0.0f;
            const float distance = glm::length(mouse - (screen[u] + along * t));
            const float d = glm::mix(depth[u], depth[w], t);
            // Edges drawn on top of each other go to the nearer one, as with vertices.
            const bool sameSpot = found && std::abs(distance - bestDistance) < 1.0f;
            if (sameSpot ? d < bestDepth : distance < bestDistance) {
                found = true;
                outA = u;
                outB = w;
                bestDistance = distance;
                bestDepth = d;
            }
        }
    }
    return found;
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
    if (m_levelMode == LevelEditMode::Part) {
        const int picked = pickLevelFace(mouseX, mouseY);
        if (picked < 0)
            return false;
        selectPart(static_cast<uint32_t>(picked), (keyMods() & SDL_KMOD_SHIFT) != 0);
        return true;
    }
    if (m_levelMode == LevelEditMode::Face) {
        const int picked = pickLevelFace(mouseX, mouseY);
        if (picked < 0)
            return false;
        const SDL_Keymod mods = keyMods();
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
            if (lineDragParam(m_faceDrag.instance, m_faceDrag.center, m_faceDrag.normal, mouseX, mouseY,
                    m_faceDrag.startParam)) {
                m_faceDrag.active = true;
                m_faceDrag.moved = false;
                m_faceDrag.extrude = (keyMods() & SDL_KMOD_ALT) != 0;
                m_faceDrag.distance = 0.0f;
                m_faceDrag.startMesh = *mesh;
                m_faceDrag.solid = m_faceDrag.extrude && mesh->isClosed();
            }
        }
        return true;
    }

    if (m_levelMode == LevelEditMode::Edge) {
        uint32_t a, b;
        if (!pickLevelEdge(mouseX, mouseY, a, b)) {
            m_edgeSelected = false;
            return false;
        }
        m_levelSelectionInstance = m_gizmo.selectedInstance;
        m_edgeSelected = true;
        m_selectedEdge = { a, b };
        // Pressing on an edge also grabs it, like faces: drag to push/pull, Alt+drag to extrude.
        const PolyMesh* mesh = selectedLevelMesh();
        if (!mesh)
            return true;
        const bool extrude = (keyMods() & SDL_KMOD_ALT) != 0;
        m_edgeDrag.instance = m_gizmo.selectedInstance;
        m_edgeDrag.a = a;
        m_edgeDrag.b = b;
        m_edgeDrag.center = (mesh->positions[a] + mesh->positions[b]) * 0.5f;
        // A border edge moves within its face's plane (resizing a plane); others along their faces' normal.
        m_edgeDrag.direction = mesh->edgeExtrudeDirection(a, b);
        if (m_edgeDrag.direction != glm::vec3(0.0f) &&
            lineDragParam(m_edgeDrag.instance, m_edgeDrag.center, m_edgeDrag.direction, mouseX, mouseY,
                m_edgeDrag.startParam)) {
            m_edgeDrag.active = true;
            m_edgeDrag.moved = false;
            m_edgeDrag.extrude = extrude;
            m_edgeDrag.offset = glm::vec3(0.0f);
            m_edgeDrag.startMesh = *mesh;
        }
        return true;
    }

    const bool shift = (keyMods() & SDL_KMOD_SHIFT) != 0;
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

bool Editor::lineDragParam(int instance, const glm::vec3& point, const glm::vec3& localAxis, float mouseX,
    float mouseY, float& param) const
{
    if (!validInstance(instance))
        return false;
    // The line in world space, parameterized in object units so the result needs no conversion.
    const glm::mat4 transform = m_models.getInstances()[instance].getTransformMatrix();
    const glm::vec3 origin = glm::vec3(transform * glm::vec4(point, 1.0f));
    const glm::vec3 axis = glm::vec3(transform * glm::vec4(localAxis, 0.0f));
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
    if (!lineDragParam(m_faceDrag.instance, m_faceDrag.center, m_faceDrag.normal, mouseX, mouseY, param))
        return;
    float distance = param - m_faceDrag.startParam;
    if (snapActive())
        distance = std::round(distance / m_faceDrag.startMesh.gridSize) * m_faceDrag.startMesh.gridSize;
    if (distance == m_faceDrag.distance)
        return;
    m_faceDrag.distance = distance;
    *mesh = m_faceDrag.startMesh;
    int selected = static_cast<int>(m_faceDrag.face);
    if (std::abs(distance) > 1e-6f) {
        std::vector<uint32_t> caps;
        if (!m_faceDrag.extrude)
            mesh->moveFace(m_faceDrag.face, distance);
        else if (!m_faceDrag.solid)
            mesh->extrudeFace(m_faceDrag.face, distance);
        // A closed shape grows or, pulled inwards, gets cut into; faces move, so the cap is looked up.
        else if (mesh->extrudeFacesSolid({ m_faceDrag.face }, distance, &caps))
            selected = caps.empty() ? -1 : static_cast<int>(caps.front());
    }
    if (m_faceDrag.solid) {
        m_selectedFace = selected;
        m_selectedFaces.clear();
        if (selected >= 0)
            m_selectedFaces.push_back(static_cast<uint32_t>(selected));
    }
    // Rebuilt once per frame in updateLevelHistory(); several motion events can arrive per frame.
    m_faceDrag.moved = true;
}

void Editor::dragLevelEdge(float mouseX, float mouseY)
{
    PolyMesh* mesh = selectedLevelMesh();
    // Anything changing the selection mid-drag (undo, another object, mode) ends the drag.
    const size_t count = m_edgeDrag.startMesh.positions.size();
    if (!mesh || m_gizmo.selectedInstance != m_edgeDrag.instance || m_levelMode != LevelEditMode::Edge ||
        m_edgeDrag.a >= count || m_edgeDrag.b >= count) {
        m_edgeDrag.active = false;
        return;
    }
    float param;
    if (!lineDragParam(m_edgeDrag.instance, m_edgeDrag.center, m_edgeDrag.direction, mouseX, mouseY, param))
        return;
    const glm::vec3 offset = edgeOffset(m_edgeDrag.direction, param - m_edgeDrag.startParam,
        m_edgeDrag.startMesh.gridSize, snapActive());
    if (offset == m_edgeDrag.offset)
        return;
    m_edgeDrag.offset = offset;
    *mesh = m_edgeDrag.startMesh;
    m_selectedEdge = { m_edgeDrag.a, m_edgeDrag.b };
    if (glm::length(offset) > 1e-6f)
        m_selectedEdge = editEdge(*mesh, m_edgeDrag.a, m_edgeDrag.b, offset, m_edgeDrag.extrude);
    m_edgeSelected = true;
    // Rebuilt once per frame in updateLevelHistory().
    m_edgeDrag.moved = true;
}

bool Editor::faceDrawValid()
{
    if (!m_faceDraw.active)
        return false;
    const PolyMesh* mesh = selectedLevelMesh();
    if (!mesh || m_levelMode != LevelEditMode::Face || m_gizmo.selectedInstance != m_faceDraw.instance ||
        m_selectedFace != static_cast<int>(m_faceDraw.face) || m_faceDraw.face >= mesh->faces.size()) {
        m_faceDraw = {};
        return false;
    }
    return true;
}

bool Editor::faceDrawPoint(float mouseX, float mouseY, glm::vec3& out)
{
    const PolyMesh& mesh = *selectedLevelMesh();
    const PolyFace& face = mesh.faces[m_faceDraw.face];
    const glm::mat4 transform = m_models.getInstances()[m_gizmo.selectedInstance].getTransformMatrix();
    const glm::mat4 inverse = glm::inverse(transform);
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height,
        getView(m_camera), sceneProjection());
    Ray localRay;
    localRay.origin = glm::vec3(inverse * glm::vec4(ray.origin, 1.0f));
    localRay.direction = glm::vec3(inverse * glm::vec4(ray.direction, 0.0f));
    const glm::vec3 n = mesh.faceNormal(face);
    const glm::vec3 p0 = mesh.positions[face.verts[0]];
    glm::vec3 hit;
    if (!rayHitsPlane(localRay, p0, n, hit))
        return false;

    const glm::mat4 mvp = sceneProjection() * getView(m_camera) * transform;
    const glm::vec2 mouse(mouseX, mouseY);
    const auto screenDistance = [&](const glm::vec3& p) {
        if ((mvp * glm::vec4(p, 1.0f)).w <= 1e-4f)
            return FLT_MAX;
        return glm::length(worldToScreen(p, mvp, m_sceneView.width, m_sceneView.height) - mouse);
    };
    // Face corners and the shape's own points win, so cuts start exactly on corners and shapes close.
    float best = kVertexPickRadius;
    bool found = false;
    const auto consider = [&](const glm::vec3& p) {
        const float d = screenDistance(p);
        if (d < best) {
            best = d;
            out = p;
            found = true;
        }
    };
    for (uint32_t v : face.verts)
        consider(mesh.positions[v]);
    for (const glm::vec3& p : m_faceDraw.points)
        consider(p);
    if (found)
        return true;
    if (snapActive()) {
        // Snapped on the two object axes across the face, the third following from the plane, so on
        // axis-aligned faces points land exactly on the shape's grid.
        const int axis = dominantAxis(n);
        out = ontoPlaneAlong(glm::round(hit / mesh.gridSize) * mesh.gridSize, axis, p0, n);
        return true;
    }
    out = hit;
    const size_t count = face.verts.size();
    for (size_t i = 0; i < count; ++i) {
        const glm::vec3 a = mesh.positions[face.verts[i]];
        const glm::vec3 ab = mesh.positions[face.verts[(i + 1) % count]] - a;
        const float len2 = glm::dot(ab, ab);
        if (len2 > 0.0f)
            consider(a + ab * std::clamp(glm::dot(hit - a, ab) / len2, 0.0f, 1.0f));
    }
    return true;
}

std::vector<glm::vec3> Editor::faceDrawOutline(const glm::vec3* hover)
{
    std::vector<glm::vec3> points = m_faceDraw.points;
    if (hover)
        points.push_back(*hover);
    if (m_faceDrawShape == FaceDrawShape::Polygon || points.size() < 2)
        return points;
    const PolyMesh& mesh = *selectedLevelMesh();
    const glm::vec3 n = mesh.faceNormal(mesh.faces[m_faceDraw.face]);
    const glm::vec3 a = points[0], b = points[1];
    if (m_faceDrawShape == FaceDrawShape::Rectangle) {
        // Sides follow the object axes across the face, like the grid.
        const int axis = dominantAxis(n);
        const int u = (axis + 1) % 3;
        glm::vec3 c1 = a, c2 = b;
        c1[u] = b[u];
        c2[u] = a[u];
        return { a, ontoPlaneAlong(c1, axis, a, n), b, ontoPlaneAlong(c2, axis, a, n) };
    }
    // Circle around a, with its first corner on b.
    const glm::vec3 radius = b - a;
    const glm::vec3 side = glm::cross(n, radius);
    const int segments = std::clamp(m_faceDrawSegments, 3, 64);
    std::vector<glm::vec3> circle;
    for (int i = 0; i < segments; ++i) {
        const float angle = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(segments);
        circle.push_back(a + radius * std::cos(angle) + side * std::sin(angle));
    }
    return circle;
}

void Editor::handleFaceDrawClick(float mouseX, float mouseY)
{
    glm::vec3 point;
    if (!faceDrawPoint(mouseX, mouseY, point))
        return;
    std::vector<glm::vec3>& points = m_faceDraw.points;
    if (m_faceDrawShape != FaceDrawShape::Polygon) {
        points.push_back(point);
        if (points.size() >= 2)
            applyFaceDraw(true);
        return;
    }
    // faceDrawPoint() snaps to the first point, so clicking it again closes the shape.
    if (points.size() >= 3 && point == points.front()) {
        applyFaceDraw(true);
        return;
    }
    if (!points.empty() && point == points.back())
        return;
    points.push_back(point);
    // A run from the border across the face back to the border is a cut and finishes by itself.
    const PolyMesh& mesh = *selectedLevelMesh();
    const uint32_t face = m_faceDraw.face;
    if (points.size() >= 2 && mesh.onFaceBorder(face, points.front()) && mesh.onFaceBorder(face, point) &&
        !mesh.onFaceBorder(face, (points[points.size() - 2] + point) * 0.5f))
        applyFaceDraw(false);
}

void Editor::applyFaceDraw(bool closed)
{
    if (!faceDrawValid())
        return;
    PolyMesh& mesh = *selectedLevelMesh();
    std::string error;
    const uint32_t piece = mesh.divideFace(m_faceDraw.face, faceDrawOutline(nullptr), closed, &error);
    if (piece == UINT32_MAX) {
        setStatus(error, true);
        // Two-click shapes start over; a polygon keeps its points so the last ones can be taken back.
        if (m_faceDrawShape != FaceDrawShape::Polygon)
            m_faceDraw.points.clear();
        return;
    }
    rebuildSelectedLevelModel("divide face");
    m_faceDraw = {};
    // The part inside the shape, ready to extrude.
    clearFaceSelection();
    selectLevelFace(piece, false);
    setStatus("Face divided");
}

bool Editor::handleFaceDrawKeys()
{
    if (!faceDrawValid())
        return false;
    if (m_keymap.pressed(EditorAction::FaceDrawCancel)) {
        m_faceDraw = {};
        setStatus("Drawing cancelled");
        return true;
    }
    if (m_keymap.pressed(EditorAction::FaceDrawRemovePoint) && !m_faceDraw.points.empty())
        m_faceDraw.points.pop_back();
    if (m_faceDrawShape == FaceDrawShape::Polygon && m_keymap.pressed(EditorAction::FaceDrawApply)) {
        // Ending on the border after starting on it makes a cut; anything else closes the shape.
        const PolyMesh& mesh = *selectedLevelMesh();
        const std::vector<glm::vec3>& points = m_faceDraw.points;
        const bool cut = points.size() >= 2 && mesh.onFaceBorder(m_faceDraw.face, points.front()) &&
            mesh.onFaceBorder(m_faceDraw.face, points.back());
        applyFaceDraw(!cut);
    }
    return true;
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
    if (m_levelMode == LevelEditMode::Part)
        for (uint32_t f : selectedPartFaces())
            drawFace(*mesh, mesh->faces[f], kFaceOutline, 2.0f);
    if (faceDrawValid()) {
        const ImVec2 mousePos = ImGui::GetMousePos();
        glm::vec3 hover;
        const bool hovering = sceneViewContains(mousePos.x, mousePos.y) &&
            faceDrawPoint(mousePos.x - origin.x, mousePos.y - origin.y, hover);
        const std::vector<glm::vec3> shape = faceDrawOutline(hovering ? &hover : nullptr);
        for (size_t i = 0; i + 1 < shape.size(); ++i)
            drawLine(shape[i], shape[i + 1], kDrawOutline, 2.0f);
        if (m_faceDrawShape != FaceDrawShape::Polygon && shape.size() > 2)
            drawLine(shape.back(), shape.front(), kDrawOutline, 2.0f);
        const auto drawDot = [&](const glm::vec3& p, float radius) {
            const glm::vec2 s = worldToScreen(p, mvp, m_sceneView.width, m_sceneView.height);
            if (s.x > -5000.0f)
                drawList->AddCircleFilled(ImVec2(origin.x + s.x, origin.y + s.y), radius, kDrawOutline);
        };
        for (const glm::vec3& p : m_faceDraw.points)
            drawDot(p, 4.0f);
        if (hovering)
            drawDot(hover, 5.5f);
    }
    if (selected && m_faceDrag.active && m_faceDrag.distance != 0.0f) {
        const glm::vec2 p = worldToScreen(mesh->faceCenter(*selected), mvp, m_sceneView.width, m_sceneView.height);
        if (p.x > -5000.0f) {
            char label[48];
            snprintf(label, sizeof(label), "%s %+g", m_faceDrag.extrude ? "Extrude" : "Push/pull", m_faceDrag.distance);
            drawList->AddText(ImVec2(origin.x + p.x + 8.0f, origin.y + p.y - 8.0f), kFaceOutline, label);
        }
    }
    if (m_levelMode == LevelEditMode::Edge) {
        const ImVec2 mousePos = ImGui::GetMousePos();
        uint32_t a, b;
        if (!m_edgeDrag.active && sceneViewContains(mousePos.x, mousePos.y) &&
            pickLevelEdge(mousePos.x - origin.x, mousePos.y - origin.y, a, b))
            drawLine(mesh->positions[a], mesh->positions[b], kFaceGroupOutline, 2.0f);
        if (selectedLevelEdge(a, b)) {
            drawLine(mesh->positions[a], mesh->positions[b], kFaceOutline, 3.0f);
            if (m_edgeDrag.active && m_edgeDrag.offset != glm::vec3(0.0f)) {
                const glm::vec3 middle = (mesh->positions[a] + mesh->positions[b]) * 0.5f;
                const glm::vec2 p = worldToScreen(middle, mvp, m_sceneView.width, m_sceneView.height);
                if (p.x > -5000.0f) {
                    // Signed, so pulling inwards reads as negative.
                    const float distance = std::copysign(glm::length(m_edgeDrag.offset),
                        glm::dot(m_edgeDrag.offset, m_edgeDrag.direction));
                    char label[48];
                    snprintf(label, sizeof(label), "%s %+g", m_edgeDrag.extrude ? "Extrude" : "Push/pull", distance);
                    drawList->AddText(ImVec2(origin.x + p.x + 8.0f, origin.y + p.y - 8.0f), kFaceOutline, label);
                }
            }
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
