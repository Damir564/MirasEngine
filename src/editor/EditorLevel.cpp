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
