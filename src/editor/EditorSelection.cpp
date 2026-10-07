#include "Editor.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <exception>
#include <limits>
#include <unordered_map>
#include "EditorStyle.h"
#include "engine/ModelManager.h"

namespace {

void insertSorted(std::vector<uint64_t>& ids, uint64_t id)
{
    const auto at = std::lower_bound(ids.begin(), ids.end(), id);
    if (at == ids.end() || *at != id)
        ids.insert(at, id);
}

std::string objectCount(size_t count)
{
    return std::to_string(count) + (count == 1 ? " object" : " objects");
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Selection set
// ---------------------------------------------------------------------------------------------

bool Editor::isSelected(int index) const
{
    return validInstance(index) &&
        std::binary_search(m_selection.begin(), m_selection.end(), m_models.getInstances()[index].id);
}

std::vector<int> Editor::selectedIndices(bool activeFirst) const
{
    std::vector<int> indices;
    if (m_selection.empty())
        return indices;
    indices.reserve(m_selection.size());
    const auto& instances = m_models.getInstances();
    for (int i = 0; i < static_cast<int>(instances.size()); ++i)
        if (std::binary_search(m_selection.begin(), m_selection.end(), instances[i].id))
            indices.push_back(i);
    if (activeFirst && hasSelection()) {
        const auto active = std::find(indices.begin(), indices.end(), m_gizmo.selectedInstance);
        if (active != indices.end())
            std::rotate(indices.begin(), active, active + 1);
    }
    return indices;
}

void Editor::setActive(int index)
{
    if (!validInstance(index)) {
        m_gizmo.deselect();
        m_activeId = 0;
        return;
    }
    m_gizmo.select(index);
    m_activeId = m_models.getInstances()[index].id;
}

void Editor::selectInstance(int index)
{
    m_selection.clear();
    if (validInstance(index))
        m_selection.push_back(m_models.getInstances()[index].id);
    setActive(index);
}

void Editor::deselectAll()
{
    m_selection.clear();
    setActive(-1);
}

void Editor::addToSelection(int index)
{
    if (!validInstance(index))
        return;
    insertSorted(m_selection, m_models.getInstances()[index].id);
    setActive(index);
}

void Editor::toggleSelection(int index)
{
    if (!validInstance(index))
        return;
    if (!isSelected(index)) {
        addToSelection(index);
        return;
    }
    const uint64_t id = m_models.getInstances()[index].id;
    m_selection.erase(std::lower_bound(m_selection.begin(), m_selection.end(), id));
    if (m_gizmo.selectedInstance == index) {
        // The most recently listed of the others takes over the gizmo.
        const std::vector<int> rest = selectedIndices();
        setActive(rest.empty() ? -1 : rest.back());
    }
}

void Editor::selectAll()
{
    const auto& instances = m_models.getInstances();
    std::vector<int> visible;
    for (int i = 0; i < static_cast<int>(instances.size()); ++i)
        if (instances[i].visible)
            visible.push_back(i);
    if (visible.empty())
        return;
    const bool keepActive = hasSelection() && instances[m_gizmo.selectedInstance].visible;
    selectIndices(visible, keepActive ? m_gizmo.selectedInstance : -1);
    setStatus("Selected " + objectCount(visible.size()));
}

void Editor::selectIndices(const std::vector<int>& indices, int active)
{
    m_selection.clear();
    for (int index : indices)
        if (validInstance(index))
            m_selection.push_back(m_models.getInstances()[index].id);
    std::sort(m_selection.begin(), m_selection.end());
    if (std::find(indices.begin(), indices.end(), active) == indices.end())
        active = indices.empty() ? -1 : indices.front();
    setActive(active);
}

void Editor::selectRange(int to)
{
    if (!hasSelection()) {
        selectInstance(to);
        return;
    }
    const int anchor = m_gizmo.selectedInstance;
    const auto& instances = m_models.getInstances();
    std::vector<int> range;
    for (int i = std::min(anchor, to); i <= std::max(anchor, to); ++i)
        if (i == anchor || !m_hierarchyFilter.IsActive() || m_hierarchyFilter.PassFilter(instances[i].name.c_str()))
            range.push_back(i);
    // The anchor stays active, so the next Shift+click ranges from it again.
    selectIndices(range, anchor);
}

void Editor::clickSelect(int index, SDL_Keymod mods)
{
    if (mods & SDL_KMOD_CTRL)
        toggleSelection(index);
    else if (mods & SDL_KMOD_SHIFT)
        addToSelection(index);
    else
        selectInstance(index);
}

void Editor::validateSelection()
{
    const auto& instances = m_models.getInstances();
    // Indices shift when objects before the active one are removed or restored; follow its id.
    if (m_activeId != 0 && (!hasSelection() || instances[m_gizmo.selectedInstance].id != m_activeId)) {
        const auto found = std::find_if(instances.begin(), instances.end(),
            [&](const ModelInstance& instance) { return instance.id == m_activeId; });
        setActive(found == instances.end() ? -1 : static_cast<int>(found - instances.begin()));
    }
    if (hasSelection())
        insertSorted(m_selection, m_activeId);
    if (m_selection.empty())
        return;

    // Drop objects that are gone (deleted by undo, a command or a scene load).
    size_t present = 0;
    for (const ModelInstance& instance : instances)
        if (std::binary_search(m_selection.begin(), m_selection.end(), instance.id))
            ++present;
    if (present != m_selection.size()) {
        std::vector<uint64_t> kept;
        kept.reserve(present);
        for (const ModelInstance& instance : instances)
            if (std::binary_search(m_selection.begin(), m_selection.end(), instance.id))
                kept.push_back(instance.id);
        std::sort(kept.begin(), kept.end());
        m_selection = std::move(kept);
    }
    if (!hasSelection() && !m_selection.empty())
        setActive(selectedIndices().front());
}

// ---------------------------------------------------------------------------------------------
// Picking and box selection
// ---------------------------------------------------------------------------------------------

int Editor::objectUnderMouse(float mouseX, float mouseY) const
{
    const Ray ray = screenToWorldRay(mouseX, mouseY, m_sceneView.width, m_sceneView.height, getView(m_camera),
        sceneProjection());
    return pickSubmesh(ray, m_models.getInstances(), [&](size_t index) { return m_models.getModel(index); })
        .instanceIndex;
}

void Editor::finishObjectMarquee(float mouseX, float mouseY)
{
    m_objectMarquee.active = false;
    const glm::vec2 lo = glm::min(m_objectMarquee.start, glm::vec2(mouseX, mouseY));
    const glm::vec2 hi = glm::max(m_objectMarquee.start, glm::vec2(mouseX, mouseY));
    if (!m_objectMarquee.additive)
        deselectAll();
    if (hi.x - lo.x < 4.0f && hi.y - lo.y < 4.0f)
        return; // a click on empty space

    const glm::mat4 viewProj = sceneProjection() * getView(m_camera);
    const auto& instances = m_models.getInstances();
    size_t found = 0;
    int first = -1;
    for (int i = 0; i < static_cast<int>(instances.size()); ++i) {
        glm::vec3 boundsMin, boundsMax;
        if (!instances[i].visible || !instanceWorldBounds(i, boundsMin, boundsMax))
            continue;
        const glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
        if ((viewProj * glm::vec4(center, 1.0f)).w <= 1e-4f)
            continue;
        const glm::vec2 p = worldToScreen(center, viewProj, m_sceneView.width, m_sceneView.height);
        if (p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y)
            continue;
        insertSorted(m_selection, instances[i].id);
        if (first < 0)
            first = i;
        ++found;
    }
    if (!hasSelection() && first >= 0)
        setActive(first);
    expandSelectionToGroups();
    setStatus(found ? "Selected " + objectCount(selectionCount()) : "No objects in the box");
}

void Editor::drawObjectMarquee()
{
    if (!m_objectMarquee.active)
        return;
    const glm::vec2 lo = glm::min(m_objectMarquee.start, m_objectMarquee.end);
    const glm::vec2 hi = glm::max(m_objectMarquee.start, m_objectMarquee.end);
    if (hi.x - lo.x < 4.0f && hi.y - lo.y < 4.0f)
        return;
    const ImVec2 origin(m_sceneView.x, m_sceneView.y);
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->PushClipRect(origin, ImVec2(origin.x + m_sceneView.width, origin.y + m_sceneView.height), false);
    const ImVec2 a(origin.x + lo.x, origin.y + lo.y);
    const ImVec2 b(origin.x + hi.x, origin.y + hi.y);
    drawList->AddRectFilled(a, b, ImGui::GetColorU32(EditorStyle::withAlpha(EditorStyle::kAccent, 0.15f)));
    drawList->AddRect(a, b, ImGui::GetColorU32(EditorStyle::kAccent));
    drawList->PopClipRect();
}

// ---------------------------------------------------------------------------------------------
// Bounds and focus
// ---------------------------------------------------------------------------------------------

bool Editor::instanceWorldBounds(int index, glm::vec3& lo, glm::vec3& hi) const
{
    if (!validInstance(index))
        return false;
    const ModelInstance& instance = m_models.getInstances()[index];
    const GPUModel* model = m_models.getModel(instance.modelIndex);
    if (!model)
        return false;
    // Level shapes from their mesh, which is current even before a deferred rebuild.
    const glm::mat4 transform = instance.getTransformMatrix();
    lo = glm::vec3(FLT_MAX);
    hi = glm::vec3(-FLT_MAX);
    const auto add = [&](const glm::vec3& p) {
        const glm::vec3 world(transform * glm::vec4(p, 1.0f));
        lo = glm::min(lo, world);
        hi = glm::max(hi, world);
    };
    if (model->polyMesh) {
        for (const glm::vec3& p : model->polyMesh->positions)
            add(p);
    }
    else {
        for (int i = 0; i < 8; ++i)
            add({ (i & 1) ? model->boundsMax.x : model->boundsMin.x, (i & 2) ? model->boundsMax.y : model->boundsMin.y,
                (i & 4) ? model->boundsMax.z : model->boundsMin.z });
    }
    return lo.x <= hi.x;
}

bool Editor::selectionWorldBounds(glm::vec3& lo, glm::vec3& hi) const
{
    lo = glm::vec3(FLT_MAX);
    hi = glm::vec3(-FLT_MAX);
    for (int index : selectedIndices()) {
        glm::vec3 objectMin, objectMax;
        if (!instanceWorldBounds(index, objectMin, objectMax))
            continue;
        lo = glm::min(lo, objectMin);
        hi = glm::max(hi, objectMax);
    }
    return lo.x <= hi.x;
}

void Editor::focusSelection()
{
    if (selectionCount() <= 1) {
        focusOnInstance(m_gizmo.selectedInstance);
        return;
    }
    glm::vec3 lo, hi;
    if (!selectionWorldBounds(lo, hi))
        return;
    // A sphere around the group's box fits the vertical field of view at radius / sin(fov / 2); 10% margin.
    const float radius = glm::length(hi - lo) * 0.5f;
    const float fit = 1.1f / std::sin(glm::radians(m_prefs.fieldOfView * 0.5f));
    m_camera.position = (lo + hi) * 0.5f - getFront(m_camera) * std::max(radius * fit, 1.0f);
}

// ---------------------------------------------------------------------------------------------
// Group actions
// ---------------------------------------------------------------------------------------------

void Editor::deleteSelection()
{
    const std::vector<int> indices = selectedIndices();
    if (indices.size() <= 1) {
        deleteInstance(m_gizmo.selectedInstance);
        return;
    }
    deselectAll();
    // Back to front, so the remaining indices stay valid.
    for (auto it = indices.rbegin(); it != indices.rend(); ++it)
        m_models.removeInstance(static_cast<size_t>(*it));
    releaseUnusedLevelModels();
    markSceneChanged();
    setStatus("Deleted " + objectCount(indices.size()));
}

void Editor::duplicateSelection()
{
    const std::vector<int> indices = selectedIndices();
    if (indices.size() <= 1) {
        duplicateInstance(m_gizmo.selectedInstance);
        return;
    }
    // Beside the group along X, a whole number of grid cells away so the copies stay on the grid.
    glm::vec3 lo(0.0f), hi(0.0f);
    selectionWorldBounds(lo, hi);
    const float offset = std::max(std::ceil((hi.x - lo.x) / m_gridSize), 1.0f) * m_gridSize;
    const auto groupNames = groupNamesForCopies(indices);
    std::vector<int> copies;
    int activeCopy = -1;
    for (int index : indices) {
        const auto copy = copyInstance(index, glm::vec3(offset, 0.0f, 0.0f));
        if (!copy)
            break;
        ModelInstance& created = m_models.getInstances()[*copy];
        if (const auto renamed = groupNames.find(created.group); renamed != groupNames.end())
            created.group = renamed->second;
        copies.push_back(static_cast<int>(*copy));
        if (index == m_gizmo.selectedInstance)
            activeCopy = static_cast<int>(*copy);
    }
    if (copies.empty())
        return;
    selectIndices(copies, activeCopy);
    setStatus("Duplicated " + objectCount(copies.size()));
}

void Editor::copySelection()
{
    m_clipboard.clear();
    // Active first: it is the active object again after pasting.
    for (int index : selectedIndices(true)) {
        const ModelInstance& instance = m_models.getInstances()[index];
        const GPUModel* model = m_models.getModel(instance.modelIndex);
        if (!model)
            continue;
        ClipboardObject object;
        object.instance = instance;
        object.modelName = model->name;
        if (!model->prefabPath.empty())
            object.prefabPath = model->prefabPath;
        else if (model->polyMesh)
            object.mesh = std::make_shared<const PolyMesh>(*model->polyMesh);
        else
            object.modelPath = model->sourcePath;
        m_clipboard.push_back(std::move(object));
    }
    setStatus("Copied " + objectCount(m_clipboard.size()));
}

void Editor::cutSelection()
{
    copySelection();
    const size_t count = m_clipboard.size();
    deleteSelection();
    setStatus("Cut " + objectCount(count));
}

void Editor::pasteClipboard()
{
    if (m_clipboard.empty()) {
        setStatus("Nothing to paste");
        return;
    }
    // Groups pasted with several objects become groups of their own; a single object joins its group.
    std::unordered_map<std::string, size_t> groupCounts;
    for (const ClipboardObject& object : m_clipboard)
        if (!object.instance.group.empty())
            ++groupCounts[object.instance.group];
    std::unordered_map<std::string, std::string> groupNames;
    std::vector<std::string> given;
    for (const auto& [group, count] : groupCounts) {
        if (count < 2)
            continue;
        // Free in the scene and not given to another pasted group.
        std::string name = uniqueGroupName(group);
        for (int n = 2; std::find(given.begin(), given.end(), name) != given.end(); ++n)
            name = uniqueGroupName(group + " (" + std::to_string(n) + ")");
        given.push_back(name);
        groupNames[group] = name;
    }
    std::vector<int> pasted;
    for (const ClipboardObject& object : m_clipboard) {
        std::optional<size_t> modelIndex;
        if (!object.prefabPath.empty()) {
            modelIndex = loadPrefabModel(object.prefabPath);
        }
        else if (object.mesh) {
            modelIndex = createLevelModel(*object.mesh, object.modelName);
        }
        else if (const auto loaded = m_models.findModelByPath(object.modelPath)) {
            modelIndex = loaded;
        }
        else {
            // Copied from a scene that has since been closed.
            try {
                modelIndex = m_models.loadModelSync(object.modelPath, object.modelName);
            }
            catch (const std::exception& e) {
                setStatus("Cannot paste " + object.instance.name + ": " + e.what(), true);
            }
        }
        if (!modelIndex)
            continue;
        const ModelInstance& source = object.instance;
        const size_t index = m_models.createInstance(*modelIndex, source.position, source.rotation, source.scale);
        ModelInstance& created = m_models.getInstances()[index];
        created.name = uniqueInstanceName(source.name);
        created.color = source.color;
        created.visible = source.visible;
        created.locked = source.locked;
        created.entity = source.entity;
        created.entityParams = source.entityParams;
        const auto renamed = groupNames.find(source.group);
        created.group = renamed != groupNames.end() ? renamed->second : source.group;
        pasted.push_back(static_cast<int>(index));
    }
    if (pasted.empty())
        return;
    markSceneChanged();
    selectIndices(pasted, pasted.front());
    setStatus("Pasted " + objectCount(pasted.size()));
}

float Editor::distanceToSurfaceBelow(const glm::vec3& origin, int ignore) const
{
    const auto& instances = m_models.getInstances();
    float best = std::numeric_limits<float>::infinity();
    for (int i = 0; i < static_cast<int>(instances.size()); ++i) {
        const ModelInstance& instance = instances[i];
        if (i == ignore || !instance.visible)
            continue;
        const GPUModel* model = m_models.getModel(instance.modelIndex);
        if (!model || !model->isValid())
            continue;
        // Unnormalized local direction: an affine map keeps the ray parameter, so local t is world t.
        const glm::mat4 inverse = glm::inverse(instance.getTransformMatrix());
        Ray local;
        local.origin = glm::vec3(inverse * glm::vec4(origin, 1.0f));
        local.direction = glm::vec3(inverse * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f));
        float t = 0.0f;
        if (!rayIntersectsAABB(local, model->boundsMin, model->boundsMax, t) || t >= best)
            continue;
        for (const SubmeshInfo& sub : model->submeshes)
            if (rayIntersectsSubmesh(local, *model, sub, best, t))
                best = t;
    }
    return best;
}

void Editor::dropSelectionToFloor()
{
    if (m_gizmo.isDragging)
        return;
    struct Item {
        int index;
        glm::vec3 lo, hi;
    };
    std::vector<Item> items;
    for (int index : selectedIndices()) {
        Item item{ index, glm::vec3(0.0f), glm::vec3(0.0f) };
        if (instanceWorldBounds(index, item.lo, item.hi))
            items.push_back(item);
    }
    // Lowest first, so selected objects stacked on each other land on the ones already dropped.
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.lo.y < b.lo.y; });

    size_t moved = 0;
    for (const Item& item : items) {
        // Rays from just above the bottom at the middle and near the corners, so an object over an edge
        // lands on it. Nothing below: the ground (y = 0).
        const glm::vec3 size = item.hi - item.lo;
        const float startY = item.lo.y + std::max(size.y * 0.01f, 1e-3f);
        const glm::vec2 center((item.lo.x + item.hi.x) * 0.5f, (item.lo.z + item.hi.z) * 0.5f);
        const glm::vec2 half = glm::vec2(size.x, size.z) * 0.45f;
        static constexpr float kOffsets[][2] = { { 0, 0 }, { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, 1 } };
        float nearest = std::numeric_limits<float>::infinity();
        for (const auto& offset : kOffsets) {
            const glm::vec3 origin(center.x + offset[0] * half.x, startY, center.y + offset[1] * half.y);
            nearest = std::min(nearest, distanceToSurfaceBelow(origin, item.index));
        }
        const float floorY = std::isfinite(nearest) ? startY - nearest : 0.0f;
        const float drop = item.lo.y - floorY;
        if (std::abs(drop) < 1e-5f)
            continue;
        m_models.getInstances()[item.index].position.y -= drop;
        ++moved;
    }
    if (moved == 0) {
        setStatus("Already resting on a surface");
        return;
    }
    markSceneChanged();
    setStatus("Dropped " + objectCount(moved));
}

void Editor::rotateSelection(float degrees)
{
    if (m_gizmo.isDragging || !hasSelection())
        return;
    auto& instances = m_models.getInstances();
    const glm::vec3 pivot = instances[m_gizmo.selectedInstance].position;
    float c = std::cos(glm::radians(degrees));
    float s = std::sin(glm::radians(degrees));
    // Exact quarter turns keep positions on the grid.
    if (std::fmod(degrees, 90.0f) == 0.0f) {
        c = std::round(c);
        s = std::round(s);
    }
    const std::vector<int> indices = selectedIndices();
    for (int index : indices) {
        ModelInstance& instance = instances[index];
        const glm::vec3 d = instance.position - pivot;
        instance.position = pivot + glm::vec3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
        instance.rotation = rotateEulerAboutAxis(instance.rotation, 1, degrees);
    }
    markSceneChanged();
    setStatus("Rotated " + objectCount(indices.size()) + (degrees < 0.0f ? " clockwise" : " counter-clockwise"));
}

void Editor::toggleSelectionVisibility()
{
    const std::vector<int> indices = selectedIndices();
    auto& instances = m_models.getInstances();
    const bool anyVisible = std::any_of(indices.begin(), indices.end(), [&](int i) { return instances[i].visible; });
    for (int index : indices)
        instances[index].visible = !anyVisible;
    markSceneChanged();
    if (anyVisible) {
        const std::string showAll = m_keymap.shortcutLabel(EditorAction::UnhideAll);
        setStatus("Hidden " + objectCount(indices.size()) + (showAll.empty() ? "" : " (" + showAll + " shows all)"));
    }
    else {
        setStatus("Shown " + objectCount(indices.size()));
    }
}

void Editor::unhideAll()
{
    size_t shown = 0;
    for (ModelInstance& instance : m_models.getInstances()) {
        if (instance.visible)
            continue;
        instance.visible = true;
        ++shown;
    }
    if (shown > 0)
        markSceneChanged();
    setStatus(shown ? "Shown " + objectCount(shown) : "Nothing is hidden");
}

// ---------------------------------------------------------------------------------------------
// Group gizmo drags
// ---------------------------------------------------------------------------------------------

void Editor::beginGroupDrag()
{
    m_groupDrag.clear();
    if (selectionCount() <= 1)
        return;
    for (int index : selectedIndices()) {
        if (index == m_gizmo.selectedInstance)
            continue;
        const ModelInstance& instance = m_models.getInstances()[index];
        m_groupDrag.push_back({ index, instance.position, instance.rotation, instance.scale });
    }
}

void Editor::dragGroup(int axis, float degrees, float scaleFactor)
{
    if (m_groupDrag.empty() || !hasSelection())
        return;
    auto& instances = m_models.getInstances();
    const glm::vec3 pivot = m_gizmo.originalPosition;
    const glm::vec3 moved = instances[m_gizmo.selectedInstance].position - pivot;
    const glm::mat3 turn = glm::mat3_cast(
        glm::angleAxis(glm::radians(degrees), gizmoAxisDirection(static_cast<GizmoAxis>(axis + 1))));
    for (const GroupDragItem& item : m_groupDrag) {
        if (!validInstance(item.index))
            continue;
        ModelInstance& instance = instances[item.index];
        switch (m_gizmo.mode) {
        case GizmoMode::Translate:
            instance.position = item.position + moved;
            break;
        case GizmoMode::Rotate:
            instance.position = pivot + turn * (item.position - pivot);
            instance.rotation = rotateEulerAboutAxis(item.rotation, axis, degrees);
            break;
        case GizmoMode::Scale:
            instance.position = item.position;
            instance.position[axis] = pivot[axis] + (item.position[axis] - pivot[axis]) * scaleFactor;
            instance.scale = item.scale;
            instance.scale[axis] = std::max(item.scale[axis] * scaleFactor, 0.01f);
            break;
        default:
            break;
        }
    }
}
