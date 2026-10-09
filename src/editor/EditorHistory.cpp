#include "Editor.h"
#include <algorithm>
#include <utility>
#include "engine/ModelManager.h"
#include "engine/SceneManager.h"

namespace {
constexpr size_t kMaxUndoSteps = 100;

// Everything an object change records except the model and the scene position.
bool sameInstanceState(const ModelInstance& a, const ModelInstance& b)
{
    return a.id == b.id && a.name == b.name && a.position == b.position && a.rotation == b.rotation &&
        a.scale == b.scale && a.visible == b.visible && a.color == b.color && a.locked == b.locked &&
        a.entity == b.entity && a.entityParams == b.entityParams && a.group == b.group && a.collision == b.collision;
}
}

void Editor::updateLevelHistory()
{
    if (m_vertexDrag.moved) {
        m_vertexDrag.moved = false;
        if (selectedLevelMesh())
            rebuildSelectedLevelModel("move vertex");
    }
    if (m_faceDrag.moved) {
        m_faceDrag.moved = false;
        if (selectedLevelMesh())
            rebuildSelectedLevelModel(m_faceDrag.extrude ? "extrude" : "push/pull");
    }
    if (m_edgeDrag.moved) {
        m_edgeDrag.moved = false;
        if (selectedLevelMesh())
            rebuildSelectedLevelModel(m_edgeDrag.extrude ? "extrude edge" : "push/pull edge");
    }
    if (m_partDrag.moved) {
        m_partDrag.moved = false;
        if (selectedLevelMesh())
            rebuildSelectedLevelModel(m_gizmo.mode == GizmoMode::Rotate ? "rotate part"
                : m_gizmo.mode == GizmoMode::Scale ? "scale part" : "move part");
    }
    if (m_levelEditPending) {
        if (ImGui::IsAnyItemActive() || m_vertexDrag.active || m_faceDrag.active || m_edgeDrag.active ||
            m_partDrag.active)
            return;
        const auto found = m_models.findModelByPath(m_levelBaselinePath);
        const GPUModel* model = found ? m_models.getModel(*found) : nullptr;
        if (model && model->polyMesh) {
            HistoryEntry entry;
            entry.action = m_levelEditAction;
            entry.levels.push_back({ m_levelBaselinePath, std::move(m_levelBaseline), *model->polyMesh });
            pushHistory(std::move(entry));
            m_levelEntryPushed = true;
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

bool Editor::editInProgress() const
{
    return m_levelEditPending || m_vertexDrag.active || m_faceDrag.active || m_edgeDrag.active || m_partDrag.active ||
        m_gizmo.isDragging || ImGui::IsAnyItemActive();
}

Editor::ObjectRecord Editor::makeObjectRecord(size_t index,
    std::unordered_map<std::string, std::shared_ptr<const PolyMesh>>& meshCopies) const
{
    const ModelInstance& instance = m_models.getInstances()[index];
    ObjectRecord record;
    record.instance = instance;
    record.index = index;
    const auto& models = m_models.getModels();
    if (instance.modelIndex < models.size() && models[instance.modelIndex]) {
        const GPUModel& model = *models[instance.modelIndex];
        record.modelPath = model.sourcePath;
        record.modelName = model.name;
        record.prefabPath = model.prefabPath;
        if (model.polyMesh) {
            // Instances of one prefab share a single copy.
            std::shared_ptr<const PolyMesh>& copy = meshCopies[model.sourcePath];
            if (!copy)
                copy = std::make_shared<const PolyMesh>(*model.polyMesh);
            record.mesh = copy;
        }
    }
    return record;
}

void Editor::resetObjectBaseline()
{
    std::unordered_map<std::string, std::shared_ptr<const PolyMesh>> meshCopies;
    m_objectBaseline.clear();
    m_objectBaseline.reserve(m_models.getInstances().size());
    for (size_t i = 0; i < m_models.getInstances().size(); ++i)
        m_objectBaseline.push_back(makeObjectRecord(i, meshCopies));
    m_objectBaselineValid = true;
    m_levelEntryPushed = false;
    m_sceneChanged = false;
}

bool Editor::objectChangedSinceBaseline()
{
    const auto& instances = m_models.getInstances();
    if (instances.size() != m_objectBaseline.size())
        return true;
    for (size_t i = 0; i < instances.size(); ++i) {
        const ObjectRecord& record = m_objectBaseline[i];
        if (!sameInstanceState(record.instance, instances[i]))
            return true;
        const GPUModel* model = m_models.getModel(instances[i].modelIndex);
        if (!model || model->sourcePath != record.modelPath)
            return true;
    }
    return false;
}

bool Editor::sameObjectState(const ObjectRecord& a, const ObjectRecord& b)
{
    return sameInstanceState(a.instance, b.instance) && a.modelPath == b.modelPath;
}

std::string Editor::describeObjectChanges(const std::vector<ObjectChange>& changes)
{
    if (changes.size() > 1) {
        const std::string count = std::to_string(changes.size());
        if (std::all_of(changes.begin(), changes.end(), [](const ObjectChange& c) { return !c.before; }))
            return "add " + count + " objects";
        if (std::all_of(changes.begin(), changes.end(), [](const ObjectChange& c) { return !c.after; }))
            return "delete " + count + " objects";
        return "edit " + count + " objects";
    }
    const ObjectChange& change = changes.front();
    if (!change.before)
        return "add " + change.after->instance.name;
    if (!change.after)
        return "delete " + change.before->instance.name;
    const ModelInstance& a = change.before->instance;
    const ModelInstance& b = change.after->instance;
    if (a.name != b.name)
        return "rename " + a.name;
    if (change.before->modelPath != change.after->modelPath)
        return "relink " + b.name;
    if (a.locked != b.locked)
        return (b.locked ? "lock " : "unlock ") + b.name;
    if (a.visible != b.visible)
        return (b.visible ? "show " : "hide ") + b.name;
    if (a.color != b.color)
        return "color " + b.name;
    if (a.entity != b.entity || a.entityParams != b.entityParams)
        return "entity " + b.name;
    if (a.group != b.group)
        return (b.group.empty() ? "ungroup " : "group ") + b.name;
    if (a.collision != b.collision)
        return "collision " + b.name;
    const bool moved = a.position != b.position;
    const bool rotated = a.rotation != b.rotation;
    const bool scaled = a.scale != b.scale;
    if (moved && !rotated && !scaled)
        return "move " + b.name;
    if (rotated && !moved && !scaled)
        return "rotate " + b.name;
    if (scaled && !moved && !rotated)
        return "scale " + b.name;
    return "transform " + b.name;
}

void Editor::updateObjectHistory()
{
    // After a scene is opened its instances appear over several frames; they are not an edit.
    if (!m_objectBaselineValid) {
        if (!m_scenes.isLoading())
            resetObjectBaseline();
        return;
    }
    if ((!m_sceneChanged && !m_levelEntryPushed) || editInProgress())
        return;
    m_sceneChanged = false;
    const bool levelPushed = std::exchange(m_levelEntryPushed, false);
    if (!objectChangedSinceBaseline()) {
        // Deleted level shapes are restored from the snapshot's meshes, so they must stay current.
        if (levelPushed)
            resetObjectBaseline();
        return;
    }

    std::vector<ObjectRecord> previous = std::move(m_objectBaseline);
    resetObjectBaseline();
    std::unordered_map<uint64_t, size_t> previousById;
    for (size_t i = 0; i < previous.size(); ++i)
        previousById.emplace(previous[i].instance.id, i);

    std::vector<ObjectChange> changes;
    std::vector<bool> stillThere(previous.size(), false);
    for (const ObjectRecord& current : m_objectBaseline) {
        const auto found = previousById.find(current.instance.id);
        if (found == previousById.end()) {
            changes.push_back({ std::nullopt, current });
            continue;
        }
        stillThere[found->second] = true;
        if (!sameObjectState(previous[found->second], current))
            changes.push_back({ previous[found->second], current });
    }
    for (size_t i = 0; i < previous.size(); ++i) {
        if (!stillThere[i])
            changes.push_back({ std::move(previous[i]), std::nullopt });
    }
    if (changes.empty())
        return; // only the order changed

    // Objects created or deleted by the same action as a geometry edit undo together with it.
    if (levelPushed && !m_undoStack.empty()) {
        m_undoStack.back().objects = std::move(changes);
        return;
    }
    HistoryEntry entry;
    entry.action = describeObjectChanges(changes);
    entry.objects = std::move(changes);
    pushHistory(std::move(entry));
}

std::optional<size_t> Editor::resolveRecordModel(const ObjectRecord& record)
{
    if (const auto found = m_models.findModelByPath(record.modelPath))
        return found;
    // A prefab loaded again since keeps being shared.
    if (!record.prefabPath.empty()) {
        const auto& models = m_models.getModels();
        for (size_t i = 0; i < models.size(); ++i) {
            if (models[i] && models[i]->polyMesh && models[i]->prefabPath == record.prefabPath)
                return i;
        }
    }
    if (!record.mesh)
        return std::nullopt;
    const auto index = createLevelModel(*record.mesh, record.modelName);
    if (!index)
        return std::nullopt;
    // The old path keeps geometry entries for this shape working; level paths are never reused.
    GPUModel* model = m_models.getModel(*index);
    model->sourcePath = record.modelPath;
    model->prefabPath = record.prefabPath;
    return index;
}

bool Editor::applyObjectChanges(const std::vector<ObjectChange>& changes, bool undo)
{
    auto& instances = m_models.getInstances();
    const auto indexOf = [&instances](uint64_t id) -> int {
        for (size_t i = 0; i < instances.size(); ++i) {
            if (instances[i].id == id)
                return static_cast<int>(i);
        }
        return -1;
    };

    // Models first, so a missing one leaves the scene untouched.
    std::vector<std::pair<const ObjectRecord*, size_t>> targets;
    for (const ObjectChange& change : changes) {
        const std::optional<ObjectRecord>& target = undo ? change.before : change.after;
        if (!target)
            continue;
        const auto model = resolveRecordModel(*target);
        if (!model) {
            setStatus("Cannot " + std::string(undo ? "undo" : "redo") + ": the model of " + target->instance.name +
                " was unloaded", true);
            releaseUnusedLevelModels(); // any shapes rebuilt above
            return false;
        }
        targets.push_back({ &*target, *model });
    }

    for (const ObjectChange& change : changes) {
        const std::optional<ObjectRecord>& target = undo ? change.before : change.after;
        const std::optional<ObjectRecord>& current = undo ? change.after : change.before;
        if (target || !current)
            continue;
        const int index = indexOf(current->instance.id);
        if (index >= 0)
            m_models.removeInstance(static_cast<size_t>(index));
    }
    // Ascending, so each restored object lands at its old position.
    std::sort(targets.begin(), targets.end(), [](const auto& a, const auto& b) { return a.first->index < b.first->index; });
    std::vector<uint64_t> selectIds;
    for (const auto& [record, modelIndex] : targets) {
        ModelInstance instance = record->instance;
        instance.modelIndex = modelIndex;
        const int index = indexOf(instance.id);
        if (index >= 0)
            instances[index] = std::move(instance);
        else
            instances.insert(instances.begin() + std::min(record->index, instances.size()), std::move(instance));
        selectIds.push_back(record->instance.id);
    }
    releaseUnusedLevelModels();

    // Face and vertex picks referred to indices that may have moved.
    m_levelSelectionInstance = -1;
    m_renamingInstance = -1;
    // Every object the step touched is selected again; the last one is active.
    std::vector<int> selected;
    for (uint64_t id : selectIds)
        if (const int index = indexOf(id); index >= 0)
            selected.push_back(index);
    if (selected.empty())
        deselectAll();
    else
        selectIndices(selected, selected.back());
    return true;
}

bool Editor::applyLevelEdits(const HistoryEntry& entry, bool undo)
{
    // Every shape is looked up first, so a deleted one leaves the others untouched.
    std::vector<size_t> modelIndices;
    for (const LevelEdit& edit : entry.levels) {
        const auto found = m_models.findModelByPath(edit.modelPath);
        const GPUModel* model = found ? m_models.getModel(*found) : nullptr;
        if (!model || !model->polyMesh) {
            setStatus("Cannot " + std::string(undo ? "undo " : "redo ") + entry.action + ": the object was deleted", true);
            return false;
        }
        modelIndices.push_back(*found);
    }
    for (size_t i = 0; i < entry.levels.size(); ++i) {
        const LevelEdit& edit = entry.levels[i];
        GPUModel* model = m_models.getModel(modelIndices[i]);
        *model->polyMesh = undo ? edit.before : edit.after;
        if (edit.modelPath == m_levelBaselinePath)
            m_levelBaseline = *model->polyMesh;
        try {
            m_models.rebuildPolyMesh(modelIndices[i]);
        }
        catch (const std::exception& e) {
            setStatus("Failed to rebuild " + model->name + ": " + e.what(), true);
        }
    }
    return true;
}

bool Editor::applyHistoryEntry(const HistoryEntry& entry, bool undo)
{
    // Undo restores deleted objects before their geometry; redo replays in the original order.
    bool ok = true;
    if (undo && !entry.objects.empty())
        ok = applyObjectChanges(entry.objects, true);
    if (ok && !entry.levels.empty())
        ok = applyLevelEdits(entry, undo);
    if (ok && !undo && !entry.objects.empty())
        ok = applyObjectChanges(entry.objects, false);
    resetObjectBaseline();
    if (ok)
        setStatus((undo ? "Undo " : "Redo ") + entry.action);
    return ok;
}

void Editor::pushHistory(HistoryEntry entry)
{
    entry.serial = ++m_historySerial;
    m_undoStack.push_back(std::move(entry));
    if (m_undoStack.size() > kMaxUndoSteps) {
        // Undoing everything now ends at the state after the dropped step.
        m_baseState = m_undoStack.front().serial;
        m_undoStack.erase(m_undoStack.begin());
    }
    m_redoStack.clear();
}

void Editor::undo()
{
    // Mid-drag the edit has no history entry yet.
    if (editInProgress())
        return;
    updateObjectHistory(); // changes made earlier this frame
    if (m_undoStack.empty()) {
        setStatus("Nothing to undo");
        return;
    }
    HistoryEntry entry = std::move(m_undoStack.back());
    m_undoStack.pop_back();
    if (applyHistoryEntry(entry, true))
        m_redoStack.push_back(std::move(entry));
}

void Editor::redo()
{
    if (editInProgress())
        return;
    updateObjectHistory();
    if (m_redoStack.empty()) {
        setStatus("Nothing to redo");
        return;
    }
    HistoryEntry entry = std::move(m_redoStack.back());
    m_redoStack.pop_back();
    if (applyHistoryEntry(entry, false))
        m_undoStack.push_back(std::move(entry));
}

void Editor::clearHistory()
{
    m_undoStack.clear();
    m_redoStack.clear();
    m_baseState = ++m_historySerial;
    m_levelEditPending = false;
    m_levelBaselinePath.clear();
    m_levelEntryPushed = false;
    m_objectBaselineValid = false;
    m_sceneChanged = false;
}
