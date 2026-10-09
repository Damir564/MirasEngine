#include "SceneManager.h"
#include "ModelLoader.h"
#include "ModelManager.h"
#include <filesystem>
#include "Log.h"

SceneManager::SceneManager(ModelManager& models, vk::Device device)
    : m_models(models), m_device(device)
{
}

SceneManager::OpenResult SceneManager::open(const std::string& path)
{
    OpenResult result;
    SceneSerializer::LoadedScene loaded = SceneSerializer::Load(path);
    if (!loaded.valid)
        return result;

    clear();
    for (auto& model : loaded.models) {
        if (model.polyMesh) {
            // Level geometry is rebuilt right away under a fresh path; the pending scene maps to it by path.
            try {
                const size_t index = m_models.addPolyMesh(std::move(*model.polyMesh), model.name);
                GPUModel* built = m_models.getModel(index);
                built->prefabPath = std::move(model.prefabPath);
                model.path = built->sourcePath;
                ++result.queuedModels;
            }
            catch (const std::exception& e) {
                LOG_ERROR("[SCENE] Failed to build level geometry '" << model.name << "': " << e.what() << "\n");
                model.path.clear(); // the file's path may now name a different level model
            }
            model.polyMesh.reset();
            continue;
        }
        if (!isBuiltinModelPath(model.path) && !std::filesystem::exists(model.path)) {
            result.missingFiles.push_back(model.path);
            continue;
        }
        m_models.loadModelAsync(model.path, model.name);
        ++result.queuedModels;
    }

    m_settings = loaded.settings;
    m_pendingScene = std::move(loaded);
    m_pendingLoad = true;
    m_currentPath = path;
    result.ok = true;
    return result;
}

bool SceneManager::save(const std::string& path)
{
    return SceneSerializer::Save(path, m_models, m_settings);
}

void SceneManager::clear()
{
    (void)m_device.waitIdle();
    auto& instances = m_models.getInstances();
    while (!instances.empty())
        m_models.removeInstance(instances.size() - 1);
    while (!m_models.getModels().empty())
        m_models.unloadModel(m_models.getModels().size() - 1);
    m_pendingLoad = false;
}

void SceneManager::update()
{
    if (m_pendingLoad)
        instantiatePendingScene();
}

void SceneManager::instantiatePendingScene()
{
    const auto& loadedModels = m_models.getModels();
    const bool allDone = m_models.getLoadingTasks().empty() && !loadedModels.empty();
    if (!allDone || loadedModels.size() < m_pendingScene.models.size())
        return;

    // Scene files index models by their own order; map that onto the manager's slots by source path.
    std::vector<int> fileToManager(m_pendingScene.models.size(), -1);
    for (size_t fi = 0; fi < m_pendingScene.models.size(); ++fi) {
        for (size_t mi = 0; mi < loadedModels.size(); ++mi) {
            if (loadedModels[mi] && loadedModels[mi]->isValid() &&
                loadedModels[mi]->sourcePath == m_pendingScene.models[fi].path) {
                fileToManager[fi] = static_cast<int>(mi);
                break;
            }
        }
    }

    // Shared material overrides need the loaded model, so they are applied by loading it once more
    // (through its cache).
    for (size_t fi = 0; fi < m_pendingScene.models.size(); ++fi) {
        auto& overrides = m_pendingScene.models[fi].materialOverrides;
        GPUModel* model = fileToManager[fi] >= 0 ? m_models.getModel(static_cast<size_t>(fileToManager[fi])) : nullptr;
        if (overrides.empty() || !model || model->polyMesh)
            continue;
        model->materialOverrides = std::move(overrides);
        try {
            m_models.reloadModel(static_cast<size_t>(fileToManager[fi]));
        }
        catch (const std::exception& e) {
            LOG_ERROR("[SCENE] Failed to apply materials to '" << model->name << "': " << e.what() << "\n");
        }
    }

    m_models.reserveInstances(m_pendingScene.instances.size());
    for (const auto& inst : m_pendingScene.instances) {
        if (inst.fileModelIndex >= fileToManager.size()) continue;
        const int managerIdx = fileToManager[inst.fileModelIndex];
        if (managerIdx < 0) continue;

        const size_t newIdx = m_models.createInstance(static_cast<size_t>(managerIdx), inst.position);
        auto& newInst = m_models.getInstances()[newIdx];
        newInst.name = inst.name;
        newInst.rotation = inst.rotation;
        newInst.scale = inst.scale;
        newInst.visible = inst.visible;
        newInst.color = inst.color;
        newInst.locked = inst.locked;
        newInst.entity = inst.entity;
        newInst.entityParams = inst.entityParams;
        newInst.group = inst.group;
        newInst.collision = inst.collision;
    }

    m_pendingLoad = false;
    LOG_INFO("[SCENE] All instances created\n");
}
