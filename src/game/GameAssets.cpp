#include "GameAssets.h"
#include <algorithm>
#include <exception>
#include <filesystem>
#include "engine/Audio.h"
#include "engine/Log.h"
#include "engine/ModelLoader.h"
#include "engine/ModelManager.h"

const char* gameModelPath(GameModelId id)
{
    switch (id) {
    case GameModelId::Enemy: return "models/enemy_grunt.glb";
    case GameModelId::Shotgun: return "models/shotgun.glb";
    case GameModelId::Medkit: return "models/medkit.glb";
    case GameModelId::Shells: return "models/shells.glb";
    case GameModelId::Exit: return "models/exit_portal.glb";
    case GameModelId::PlayerStart: return "models/player_start.glb";
    default: return "";
    }
}

void GameAssets::load(ModelManager& models, AudioSystem& audio)
{
    if (audio.findSound("shotgun_fire") < 0)
        registerGameSounds(audio);

    for (size_t i = 0; i < m_models.size(); ++i) {
        GameModel& model = m_models[i];
        if (model.valid())
            continue;
        model = {};
        model.path = gameModelPath(static_cast<GameModelId>(i));
        try {
            model.data = loadAnimatedModel(model.path);
            const std::string stem = std::filesystem::path(model.path).stem().string();
            for (size_t m = 0; m < model.data.meshes.size(); ++m) {
                // A source path no scene or file uses, so they are never confused with scene models.
                const std::string source = "game:" + model.path + "#" + std::to_string(m);
                const size_t index = models.addMesh(std::move(model.data.meshes[m]), stem + "/" + std::to_string(m), source);
                model.meshModels.push_back(index);
                m_owned.emplace_back(index, source);
            }
            model.data.meshes.clear();
        }
        catch (const std::exception& e) {
            LOG_ERROR("[GAME] Can't load " << model.path << ": " << e.what() << "\n");
            model = {};
        }
    }

    if (!m_cube) {
        if (const auto found = models.findModelByPath(kBuiltinCubePath)) {
            m_cube = *found;
        }
        else {
            try {
                m_cube = models.loadModelSync(kBuiltinCubePath, "Cube");
                m_owned.emplace_back(*m_cube, kBuiltinCubePath);
            }
            catch (const std::exception& e) {
                LOG_ERROR("[GAME] Can't create the particle cube: " << e.what() << "\n");
            }
        }
    }
    m_loaded = true;
}

void GameAssets::release(ModelManager& models)
{
    // Highest index first: unloading shifts the models after it.
    std::sort(m_owned.begin(), m_owned.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    size_t released = 0;
    for (const auto& [index, source] : m_owned) {
        const GPUModel* model = models.getModel(index);
        if (model && model->sourcePath == source) {
            models.unloadModel(index);
            ++released;
        }
        else if (const auto found = models.findModelByPath(source)) {
            models.unloadModel(*found);
            ++released;
        }
    }
    if (!m_owned.empty())
        LOG_INFO("[GAME] Released " << released << " of " << m_owned.size() << " game models, "
            << models.getModels().size() << " models left\n");
    m_owned.clear();
    for (GameModel& model : m_models)
        model = {};
    m_cube.reset();
    m_loaded = false;
}
