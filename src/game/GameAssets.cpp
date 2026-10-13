#include "GameAssets.h"
#include <algorithm>
#include <exception>
#include <filesystem>
#include "engine/Audio.h"
#include "engine/Log.h"
#include "engine/ModelLoader.h"
#include "engine/ModelManager.h"
#include "engine/Renderer.h"

const char* gameModelPath(GameModelId id)
{
    switch (id) {
    case GameModelId::Enemy: return "models/enemy_grunt.glb";
    case GameModelId::Shotgun: return "models/shotgun.glb";
    case GameModelId::Pistol: return "models/pistol.glb";
    case GameModelId::Magazine: return "models/magazine.glb";
    case GameModelId::Hands: return "models/hands.glb";
    case GameModelId::ShellItem: return "models/shell.glb";
    case GameModelId::RoundItem: return "models/round.glb";
    case GameModelId::PlayerBody: return "models/player_body.glb";
    case GameModelId::Medkit: return "models/medkit.glb";
    case GameModelId::Shells: return "models/shells.glb";
    case GameModelId::RoundsBox: return "models/rounds_box.glb";
    case GameModelId::Exit: return "models/exit_portal.glb";
    case GameModelId::PlayerStart: return "models/player_start.glb";
    default: return "";
    }
}

void GameAssets::load(ModelManager& models, AudioSystem& audio)
{
    if (audio.findSound("shotgun_fire") < 0)
        registerGameSounds(audio);

    for (size_t i = 0; i < m_models.size(); ++i)
        loadModel(models, static_cast<GameModelId>(i));

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

void emitModelNodes(const GameModel& model, const std::vector<glm::mat4>& nodeMatrices, const glm::mat4& world,
    const glm::vec3& tint, bool castShadow, std::vector<DynamicInstance>& out, int tintedNode)
{
    for (size_t n = 0; n < model.data.nodes.size() && n < nodeMatrices.size(); ++n) {
        const int mesh = model.data.nodes[n].mesh;
        if (mesh < 0 || mesh >= static_cast<int>(model.meshModels.size()))
            continue;
        const glm::mat4 matrix = world * nodeMatrices[n];
        // Parts scaled away (the muzzle flash between shots) are not drawn.
        const float size = std::max({ glm::length(glm::vec3(matrix[0])), glm::length(glm::vec3(matrix[1])),
            glm::length(glm::vec3(matrix[2])) });
        if (size < 0.01f)
            continue;
        DynamicInstance& part = out.emplace_back();
        part.instance.modelIndex = model.meshModels[mesh];
        part.instance.position = glm::vec3(matrix[3]);
        part.instance.color = tintedNode < 0 || tintedNode == static_cast<int>(n) ? tint : glm::vec3(1.0f);
        part.transform = matrix;
        part.castShadow = castShadow;
    }
}

bool GameAssets::intact(ModelManager& models) const
{
    for (const auto& [index, source] : m_owned) {
        const GPUModel* model = models.getModel(index);
        if (!model || model->sourcePath != source)
            return false;
    }
    return true;
}

void GameAssets::loadModel(ModelManager& models, GameModelId id)
{
    GameModel& model = m_models[static_cast<size_t>(id)];
    if (model.valid())
        return;
    model = {};
    model.path = gameModelPath(id);
    try {
        model.data = loadAnimatedModel(model.path);
        const std::string stem = std::filesystem::path(model.path).stem().string();
        for (size_t m = 0; m < model.data.meshes.size(); ++m) {
            // A source path no scene or file uses, so they are never confused with scene models.
            const std::string source = kRuntimeModelPrefix + model.path + "#" + std::to_string(m);
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
