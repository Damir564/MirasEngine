#pragma once
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>
#include "engine/Animation.h"

class AudioSystem;
class ModelManager;

// An animated model the game draws itself: the node hierarchy and clips, with one ModelManager model per mesh.
struct GameModel {
    std::string path;
    AnimatedModelData data;          // meshes are moved out once uploaded
    std::vector<size_t> meshModels;  // ModelManager index of each data mesh

    bool valid() const { return !data.nodes.empty(); }
};

enum class GameModelId { Enemy, Shotgun, Medkit, Shells, Exit, PlayerStart, Count };

// The game's own models (weapon, enemies, pickups) and sounds. The models are added to the shared
// ModelManager after the level has loaded and are released before it is closed, so they never end up in
// a scene (the editor shares the ModelManager while playing a scene).
class GameAssets {
public:
    // Loads whatever is missing; a model that fails to load is logged and left invalid.
    void load(ModelManager& models, AudioSystem& audio);
    // Unloads the models this added. Must run before the scene's models are cleared.
    void release(ModelManager& models);

    const GameModel& model(GameModelId id) const { return m_models[static_cast<size_t>(id)]; }
    // The built-in cube, for particles.
    std::optional<size_t> cube() const { return m_cube; }
    bool loaded() const { return m_loaded; }

private:
    std::array<GameModel, static_cast<size_t>(GameModelId::Count)> m_models;
    std::optional<size_t> m_cube;
    // (index, source path) of every model added, to find them again when unloading.
    std::vector<std::pair<size_t, std::string>> m_owned;
    bool m_loaded = false;
};

// Path of each GameModelId, next to the executable.
const char* gameModelPath(GameModelId id);
// Synthesizes the game's sounds into the audio system (WAV files in sounds/ replace them by name).
void registerGameSounds(AudioSystem& audio);
