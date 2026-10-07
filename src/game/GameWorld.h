#pragma once
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>
#include "engine/Animation.h"
#include "engine/Audio.h"
#include "GameAssets.h"

class ModelManager;
class PhysicsWorld;
struct ModelInstance;
struct DynamicInstance;

// What a scene's entity object asks the game to create.
struct SpawnInfo {
    std::string type;   // EntityTypes.h id
    std::string name;
    glm::vec3 position{ 0.0f };
    float yaw = 0.0f;   // radians; the entity faces (sin yaw, 0, cos yaw)
    std::string params; // "key=value" pairs
    glm::vec3 color{ 1.0f }; // the scene object's tint
};

enum class EnemyState { Idle, Chase, Attack, Hurt, Dead };

struct Entity {
    enum class Kind { Enemy, HealthPack, AmmoPack, Exit };
    uint32_t id = 0;
    Kind kind = Kind::Enemy;
    std::string name;
    glm::vec3 position{ 0.0f };
    float yaw = 0.0f;
    const GameModel* model = nullptr;
    Animator animator;
    bool active = true; // pickups: not taken yet

    // Enemies
    EnemyState state = EnemyState::Idle;
    float stateTime = 0.0f;
    float health = 0.0f;
    float maxHealth = 0.0f;
    float speed = 0.0f;
    float damage = 0.0f;
    float attackDelay = 0.35f; // after each swing
    float sightRange = 0.0f;
    float hearingRange = 0.0f;
    float size = 1.0f;         // body scale: model, capsule and reach
    glm::vec3 tint{ 1.0f };
    int character = -1;      // PhysicsWorld character; -1 once dead
    bool alerted = false;
    bool hunter = false;       // always knows where the player is
    bool attackLanded = false;
    float attackCooldown = 0.0f;
    float thinkTimer = 0.0f;
    float hurtFlash = 0.0f;
    float soundCooldown = 0.0f;
    float stepTimer = 0.0f;
    glm::vec3 lastSeen{ 0.0f };
    glm::vec3 stuckCheckPosition{ 0.0f };
    float stuckTimer = 0.0f;
    float sidestepTime = 0.0f;
    float sidestepSign = 1.0f;

    // Pickups and the exit
    float amount = 0.0f;
    std::string nextLevel;
};

struct PlayerState {
    float health = 100.0f;
    int shells = 8;    // loaded
    int reserve = 24;  // carried
    bool alive = true;
    float damageFlash = 0.0f; // 1 right after being hit, fading
    float hitMarker = 0.0f;   // 1 right after hitting an enemy
};

// Everything that happens in a level: the entities spawned from the scene's entity objects, the player's
// shotgun and health, effects and sounds. Game owns the camera, input and the player's movement.
class GameWorld {
public:
    static constexpr int kMagazineSize = 8;
    static constexpr int kMaxReserve = 64;
    static constexpr float kMaxHealth = 100.0f;

    GameWorld(ModelManager& models, PhysicsWorld& physics, AudioSystem& audio, const GameAssets& assets);

    // Creates the entity a scene object stands for. Types without a factory are ignored.
    void spawn(const SpawnInfo& info);
    // Spawns every entity object of the scene; the player start (if any) is returned instead of spawned.
    bool spawnFromScene(const std::vector<ModelInstance>& instances, SpawnInfo& playerStart);
    // Removes every entity and effect and resets the player.
    void clear();

    struct Input {
        float dt = 0.0f;
        glm::vec3 eye{ 0.0f };      // camera position
        glm::vec3 forward{ 0.0f, 0.0f, -1.0f };
        glm::vec3 playerFeet{ 0.0f };
        bool moving = false;        // walking on the ground (for footsteps and weapon bob)
        bool sprinting = false;
        bool fire = false;          // trigger held
        bool reload = false;        // reload pressed this frame
    };
    void update(const Input& input);
    // Entities, effects and the weapon (in front of the camera) for this frame.
    void collectDrawables(const glm::mat4& cameraWorld, std::vector<DynamicInstance>& out);

    const PlayerState& player() const { return m_player; }
    // Upward camera kick from firing, in degrees; Game adds it to the pitch and it recovers by itself.
    float recoil() const { return m_recoil; }
    int enemiesTotal() const { return m_enemiesTotal; }
    int enemiesKilled() const { return m_enemiesKilled; }
    bool exitUnlocked() const { return m_exitUnlocked; }
    bool levelComplete() const { return m_levelComplete; }
    const std::string& nextLevel() const { return m_nextLevel; }
    float levelTime() const { return m_levelTime; }
    bool reloading() const { return m_weapon.action == WeaponAction::Reloading; }

    struct Message {
        std::string text;
        float age = 0.0f;
    };
    const std::vector<Message>& messages() const { return m_messages; }
    void showMessage(const std::string& text);

    // For the autoplay bot: the nearest living enemy, its aim point and whether it is in sight.
    bool nearestEnemy(const glm::vec3& eye, glm::vec3& aimPoint, bool& visible) const;
    // The exit's position; false when the level has none.
    bool exitPosition(glm::vec3& position) const;
    const std::vector<Entity>& entities() const { return m_entities; }
    // Damage to the player (e.g. falling out of the level kills).
    void damagePlayer(float amount);

private:
    using SpawnFunction = std::function<void(GameWorld&, const SpawnInfo&)>;
    void registerEntity(const std::string& type, SpawnFunction function);
    Entity& addEntity(Entity::Kind kind, const SpawnInfo& info, GameModelId model);
    void spawnEnemy(const SpawnInfo& info);
    void spawnPickup(const SpawnInfo& info, Entity::Kind kind);
    void spawnExit(const SpawnInfo& info);

    void updateEnemy(Entity& enemy, const Input& input);
    void setEnemyState(Entity& enemy, EnemyState state);
    void damageEnemy(Entity& enemy, float amount);
    bool lineOfSight(const glm::vec3& from, const glm::vec3& to) const;
    void alertNearby(const glm::vec3& position, float radius);
    void updatePickups(const Input& input);
    void updateExit(const Input& input);

    enum class WeaponAction { Idle, Firing, Reloading };
    void updateWeapon(const Input& input);
    void fire(const Input& input);
    // Distance along the ray to the enemy's body (a vertical capsule), or a negative value on a miss.
    static float rayHitsEnemy(const glm::vec3& origin, const glm::vec3& dir, const Entity& enemy);

    void spawnParticles(const glm::vec3& position, const glm::vec3& normal, const glm::vec3& color, int count,
        float speed, float size, float life, bool gravity);
    void updateParticles(float dt);
    void playAt(const std::string& sound, const glm::vec3& position, float volume = 1.0f, float pitch = 1.0f);
    void play(const std::string& sound, float volume = 1.0f, float pitch = 1.0f);
    float random(float lo, float hi);

    // Adds the parts of an animated model posed by the animator, placed by `world`. The tint applies to
    // every part, or only to the node `tintedNode` when it is set.
    void addModel(const GameModel& model, Animator& animator, const glm::mat4& world, const glm::vec3& tint,
        bool castShadow, std::vector<DynamicInstance>& out, int tintedNode = -1);

    ModelManager& m_models;
    PhysicsWorld& m_physics;
    AudioSystem& m_audio;
    const GameAssets& m_assets;
    std::unordered_map<std::string, SpawnFunction> m_factories;

    std::vector<Entity> m_entities;
    uint32_t m_nextEntityId = 1;
    PlayerState m_player;
    int m_enemiesTotal = 0;
    int m_enemiesKilled = 0;
    bool m_exitUnlocked = false;
    bool m_levelComplete = false;
    std::string m_nextLevel;
    float m_levelTime = 0.0f;
    float m_exitMessageCooldown = 0.0f;
    float m_pickupMessageCooldown = 0.0f;
    bool m_hasExit = false;
    float m_completeTimer = 0.0f; // levels without an exit end shortly after the last kill
    glm::vec3 m_lastPlayerFeet{ 0.0f };

    struct Weapon {
        Animator animator;
        WeaponAction action = WeaponAction::Idle;
        float cooldown = 0.0f;   // until the next shot is possible
        float actionTime = 0.0f;
        bool reloadRequested = false;
        float emptyClickCooldown = 0.0f;
        float bobPhase = 0.0f;
        float bobAmount = 0.0f;
    };
    Weapon m_weapon;
    float m_recoil = 0.0f;
    float m_stepTimer = 0.0f;

    struct Particle {
        glm::vec3 position{ 0.0f };
        glm::vec3 velocity{ 0.0f };
        glm::vec3 color{ 1.0f };
        float size = 0.05f;
        float life = 0.0f;
        float maxLife = 1.0f;
        bool gravity = true;
    };
    std::vector<Particle> m_particles;

    struct DelayedSound {
        float delay;
        std::string name;
        float volume;
    };
    std::vector<DelayedSound> m_delayedSounds;
    std::vector<Message> m_messages;
    AudioSystem::VoiceId m_ambience = 0;

    std::mt19937 m_random{ 1234u };
    std::vector<glm::mat4> m_nodeMatrices; // scratch
};
