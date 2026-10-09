#pragma once
#include <array>
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
#include "WeaponTuning.h"

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
    bool alive = true;
    float damageFlash = 0.0f; // 1 right after being hit, fading
    glm::vec3 hitFrom{ 0.0f }; // where the last hit came from
    float hitMarker = 0.0f;   // 1 right after hitting an enemy
};

// Everything that happens in a level: the entities spawned from the scene's entity objects, the player's
// hands and guns (Arms, presented in GameArms.cpp) and health, effects and sounds. Game owns the camera,
// input and the player's movement.
class GameWorld {
public:
    // Shell / round / magazine slots the body model shows (player_body.glb).
    static constexpr int kBandolierLoops = 8;
    static constexpr int kPocketSlots = 16;
    static constexpr int kBoxRounds = 30;
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
        bool fire = false;          // main hand's trigger held; a shot needs a fresh pull
        // Pressed this frame:
        bool secondary = false;     // the off hand's gun fires, or the main hand's gun is pumped / racked
        bool reload = false;        // context reload (Arms ActionKind::Reload)
        bool forceReload = false;   // reload even with a gun in each hand (drops the off hand's)
        bool unload = false;        // pistol magazine out
        int zone = -1;              // a storage zone's button (Zone), -1 = none
        bool putBack = false;       // the held item back where it came from
        bool pickUp = false;        // the nearest dropped item
        bool toggleOffGun = false;  // the other gun into / out of the off hand
        ItemKind swapTo = ItemKind::Empty; // the main hand takes this
        float jacket = 0.0f;        // 0 closed .. 1 held open to look at what is carried
        bool restock = false;       // held while the jacket is open: move shells from the pocket to the bandolier
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
    // The hands, guns and ammunition (read-only: changes go through Input).
    const Arms& arms() const { return m_arms; }
    // Where the main hand's gun's shots leave and fly, in world space (along its barrel, not the view).
    const glm::vec3& barrelOrigin() const { return m_view.barrelOrigin[0]; }
    const glm::vec3& barrelDirection() const { return m_view.barrelDirection[0]; }

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

    // GameArms.cpp: input -> Arms requests, Arms events -> sounds, shots and dropped items, and the poses
    // of the hands and what they hold.
    void resetArms();
    void updateWeapon(const Input& input);
    void requestFromInput(const Input& input);
    void handleArmsEvents(const glm::mat4& cameraWorld);
    // Poses both hands and what they hold (camera space) and the barrels (world space) for this frame.
    void poseArms(float dt, const glm::mat4& cameraWorld);
    // Where the off hand is heading (camera space): `key` names the target (a new one starts a move),
    // `phaseLeft` is how long the move may take.
    NodePose offHandTarget(int& key, float& phaseLeft) const;
    // What a hand shows: during a swap or draw, the item on its way in.
    ItemKind shownItem(Hand hand) const;
    void shoot(Hand hand, ItemKind gun, const glm::mat4& cameraWorld);
    void updateDebris(float dt);
    void collectArms(const glm::mat4& cameraWorld, std::vector<DynamicInstance>& out);
    void drawShotgun(const glm::mat4& world, float pump, float flash, std::vector<DynamicInstance>& out);
    void drawPistol(const glm::mat4& world, float slide, float flash, bool withMagazine, std::vector<DynamicInstance>& out);
    void drawMagazine(const glm::mat4& world, int magazineId, std::vector<DynamicInstance>& out);
    void drawItem(GameModelId id, const glm::mat4& world, const glm::vec3& tint, std::vector<DynamicInstance>& out);
    // The nearest dropped item within reach of the player's feet, -1 when none.
    int nearestGroundItem() const;
    void addShells(int amount, int& added);
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
    // The same for model-space node matrices posed some other way.
    void emitNodes(const GameModel& model, const std::vector<glm::mat4>& nodeMatrices, const glm::mat4& world,
        const glm::vec3& tint, bool castShadow, std::vector<DynamicInstance>& out, int tintedNode = -1);

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

    Arms m_arms;
    WeaponTuning m_tuning;

    // How the hands and guns look this frame (GameArms.cpp); the state itself is in m_arms.
    struct HandView {
        NodePose pose;          // camera space
        NodePose from;          // where the current move started
        int key = -1;           // which target the hand is heading for; a new one starts a move
        float time = 0.0f;
        float duration = 0.1f;
    };
    struct ArmsView {
        std::array<HandView, 2> hands;
        std::array<NodePose, 2> items;       // the held items, camera space
        std::array<float, 2> shotTime{ 10.0f, 10.0f }; // since each hand last fired
        bool triggerHeld = false;
        float messageCooldown = 0.0f;
        std::string lastMessage;
        glm::vec2 sway{ 0.0f };              // radians the guns lag behind the view (yaw, pitch)
        glm::vec3 lastForward{ 0.0f };
        float bobPhase = 0.0f;
        float bobAmount = 0.0f;
        float tilt = 0.0f;                   // shotgun rolled to show the loading port
        float time = 0.0f;
        glm::mat4 bodyToCamera{ 1.0f };      // body space (eyes, yaw only) -> camera space
        std::array<glm::vec3, 2> barrelOrigin{};
        std::array<glm::vec3, 2> barrelDirection{ glm::vec3(0, 0, -1), glm::vec3(0, 0, -1) };
    };
    ArmsView m_view;
    // The guns' step bob in camera space.
    glm::vec3 weaponBob() const;
    // Node indices in the models GameArms.cpp poses (-1 when missing).
    struct Rigs {
        int shotgunGun = -1, shotgunPump = -1, shotgunFlash = -1;
        int pistolSlide = -1, pistolFlash = -1;
        int magazineFollower = -1;
        std::array<int, 8> magazineRounds{};
        int handR = -1, handL = -1;
        glm::vec3 shotgunMuzzle{ 0.0f, 0.022f, -0.71f }; // in `gun` space
        glm::vec3 pistolMuzzle{ 0.0f, 0.07f, -0.155f };
        glm::mat4 pistolMagSeat{ 1.0f };
    };
    Rigs m_rigs;
    std::vector<NodePose> m_scratchPose;
    std::vector<glm::mat4> m_scratchMatrices;

    // The player's body (PlayerBody model): jacket panels, legs, one node per shell / round it can show,
    // magazine anchors and the holster.
    struct BodyRig {
        int flapL = -1, flapR = -1, legL = -1, legR = -1, holsterGun = -1;
        std::array<int, kBandolierLoops> band{};
        std::array<int, kPocketSlots> pocket{};
        std::array<int, kBoxRounds> boxRounds{};
        std::array<int, 2> bandMags{ -1, -1 };
        std::array<int, 2> pocketMags{ -1, -1 };
        int rightPocketMag = -1;
    };
    BodyRig m_body;
    // Body-space matrix of a body node (falls back to the body's origin).
    glm::mat4 bodyNode(int node) const;
    float m_jacket = 0.0f;
    std::vector<NodePose> m_bodyPose;
    std::vector<glm::mat4> m_bodyMatrices;
    void poseBody();
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

    // Things that fell: spent shells and casings (cosmetic, they fade) and dropped items (an Arms ground item:
    // they stay until picked up).
    struct Debris {
        ItemKind item = ItemKind::Shell;
        int ground = -1;          // Arms ground item id; -1 = cosmetic
        glm::vec3 position{ 0.0f };
        glm::vec3 velocity{ 0.0f };
        glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };
        glm::vec3 spin{ 0.0f };   // radians per second about each axis
        glm::vec3 tint{ 1.0f };
        float floor = 0.0f;       // height it comes to rest at
        float life = 0.0f;
        bool landed = false;
    };
    std::vector<Debris> m_debris;
    void spawnDebris(ItemKind item, int ground, const glm::vec3& position, const glm::vec3& velocity,
        const glm::vec3& tint = glm::vec3(1.0f));

    // Pellet streaks, so the player can learn where the barrel points.
    struct Tracer {
        glm::vec3 from{ 0.0f };
        glm::vec3 to{ 0.0f };
        float life = 0.0f;
    };
    std::vector<Tracer> m_tracers;

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
