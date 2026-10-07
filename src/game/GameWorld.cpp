#include "GameWorld.h"
#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include "engine/EntityTypes.h"
#include "engine/Log.h"
#include "engine/ModelManager.h"
#include "engine/Physics.h"
#include "engine/Renderer.h"

namespace {

constexpr float kPi = 3.14159265f;
constexpr glm::vec3 kUp(0.0f, 1.0f, 0.0f);

// Enemies
constexpr float kEnemyRadius = 0.42f;
constexpr float kEnemyHeight = 1.85f;
constexpr float kEnemyEyeHeight = 1.6f;
constexpr float kSightRange = 30.0f;
constexpr float kHearingRange = 24.0f;
constexpr float kAttackRange = 1.9f;  // starts a swing
constexpr float kHitRange = 2.5f;     // the swing still lands
constexpr float kTurnSpeed = 6.0f;    // radians per second

// Shotgun
constexpr int kPellets = 9;
constexpr float kSpreadDegrees = 5.0f;
constexpr float kShotRange = 70.0f;
constexpr float kFireInterval = 0.85f;
constexpr float kShellReloadTime = 0.5f;
constexpr float kPelletDamageNear = 13.0f;
constexpr float kPelletDamageFar = 4.0f;

// Pickups and the exit
constexpr float kPickupRadius = 1.1f;
constexpr float kExitRadius = 1.6f;

glm::vec3 forwardFromYaw(float yaw)
{
    return glm::vec3(std::sin(yaw), 0.0f, std::cos(yaw));
}

float yawOf(const glm::vec3& direction)
{
    return std::atan2(direction.x, direction.z);
}

// Turns `current` towards `target` (radians) by at most maxStep, the short way round.
float approachAngle(float current, float target, float maxStep)
{
    const float delta = std::remainder(target - current, 2.0f * kPi);
    return current + std::clamp(delta, -maxStep, maxStep);
}

glm::vec3 flat(const glm::vec3& v)
{
    return glm::vec3(v.x, 0.0f, v.z);
}

} // namespace

GameWorld::GameWorld(ModelManager& models, PhysicsWorld& physics, AudioSystem& audio, const GameAssets& assets)
    : m_models(models), m_physics(physics), m_audio(audio), m_assets(assets)
{
    // The entity factory: one spawn function per entity type a scene can hold (EntityTypes.h).
    registerEntity(kEnemyEntity, [](GameWorld& world, const SpawnInfo& info) { world.spawnEnemy(info); });
    registerEntity(kHealthEntity, [](GameWorld& world, const SpawnInfo& info) {
        world.spawnPickup(info, Entity::Kind::HealthPack);
    });
    registerEntity(kAmmoEntity, [](GameWorld& world, const SpawnInfo& info) {
        world.spawnPickup(info, Entity::Kind::AmmoPack);
    });
    registerEntity(kExitEntity, [](GameWorld& world, const SpawnInfo& info) { world.spawnExit(info); });
}

void GameWorld::registerEntity(const std::string& type, SpawnFunction function)
{
    m_factories[type] = std::move(function);
}

// ---------------------------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------------------------

void GameWorld::spawn(const SpawnInfo& info)
{
    const auto factory = m_factories.find(info.type);
    if (factory == m_factories.end()) {
        LOG_ERROR("[GAME] No entity type '" << info.type << "' (object " << info.name << ")\n");
        return;
    }
    factory->second(*this, info);
}

bool GameWorld::spawnFromScene(const std::vector<ModelInstance>& instances, SpawnInfo& playerStart)
{
    bool foundStart = false;
    for (const ModelInstance& instance : instances) {
        if (instance.entity.empty())
            continue;
        SpawnInfo info{ instance.entity, instance.name, instance.position, glm::radians(instance.rotation.y),
            instance.entityParams, instance.color };
        if (instance.entity == kPlayerStartEntity) {
            if (!foundStart)
                playerStart = info;
            foundStart = true;
            continue;
        }
        spawn(info);
    }
    m_exitUnlocked = m_enemiesTotal == 0;
    LOG_INFO("[GAME] Spawned " << m_entities.size() << " entities (" << m_enemiesTotal << " enemies)\n");
    return foundStart;
}

Entity& GameWorld::addEntity(Entity::Kind kind, const SpawnInfo& info, GameModelId model)
{
    Entity entity;
    entity.id = m_nextEntityId++;
    entity.kind = kind;
    entity.name = info.name;
    entity.position = info.position;
    entity.yaw = info.yaw;
    const GameModel& gameModel = m_assets.model(model);
    entity.model = gameModel.valid() ? &gameModel : nullptr;
    entity.animator.reset(entity.model ? &entity.model->data : nullptr);
    m_entities.push_back(std::move(entity));
    return m_entities.back();
}

void GameWorld::spawnEnemy(const SpawnInfo& info)
{
    Entity& enemy = addEntity(Entity::Kind::Enemy, info, GameModelId::Enemy);
    enemy.maxHealth = enemy.health = std::max(entityParam(info.params, "health", 60.0f), 1.0f);
    enemy.speed = std::clamp(entityParam(info.params, "speed", 3.4f), 0.0f, 12.0f);
    enemy.damage = std::max(entityParam(info.params, "damage", 12.0f), 0.0f);
    enemy.attackDelay = std::clamp(entityParam(info.params, "cooldown", 0.35f), 0.0f, 5.0f);
    enemy.sightRange = std::clamp(entityParam(info.params, "sight", kSightRange), 1.0f, 100.0f);
    enemy.hearingRange = std::clamp(entityParam(info.params, "hearing", kHearingRange), 0.0f, 100.0f);
    enemy.size = std::clamp(entityParam(info.params, "size", 1.0f), 0.5f, 2.5f);
    enemy.tint = info.color;
    enemy.hunter = entityParam(info.params, "alert", 0.0f) != 0.0f;
    enemy.character = m_physics.addCharacter(info.position + glm::vec3(0.0f, 0.05f, 0.0f), kEnemyRadius * enemy.size,
        kEnemyHeight * enemy.size);
    enemy.thinkTimer = random(0.0f, 0.25f);
    enemy.stuckCheckPosition = info.position;
    // Not all in step.
    enemy.animator.play("idle", true, 0.0f, random(0.85f, 1.15f));
    enemy.animator.update(random(0.0f, 2.0f));
    ++m_enemiesTotal;
}

void GameWorld::spawnPickup(const SpawnInfo& info, Entity::Kind kind)
{
    const bool health = kind == Entity::Kind::HealthPack;
    Entity& pickup = addEntity(kind, info, health ? GameModelId::Medkit : GameModelId::Shells);
    pickup.amount = std::max(entityParam(info.params, "amount", health ? 25.0f : 8.0f), 1.0f);
    pickup.animator.play("idle");
    pickup.animator.update(random(0.0f, 3.0f));
}

void GameWorld::spawnExit(const SpawnInfo& info)
{
    Entity& exit = addEntity(Entity::Kind::Exit, info, GameModelId::Exit);
    exit.nextLevel = entityParamText(info.params, "next");
    exit.animator.play("idle");
    m_hasExit = true;
}

void GameWorld::clear()
{
    for (Entity& entity : m_entities)
        if (entity.character >= 0)
            m_physics.removeCharacter(entity.character);
    m_entities.clear();
    m_particles.clear();
    m_messages.clear();
    m_delayedSounds.clear();
    m_player = {};
    m_enemiesTotal = 0;
    m_enemiesKilled = 0;
    m_exitUnlocked = false;
    m_levelComplete = false;
    m_nextLevel.clear();
    m_levelTime = 0.0f;
    m_recoil = 0.0f;
    m_stepTimer = 0.0f;
    m_completeTimer = 0.0f;
    m_hasExit = false;

    m_weapon = {};
    const GameModel& shotgun = m_assets.model(GameModelId::Shotgun);
    m_weapon.animator.reset(shotgun.valid() ? &shotgun.data : nullptr);
    m_weapon.animator.play("idle");

    if (m_ambience)
        m_audio.stop(m_ambience);
    m_ambience = 0;
}

// ---------------------------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------------------------

void GameWorld::update(const Input& input)
{
    const float dt = input.dt;
    m_lastPlayerFeet = input.playerFeet;
    if (m_player.alive && !m_levelComplete)
        m_levelTime += dt;

    m_audio.setListener(input.eye, input.forward, kUp);
    if (!m_ambience || !m_audio.isPlaying(m_ambience)) {
        AudioSystem::PlayParams music;
        music.volume = 0.5f;
        music.loop = true;
        music.group = AudioSystem::Group::Music;
        m_ambience = m_audio.play("ambience", music);
    }
    for (DelayedSound& sound : m_delayedSounds) {
        sound.delay -= dt;
        if (sound.delay <= 0.0f)
            play(sound.name, sound.volume);
    }
    std::erase_if(m_delayedSounds, [](const DelayedSound& s) { return s.delay <= 0.0f; });
    for (Message& message : m_messages)
        message.age += dt;
    std::erase_if(m_messages, [](const Message& m) { return m.age > 3.5f; });

    m_player.damageFlash = std::max(0.0f, m_player.damageFlash - dt * 1.5f);
    m_player.hitMarker = std::max(0.0f, m_player.hitMarker - dt * 6.0f);
    m_recoil *= std::exp(-dt * 9.0f);
    m_exitMessageCooldown -= dt;
    m_pickupMessageCooldown -= dt;

    if (m_player.alive) {
        updateWeapon(input);
        if (input.moving) {
            m_stepTimer -= dt;
            if (m_stepTimer <= 0.0f) {
                play("footstep", 0.35f, random(0.85f, 1.1f));
                m_stepTimer = input.sprinting ? 0.3f : 0.42f;
            }
        }
    }
    for (Entity& entity : m_entities)
        if (entity.kind == Entity::Kind::Enemy)
            updateEnemy(entity, input);
    if (m_player.alive) {
        updatePickups(input);
        updateExit(input);
    }
    updateParticles(dt);
}

void GameWorld::showMessage(const std::string& text)
{
    m_messages.push_back({ text, 0.0f });
    if (m_messages.size() > 4)
        m_messages.erase(m_messages.begin());
}

void GameWorld::damagePlayer(float amount)
{
    if (!m_player.alive || m_levelComplete)
        return;
    m_player.health -= amount;
    m_player.damageFlash = 1.0f;
    play("player_hurt", 0.9f, random(0.95f, 1.05f));
    if (m_player.health <= 0.0f) {
        m_player.health = 0.0f;
        m_player.alive = false;
        LOG_INFO("[GAME] The player died after " << m_levelTime << " s\n");
    }
}

// ---------------------------------------------------------------------------------------------
// Enemies
// ---------------------------------------------------------------------------------------------

void GameWorld::setEnemyState(Entity& enemy, EnemyState state)
{
    enemy.state = state;
    enemy.stateTime = 0.0f;
    switch (state) {
    case EnemyState::Idle:
        enemy.animator.play("idle", true, 0.3f);
        break;
    case EnemyState::Chase:
        enemy.animator.play("walk", true, 0.2f);
        break;
    case EnemyState::Attack:
        enemy.attackLanded = false;
        enemy.animator.play("attack", false, 0.1f, 1.0f, true);
        playAt("enemy_attack", enemy.position + glm::vec3(0.0f, 1.4f, 0.0f), 0.8f, random(0.9f, 1.1f));
        break;
    case EnemyState::Hurt:
        enemy.animator.play("hit", false, 0.05f, 1.0f, true);
        break;
    case EnemyState::Dead:
        enemy.animator.play("death", false, 0.1f);
        break;
    }
}

bool GameWorld::lineOfSight(const glm::vec3& from, const glm::vec3& to) const
{
    const glm::vec3 delta = to - from;
    const float distance = glm::length(delta);
    float hit = 0.0f;
    return distance < 0.2f || !m_physics.castRay(from, delta, distance - 0.15f, hit);
}

void GameWorld::alertNearby(const glm::vec3& position, float radius)
{
    for (Entity& enemy : m_entities) {
        if (enemy.kind != Entity::Kind::Enemy || enemy.state == EnemyState::Dead || enemy.alerted)
            continue;
        if (glm::distance(enemy.position, position) < std::min(radius, enemy.hearingRange)) {
            enemy.alerted = true;
            enemy.lastSeen = m_lastPlayerFeet;
        }
    }
}

void GameWorld::updateEnemy(Entity& enemy, const Input& input)
{
    const float dt = input.dt;
    enemy.animator.update(dt);
    enemy.hurtFlash = std::max(0.0f, enemy.hurtFlash - dt * 4.0f);
    enemy.soundCooldown -= dt;
    enemy.attackCooldown -= dt;
    enemy.stateTime += dt;
    if (enemy.state == EnemyState::Dead || enemy.character < 0)
        return;

    enemy.position = m_physics.characterPosition(enemy.character);
    const glm::vec3 toPlayer = input.playerFeet - enemy.position;
    const float playerDistance = glm::length(flat(toPlayer));
    const glm::vec3 eye = enemy.position + glm::vec3(0.0f, kEnemyEyeHeight * enemy.size, 0.0f);
    // Bigger bodies reach further.
    const float attackRange = kAttackRange * std::max(enemy.size, 1.0f);
    const float hitRange = kHitRange * std::max(enemy.size, 1.0f);

    // Senses, a few times a second.
    enemy.thinkTimer -= dt;
    if (enemy.thinkTimer <= 0.0f && m_player.alive) {
        enemy.thinkTimer = 0.2f;
        if (enemy.hunter) {
            enemy.alerted = true;
            enemy.lastSeen = input.playerFeet;
        }
        const glm::vec3 facing = forwardFromYaw(enemy.yaw);
        const bool inView = playerDistance < 5.0f ||
            glm::dot(facing, flat(toPlayer) / std::max(playerDistance, 1e-3f)) > -0.1f;
        if (glm::length(toPlayer) < enemy.sightRange && (enemy.alerted || inView) && lineOfSight(eye, input.eye)) {
            enemy.lastSeen = input.playerFeet;
            if (!enemy.alerted) {
                enemy.alerted = true;
                playAt("enemy_alert", eye, 1.0f, random(0.9f, 1.1f));
                enemy.soundCooldown = 1.0f;
            }
        }
    }

    glm::vec3 velocity(0.0f);
    switch (enemy.state) {
    case EnemyState::Idle:
        if (enemy.alerted && m_player.alive)
            setEnemyState(enemy, EnemyState::Chase);
        break;
    case EnemyState::Hurt:
        if (enemy.stateTime > 0.35f)
            setEnemyState(enemy, EnemyState::Chase);
        break;
    case EnemyState::Attack: {
        if (playerDistance > 0.1f)
            enemy.yaw = approachAngle(enemy.yaw, yawOf(toPlayer), kTurnSpeed * dt);
        if (!enemy.attackLanded && enemy.animator.normalizedTime() >= 0.5f) {
            enemy.attackLanded = true;
            const bool facing = glm::dot(forwardFromYaw(enemy.yaw), flat(toPlayer) / std::max(playerDistance, 1e-3f)) > 0.3f;
            if (playerDistance < hitRange && std::abs(toPlayer.y) < 1.5f * std::max(enemy.size, 1.0f) && facing)
                damagePlayer(enemy.damage);
        }
        if (enemy.animator.finished()) {
            enemy.attackCooldown = enemy.attackDelay;
            setEnemyState(enemy, m_player.alive ? EnemyState::Chase : EnemyState::Idle);
        }
        break;
    }
    case EnemyState::Chase: {
        if (!m_player.alive) {
            enemy.alerted = false;
            setEnemyState(enemy, EnemyState::Idle);
            break;
        }
        if (playerDistance < attackRange && enemy.attackCooldown <= 0.0f && std::abs(toPlayer.y) < 1.5f * std::max(enemy.size, 1.0f)) {
            setEnemyState(enemy, EnemyState::Attack);
            break;
        }
        const glm::vec3 toTarget = flat(enemy.lastSeen - enemy.position);
        const float targetDistance = glm::length(toTarget);
        if (targetDistance < 0.6f && playerDistance > attackRange) {
            // Reached where the player was last seen without finding them.
            enemy.alerted = false;
            setEnemyState(enemy, EnemyState::Idle);
            break;
        }
        const glm::vec3 direction = targetDistance > 1e-3f ? toTarget / targetDistance : forwardFromYaw(enemy.yaw);
        enemy.yaw = approachAngle(enemy.yaw, yawOf(direction), kTurnSpeed * dt);
        if (playerDistance > 1.2f)
            velocity = direction * enemy.speed;
        // Keep out of each other's way.
        for (const Entity& other : m_entities) {
            if (&other == &enemy || other.kind != Entity::Kind::Enemy || other.state == EnemyState::Dead)
                continue;
            const glm::vec3 away = flat(enemy.position - other.position);
            const float distance = glm::length(away);
            if (distance > 1e-3f && distance < 1.1f)
                velocity += away / distance * (1.1f - distance) * 4.0f;
        }
        // Stuck on a corner: step aside for a moment.
        if (enemy.sidestepTime > 0.0f) {
            enemy.sidestepTime -= dt;
            velocity += glm::vec3(-direction.z, 0.0f, direction.x) * enemy.sidestepSign * enemy.speed;
        }
        enemy.stuckTimer += dt;
        if (enemy.stuckTimer > 0.8f) {
            if (glm::length(flat(enemy.position - enemy.stuckCheckPosition)) < 0.3f && playerDistance > 1.5f) {
                enemy.sidestepTime = 0.7f;
                enemy.sidestepSign = random(0.0f, 1.0f) < 0.5f ? -1.0f : 1.0f;
            }
            enemy.stuckTimer = 0.0f;
            enemy.stuckCheckPosition = enemy.position;
        }
        enemy.animator.setSpeed(std::clamp(enemy.speed / 3.0f, 0.6f, 1.8f));
        enemy.stepTimer -= dt;
        if (enemy.stepTimer <= 0.0f) {
            playAt("footstep", enemy.position, 0.5f, random(0.6f, 0.7f));
            enemy.stepTimer = 0.4f;
        }
        break;
    }
    case EnemyState::Dead:
        break;
    }
    // Characters only collide with the level, so an enemy the player walks into steps back.
    constexpr float kPersonalSpace = 0.85f;
    if (m_player.alive && playerDistance > 1e-3f && playerDistance < kPersonalSpace && std::abs(toPlayer.y) < 1.5f)
        velocity -= flat(toPlayer) / playerDistance * (kPersonalSpace - playerDistance) * 8.0f;
    m_physics.moveCharacter(enemy.character, dt, velocity, false);
    enemy.position = m_physics.characterPosition(enemy.character);
}

void GameWorld::damageEnemy(Entity& enemy, float amount)
{
    if (enemy.state == EnemyState::Dead)
        return;
    enemy.health -= amount;
    enemy.hurtFlash = 1.0f;
    enemy.alerted = true;
    enemy.lastSeen = m_lastPlayerFeet;
    const glm::vec3 chest = enemy.position + glm::vec3(0.0f, 1.3f, 0.0f);
    if (enemy.health > 0.0f) {
        if (enemy.soundCooldown <= 0.0f) {
            playAt("enemy_hurt", chest, 1.0f, random(0.9f, 1.15f));
            enemy.soundCooldown = 0.4f;
        }
        // A swing past its start still lands.
        if (enemy.state != EnemyState::Attack || enemy.animator.normalizedTime() < 0.3f)
            setEnemyState(enemy, EnemyState::Hurt);
        return;
    }

    setEnemyState(enemy, EnemyState::Dead);
    if (enemy.character >= 0) {
        m_physics.removeCharacter(enemy.character);
        enemy.character = -1;
    }
    playAt("enemy_death", chest, 1.0f, random(0.9f, 1.1f));
    spawnParticles(chest, kUp, glm::vec3(0.4f, 0.02f, 0.02f), 14, 3.0f, 0.07f, 0.8f, true);
    ++m_enemiesKilled;
    LOG_INFO("[GAME] Killed " << enemy.name << " (" << m_enemiesKilled << "/" << m_enemiesTotal << ")\n");
    if (m_enemiesKilled >= m_enemiesTotal && !m_exitUnlocked) {
        m_exitUnlocked = true;
        play("exit_unlock", 0.8f);
        showMessage(m_hasExit ? "All enemies are dead. The exit is open!" : "All enemies are dead!");
    }
}

float GameWorld::rayHitsEnemy(const glm::vec3& origin, const glm::vec3& dir, const Entity& enemy)
{
    const float radius = kEnemyRadius * enemy.size;
    // A vertical capsule from the feet to the top of the head.
    const glm::vec3 bottom = enemy.position + glm::vec3(0.0f, radius, 0.0f);
    const glm::vec3 top = enemy.position + glm::vec3(0.0f, kEnemyHeight * enemy.size - radius, 0.0f);
    float best = -1.0f;
    const auto consider = [&](float t) {
        if (t >= 0.0f && (best < 0.0f || t < best))
            best = t;
    };
    const glm::vec2 o(origin.x - bottom.x, origin.z - bottom.z);
    const glm::vec2 d(dir.x, dir.z);
    const float a = glm::dot(d, d);
    if (a > 1e-8f) {
        const float b = 2.0f * glm::dot(o, d);
        const float c = glm::dot(o, o) - radius * radius;
        const float discriminant = b * b - 4.0f * a * c;
        if (discriminant >= 0.0f) {
            const float t = (-b - std::sqrt(discriminant)) / (2.0f * a);
            const float y = origin.y + dir.y * t;
            if (y >= bottom.y && y <= top.y)
                consider(t);
        }
    }
    for (const glm::vec3& center : { bottom, top }) {
        const glm::vec3 oc = origin - center;
        const float b = glm::dot(oc, dir);
        const float c = glm::dot(oc, oc) - radius * radius;
        const float discriminant = b * b - c;
        if (discriminant >= 0.0f)
            consider(-b - std::sqrt(discriminant));
    }
    return best;
}

// ---------------------------------------------------------------------------------------------
// Weapon
// ---------------------------------------------------------------------------------------------

void GameWorld::updateWeapon(const Input& input)
{
    const float dt = input.dt;
    Weapon& weapon = m_weapon;
    weapon.animator.update(dt);
    weapon.cooldown -= dt;
    weapon.emptyClickCooldown -= dt;
    const float targetBob = input.moving ? (input.sprinting ? 1.4f : 1.0f) : 0.0f;
    weapon.bobAmount += (targetBob - weapon.bobAmount) * std::min(dt * 8.0f, 1.0f);
    weapon.bobPhase += dt * (input.sprinting ? 11.0f : 8.0f) * (input.moving ? 1.0f : 0.0f);
    if (input.reload)
        weapon.reloadRequested = true;

    const auto startReload = [&] {
        weapon.action = WeaponAction::Reloading;
        weapon.actionTime = 0.0f;
        weapon.reloadRequested = false;
        weapon.animator.play("reload", false, 0.08f, 1.0f, true);
    };

    switch (weapon.action) {
    case WeaponAction::Firing:
        weapon.actionTime += dt;
        if (weapon.animator.finished()) {
            weapon.action = WeaponAction::Idle;
            weapon.animator.play("idle", true, 0.1f);
        }
        break;
    case WeaponAction::Reloading:
        weapon.actionTime += dt;
        if (weapon.actionTime >= kShellReloadTime) {
            ++m_player.shells;
            --m_player.reserve;
            play("shell_insert", 0.8f, random(0.95f, 1.05f));
            weapon.actionTime = 0.0f;
            // Holding the trigger stops reloading after this shell.
            if (m_player.shells >= kMagazineSize || m_player.reserve <= 0 || input.fire) {
                weapon.action = WeaponAction::Idle;
                weapon.cooldown = 0.0f;
                weapon.animator.play("idle", true, 0.1f);
            }
            else {
                weapon.animator.play("reload", false, 0.05f, 1.0f, true);
            }
        }
        break;
    case WeaponAction::Idle:
        break;
    }

    if (weapon.action == WeaponAction::Reloading || weapon.cooldown > 0.0f)
        return;
    if (input.fire) {
        if (m_player.shells > 0) {
            fire(input);
        }
        else if (m_player.reserve > 0) {
            startReload();
        }
        else if (weapon.emptyClickCooldown <= 0.0f) {
            play("shotgun_empty", 0.8f);
            weapon.emptyClickCooldown = 0.45f;
            showMessage("Out of shells");
        }
    }
    else if (weapon.reloadRequested || (m_player.shells == 0 && m_player.reserve > 0)) {
        if (m_player.shells < kMagazineSize && m_player.reserve > 0)
            startReload();
        else
            weapon.reloadRequested = false;
    }
}

void GameWorld::fire(const Input& input)
{
    Weapon& weapon = m_weapon;
    --m_player.shells;
    weapon.cooldown = kFireInterval;
    weapon.action = WeaponAction::Firing;
    weapon.actionTime = 0.0f;
    weapon.animator.play("fire", false, 0.0f, 1.0f, true);
    play("shotgun_fire", 1.0f, random(0.96f, 1.04f));
    m_delayedSounds.push_back({ 0.32f, "shotgun_pump", 0.75f });
    m_recoil += 4.5f;
    // Everyone nearby hears it.
    alertNearby(input.eye, kHearingRange);

    const glm::vec3 forward = glm::normalize(input.forward);
    const glm::vec3 right = glm::normalize(glm::cross(forward, std::abs(forward.y) > 0.99f ? glm::vec3(1, 0, 0) : kUp));
    const glm::vec3 up = glm::cross(right, forward);
    const float spread = std::tan(glm::radians(kSpreadDegrees));
    std::vector<float> damage(m_entities.size(), 0.0f);
    bool impactPlayed = false;
    for (int pellet = 0; pellet < kPellets; ++pellet) {
        // Uniform over the cone's cross-section.
        const float radius = spread * std::sqrt(random(0.0f, 1.0f));
        const float angle = random(0.0f, 2.0f * kPi);
        const glm::vec3 dir = glm::normalize(forward + right * (radius * std::cos(angle)) + up * (radius * std::sin(angle)));

        float wallDistance = kShotRange;
        glm::vec3 normal(0.0f, 1.0f, 0.0f);
        const bool wall = m_physics.castRay(input.eye, dir, kShotRange, wallDistance, &normal);
        int target = -1;
        float targetDistance = wallDistance;
        for (size_t i = 0; i < m_entities.size(); ++i) {
            const Entity& entity = m_entities[i];
            if (entity.kind != Entity::Kind::Enemy || entity.state == EnemyState::Dead)
                continue;
            const float t = rayHitsEnemy(input.eye, dir, entity);
            if (t >= 0.0f && t < targetDistance) {
                target = static_cast<int>(i);
                targetDistance = t;
            }
        }
        if (target >= 0) {
            const Entity& enemy = m_entities[target];
            const glm::vec3 point = input.eye + dir * targetDistance;
            const float falloff = std::clamp((targetDistance - 6.0f) / 24.0f, 0.0f, 1.0f);
            float amount = kPelletDamageNear + (kPelletDamageFar - kPelletDamageNear) * falloff;
            if (point.y > enemy.position.y + 1.55f)
                amount *= 1.6f; // head
            damage[target] += amount;
            spawnParticles(point, -dir, glm::vec3(0.45f, 0.03f, 0.02f), 4, 2.5f, 0.06f, 0.5f, true);
        }
        else if (wall) {
            const glm::vec3 point = input.eye + dir * wallDistance + normal * 0.02f;
            spawnParticles(point, normal, glm::vec3(1.0f, 0.72f, 0.3f), 3, 4.0f, 0.025f, 0.25f, true);
            spawnParticles(point, normal, glm::vec3(0.5f, 0.48f, 0.45f), 2, 1.0f, 0.07f, 0.6f, false);
            if (!impactPlayed) {
                playAt("impact", point, 0.6f, random(0.8f, 1.2f));
                impactPlayed = true;
            }
        }
    }
    for (size_t i = 0; i < damage.size(); ++i) {
        if (damage[i] <= 0.0f)
            continue;
        m_player.hitMarker = 1.0f;
        playAt("flesh_hit", m_entities[i].position + glm::vec3(0.0f, 1.2f, 0.0f), 0.9f, random(0.9f, 1.1f));
        damageEnemy(m_entities[i], damage[i]);
    }
}

// ---------------------------------------------------------------------------------------------
// Pickups and the exit
// ---------------------------------------------------------------------------------------------

void GameWorld::updatePickups(const Input& input)
{
    for (Entity& pickup : m_entities) {
        if (pickup.kind != Entity::Kind::HealthPack && pickup.kind != Entity::Kind::AmmoPack)
            continue;
        pickup.animator.update(input.dt);
        if (!pickup.active)
            continue;
        const glm::vec3 offset = input.playerFeet - pickup.position;
        if (glm::length(flat(offset)) > kPickupRadius || std::abs(offset.y) > 1.5f)
            continue;
        if (pickup.kind == Entity::Kind::HealthPack) {
            if (m_player.health >= kMaxHealth) {
                if (m_pickupMessageCooldown <= 0.0f) {
                    showMessage("Health is full");
                    m_pickupMessageCooldown = 3.0f;
                }
                continue;
            }
            m_player.health = std::min(kMaxHealth, m_player.health + pickup.amount);
            play("pickup_health", 0.8f);
            showMessage("+" + std::to_string(static_cast<int>(pickup.amount)) + " health");
        }
        else {
            if (m_player.reserve >= kMaxReserve) {
                if (m_pickupMessageCooldown <= 0.0f) {
                    showMessage("Can't carry more shells");
                    m_pickupMessageCooldown = 3.0f;
                }
                continue;
            }
            m_player.reserve = std::min(kMaxReserve, m_player.reserve + static_cast<int>(pickup.amount));
            play("pickup_ammo", 0.8f);
            showMessage("+" + std::to_string(static_cast<int>(pickup.amount)) + " shells");
        }
        pickup.active = false;
    }
}

void GameWorld::updateExit(const Input& input)
{
    for (Entity& exit : m_entities) {
        if (exit.kind != Entity::Kind::Exit)
            continue;
        exit.animator.update(input.dt);
        const glm::vec3 offset = input.playerFeet - exit.position;
        if (glm::length(flat(offset)) > kExitRadius || offset.y < -1.0f || offset.y > 2.5f)
            continue;
        if (m_exitUnlocked) {
            if (!m_levelComplete) {
                m_levelComplete = true;
                m_nextLevel = exit.nextLevel;
                play("level_complete", 0.9f);
                LOG_INFO("[GAME] Level complete in " << m_levelTime << " s, " << m_enemiesKilled << "/"
                    << m_enemiesTotal << " enemies\n");
            }
        }
        else if (m_exitMessageCooldown <= 0.0f) {
            const int left = m_enemiesTotal - m_enemiesKilled;
            showMessage("The exit is sealed: " + std::to_string(left) + (left == 1 ? " enemy left" : " enemies left"));
            m_exitMessageCooldown = 3.0f;
        }
    }
    // A level without an exit ends a moment after its last enemy dies.
    if (!m_hasExit && m_exitUnlocked && m_enemiesTotal > 0 && !m_levelComplete) {
        m_completeTimer += input.dt;
        if (m_completeTimer > 2.0f) {
            m_levelComplete = true;
            play("level_complete", 0.9f);
        }
    }
}

bool GameWorld::nearestEnemy(const glm::vec3& eye, glm::vec3& aimPoint, bool& visible) const
{
    float best = -1.0f;
    for (const Entity& enemy : m_entities) {
        if (enemy.kind != Entity::Kind::Enemy || enemy.state == EnemyState::Dead)
            continue;
        const glm::vec3 chest = enemy.position + glm::vec3(0.0f, 1.2f, 0.0f);
        const bool sees = lineOfSight(eye, chest);
        // Visible enemies first, then the closest.
        const float score = glm::distance(eye, chest) + (sees ? 0.0f : 1000.0f);
        if (best < 0.0f || score < best) {
            best = score;
            aimPoint = chest;
            visible = sees;
        }
    }
    return best >= 0.0f;
}

bool GameWorld::exitPosition(glm::vec3& position) const
{
    for (const Entity& entity : m_entities) {
        if (entity.kind == Entity::Kind::Exit) {
            position = entity.position;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Effects and sound
// ---------------------------------------------------------------------------------------------

void GameWorld::spawnParticles(const glm::vec3& position, const glm::vec3& normal, const glm::vec3& color, int count,
    float speed, float size, float life, bool gravity)
{
    for (int i = 0; i < count; ++i) {
        Particle particle;
        particle.position = position;
        const glm::vec3 scatter(random(-1.0f, 1.0f), random(-1.0f, 1.0f), random(-1.0f, 1.0f));
        particle.velocity = (normal * random(0.4f, 1.0f) + scatter * 0.6f) * speed;
        particle.color = color * random(0.8f, 1.1f);
        particle.size = size * random(0.6f, 1.3f);
        particle.maxLife = particle.life = life * random(0.6f, 1.2f);
        particle.gravity = gravity;
        m_particles.push_back(particle);
    }
    // A long fight must not pile up particles.
    if (m_particles.size() > 400)
        m_particles.erase(m_particles.begin(), m_particles.begin() + (m_particles.size() - 400));
}

void GameWorld::updateParticles(float dt)
{
    for (Particle& particle : m_particles) {
        if (particle.gravity)
            particle.velocity.y -= 9.81f * dt;
        particle.velocity *= std::exp(-dt * 1.5f);
        particle.position += particle.velocity * dt;
        particle.life -= dt;
    }
    std::erase_if(m_particles, [](const Particle& p) { return p.life <= 0.0f; });
}

void GameWorld::playAt(const std::string& sound, const glm::vec3& position, float volume, float pitch)
{
    AudioSystem::PlayParams params;
    params.volume = volume;
    params.pitch = pitch;
    params.positional = true;
    params.position = position;
    params.minDistance = 2.5f;
    params.maxDistance = 45.0f;
    m_audio.play(sound, params);
}

void GameWorld::play(const std::string& sound, float volume, float pitch)
{
    AudioSystem::PlayParams params;
    params.volume = volume;
    params.pitch = pitch;
    m_audio.play(sound, params);
}

float GameWorld::random(float lo, float hi)
{
    return std::uniform_real_distribution<float>(lo, hi)(m_random);
}

// ---------------------------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------------------------

void GameWorld::addModel(const GameModel& model, Animator& animator, const glm::mat4& world, const glm::vec3& tint,
    bool castShadow, std::vector<DynamicInstance>& out, int tintedNode)
{
    animator.evaluate(m_nodeMatrices);
    for (size_t n = 0; n < model.data.nodes.size() && n < m_nodeMatrices.size(); ++n) {
        const int mesh = model.data.nodes[n].mesh;
        if (mesh < 0 || mesh >= static_cast<int>(model.meshModels.size()))
            continue;
        const glm::mat4 matrix = world * m_nodeMatrices[n];
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

void GameWorld::collectDrawables(const glm::mat4& cameraWorld, std::vector<DynamicInstance>& out)
{
    for (Entity& entity : m_entities) {
        if (!entity.model || !entity.active)
            continue;
        const glm::mat4 world = glm::rotate(glm::translate(glm::mat4(1.0f), entity.position), entity.yaw, kUp);
        switch (entity.kind) {
        case Entity::Kind::Enemy: {
            const glm::vec3 flash = glm::mix(entity.tint, glm::vec3(1.0f, 0.35f, 0.3f), entity.hurtFlash);
            addModel(*entity.model, entity.animator, glm::scale(world, glm::vec3(entity.size)), flash, true, out);
            break;
        }
        case Entity::Kind::Exit: {
            // The field glows green once the exit is open, red while it is sealed.
            const int field = entity.model->data.findNode("field");
            const glm::vec3 tint = m_exitUnlocked ? glm::vec3(1.0f) : glm::vec3(1.0f, 0.22f, 0.18f);
            addModel(*entity.model, entity.animator, world, tint, true, out, field);
            break;
        }
        default:
            addModel(*entity.model, entity.animator, world, glm::vec3(1.0f), true, out);
            break;
        }
    }

    if (const std::optional<size_t> cube = m_assets.cube()) {
        for (const Particle& particle : m_particles) {
            DynamicInstance& part = out.emplace_back();
            part.instance.modelIndex = *cube;
            part.instance.position = particle.position;
            part.instance.color = particle.color;
            const float size = particle.size * std::clamp(particle.life / particle.maxLife * 1.5f, 0.2f, 1.0f);
            part.transform = glm::scale(glm::translate(glm::mat4(1.0f), particle.position), glm::vec3(size));
            part.castShadow = false;
        }
    }

    const GameModel& shotgun = m_assets.model(GameModelId::Shotgun);
    if (m_player.alive && shotgun.valid()) {
        const Weapon& weapon = m_weapon;
        const glm::vec3 bob(std::sin(weapon.bobPhase) * 0.012f * weapon.bobAmount,
            -std::abs(std::cos(weapon.bobPhase)) * 0.014f * weapon.bobAmount, 0.0f);
        addModel(shotgun, m_weapon.animator, cameraWorld * glm::translate(glm::mat4(1.0f), bob), glm::vec3(1.0f),
            false, out);
    }
}
