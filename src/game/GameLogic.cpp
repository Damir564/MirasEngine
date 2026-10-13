// Logic entities (trigger, relay, timer, counter) and the inputs and outputs wired between entities
// (EntityConnection): an output queues each of its connections' inputs, updateLogic() delivers them when due.
#include "GameWorld.h"
#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include "engine/Log.h"
#include "engine/ModelManager.h"
#include "engine/Physics.h"
#include "engine/Renderer.h"

namespace {

// The use key reaches a door part this far from the eyes, looking at it within about 60 degrees.
constexpr float kDoorReach = 2.4f;
constexpr float kDoorFacing = 0.5f;

// Inputs per tick at most: a chain of zero-delay wires that loops would otherwise never end.
constexpr int kMaxInputsPerTick = 256;
// The player's point that triggers test: a little above the feet, so a volume lying on the floor counts.
constexpr float kTriggerProbeHeight = 0.5f;

bool insideTrigger(const Entity& trigger, const glm::vec3& point)
{
    const glm::vec3 center = trigger.position + glm::vec3(0.0f, trigger.halfSize.y, 0.0f);
    const glm::vec3 d = point - center;
    // The entity faces +Z turned by yaw: its +Z is (sin, 0, cos), its +X (cos, 0, -sin).
    const float s = std::sin(trigger.yaw), c = std::cos(trigger.yaw);
    const glm::vec3 local(d.x * c - d.z * s, d.y, d.x * s + d.z * c);
    return std::abs(local.x) <= trigger.halfSize.x && std::abs(local.y) <= trigger.halfSize.y &&
        std::abs(local.z) <= trigger.halfSize.z;
}

} // namespace

void GameWorld::spawnLogic(const SpawnInfo& info, Entity::Kind kind)
{
    Entity& entity = addEntity(kind, info, GameModelId::Count);
    entity.enabled = entityParam(info.params, "enabled", 1.0f) != 0.0f;
    switch (kind) {
    case Entity::Kind::Trigger:
        entity.halfSize = 0.5f * glm::vec3(std::max(entityParam(info.params, "size_x", 2.0f), 0.1f),
            std::max(entityParam(info.params, "size_y", 2.0f), 0.1f),
            std::max(entityParam(info.params, "size_z", 2.0f), 0.1f));
        entity.once = entityParam(info.params, "once", 1.0f) != 0.0f;
        break;
    case Entity::Kind::Timer:
        entity.interval = std::max(entityParam(info.params, "interval", 1.0f), 0.05f);
        entity.once = entityParam(info.params, "repeat", 1.0f) == 0.0f;
        entity.enabled = entityParam(info.params, "running", 0.0f) != 0.0f;
        entity.timeLeft = entity.interval;
        break;
    case Entity::Kind::Counter:
        entity.target = std::max(static_cast<int>(std::lround(entityParam(info.params, "target", 3.0f))), 1);
        break;
    default:
        break;
    }
}

void GameWorld::fireOutput(const Entity& source, std::string_view output)
{
    for (const EntityConnection& connection : source.outputs) {
        if (connection.output == output)
            m_pendingInputs.push_back({ connection.delay, source.name + "." + connection.output, connection.target,
                connection.input });
    }
}

void GameWorld::updateLogic(const Input& input)
{
    const float dt = input.dt;
    const glm::vec3 probe = input.playerFeet + glm::vec3(0.0f, kTriggerProbeHeight, 0.0f);
    for (Entity& entity : m_entities) {
        if (entity.kind == Entity::Kind::Trigger) {
            const bool inside = entity.enabled && m_player.alive && insideTrigger(entity, probe);
            if (inside && !entity.inside) {
                fireOutput(entity, "on_enter");
                if (entity.once)
                    entity.enabled = false;
            }
            else if (!inside && entity.inside) {
                fireOutput(entity, "on_exit");
            }
            entity.inside = inside;
        }
        else if (entity.kind == Entity::Kind::Door) {
            updateDoor(entity, dt);
        }
        else if (entity.kind == Entity::Kind::Timer && entity.enabled) {
            entity.timeLeft -= dt;
            if (entity.timeLeft <= 0.0f) {
                fireOutput(entity, "on_timer");
                entity.timeLeft = std::max(entity.timeLeft + entity.interval, 0.0f);
                if (entity.once)
                    entity.enabled = false;
            }
        }
    }

    // Due inputs in the order they were fired; zero-delay inputs they cause are delivered this tick too.
    for (PendingInput& pending : m_pendingInputs)
        pending.delay -= dt;
    int delivered = 0;
    for (size_t i = 0; i < m_pendingInputs.size();) {
        if (m_pendingInputs[i].delay > 0.0f) {
            ++i;
            continue;
        }
        if (++delivered > kMaxInputsPerTick) {
            LOG_ERROR("[IO] More than " << kMaxInputsPerTick << " inputs in one tick (a loop of zero-delay outputs?); "
                "the rest waits for the next tick\n");
            break;
        }
        const PendingInput pending = std::move(m_pendingInputs[i]);
        m_pendingInputs.erase(m_pendingInputs.begin() + static_cast<std::ptrdiff_t>(i));
        const bool found = std::any_of(m_entities.begin(), m_entities.end(),
            [&](const Entity& e) { return e.name == pending.target; });
        LOG_INFO("[IO] " << pending.source << " -> " << pending.target << "." << pending.input
            << (found ? "" : " (no entity with that name)") << "\n");
        for (Entity& entity : m_entities)
            if (entity.name == pending.target)
                receiveInput(entity, pending.input);
    }
}

void GameWorld::receiveInput(Entity& entity, std::string_view input)
{
    switch (entity.kind) {
    case Entity::Kind::Enemy:
        if (input == "alert")
            alertEnemy(entity);
        else if (input == "kill")
            damageEnemy(entity, entity.health + 1.0f);
        break;
    case Entity::Kind::Trigger:
        if (input == "enable")
            entity.enabled = true;
        else if (input == "disable")
            entity.enabled = false;
        else if (input == "toggle")
            entity.enabled = !entity.enabled;
        break;
    case Entity::Kind::Relay:
        if (input == "trigger" && entity.enabled)
            fireOutput(entity, "on_trigger");
        else if (input == "enable")
            entity.enabled = true;
        else if (input == "disable")
            entity.enabled = false;
        break;
    case Entity::Kind::Timer:
        if (input == "start" || (input == "toggle" && !entity.enabled)) {
            if (!entity.enabled)
                entity.timeLeft = entity.interval;
            entity.enabled = true;
        }
        else if (input == "stop" || input == "toggle") {
            entity.enabled = false;
        }
        else if (input == "fire") {
            fireOutput(entity, "on_timer");
        }
        break;
    case Entity::Kind::Door:
        if (input == "open" || input == "close" || input == "toggle")
            moveDoor(entity, input == "toggle" ? !entity.opening : input == "open");
        else if (input == "lock")
            entity.locked = true;
        else if (input == "unlock")
            entity.locked = false;
        break;
    case Entity::Kind::Exit:
        if (input == "unlock")
            unlockExit();
        else if (input == "lock")
            m_exitUnlocked = false;
        break;
    case Entity::Kind::Counter: {
        const int before = entity.count;
        if (input == "add")
            ++entity.count;
        else if (input == "subtract")
            --entity.count;
        else if (input == "reset")
            entity.count = 0;
        if (before < entity.target && entity.count >= entity.target)
            fireOutput(entity, "on_reached");
        break;
    }
    default:
        break;
    }
}

// ---------------------------------------------------------------------------------------------
// Doors
// ---------------------------------------------------------------------------------------------

std::vector<uint64_t> GameWorld::movableInstanceIds(const std::vector<ModelInstance>& instances)
{
    std::vector<uint64_t> ids;
    for (const ModelInstance& door : instances) {
        if (door.entity != kDoorEntity)
            continue;
        const std::string group = entityParamText(door.entityParams, "group");
        if (group.empty())
            continue;
        for (const ModelInstance& part : instances)
            if (part.group == group && part.entity.empty())
                ids.push_back(part.id);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

void GameWorld::spawnDoor(const SpawnInfo& info)
{
    Entity& door = addEntity(Entity::Kind::Door, info, GameModelId::Count);
    door.group = entityParamText(info.params, "group");
    door.swing = glm::radians(std::clamp(entityParam(info.params, "angle", 0.0f), -180.0f, 180.0f));
    door.slide = glm::vec3(entityParam(info.params, "move_x", 0.0f), entityParam(info.params, "move_y", 3.0f),
        entityParam(info.params, "move_z", 0.0f));
    door.moveTime = std::max(entityParam(info.params, "time", 1.0f), 0.05f);
    door.usable = entityParam(info.params, "use", 1.0f) != 0.0f;
    door.locked = entityParam(info.params, "locked", 0.0f) != 0.0f;
    door.opening = entityParam(info.params, "open", 0.0f) != 0.0f;
    door.openness = door.previousOpenness = door.opening ? 1.0f : 0.0f;
    if (door.group.empty())
        LOG_ERROR("[GAME] Door " << door.name << " has no group: nothing moves\n");
}

void GameWorld::attachDoorParts(const std::vector<ModelInstance>& instances)
{
    m_movableIds = movableInstanceIds(instances);
    for (Entity& door : m_entities) {
        if (door.kind != Entity::Kind::Door || door.group.empty())
            continue;
        for (size_t i = 0; i < instances.size(); ++i)
            if (instances[i].group == door.group && instances[i].entity.empty())
                door.parts.push_back(i);
        if (door.parts.empty())
            LOG_ERROR("[GAME] Door " << door.name << ": no objects in group '" << door.group << "'\n");
        placeDoorParts(door);
    }
}

glm::mat4 GameWorld::doorMotion(const Entity& door, float openness) const
{
    if (door.swing != 0.0f) {
        const glm::mat4 toHinge = glm::translate(glm::mat4(1.0f), door.position);
        return glm::rotate(toHinge, door.swing * openness, glm::vec3(0.0f, 1.0f, 0.0f)) *
            glm::translate(glm::mat4(1.0f), -door.position);
    }
    const glm::mat4 frame = glm::rotate(glm::mat4(1.0f), door.yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    return glm::translate(glm::mat4(1.0f), glm::vec3(frame * glm::vec4(door.slide * openness, 0.0f)));
}

void GameWorld::placeDoorParts(const Entity& door)
{
    const glm::mat4 motion = doorMotion(door, door.openness);
    const glm::quat turn = glm::angleAxis(door.swing * door.openness, glm::vec3(0.0f, 1.0f, 0.0f));
    const auto& instances = m_models.getInstances();
    for (size_t index : door.parts) {
        const ModelInstance& part = instances[index];
        m_physics.moveBody(part.id, glm::vec3(motion * glm::vec4(part.position, 1.0f)),
            turn * glm::quat(glm::radians(part.rotation)));
    }
}

void GameWorld::moveDoor(Entity& door, bool open)
{
    if (door.opening == open)
        return;
    door.opening = open;
    playAt("door_move", door.position + glm::vec3(0.0f, 1.2f, 0.0f), 0.8f, random(0.95f, 1.05f));
}

void GameWorld::updateDoor(Entity& door, float dt)
{
    const float target = door.opening ? 1.0f : 0.0f;
    if (door.openness == target)
        return;
    const float step = dt / door.moveTime;
    door.openness = door.opening ? std::min(door.openness + step, 1.0f) : std::max(door.openness - step, 0.0f);
    placeDoorParts(door);
    if (door.openness == target) {
        playAt("door_stop", door.position + glm::vec3(0.0f, 1.2f, 0.0f), 0.8f, random(0.95f, 1.05f));
        fireOutput(door, door.opening ? "on_opened" : "on_closed");
    }
}

int GameWorld::doorInReach() const
{
    const auto& instances = m_models.getInstances();
    int best = -1;
    float bestDistance = kDoorReach;
    for (size_t i = 0; i < m_entities.size(); ++i) {
        const Entity& door = m_entities[i];
        if (door.kind != Entity::Kind::Door || !door.usable)
            continue;
        const glm::mat4 motion = doorMotion(door, door.openness);
        for (size_t index : door.parts) {
            const ModelInstance& part = instances[index];
            const GPUModel* model = m_models.getModel(part.modelIndex);
            const glm::vec3 local = model ? (model->boundsMin + model->boundsMax) * 0.5f : glm::vec3(0.0f);
            const glm::vec3 center(motion * part.getTransformMatrix() * glm::vec4(local, 1.0f));
            const glm::vec3 toPart = center - m_lastEye;
            const float distance = glm::length(toPart);
            if (distance < bestDistance && glm::dot(toPart / std::max(distance, 1e-3f), m_lastForward) > kDoorFacing) {
                best = static_cast<int>(i);
                bestDistance = distance;
            }
        }
    }
    return best;
}

void GameWorld::useDoor(Entity& door)
{
    if (door.locked) {
        if (m_pickupMessageCooldown <= 0.0f) {
            showMessage("It's locked");
            m_pickupMessageCooldown = 1.0f;
        }
        play("reject", 0.5f);
        return;
    }
    moveDoor(door, !door.opening);
}

void GameWorld::unlockExit()
{
    if (m_exitUnlocked)
        return;
    m_exitUnlocked = true;
    play("exit_unlock", 0.8f);
    if (m_hasExit)
        showMessage("The exit is open!");
    for (const Entity& exit : m_entities)
        if (exit.kind == Entity::Kind::Exit)
            fireOutput(exit, "on_unlock");
}

void GameWorld::collectDoors(float blend, std::vector<DynamicInstance>& out) const
{
    const auto& instances = m_models.getInstances();
    for (const Entity& door : m_entities) {
        if (door.kind != Entity::Kind::Door)
            continue;
        const glm::mat4 motion = doorMotion(door, glm::mix(door.previousOpenness, door.openness, blend));
        for (size_t index : door.parts) {
            DynamicInstance& part = out.emplace_back();
            part.instance = instances[index];
            part.transform = motion * part.instance.getTransformMatrix();
        }
    }
}
