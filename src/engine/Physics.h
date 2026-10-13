#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class ModelManager;
namespace JPH { class CharacterVirtual; }

// Jolt-backed collision world: the loaded scene as static triangle meshes plus a walking player capsule.
class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    // Replaces all static colliders with the visible instances of the scene (not game entity markers), each
    // shaped by its ModelInstance::collision. Instances whose ModelInstance::id is in `movable` (sorted) get a
    // kinematic body instead, placed with moveBody().
    void buildStaticScene(ModelManager& models, std::span<const uint64_t> movable = {});
    // Places a movable instance's body (its ModelInstance position and rotation as the game moves it).
    void moveBody(uint64_t instanceId, const glm::vec3& position, const glm::quat& rotation);
    // Removes the colliders and every character.
    void clear();

    // First static collider hit along the ray within maxDistance: its distance and surface normal.
    bool castRay(const glm::vec3& origin, const glm::vec3& direction, float maxDistance, float& hitDistance,
        glm::vec3* hitNormal = nullptr) const;

    // More walking capsules (e.g. enemies), moved like the player. They collide with the static scene only.
    // Ids stay valid until removeCharacter() or clear().
    int addCharacter(const glm::vec3& feetPosition, float radius, float height);
    void removeCharacter(int id);
    void moveCharacter(int id, float dt, const glm::vec3& horizontalVelocity, bool jump);
    glm::vec3 characterPosition(int id) const;

    // The player's position is the bottom of the capsule (its feet).
    void spawnPlayer(const glm::vec3& feetPosition);
    // horizontalVelocity is the target walking velocity in m/s (its y is ignored); the player accelerates
    // toward it on the ground and keeps its momentum in the air.
    void updatePlayer(float dt, const glm::vec3& horizontalVelocity, bool jump);
    glm::vec3 playerPosition() const;
    bool playerOnGround() const;

private:
    struct Impl;
    void updateCharacter(JPH::CharacterVirtual* character, float dt, const glm::vec3& horizontalVelocity, bool jump);

    std::unique_ptr<Impl> m_impl;
};
