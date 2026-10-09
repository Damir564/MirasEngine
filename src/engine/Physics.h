#pragma once
#include <memory>
#include <glm/glm.hpp>

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
    // shaped by its ModelInstance::collision.
    void buildStaticScene(ModelManager& models);
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
    bool characterOnGround(int id) const;

    // The player's position is the bottom of the capsule (its feet).
    void spawnPlayer(const glm::vec3& feetPosition);
    bool hasPlayer() const;
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
