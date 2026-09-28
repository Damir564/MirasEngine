#pragma once
#include <memory>
#include <glm/glm.hpp>

class ModelManager;

// Jolt-backed collision world: the loaded scene as static triangle meshes plus a walking player capsule.
class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    // Replaces all static colliders with the visible instances of the scene.
    void buildStaticScene(ModelManager& models);
    void clear();

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
    std::unique_ptr<Impl> m_impl;
};
