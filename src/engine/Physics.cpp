#include "Physics.h"
// Jolt.h must come before any other Jolt header.
#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "ModelManager.h"

namespace {

constexpr float kCapsuleRadius = 0.3f;
constexpr float kCapsuleHalfHeight = 0.6f; // cylinder part; total height 1.8 m
constexpr float kJumpSpeed = 5.0f;
constexpr float kMaxSlopeDegrees = 50.0f;
// Horizontal acceleration in m/s^2. Air control is weak so a jump keeps the momentum it started with.
constexpr float kGroundAcceleration = 30.0f;
constexpr float kGroundFriction = 20.0f;
constexpr float kAirAcceleration = 4.0f;
// Long frames (window drag, loading hitch) would otherwise let the character pass through thin geometry.
constexpr float kMaxStep = 1.0f / 30.0f;

namespace Layers {
constexpr JPH::ObjectLayer NonMoving = 0;
constexpr JPH::ObjectLayer Moving = 1;
}

namespace BroadPhaseLayers {
constexpr JPH::BroadPhaseLayer NonMoving(0);
constexpr JPH::BroadPhaseLayer Moving(1);
constexpr JPH::uint Count = 2;
}

class BroadPhaseLayerMap final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::Count; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override
    {
        return layer == Layers::NonMoving ? BroadPhaseLayers::NonMoving : BroadPhaseLayers::Moving;
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override
    {
        return layer == BroadPhaseLayers::NonMoving ? "NonMoving" : "Moving";
    }
#endif
};

class ObjectVsBroadPhaseFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer broadPhase) const override
    {
        return layer == Layers::Moving || broadPhase == BroadPhaseLayers::Moving;
    }
};

class ObjectPairFilter final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override
    {
        return a == Layers::Moving || b == Layers::Moving;
    }
};

// Jolt's allocator, factory and type registry are process-wide; keep them alive while any world exists.
std::mutex g_joltMutex;
int g_joltUsers = 0;

void acquireJolt()
{
    std::lock_guard lock(g_joltMutex);
    if (g_joltUsers++ == 0) {
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    }
}

void releaseJolt()
{
    std::lock_guard lock(g_joltMutex);
    if (--g_joltUsers == 0) {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
}

JPH::Vec3 gravity() { return JPH::Vec3(0.0f, -9.81f, 0.0f); }
JPH::Vec3 toJolt(const glm::vec3& v) { return JPH::Vec3(v.x, v.y, v.z); }
glm::vec3 toGlm(JPH::Vec3Arg v) { return glm::vec3(v.GetX(), v.GetY(), v.GetZ()); }

JPH::Vec3 moveTowards(JPH::Vec3Arg current, JPH::Vec3Arg target, float maxDelta)
{
    const JPH::Vec3 delta = target - current;
    const float length = delta.Length();
    return length <= maxDelta ? target : current + delta * (maxDelta / length);
}

JPH::ShapeRefC createMeshShape(const GPUModel& model)
{
    const std::vector<glm::vec3>& tris = model.collisionTriangles;
    JPH::TriangleList triangles;
    triangles.reserve(tris.size() / 3);
    for (size_t i = 0; i + 2 < tris.size(); i += 3) {
        triangles.push_back(JPH::Triangle(JPH::Float3(tris[i].x, tris[i].y, tris[i].z),
            JPH::Float3(tris[i + 1].x, tris[i + 1].y, tris[i + 1].z),
            JPH::Float3(tris[i + 2].x, tris[i + 2].y, tris[i + 2].z)));
    }
    if (triangles.empty())
        return nullptr;
    JPH::MeshShapeSettings settings(triangles);
    JPH::ShapeSettings::ShapeResult result = settings.Create();
    if (result.HasError()) {
        std::cerr << "[Physics] Mesh shape for '" << model.name << "' failed: " << result.GetError().c_str() << "\n";
        return nullptr;
    }
    return result.Get();
}

} // namespace

struct PhysicsWorld::Impl {
    Impl()
    {
        acquireJolt();
        tempAllocator = std::make_unique<JPH::TempAllocatorImpl>(10 * 1024 * 1024);
        system = std::make_unique<JPH::PhysicsSystem>();
        system->Init(65536, 0, 65536, 10240, broadPhaseLayers, objectVsBroadPhase, objectPairs);
        system->SetGravity(gravity());
    }

    ~Impl()
    {
        clear();
        system.reset();
        tempAllocator.reset();
        releaseJolt();
    }

    void clear()
    {
        character = nullptr;
        JPH::BodyInterface& bodies = system->GetBodyInterface();
        if (!staticBodies.empty()) {
            bodies.RemoveBodies(staticBodies.data(), static_cast<int>(staticBodies.size()));
            bodies.DestroyBodies(staticBodies.data(), static_cast<int>(staticBodies.size()));
            staticBodies.clear();
        }
    }

    BroadPhaseLayerMap broadPhaseLayers;
    ObjectVsBroadPhaseFilter objectVsBroadPhase;
    ObjectPairFilter objectPairs;
    std::unique_ptr<JPH::TempAllocatorImpl> tempAllocator;
    std::unique_ptr<JPH::PhysicsSystem> system;
    std::vector<JPH::BodyID> staticBodies;
    JPH::Ref<JPH::CharacterVirtual> character;
};

PhysicsWorld::PhysicsWorld() : m_impl(std::make_unique<Impl>()) {}
PhysicsWorld::~PhysicsWorld() = default;

void PhysicsWorld::clear()
{
    m_impl->clear();
}

void PhysicsWorld::buildStaticScene(ModelManager& models)
{
    clear();
    JPH::BodyInterface& bodies = m_impl->system->GetBodyInterface();
    // Instances usually share models, so each mesh shape is built once.
    std::unordered_map<size_t, JPH::ShapeRefC> shapes;

    for (const ModelInstance& instance : models.getInstances()) {
        if (!instance.visible)
            continue;
        if (glm::any(glm::lessThan(glm::abs(instance.scale), glm::vec3(1e-4f))))
            continue;
        const GPUModel* model = models.getModel(instance.modelIndex);
        if (!model || !model->isValid())
            continue;
        auto [it, inserted] = shapes.try_emplace(instance.modelIndex);
        if (inserted)
            it->second = createMeshShape(*model);
        if (!it->second)
            continue;

        JPH::ShapeRefC shape = it->second;
        if (instance.scale != glm::vec3(1.0f))
            shape = new JPH::ScaledShape(shape, toJolt(instance.scale));

        const glm::quat rotation = glm::quat(glm::radians(instance.rotation));
        JPH::BodyCreationSettings settings(shape, JPH::RVec3(toJolt(instance.position)),
            JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w).Normalized(),
            JPH::EMotionType::Static, Layers::NonMoving);
        JPH::Body* body = bodies.CreateBody(settings);
        if (!body) {
            std::cerr << "[Physics] Body limit reached; remaining instances have no collision\n";
            break;
        }
        m_impl->staticBodies.push_back(body->GetID());
    }

    if (!m_impl->staticBodies.empty()) {
        const int count = static_cast<int>(m_impl->staticBodies.size());
        JPH::BodyInterface::AddState state = bodies.AddBodiesPrepare(m_impl->staticBodies.data(), count);
        bodies.AddBodiesFinalize(m_impl->staticBodies.data(), count, state, JPH::EActivation::DontActivate);
    }
    m_impl->system->OptimizeBroadPhase();
}

void PhysicsWorld::spawnPlayer(const glm::vec3& feetPosition)
{
    JPH::Ref<JPH::CharacterVirtualSettings> settings = new JPH::CharacterVirtualSettings();
    // Offset the capsule so the character's origin is at its feet.
    settings->mShape = new JPH::RotatedTranslatedShape(JPH::Vec3(0.0f, kCapsuleHalfHeight + kCapsuleRadius, 0.0f),
        JPH::Quat::sIdentity(), new JPH::CapsuleShape(kCapsuleHalfHeight, kCapsuleRadius));
    settings->mMaxSlopeAngle = JPH::DegreesToRadians(kMaxSlopeDegrees);
    // Only contacts below the center of the bottom sphere count as standing on something.
    settings->mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -kCapsuleRadius);
    m_impl->character = new JPH::CharacterVirtual(settings, JPH::RVec3(toJolt(feetPosition)),
        JPH::Quat::sIdentity(), m_impl->system.get());
}

bool PhysicsWorld::hasPlayer() const
{
    return m_impl->character != nullptr;
}

void PhysicsWorld::updatePlayer(float dt, const glm::vec3& horizontalVelocity, bool jump)
{
    JPH::CharacterVirtual* character = m_impl->character.GetPtr();
    if (!character || dt <= 0.0f)
        return;
    dt = std::min(dt, kMaxStep);

    character->UpdateGroundVelocity();
    const JPH::Vec3 up = JPH::Vec3::sAxisY();
    const JPH::Vec3 vertical = up * up.Dot(character->GetLinearVelocity());
    const JPH::Vec3 groundVelocity = character->GetGroundVelocity();
    // Without this check a jump would be cancelled on the next frame while still touching the ground.
    const bool movingTowardsGround = vertical.GetY() - groundVelocity.GetY() < 0.1f;

    const JPH::Vec3 desired(horizontalVelocity.x, 0.0f, horizontalVelocity.z);
    const bool hasInput = !desired.IsNearZero();
    const bool grounded = character->GetGroundState() == JPH::CharacterVirtual::EGroundState::OnGround &&
        movingTowardsGround;

    // Horizontal speed is measured relative to what the player stands on, so moving platforms carry them.
    const JPH::Vec3 horizontalGround(groundVelocity.GetX(), 0.0f, groundVelocity.GetZ());
    JPH::Vec3 horizontal = character->GetLinearVelocity() - vertical - (grounded ? horizontalGround : JPH::Vec3::sZero());
    if (grounded) {
        const float rate = hasInput ? kGroundAcceleration : kGroundFriction;
        horizontal = moveTowards(horizontal, desired, rate * dt);
    } else if (hasInput) {
        // Air steering can redirect momentum but not build speed beyond the faster of momentum and input.
        const JPH::Vec3 steered = moveTowards(horizontal, desired, kAirAcceleration * dt);
        const float maxSpeed = std::max(horizontal.Length(), desired.Length());
        horizontal = steered.Length() > maxSpeed ? steered.Normalized() * maxSpeed : steered;
    }

    JPH::Vec3 velocity;
    if (grounded) {
        velocity = groundVelocity + horizontal;
        if (jump)
            velocity += up * kJumpSpeed;
    } else {
        velocity = vertical + horizontal;
    }
    velocity += gravity() * dt;
    character->SetLinearVelocity(velocity);

    JPH::CharacterVirtual::ExtendedUpdateSettings updateSettings;
    character->ExtendedUpdate(dt, gravity(), updateSettings,
        m_impl->system->GetDefaultBroadPhaseLayerFilter(Layers::Moving),
        m_impl->system->GetDefaultLayerFilter(Layers::Moving),
        {}, {}, *m_impl->tempAllocator);
}

glm::vec3 PhysicsWorld::playerPosition() const
{
    if (!m_impl->character)
        return glm::vec3(0.0f);
    return toGlm(JPH::Vec3(m_impl->character->GetPosition()));
}

bool PhysicsWorld::playerOnGround() const
{
    return m_impl->character &&
        m_impl->character->GetGroundState() == JPH::CharacterVirtual::EGroundState::OnGround;
}
