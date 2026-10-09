#pragma once
#include <vector>
#include <glm/glm.hpp>

struct ModelInstance;

// A point or spot light: a scene object whose entity is kLightEntity (EntityTypes.h), configured by its
// entity parameters, colored by its color and aimed by its rotation.
struct SceneLight {
    glm::vec3 position{ 0.0f };
    glm::vec3 direction{ 0.0f, -1.0f, 0.0f }; // spot: where the cone points (the object's -Y axis)
    glm::vec3 color{ 1.0f };                  // linear
    float intensity = 15.0f;
    float range = 12.0f;                      // meters; nothing is lit beyond it
    bool spot = false;
    float coneDegrees = 35.0f;                // spot: from the axis to the edge of the cone
    float softness = 0.25f;                   // spot: part of the cone that fades out towards its edge
    bool shadows = false;
};

SceneLight lightFromInstance(const ModelInstance& instance);
// The visible light objects among `instances`, appended to `out`.
void collectSceneLights(const std::vector<ModelInstance>& instances, std::vector<SceneLight>& out);
