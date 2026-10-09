#include "Lights.h"
#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include "EntityTypes.h"
#include "ModelManager.h"

SceneLight lightFromInstance(const ModelInstance& instance)
{
    const std::string& params = instance.entityParams;
    SceneLight light;
    light.position = instance.position;
    light.direction = glm::normalize(glm::quat(glm::radians(instance.rotation)) * glm::vec3(0.0f, -1.0f, 0.0f));
    // The color picker works in sRGB.
    light.color = glm::pow(glm::clamp(instance.color, glm::vec3(0.0f), glm::vec3(1.0f)), glm::vec3(2.2f));
    light.intensity = std::max(entityParam(params, "intensity", light.intensity), 0.0f);
    light.range = std::clamp(entityParam(params, "range", light.range), 0.1f, 1000.0f);
    light.spot = entityParam(params, "spot", 0.0f) != 0.0f;
    light.coneDegrees = std::clamp(entityParam(params, "cone", light.coneDegrees), 1.0f, 89.0f);
    light.softness = std::clamp(entityParam(params, "softness", light.softness), 0.0f, 1.0f);
    light.shadows = entityParam(params, "shadows", 0.0f) != 0.0f;
    return light;
}

void collectSceneLights(const std::vector<ModelInstance>& instances, std::vector<SceneLight>& out)
{
    for (const ModelInstance& instance : instances) {
        if (!instance.visible || instance.entity != kLightEntity)
            continue;
        const SceneLight light = lightFromInstance(instance);
        if (light.intensity > 0.0f && glm::dot(light.color, glm::vec3(1.0f)) > 0.0f)
            out.push_back(light);
    }
}
