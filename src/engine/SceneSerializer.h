#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <glm/glm.hpp>

// Forward declarations - adjust these includes to match your project
#include "ModelManager.h" // For ModelManager, ModelInstance, GPUModel

struct SceneFileHeader {
    char magic[4] = { 'S', 'C', 'N', 'E' };
    uint32_t version = 2;
    uint32_t modelCount = 0;
    uint32_t instanceCount = 0;
};

struct SceneLayerEntry {
    uint32_t nameLength = 0;
    bool visible = true;
};

struct SceneHierarchyEntry {
    uint32_t nameLength = 0;
    bool visible = true;
};

struct SceneModelEntry {
    uint32_t pathLength = 0;
    uint32_t nameLength = 0;
    uint32_t layerCount = 0;
    uint32_t hierarchyCount = 0;
    uint32_t viewMode = 0;
    // followed by: char path[pathLength], char name[nameLength]
};

struct SceneInstanceEntry {
    uint32_t modelIndex = 0; // index into the model list in this file
    uint32_t nameLength = 0;
    float posX, posY, posZ;
    float rotX, rotY, rotZ;
    float scaleX, scaleY, scaleZ;
    bool visible = true;
    // followed by: char name[nameLength]
};

class SceneSerializer {
public:
    static bool Save(const std::string& filepath, ModelManager& modelManager) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[SCENE] Failed to open file for writing: " << filepath << "\n";
            return false;
        }

        const auto& models = modelManager.getModels();
        const auto& instances = modelManager.getInstances();

        // Build unique model path list (deduplicate)
        struct ModelEntry {
            std::string path;
            std::string name;
            std::vector<IfcTypeLayer> layers;
            std::vector<IfcTreeNode> hierarchy;
            uint32_t viewMode = 0;
        };
        std::vector<ModelEntry> uniqueModels;
        // Map from modelManager model index -> file model index
        std::vector<uint32_t> modelIndexMap(models.size(), 0);

        for (size_t i = 0; i < models.size(); ++i) {
            if (!models[i] || !models[i]->isValid()) continue;

            bool found = false;
            for (size_t j = 0; j < uniqueModels.size(); ++j) {
                if (uniqueModels[j].path == models[i]->sourcePath) {
                    modelIndexMap[i] = static_cast<uint32_t>(j);
                    found = true;
                    break;
                }
            }
            if (!found) {
                modelIndexMap[i] = static_cast<uint32_t>(uniqueModels.size());
                ModelEntry modelEntry;
                modelEntry.path = models[i]->sourcePath;
                modelEntry.name = models[i]->name;
                modelEntry.layers = models[i]->ifcInfo.layers;
                modelEntry.hierarchy = models[i]->ifcInfo.tree;
                modelEntry.viewMode = static_cast<uint32_t>(models[i]->ifcInfo.viewMode);
                uniqueModels.push_back(std::move(modelEntry));
            }
        }

        // Count valid instances
        uint32_t validInstanceCount = 0;
        for (const auto& inst : instances) {
            if (inst.modelIndex < models.size() && models[inst.modelIndex] && models[inst.modelIndex]->isValid()) {
                validInstanceCount++;
            }
        }

        // Write header
        SceneFileHeader header;
        header.modelCount = static_cast<uint32_t>(uniqueModels.size());
        header.instanceCount = validInstanceCount;
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // Write models
        for (const auto& model : uniqueModels) {
            SceneModelEntry entry;
            entry.pathLength = static_cast<uint32_t>(model.path.size());
            entry.nameLength = static_cast<uint32_t>(model.name.size());
            entry.layerCount = static_cast<uint32_t>(model.layers.size());
            entry.hierarchyCount = static_cast<uint32_t>(model.hierarchy.size());
            entry.viewMode = model.viewMode;
            file.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
            file.write(model.path.data(), entry.pathLength);
            file.write(model.name.data(), entry.nameLength);

            for (const auto& layer : model.layers) {
                SceneLayerEntry le;
                le.nameLength = static_cast<uint32_t>(layer.typeName.size());
                le.visible = layer.visible;
                file.write(reinterpret_cast<const char*>(&le), sizeof(le));
                file.write(layer.typeName.data(), le.nameLength);
            }

            for (const auto& node : model.hierarchy) {
                SceneHierarchyEntry he;
                he.nameLength = static_cast<uint32_t>(node.fullPath.size());
                he.visible = node.visible;
                file.write(reinterpret_cast<const char*>(&he), sizeof(he));
                file.write(node.fullPath.data(), he.nameLength);
            }
        }

        // Write instances
        for (const auto& inst : instances) {
            if (inst.modelIndex >= models.size() || !models[inst.modelIndex] || !models[inst.modelIndex]->isValid()) {
                continue;
            }

            SceneInstanceEntry entry;
            entry.modelIndex = modelIndexMap[inst.modelIndex];
            entry.nameLength = static_cast<uint32_t>(inst.name.size());
            entry.posX = inst.position.x;
            entry.posY = inst.position.y;
            entry.posZ = inst.position.z;
            entry.rotX = inst.rotation.x;
            entry.rotY = inst.rotation.y;
            entry.rotZ = inst.rotation.z;
            entry.scaleX = inst.scale.x;
            entry.scaleY = inst.scale.y;
            entry.scaleZ = inst.scale.z;
            entry.visible = inst.visible;

            file.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
            file.write(inst.name.data(), entry.nameLength);
        }

        file.close();
        std::cout << "[SCENE] Saved: " << uniqueModels.size() << " models, "
            << validInstanceCount << " instances to " << filepath << "\n";
        return true;
    }

    struct LoadedScene {
        struct LoadedLayer {
            std::string name;
            bool visible = true;
        };

        struct LoadedHierarchyNode {
            std::string fullPath;
            bool visible = true;
        };

        struct LoadedModel {
            std::string path;
            std::string name;
            std::vector<LoadedLayer> layers;
            std::vector<LoadedHierarchyNode> hierarchyNodes;
            uint32_t viewMode = 0;
        };
        struct LoadedInstance {
            uint32_t fileModelIndex; // index into loadedModels
            std::string name;
            glm::vec3 position;
            glm::vec3 rotation;
            glm::vec3 scale;
            bool visible;
        };

        std::vector<LoadedModel> models;
        std::vector<LoadedInstance> instances;
        bool valid = false;
    };

    static LoadedScene Load(const std::string& filepath) {
        LoadedScene scene;
        scene.valid = false;

        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[SCENE] Failed to open file for reading: " << filepath << "\n";
            return scene;
        }

        // Read header
        SceneFileHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (header.magic[0] != 'S' || header.magic[1] != 'C' ||
            header.magic[2] != 'N' || header.magic[3] != 'E') {
            std::cerr << "[SCENE] Invalid scene file magic\n";
            return scene;
        }

        if (header.version != 1 && header.version != 2) {
            std::cerr << "[SCENE] Unsupported scene version: " << header.version << "\n";
            return scene;
        }

        // Read models
        scene.models.resize(header.modelCount);
        for (uint32_t i = 0; i < header.modelCount; ++i) {
            SceneModelEntry entry;
            file.read(reinterpret_cast<char*>(&entry), sizeof(entry));

            scene.models[i].path.resize(entry.pathLength);
            file.read(scene.models[i].path.data(), entry.pathLength);

            scene.models[i].name.resize(entry.nameLength);
            file.read(scene.models[i].name.data(), entry.nameLength);

            scene.models[i].viewMode = entry.viewMode;

            scene.models[i].layers.resize(entry.layerCount);
            for (uint32_t l = 0; l < entry.layerCount; ++l) {
                SceneLayerEntry le{};
                file.read(reinterpret_cast<char*>(&le), sizeof(le));

                scene.models[i].layers[l].visible = le.visible;
                scene.models[i].layers[l].name.resize(le.nameLength);
                file.read(scene.models[i].layers[l].name.data(), le.nameLength);
            }
            scene.models[i].hierarchyNodes.resize(entry.hierarchyCount);
            for (uint32_t h = 0; h < entry.hierarchyCount; ++h) {
                SceneHierarchyEntry he{};
                file.read(reinterpret_cast<char*>(&he), sizeof(he));
                scene.models[i].hierarchyNodes[h].visible = he.visible;
                scene.models[i].hierarchyNodes[h].fullPath.resize(he.nameLength);
                file.read(scene.models[i].hierarchyNodes[h].fullPath.data(), he.nameLength);
            }

            
        }

        // Read instances
        scene.instances.resize(header.instanceCount);
        for (uint32_t i = 0; i < header.instanceCount; ++i) {
            SceneInstanceEntry entry;
            file.read(reinterpret_cast<char*>(&entry), sizeof(entry));

            scene.instances[i].fileModelIndex = entry.modelIndex;
            scene.instances[i].position = glm::vec3(entry.posX, entry.posY, entry.posZ);
            scene.instances[i].rotation = glm::vec3(entry.rotX, entry.rotY, entry.rotZ);
            scene.instances[i].scale = glm::vec3(entry.scaleX, entry.scaleY, entry.scaleZ);
            scene.instances[i].visible = entry.visible;

            scene.instances[i].name.resize(entry.nameLength);
            file.read(scene.instances[i].name.data(), entry.nameLength);
        }

        scene.valid = true;
        std::cout << "[SCENE] Loaded: " << scene.models.size() << " models, "
            << scene.instances.size() << " instances from " << filepath << "\n";
        return scene;
    }
};