#pragma once

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <memory>
#include <vector>
#include <future>
#include <mutex>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include "ModelTypes.h"
#include "Buffers.h"

class TextureImage;

struct GPUModel {
    std::string name;
    std::string sourcePath;

    std::unique_ptr<VertexBuffer> vertexBuffer;
    std::unique_ptr<IndexBuffer> indexBuffer;

    std::vector<std::unique_ptr<TextureImage>> textures;
    std::vector<vk::DescriptorSet> textureDescriptorSets;

    std::vector<SubmeshInfo> submeshes;
    size_t vertexCount = 0;
    size_t indexCount = 0;

    glm::vec3 boundsCenter{ 0.0f };
    float boundsRadius{ 1.0f };

    bool isValid() const { return vertexBuffer != nullptr && indexBuffer != nullptr; }
};

struct ModelInstance {
    size_t modelIndex;
    glm::vec3 position{ 0.0f };
    glm::vec3 rotation{ 0.0f };
    glm::vec3 scale{ 1.0f };
    bool visible = true;
    std::string name;

    glm::mat4 getTransformMatrix() const {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        t = glm::rotate(t, glm::radians(rotation.x), glm::vec3(1, 0, 0));
        t = glm::rotate(t, glm::radians(rotation.y), glm::vec3(0, 1, 0));
        t = glm::rotate(t, glm::radians(rotation.z), glm::vec3(0, 0, 1));
        t = glm::scale(t, scale);
        return t;
    }
};

enum class LoadingState {
    Idle,
    LoadingCPU,
    UploadingGPU,
    Complete,
    Failed
};

struct LoadingTask {
    std::string path;
    std::string name;
    LoadingState state = LoadingState::Idle;
    std::string errorMessage;
    float progress = 0.0f;
    std::future<Mesh> meshFuture;
    Mesh loadedMesh;
};

class ModelManager {
public:
    ModelManager(VmaAllocator allocator, vk::Device device, vk::CommandPool cmdPool,
        vk::Queue queue, vk::DescriptorPool descriptorPool,
        vk::DescriptorSetLayout textureSetLayout, vk::Sampler textureSampler);
    ~ModelManager();

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    void loadModelAsync(const std::string& path, const std::string& name = "");
    size_t loadModelSync(const std::string& path, const std::string& name = "");
    void unloadModel(size_t modelIndex);

    size_t createInstance(size_t modelIndex, const glm::vec3& position = glm::vec3(0.0f));
    void removeInstance(size_t instanceIndex);
    ModelInstance* getInstance(size_t instanceIndex);

    const std::vector<std::unique_ptr<GPUModel>>& getModels() const { return m_models; }
    std::vector<ModelInstance>& getInstances() { return m_instances; }
    const std::vector<ModelInstance>& getInstances() const { return m_instances; }
    GPUModel* getModel(size_t index);

    void update();

    bool hasActiveTasks() const { return !m_loadingTasks.empty(); }
    const std::vector<LoadingTask>& getLoadingTasks() const { return m_loadingTasks; }

    vk::DescriptorSet getDefaultBaseColorSet() const { return m_defaultBaseColorSet; }
    vk::DescriptorSet getDefaultNormalSet() const { return m_defaultNormalSet; }
    vk::DescriptorSet getDefaultMRSet() const { return m_defaultMRSet; }

private:
    size_t uploadModelToGPU(Mesh& mesh, const std::string& name, const std::string& path);
    void createDefaultTextures();
    vk::DescriptorSet allocateTextureDescriptorSet(vk::ImageView view);

    VmaAllocator m_allocator;
    vk::Device m_device;
    vk::CommandPool m_cmdPool;
    vk::Queue m_queue;
    vk::DescriptorPool m_descriptorPool;
    vk::DescriptorSetLayout m_textureSetLayout;
    vk::Sampler m_textureSampler;

    std::vector<std::unique_ptr<GPUModel>> m_models;
    std::vector<ModelInstance> m_instances;
    std::vector<LoadingTask> m_loadingTasks;

    std::unique_ptr<TextureImage> m_defaultBaseColor;
    std::unique_ptr<TextureImage> m_defaultNormal;
    std::unique_ptr<TextureImage> m_defaultMR;
    vk::DescriptorSet m_defaultBaseColorSet;
    vk::DescriptorSet m_defaultNormalSet;
    vk::DescriptorSet m_defaultMRSet;

    std::mutex m_mutex;
    size_t m_nextInstanceId = 0;
};