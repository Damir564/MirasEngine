#pragma once

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <memory>
#include <vector>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include "ModelTypes.h"
#include "Buffers.h"
#include "PolyMesh.h"

class TextureImage;

// Range [begin, end) of positions in GPUModel::drawOrder.
struct DrawGroup {
    uint32_t begin = 0;
    uint32_t end = 0;
};

struct GPUModel {
    std::string name;
    std::string sourcePath;

    std::unique_ptr<DeviceBuffer> vertexBuffer;
    std::unique_ptr<DeviceBuffer> indexBuffer;

    std::vector<std::unique_ptr<TextureImage>> textures;
    std::vector<vk::DescriptorSet> textureDescriptorSets;

    // indexOffset/vertexOffset refer to the GPU buffers: vertexOffset is 0 (folded into the indices)
    // and the index buffer stores the submeshes in drawOrder.
    std::vector<SubmeshInfo> submeshes;
    // Submesh indices sorted by alpha mode, then material, then spatially (Morton order). Neighbours
    // with the same material are adjacent in the index buffer, so they can be drawn with one command.
    std::vector<uint32_t> drawOrder;
    // Runs of drawOrder sharing an identical material.
    std::vector<DrawGroup> materialGroups;
    // What the shadow pass draws: all opaque submeshes as one group (material does not matter there),
    // then each alpha-masked material. Blended submeshes cast no shadow.
    std::vector<DrawGroup> shadowGroups;
    size_t vertexCount = 0;
    size_t indexCount = 0;
    // CPU copy of the geometry for exact ray picking; indices are relative to each submesh's vertexOffset.
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;

    glm::vec3 boundsCenter{ 0.0f };
    float boundsRadius{ 1.0f };
    glm::vec3 boundsMin{ 0.0f };
    glm::vec3 boundsMax{ 0.0f };

    // Model-space triangle soup (3 positions per triangle) kept on the CPU for physics collision.
    std::vector<glm::vec3> collisionTriangles;

    // Editable source geometry of level models (null for models loaded from files).
    std::unique_ptr<PolyMesh> polyMesh;
    // Where each of textures came from, so a rebuild can keep them when only geometry or UVs changed.
    std::vector<std::string> texturePaths;

    bool isValid() const { return vertexBuffer != nullptr && indexBuffer != nullptr; }
};

struct ModelInstance {
    size_t modelIndex = 0;
    std::string name;
    glm::vec3 position{ 0.0f };
    glm::vec3 rotation{ 0.0f };
    glm::vec3 scale{ 1.0f };
    bool visible = true;
    // Multiplied into every submesh's base color.
    glm::vec3 color{ 1.0f };

    glm::mat4 getTransformMatrix() const {
        glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 R = glm::mat4_cast(
            glm::quat(glm::radians(rotation))
        );
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        return T * R * S;
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
    // Uploads a level mesh as a new model with a fresh "level:#<n>" source path and keeps the mesh on it.
    // Throws like loadModelSync() when the upload fails.
    size_t addPolyMesh(PolyMesh mesh, const std::string& name);
    // Re-uploads a level model after its polyMesh changed (waits for the GPU to go idle); its index,
    // name, path and instances are kept.
    bool rebuildPolyMesh(size_t modelIndex);

    size_t createInstance(size_t modelIndex, const glm::vec3& position = glm::vec3(0.0f), const glm::vec3& rotation = glm::vec3(0.0f), const glm::vec3& scale = glm::vec3(1.0f));
    void removeInstance(size_t instanceIndex);
    void reserveInstances(size_t additional) { m_instances.reserve(m_instances.size() + additional); }
    ModelInstance* getInstance(size_t instanceIndex);

    const std::vector<std::unique_ptr<GPUModel>>& getModels() const { return m_models; }
    std::vector<ModelInstance>& getInstances() { return m_instances; }
    const std::vector<ModelInstance>& getInstances() const { return m_instances; }
    GPUModel* getModel(size_t index);
    // Index of a loaded model with this source path.
    std::optional<size_t> findModelByPath(const std::string& path) const;

    void update();

    bool hasActiveTasks() const { return !m_loadingTasks.empty(); }
    const std::vector<LoadingTask>& getLoadingTasks() const { return m_loadingTasks; }

    vk::DescriptorSet getDefaultBaseColorSet() const { return m_defaultBaseColorSet; }
    vk::DescriptorSet getDefaultNormalSet() const { return m_defaultNormalSet; }
    vk::DescriptorSet getDefaultMRSet() const { return m_defaultMRSet; }

private:
    size_t uploadModelToGPU(Mesh& mesh, const std::string& name, const std::string& path);
    static void buildDrawOrder(GPUModel& model);
    void createDefaultTextures();
    vk::DescriptorSet allocateTextureDescriptorSet(vk::ImageView view);
    // Returns a model's sets to the shared pool; the GPU must be done with them.
    void freeTextureDescriptorSets(GPUModel& model);

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
    uint32_t m_nextLevelModelId = 1;

    bool m_canGenerateMipsSrgb = false;
    bool m_canGenerateMipsUnorm = false;
};