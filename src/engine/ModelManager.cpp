#include "ModelManager.h"
#include <iostream>
#include <algorithm>
#include <filesystem>
#include "stb_image.h"
#include "Shadow.h"
#include "IfcLayerInfo.h"

// TextureImage class - move from main.cpp or keep inline here
class TextureImage {
public:
    TextureImage(VmaAllocator allocator, vk::Device device, vk::CommandPool cmdPool,
        vk::Queue queue, const TextureData& data, vk::Format format)
        : m_allocator(allocator), m_device(device) {

        vk::DeviceSize imageSize = data.width * data.height * 4;

        VkBufferCreateInfo stagingInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        stagingInfo.size = imageSize;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo stagingAllocInfo = {};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

        VkBuffer stagingBuffer;
        VmaAllocation stagingAlloc;
        vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAlloc, nullptr);

        void* mapped;
        vmaMapMemory(allocator, stagingAlloc, &mapped);
        memcpy(mapped, data.pixels, static_cast<size_t>(imageSize));
        vmaUnmapMemory(allocator, stagingAlloc);

        VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = data.width;
        imageInfo.extent.height = data.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = static_cast<VkFormat>(format);
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo imageAllocInfo = {};
        imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkImage rawImage;
        vmaCreateImage(allocator, &imageInfo, &imageAllocInfo, &rawImage, &m_allocation, nullptr);
        m_image = vk::Image(rawImage);

        vk::CommandBufferAllocateInfo allocInfo(cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::UniqueCommandBuffer cmd = std::move(device.allocateCommandBuffersUnique(allocInfo).value[0]);

        cmd->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

        transitionLayout(cmd.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy region{};
        region.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
        region.imageExtent = vk::Extent3D{ (uint32_t)data.width, (uint32_t)data.height, 1 };
        cmd->copyBufferToImage(vk::Buffer(stagingBuffer), m_image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

        transitionLayout(cmd.get(), vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        cmd->end();
        vk::SubmitInfo submitInfo{};
        submitInfo.setCommandBufferCount(1);
        submitInfo.setPCommandBuffers(&cmd.get());
        queue.submit(submitInfo, nullptr);
        queue.waitIdle();

        vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);

        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = m_image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        m_view = device.createImageView(viewInfo).value;
    }

    ~TextureImage() {
        if (m_view) m_device.destroyImageView(m_view);
        if (m_image) vmaDestroyImage(m_allocator, VkImage(m_image), m_allocation);
    }

    TextureImage(const TextureImage&) = delete;
    TextureImage& operator=(const TextureImage&) = delete;

    TextureImage(TextureImage&& other) noexcept
        : m_allocator(other.m_allocator), m_device(other.m_device), m_image(other.m_image),
        m_allocation(other.m_allocation), m_view(other.m_view) {
        other.m_image = nullptr; other.m_view = nullptr; other.m_allocation = nullptr;
    }

    vk::ImageView getView() const { return m_view; }

private:
    void transitionLayout(vk::CommandBuffer cmd, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_image;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eNone;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        cmd.pipelineBarrier(sourceStage, destinationStage, {}, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    VmaAllocator m_allocator;
    vk::Device m_device;
    vk::Image m_image;
    VmaAllocation m_allocation = nullptr;
    vk::ImageView m_view;
};

// External function declarations - these remain in main.cpp
extern Mesh loadModelSmart(const std::string& path, IfcInfo* outIfcInfo = nullptr);
// extern SceneBounds calculateSceneBounds(const std::vector<Vertex>& vertices, const std::vector<InstanceData>& instances);

void TextureData::free() {
    if (pixels) {
        if (fromCache) {
            ::free(pixels);
        }
        else {
            stbi_image_free(pixels);
        }
        pixels = nullptr;
    }
}

ModelManager::ModelManager(VmaAllocator allocator, vk::Device device, vk::CommandPool cmdPool,
    vk::Queue queue, vk::DescriptorPool descriptorPool,
    vk::DescriptorSetLayout textureSetLayout, vk::Sampler textureSampler)
    : m_allocator(allocator)
    , m_device(device)
    , m_cmdPool(cmdPool)
    , m_queue(queue)
    , m_descriptorPool(descriptorPool)
    , m_textureSetLayout(textureSetLayout)
    , m_textureSampler(textureSampler)
{
    createDefaultTextures();
}

ModelManager::~ModelManager() {
    m_device.waitIdle();
    m_models.clear();
    m_instances.clear();
}

void ModelManager::createDefaultTextures() {
    // White default texture
    TextureData whiteData;
    whiteData.width = 1; whiteData.height = 1; whiteData.channels = 4;
    static unsigned char whitePixels[] = { 255, 255, 255, 255 };
    whiteData.pixels = whitePixels;
    m_defaultBaseColor = std::make_unique<TextureImage>(
        m_allocator, m_device, m_cmdPool, m_queue, whiteData, vk::Format::eR8G8B8A8Srgb);
    m_defaultBaseColorSet = allocateTextureDescriptorSet(m_defaultBaseColor->getView());

    // Normal default
    TextureData normalData;
    normalData.width = 1; normalData.height = 1; normalData.channels = 4;
    static unsigned char normalPixels[] = { 128, 128, 255, 255 };
    normalData.pixels = normalPixels;
    m_defaultNormal = std::make_unique<TextureImage>(
        m_allocator, m_device, m_cmdPool, m_queue, normalData, vk::Format::eR8G8B8A8Unorm);
    m_defaultNormalSet = allocateTextureDescriptorSet(m_defaultNormal->getView());

    // Metallic-Roughness default
    TextureData mrData;
    mrData.width = 1; mrData.height = 1; mrData.channels = 4;
    static unsigned char mrPixels[] = { 0, 128, 0, 255 };
    mrData.pixels = mrPixels;
    m_defaultMR = std::make_unique<TextureImage>(
        m_allocator, m_device, m_cmdPool, m_queue, mrData, vk::Format::eR8G8B8A8Unorm);
    m_defaultMRSet = allocateTextureDescriptorSet(m_defaultMR->getView());
}

vk::DescriptorSet ModelManager::allocateTextureDescriptorSet(vk::ImageView view) {
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_textureSetLayout;

    vk::DescriptorSet set = m_device.allocateDescriptorSets(allocInfo).value[0];

    vk::DescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = view;
    imageInfo.sampler = m_textureSampler;

    vk::WriteDescriptorSet write{};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;

    m_device.updateDescriptorSets(1, &write, 0, nullptr);

    return set;
}

void ModelManager::loadModelAsync(const std::string& path, const std::string& name) {
    std::lock_guard<std::mutex> lock(m_mutex);

    LoadingTask task;
    task.path = path;
    task.name = name.empty() ? std::filesystem::path(path).stem().string() : name;
    task.state = LoadingState::LoadingCPU;

    task.meshFuture = std::async(std::launch::async, [path]() {
        return loadModelSmart(path);
        });

    m_loadingTasks.push_back(std::move(task));
}

size_t ModelManager::loadModelSync(const std::string& path, const std::string& name) {
    std::string modelName = name.empty() ? std::filesystem::path(path).stem().string() : name;

    std::cout << "[ModelManager] Loading model synchronously: " << path << "\n";

    Mesh mesh = loadModelSmart(path);
    return uploadModelToGPU(mesh, modelName, path);
}

size_t ModelManager::uploadModelToGPU(Mesh& mesh, const std::string& name, const std::string& path) {
    auto gpuModel = std::make_unique<GPUModel>();
    gpuModel->name = name;
    gpuModel->sourcePath = path;
    gpuModel->submeshes = mesh.submeshes;
    gpuModel->vertexCount = mesh.vertices.size();
    gpuModel->indexCount = mesh.indices.size();

    // Create buffers
    gpuModel->vertexBuffer = std::make_unique<VertexBuffer>(m_allocator, m_device, mesh.vertices);
    gpuModel->indexBuffer = std::make_unique<IndexBuffer>(m_allocator, m_device, mesh.indices);

    // Upload textures
    for (auto& texData : mesh.textureData) {
        vk::Format fmt = texData.isLinear ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;
        auto tex = std::make_unique<TextureImage>(m_allocator, m_device, m_cmdPool, m_queue, texData, fmt);

        vk::DescriptorSet set = allocateTextureDescriptorSet(tex->getView());
        gpuModel->textureDescriptorSets.push_back(set);
        gpuModel->textures.push_back(std::move(tex));

        texData.free();
    }

    if (!mesh.vertices.empty()) {
        glm::vec3 bmin(FLT_MAX);
        glm::vec3 bmax(-FLT_MAX);

        for (const auto& v : mesh.vertices) {
            bmin = glm::min(bmin, v.position);
            bmax = glm::max(bmax, v.position);
        }

        gpuModel->boundsMin = bmin;
        gpuModel->boundsMax = bmax;
        gpuModel->boundsCenter = (bmin + bmax) * 0.5f;
        gpuModel->boundsRadius = glm::length(bmax - bmin) * 0.5f;
    }

    size_t index = m_models.size();
    m_models.push_back(std::move(gpuModel));

    std::cout << "[ModelManager] Model '" << name << "' loaded: "
        << mesh.vertices.size() << " verts, "
        << mesh.indices.size() << " indices, "
        << m_models.back()->textures.size() << " textures\n";

    return index;
}

void ModelManager::unloadModel(size_t modelIndex) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (modelIndex >= m_models.size()) return;

    // Remove instances using this model
    m_instances.erase(
        std::remove_if(m_instances.begin(), m_instances.end(),
            [modelIndex](const ModelInstance& inst) { return inst.modelIndex == modelIndex; }),
        m_instances.end());

    // Update indices
    for (auto& inst : m_instances) {
        if (inst.modelIndex > modelIndex) inst.modelIndex--;
    }

    m_device.waitIdle();
    m_models.erase(m_models.begin() + modelIndex);
}

size_t ModelManager::createInstance(size_t modelIndex, const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (modelIndex >= m_models.size()) {
        throw std::runtime_error("Invalid model index");
    }

    ModelInstance inst;
    inst.modelIndex = modelIndex;
    inst.position = position;
    inst.rotation = rotation;
    inst.scale = scale;
    inst.name = m_models[modelIndex]->name + "_" + std::to_string(m_nextInstanceId++);

    m_instances.push_back(inst);
    return m_instances.size() - 1;
}

void ModelManager::removeInstance(size_t instanceIndex) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (instanceIndex < m_instances.size()) {
        m_instances.erase(m_instances.begin() + instanceIndex);
    }
}

ModelInstance* ModelManager::getInstance(size_t instanceIndex) {
    if (instanceIndex < m_instances.size()) {
        return &m_instances[instanceIndex];
    }
    return nullptr;
}

GPUModel* ModelManager::getModel(size_t index) {
    if (index < m_models.size()) {
        return m_models[index].get();
    }
    return nullptr;
}

void ModelManager::update() {
    std::lock_guard<std::mutex> lock(m_mutex);

    for (auto it = m_loadingTasks.begin(); it != m_loadingTasks.end(); ) {
        LoadingTask& task = *it;

        if (task.state == LoadingState::LoadingCPU) {
            if (task.meshFuture.valid() &&
                task.meshFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    task.loadedMesh = task.meshFuture.get();
                    task.state = LoadingState::UploadingGPU;
                }
                catch (const std::exception& e) {
                    task.state = LoadingState::Failed;
                    task.errorMessage = e.what();
                    std::cerr << "[ModelManager] Failed: " << e.what() << "\n";
                }
            }
        }

        if (task.state == LoadingState::UploadingGPU) {
            try {
                m_mutex.unlock();
                uploadModelToGPU(task.loadedMesh, task.name, task.path);
                m_mutex.lock();
                task.state = LoadingState::Complete;
            }
            catch (const std::exception& e) {
                task.state = LoadingState::Failed;
                task.errorMessage = e.what();
            }
        }

        if (task.state == LoadingState::Complete || task.state == LoadingState::Failed) {
            it = m_loadingTasks.erase(it);
        }
        else {
            ++it;
        }
    }
}
