#include <iostream>
#include <vulkan/vulkan.hpp>
#include <volk.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#define VKB_DISABLE_DEBUG_BREAK
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/quaternion.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "stb_image.h"
//#define FASTGLTF_USE_CUSTOM_SMALLVECTOR 0
//#define FASTGLTF_ENABLE_GLM_EXT 1 
#define FASTGLTF_USE_STD_MODULE 0
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <future>

struct Camera {
	glm::vec3 position{ 0.0f, 0.0f, 0.0f };
	float yaw = -90.0f; // look forward
	float pitch = 0.0f;
	float speed = 10.0f; 
	float sensitivity = 0.1f;
};

glm::mat4 getView(const Camera& cam) {
	glm::vec3 front{
		cos(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch)),
		sin(glm::radians(cam.pitch)),
		sin(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch))
	};

	return glm::lookAt(
		cam.position,
		cam.position + glm::normalize(front),
		glm::vec3(0, 1, 0)
	);
}

glm::mat4 getProjection(float width, float height) {
	glm::mat4 proj = glm::perspective(
		glm::radians(60.0f),
		width / height,
		0.1f,
		10000.0f
	);
	proj[1][1] *= -1; // Vulkan clip space fix
	return proj;
}

struct MeshPushConstants {
	// camera
	glm::mat4 view;
	glm::mat4 proj;
	// material
	glm::vec4 baseColor{ 1.0f, 1.0f, 1.0f, 1.0f };
	float metallic{ 0.0f };
	float roughness{ 0.5f };
	float time;
};

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texCoord;
	glm::vec3 tangent;

	static vk::VertexInputBindingDescription2EXT getBindingDescription(uint32_t binding = 0) {
		vk::VertexInputBindingDescription2EXT desc{};
		desc.binding = binding;
		desc.stride = sizeof(Vertex);
		desc.inputRate = vk::VertexInputRate::eVertex;
		desc.divisor = 1;
		return desc;
	}

	static std::array<vk::VertexInputAttributeDescription2EXT, 5> getAttributeDescriptions(uint32_t locationOffset = 0) {
		std::array<vk::VertexInputAttributeDescription2EXT, 5> attributes{};

		// position
		attributes[0].location = locationOffset + 0;
		attributes[0].binding = 0;
		attributes[0].format = vk::Format::eR32G32B32Sfloat;
		attributes[0].offset = offsetof(Vertex, position);

		// normal
		attributes[1].location = locationOffset + 1;
		attributes[1].binding = 0;
		attributes[1].format = vk::Format::eR32G32B32Sfloat;
		attributes[1].offset = offsetof(Vertex, normal);

		// texCoord
		attributes[2].location = locationOffset + 2;
		attributes[2].binding = 0;
		attributes[2].format = vk::Format::eR32G32Sfloat;
		attributes[2].offset = offsetof(Vertex, texCoord);

		// Tangent (Loc 3)
		attributes[3].location = locationOffset + 3;
		attributes[3].binding = 0;
		attributes[3].format = vk::Format::eR32G32B32Sfloat;
		attributes[3].offset = offsetof(Vertex, tangent);

		// Instance buffer attribute
		attributes[4].location = locationOffset + 4;             // matches shader
		attributes[4].binding = 1;              // instance buffer binding
		attributes[4].format = vk::Format::eR32G32B32Sfloat; // vec3
		attributes[4].offset = 0;               // offset inside InstanceData struct

		return attributes;
	}
};

class VertexBuffer {
public:
	VertexBuffer(VmaAllocator allocator,
		vk::Device device,
		const std::vector<Vertex>& vertices)
		: m_allocator(allocator), m_device(device), m_vertexCount(static_cast<uint32_t>(vertices.size()))
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(Vertex) * vertices.size();
		bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;  // Changed to CPU_TO_GPU
		allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT; // auto-mapped

		VkBuffer rawBuffer;
		if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &rawBuffer, &m_allocation, nullptr) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create vertex buffer");
		}

		m_buffer = vk::Buffer(rawBuffer);

		// Copy vertex data
		void* mapped;
		vmaMapMemory(m_allocator, m_allocation, &mapped);
		memcpy(mapped, vertices.data(), sizeof(Vertex) * vertices.size());
		vmaUnmapMemory(m_allocator, m_allocation);
	}

	~VertexBuffer() {
		cleanup();
		//if (m_buffer && m_allocation) {
		//	vmaDestroyBuffer(m_allocator, VkBuffer(m_buffer), m_allocation);
		//}
	}

	// Disable copy (unique ownership)
	VertexBuffer(const VertexBuffer&) = delete;
	VertexBuffer& operator=(const VertexBuffer&) = delete;

	// Enable move
	VertexBuffer(VertexBuffer&& other) noexcept
		: m_allocator(other.m_allocator),
		m_device(other.m_device),
		m_buffer(other.m_buffer),
		m_allocation(other.m_allocation),
		m_vertexCount(other.m_vertexCount)
	{
		other.m_buffer = nullptr;
		other.m_allocation = nullptr;
		other.m_vertexCount = 0;
	}

	VertexBuffer& operator=(VertexBuffer&& other) noexcept {
		if (this != &other) {
			cleanup();
			m_allocator = other.m_allocator;
			m_device = other.m_device;
			m_buffer = other.m_buffer;
			m_allocation = other.m_allocation;
			m_vertexCount = other.m_vertexCount;

			other.m_buffer = nullptr;
			other.m_allocation = nullptr;
			other.m_vertexCount = 0;
		}
		return *this;
	}

	vk::Buffer& getBuffer() { return m_buffer; }
	uint32_t getVertexCount() const { return m_vertexCount; }

private:
	void cleanup() {
		if (m_buffer && m_allocation) {
			vmaDestroyBuffer(m_allocator, VkBuffer(m_buffer), m_allocation);
			m_buffer = nullptr;
			m_allocation = nullptr;
		}
	}

	VmaAllocator m_allocator;
	vk::Device m_device;
	vk::Buffer m_buffer;
	VmaAllocation m_allocation;
	uint32_t m_vertexCount;
};

class IndexBuffer {
public:
	IndexBuffer(VmaAllocator allocator,
		vk::Device device,
		const std::vector<uint32_t>& indices)
		: m_allocator(allocator), m_device(device),
		m_indexCount(static_cast<uint32_t>(indices.size()))
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(uint32_t) * indices.size();
		bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
		allocInfo.requiredFlags =
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer rawBuffer;
		if (vmaCreateBuffer(
			m_allocator,
			&bufferInfo,
			&allocInfo,
			&rawBuffer,
			&m_allocation,
			nullptr) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create index buffer");
		}

		m_buffer = vk::Buffer(rawBuffer);

		void* mapped;
		vmaMapMemory(m_allocator, m_allocation, &mapped);
		memcpy(mapped, indices.data(), bufferInfo.size);
		vmaUnmapMemory(m_allocator, m_allocation);
	}

	~IndexBuffer() {
		if (m_buffer && m_allocation) {
			vmaDestroyBuffer(
				m_allocator,
				VkBuffer(m_buffer),
				m_allocation);
		}
	}

	vk::Buffer getBuffer() const { return m_buffer; }
	uint32_t getIndexCount() const { return m_indexCount; }

private:
	VmaAllocator m_allocator;
	vk::Device   m_device;
	vk::Buffer   m_buffer;
	VmaAllocation m_allocation;
	uint32_t     m_indexCount;
};

struct InstanceData {
	glm::vec3 offset; // Offset for this instance
};

// Example: 4 instances, spread out
std::vector<InstanceData> instances = {
	{{0.0f, 0.0f, 0.0f}},
	//{{ 0.5f, -0.4f}},
	//{{-0.5f, 0.4f}},
	//{{ 0.5f, 0.4f}}
};

class InstanceBuffer {
public:
	InstanceBuffer(VmaAllocator allocator, vk::Device device, const std::vector<InstanceData>& data)
		: m_allocator(allocator), m_device(device), m_instanceCount(static_cast<uint32_t>(data.size()))
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(InstanceData) * data.size();
		bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
		allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer rawBuffer;
		if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &rawBuffer, &m_allocation, nullptr) != VK_SUCCESS)
			throw std::runtime_error("Failed to create instance buffer");

		m_buffer = vk::Buffer(rawBuffer);

		void* mapped;
		vmaMapMemory(m_allocator, m_allocation, &mapped);
		memcpy(mapped, data.data(), sizeof(InstanceData) * data.size());
		vmaUnmapMemory(m_allocator, m_allocation);
	}

	~InstanceBuffer() {
		if (m_buffer && m_allocation)
			vmaDestroyBuffer(m_allocator, VkBuffer(m_buffer), m_allocation);
	}

	vk::Buffer getBuffer() const { return m_buffer; }
	uint32_t getInstanceCount() const { return m_instanceCount; }

private:
	VmaAllocator m_allocator;
	vk::Device m_device;
	vk::Buffer m_buffer;
	VmaAllocation m_allocation;
	uint32_t m_instanceCount;
};

// Holds raw CPU pixel data
struct TextureData {
	int width, height, channels;
	unsigned char* pixels = nullptr;
	std::string path;

	const unsigned char* encodedData = nullptr;
	size_t encodedSize = 0;

	// Helper to free CPU memory
	void free() {
		if (pixels) {
			stbi_image_free(pixels);
			pixels = nullptr;
		}
	}

	bool isLinear = false;
};

void decodeTextureParallel(TextureData& tex) {
	if (tex.encodedData != nullptr && tex.encodedSize > 0) {
		// Load from Memory (Embedded)
		tex.pixels = stbi_load_from_memory(
			tex.encodedData,
			static_cast<int>(tex.encodedSize),
			&tex.width, &tex.height, &tex.channels, 4);
	}
	else if (!tex.path.empty()) {
		// Load from File
		tex.pixels = stbi_load(
			tex.path.c_str(),
			&tex.width, &tex.height, &tex.channels, 4);
	}

	if (!tex.pixels) {
		std::cerr << "Texture failed to load: " << (tex.path.empty() ? "Embedded" : tex.path) << "\n";
		// Create a 1x1 magenta fallback so the app doesn't crash
		tex.width = 1; tex.height = 1; tex.channels = 4;
		tex.pixels = (unsigned char*)malloc(4);
		tex.pixels[0] = 255; tex.pixels[1] = 0; tex.pixels[2] = 255; tex.pixels[3] = 255;
	}
}

TextureData prepareTextureInfo(const fastgltf::Asset& asset, const fastgltf::Image& image, const std::string& modelPath) {
	TextureData texData{};

	std::visit(fastgltf::visitor{
		[&](const fastgltf::sources::URI& uri) {
			// Store Path only
			std::string pathString = std::string(uri.uri.path());
			texData.path = (std::filesystem::path(modelPath).parent_path() / pathString).string();
		},
		[&](const fastgltf::sources::Array& array) {
			// Store Pointer only
			texData.encodedData = reinterpret_cast<const unsigned char*>(array.bytes.data());
			texData.encodedSize = array.bytes.size();
		},
		[&](const fastgltf::sources::BufferView& view) {
			// Store Pointer only
			auto& bufferView = asset.bufferViews[view.bufferViewIndex];
			auto& buffer = asset.buffers[bufferView.bufferIndex];

			std::visit(fastgltf::visitor{
				[&](const fastgltf::sources::Array& bufferArray) {
					texData.encodedData = reinterpret_cast<const unsigned char*>(bufferArray.bytes.data() + bufferView.byteOffset);
					texData.encodedSize = bufferView.byteLength;
				},
				[](auto&) {}
			}, buffer.data);
		},
		[](auto&) {}
		}, image.data);

	return texData;
}

class TextureImage {
public:
	TextureImage(VmaAllocator allocator, vk::Device device, vk::CommandPool cmdPool, vk::Queue queue, const TextureData& data, vk::Format format)
		: m_allocator(allocator), m_device(device) {

		vk::DeviceSize imageSize = data.width * data.height * 4;

		// 1. Staging Buffer
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

		// 2. Create Image (GPU)
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

		// 3. Transition & Copy (Immediate Submit)
		vk::CommandBufferAllocateInfo allocInfo(cmdPool, vk::CommandBufferLevel::ePrimary, 1);
		vk::UniqueCommandBuffer cmd = std::move(device.allocateCommandBuffersUnique(allocInfo).value[0]);

		cmd->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

		// Transition Undefined -> TransferDst
		transitionLayout(cmd.get(), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

		// Copy Buffer -> Image
		vk::BufferImageCopy region{};
		region.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
		region.imageExtent = vk::Extent3D{ (uint32_t)data.width, (uint32_t)data.height, 1 };
		cmd->copyBufferToImage(vk::Buffer(stagingBuffer), m_image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

		// Transition TransferDst -> ShaderReadOnly
		transitionLayout(cmd.get(), vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

		cmd->end();
		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBufferCount(1);
		submitInfo.setPCommandBuffers(&cmd.get());
		queue.submit(submitInfo, nullptr);
		queue.waitIdle();

		vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);

		// 4. Create View
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

	// Disable copy
	TextureImage(const TextureImage&) = delete;
	TextureImage& operator=(const TextureImage&) = delete;

	// Enable move
	TextureImage(TextureImage&& other) noexcept
		: m_allocator(other.m_allocator), m_device(other.m_device), m_image(other.m_image),
		m_allocation(other.m_allocation), m_view(other.m_view) {
		other.m_image = nullptr; other.m_view = nullptr;
	}

	vk::ImageView getView() { return m_view; }

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
	VmaAllocation m_allocation;
	vk::ImageView m_view;
};

std::vector<uint32_t> loadSpirv(const std::filesystem::path& path)
{
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file)
		throw std::runtime_error("Failed to open SPIR-V file");

	size_t size = file.tellg();
	if (size % 4 != 0)
		throw std::runtime_error("Invalid SPIR-V size");

	std::vector<uint32_t> code(size / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);
	return code;
}



struct Material {
	glm::vec4 baseColorFactor{ 1.0f };
	float metallicFactor{ 1.0f };
	float roughnessFactor{ 1.0f };
	int baseColorTextureIndex = -1;
	int normalTextureIndex = -1;
	int metallicRoughnessTextureIndex = -1;
};

struct SubmeshInfo {
	uint32_t indexOffset;   // starting index in the global index buffer
	uint32_t indexCount;    // number of indices
	uint32_t vertexOffset;  // optional: starting vertex (needed if using drawIndexed with baseVertex)
	Material material;
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<SubmeshInfo> submeshes;
	std::vector<TextureData> textureData;
};

TextureData loadMaterialTexture(const aiScene* scene, const aiMaterial* mat, const std::string& modelPath, aiTextureType type) {
	TextureData texture{};
	aiString path;

	if (mat->GetTexture(type, 0, &path) == AI_SUCCESS) {
		// 1. Check for Embedded Texture (marked with *)
		const aiTexture* embeddedTex = scene->GetEmbeddedTexture(path.C_Str());
		if (embeddedTex) {
			if (embeddedTex->mHeight == 0) { // Compressed (png/jpg)
				texture.pixels = stbi_load_from_memory(
					reinterpret_cast<unsigned char*>(embeddedTex->pcData),
					embeddedTex->mWidth, &texture.width, &texture.height, &texture.channels, 4);
			}
			else { // Raw data
				texture.pixels = stbi_load_from_memory(
					reinterpret_cast<unsigned char*>(embeddedTex->pcData),
					embeddedTex->mWidth * embeddedTex->mHeight,
					&texture.width, &texture.height, &texture.channels, 4);
			}
		}
		// 2. Load from File
		else {
			std::string fullPath = path.C_Str();
			// Simple path fix: assume texture is in same dir as model
			std::string baseDir = modelPath.substr(0, modelPath.find_last_of("/\\") + 1);
			texture.pixels = stbi_load((baseDir + fullPath).c_str(),
				&texture.width, &texture.height, &texture.channels, 4); // Force 4 channels (RGBA)
			texture.path = fullPath;
		}
	}
	return texture;
}


inline glm::mat4 aiMatrix4x4ToGlm(const aiMatrix4x4& from) {
	glm::mat4 to;
	to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
	to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
	to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
	to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
	return to;
}

const int MAX_NODES = 1000000000;
int g_NodesCounter = 0;

void processNode(aiNode* node, const aiScene* scene, glm::mat4 parentTransform, Mesh& result,
	std::unordered_map<std::string, int>& textureCache, const std::string& path) {
	++g_NodesCounter;

	// 1. Transform
	glm::mat4 nodeTransform = aiMatrix4x4ToGlm(node->mTransformation);
	glm::mat4 globalTransform = parentTransform * nodeTransform;
	glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(globalTransform)));

	uint32_t vertexOffset = static_cast<uint32_t>(result.vertices.size());
	uint32_t indexOffset = static_cast<uint32_t>(result.indices.size());

	for (unsigned int m = 0; m < node->mNumMeshes; ++m) {
		const aiMesh* mesh = scene->mMeshes[node->mMeshes[m]];

		SubmeshInfo info{};
		info.vertexOffset = vertexOffset;
		info.indexOffset = indexOffset;
		info.indexCount = mesh->mNumFaces * 3;

		// 2. Vertices
		for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
			Vertex vertex{};
			glm::vec4 pos = globalTransform * glm::vec4(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z, 1.0f);
			vertex.position = glm::vec3(pos);
			vertex.normal = mesh->HasNormals() ? glm::normalize(normalMatrix * glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z)) : glm::vec3(0.0f);
			vertex.texCoord = mesh->HasTextureCoords(0) ? glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : glm::vec2(0.0f);
			vertex.tangent = mesh->HasTangentsAndBitangents()
				? glm::normalize(normalMatrix * glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z))
				: glm::vec3(0.0f);
			result.vertices.push_back(vertex);
		}

		// 3. Indices
		for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
			const aiFace& face = mesh->mFaces[i];
			if (face.mNumIndices != 3) continue;
			result.indices.push_back(face.mIndices[1]);
			result.indices.push_back(face.mIndices[2]);
			result.indices.push_back(face.mIndices[0]);
		}

		vertexOffset += mesh->mNumVertices;
		indexOffset += mesh->mNumFaces * 3;

		// 4. Materials
		if (mesh->mMaterialIndex >= 0) {
			aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

			// A. Factors
			aiColor4D color;
			if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_BASE_COLOR, &color)) {
				info.material.baseColorFactor = glm::vec4(color.r, color.g, color.b, color.a);
			}
			else if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &color)) {
				info.material.baseColorFactor = glm::vec4(color.r, color.g, color.b, color.a);
			}

			// Default to non-metal for FBX
			float metallic = 1.0f;
			float roughness = 1.0f;

			aiGetMaterialFloat(material, AI_MATKEY_METALLIC_FACTOR, &metallic);
			aiGetMaterialFloat(material, AI_MATKEY_ROUGHNESS_FACTOR, &roughness);

			info.material.metallicFactor = metallic;
			info.material.roughnessFactor = roughness;

			aiString texPath;

			// B. Base Color Texture
			// Try PBR first, then Diffuse
			aiTextureType type = aiTextureType_BASE_COLOR;
			if (material->GetTextureCount(type) == 0) type = aiTextureType_DIFFUSE;

			if (material->GetTexture(type, 0, &texPath) == AI_SUCCESS) {
				std::string key = texPath.C_Str();
				if (textureCache.find(key) != textureCache.end()) {
					info.material.baseColorTextureIndex = textureCache[key];
				}
				else {
					TextureData tex = loadMaterialTexture(scene, material, path, type);
					if (tex.pixels) {
						int newIdx = (int)result.textureData.size();
						tex.isLinear = false;
						result.textureData.push_back(tex);
						textureCache[key] = newIdx;
						info.material.baseColorTextureIndex = newIdx;
					}
				}
			}

			// C. Normal Map
			if (material->GetTexture(aiTextureType_NORMALS, 0, &texPath) == AI_SUCCESS) {
				std::string key = texPath.C_Str();
				if (textureCache.find(key) != textureCache.end()) {
					info.material.normalTextureIndex = textureCache[key];
				}
				else {
					TextureData tex = loadMaterialTexture(scene, material, path, aiTextureType_NORMALS);
					if (tex.pixels) {
						int newIdx = (int)result.textureData.size();
						tex.isLinear = true;
						result.textureData.push_back(tex);
						textureCache[key] = newIdx;
						info.material.normalTextureIndex = newIdx;
					}
				}
			}

			// D. Metallic-Roughness (NEW: Support for glTF via Assimp)
			// Assimp maps the glTF packed texture to aiTextureType_UNKNOWN (index 0) usually.
			// We check UNKNOWN. If loading FBX, this is usually empty, so it's safe.
			if (material->GetTexture(aiTextureType_UNKNOWN, 0, &texPath) == AI_SUCCESS) {
				std::string key = texPath.C_Str();
				if (textureCache.find(key) != textureCache.end()) {
					info.material.metallicRoughnessTextureIndex = textureCache[key];
				}
				else {
					// Use UNKNOWN type
					TextureData tex = loadMaterialTexture(scene, material, path, aiTextureType_UNKNOWN);
					if (tex.pixels) {
						int newIdx = (int)result.textureData.size();
						tex.isLinear = true;
						result.textureData.push_back(tex);
						textureCache[key] = newIdx;
						info.material.metallicRoughnessTextureIndex = newIdx;
					}
				}
			}
		}

		result.submeshes.push_back(info);
	}

	for (unsigned int i = 0; i < node->mNumChildren; ++i) {
		processNode(node->mChildren[i], scene, globalTransform, result, textureCache, path);
	}
}

void calculateTotalAssimpVertices(const aiNode* node, const aiScene* scene, size_t& totalVerts, size_t& totalIndices) {
	// Count vertices/indices for meshes attached to THIS node
	for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
		const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		totalVerts += mesh->mNumVertices;
		// Since we use aiProcess_Triangulate, every face has 3 indices
		totalIndices += mesh->mNumFaces * 3;
	}

	// Recurse into children
	for (unsigned int i = 0; i < node->mNumChildren; ++i) {
		calculateTotalAssimpVertices(node->mChildren[i], scene, totalVerts, totalIndices);
	}
}

Mesh loadWithAssimp(const std::string& path) {
	Assimp::Importer importer;

	const aiScene* scene = importer.ReadFile(
		path,
		aiProcess_Triangulate 
		| aiProcess_FlipUVs 
		| aiProcess_CalcTangentSpace 
		| aiProcess_GenSmoothNormals
		| aiProcess_GlobalScale
	);

	if (!scene || !scene->HasMeshes()) {
		throw std::runtime_error("Failed to load model: " + path);
	}

	Mesh result;

	size_t totalVerts = 0;
	size_t totalIndices = 0;

	// Calculate exact counts to prevent std::vector reallocation resizing
	calculateTotalAssimpVertices(scene->mRootNode, scene, totalVerts, totalIndices);

	result.vertices.reserve(totalVerts);
	result.indices.reserve(totalIndices);

	std::unordered_map<std::string, int> textureCache;

	// Start recursion from the root node with an identity matrix
	processNode(scene->mRootNode, scene, glm::mat4(1.0f), result, textureCache, path);

	return result;
}

// =============================================================
// Helper: Load Texture from fastgltf Image
// =============================================================
TextureData loadTexture(const fastgltf::Asset& asset, const fastgltf::Image& image, const std::string& modelPath) {
	TextureData texData{};

	std::visit(fastgltf::visitor{
		[&](const fastgltf::sources::URI& uri) {
			// 1. Load from File (URI)
			// Handle both absolute and relative paths
			std::string pathString = std::string(uri.uri.path()); // fastgltf 0.9+ uses .path()
			std::filesystem::path fullPath = std::filesystem::path(modelPath).parent_path() / pathString;

			texData.pixels = stbi_load(fullPath.string().c_str(), &texData.width, &texData.height, &texData.channels, 4);
			if (!texData.pixels) {
				std::cerr << "Failed to load texture: " << fullPath << "\n";
			}
		},
		[&](const fastgltf::sources::Array& array) {
			// 2. Load from Embedded Array (e.g. base64 decoded)
			texData.pixels = stbi_load_from_memory(
				reinterpret_cast<const stbi_uc*>(array.bytes.data()),
				static_cast<int>(array.bytes.size()),
				&texData.width, &texData.height, &texData.channels, 4);
		},
		[&](const fastgltf::sources::BufferView& view) {
			// 3. Load from BufferView (GLB binary chunk)
			auto& bufferView = asset.bufferViews[view.bufferViewIndex];
			auto& buffer = asset.buffers[bufferView.bufferIndex];

			std::visit(fastgltf::visitor{
				[&](const fastgltf::sources::Array& bufferArray) {
					const stbi_uc* data = reinterpret_cast<const stbi_uc*>(bufferArray.bytes.data() + bufferView.byteOffset);
					texData.pixels = stbi_load_from_memory(
						data,
						static_cast<int>(bufferView.byteLength),
						&texData.width, &texData.height, &texData.channels, 4);
				},
				[](auto&) {} // Handle other buffer types if necessary
			}, buffer.data);
		},
		[](auto&) {} // Fallback
		}, image.data);

	return texData;
}

// =============================================================
// Helper: Convert glTF transform to GLM Matrix
// =============================================================
glm::mat4 getTransformMatrix(const fastgltf::Node& node, glm::mat4& parentMatrix) {
	glm::mat4 localMatrix(1.0f);

	std::visit(fastgltf::visitor{
		[&](const fastgltf::math::fmat4x4& matrix) {
			memcpy(&localMatrix, matrix.data(), sizeof(float) * 16);
		},
		[&](const fastgltf::TRS& trs) {
			glm::vec3 translation(trs.translation[0], trs.translation[1], trs.translation[2]);
			// GLM quat constructor is (w, x, y, z). glTF is (x, y, z, w).
			glm::quat rotation(trs.rotation[3], trs.rotation[0], trs.rotation[1], trs.rotation[2]);
			glm::vec3 scale(trs.scale[0], trs.scale[1], trs.scale[2]);

			glm::mat4 mT = glm::translate(glm::mat4(1.0f), translation);
			glm::mat4 mR = glm::mat4_cast(rotation);
			glm::mat4 mS = glm::scale(glm::mat4(1.0f), scale);

			localMatrix = mT * mR * mS;
		}
		}, node.transform);

	return parentMatrix * localMatrix;
}

// =============================================================
// Helper: Recursive Node Processor
// =============================================================
void processFastGltfNode(fastgltf::Asset& asset, size_t nodeIndex, const glm::mat4& parentTransform,
	Mesh& result, std::unordered_map<std::string, int>& textureCache, const std::string& path)
{
	auto& node = asset.nodes[nodeIndex];
	glm::mat4 globalTransform = getTransformMatrix(node, const_cast<glm::mat4&>(parentTransform));

	if (node.meshIndex.has_value()) {
		auto& mesh = asset.meshes[node.meshIndex.value()];
		glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(globalTransform)));

		for (const auto& primitive : mesh.primitives) {
			SubmeshInfo sub{};
			sub.vertexOffset = static_cast<uint32_t>(result.vertices.size());
			sub.indexOffset = static_cast<uint32_t>(result.indices.size());

			// === POSITION ===
			auto posAttrIt = primitive.findAttribute("POSITION");
			if (posAttrIt == primitive.attributes.end()) continue;
			auto& posAccessor = asset.accessors[posAttrIt->accessorIndex];

			// === INDICES ===
			std::vector<uint32_t> localIndices;
			if (primitive.indicesAccessor.has_value()) {
				auto& idxAccessor = asset.accessors[primitive.indicesAccessor.value()];
				sub.indexCount = static_cast<uint32_t>(idxAccessor.count);
				fastgltf::iterateAccessor<std::uint32_t>(asset, idxAccessor, [&](std::uint32_t idx) {
					localIndices.push_back(idx);
					});
			}
			else {
				sub.indexCount = static_cast<uint32_t>(posAccessor.count);
				for (size_t i = 0; i < posAccessor.count; ++i) localIndices.push_back((uint32_t)i);
			}

			// === DATA ARRAYS (Read into Safe Types) ===
			size_t vCount = posAccessor.count;
			std::vector<glm::vec3> positions(vCount);
			std::vector<glm::vec3> normals(vCount, glm::vec3(0.0f));
			std::vector<glm::vec2> texcoords(vCount, glm::vec2(0.0f));
			std::vector<glm::vec3> tangents(vCount, glm::vec3(0.0f));

			// Use fastgltf types for reading to ensure binary safety, then cast to GLM
			fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, posAccessor,
				[&](fastgltf::math::fvec3 v, size_t i) {
					positions[i] = glm::vec3(v.x(), v.y(), v.z());
				});

			if (auto it = primitive.findAttribute("NORMAL"); it != primitive.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, asset.accessors[it->accessorIndex],
					[&](fastgltf::math::fvec3 v, size_t i) {
						normals[i] = glm::vec3(v.x(), v.y(), v.z());
					});
			}
			if (auto it = primitive.findAttribute("TEXCOORD_0"); it != primitive.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(asset, asset.accessors[it->accessorIndex],
					[&](fastgltf::math::fvec2 v, size_t i) {
						texcoords[i] = glm::vec2(v.x(), v.y());
					});
			}
			if (auto it = primitive.findAttribute("TANGENT"); it != primitive.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(asset, asset.accessors[it->accessorIndex],
					[&](fastgltf::math::fvec4 v, size_t i) {
						tangents[i] = glm::vec3(v.x(), v.y(), v.z());
					});
			}

			// === TRANSFORM AND PUSH ===
			for (size_t i = 0; i < vCount; ++i) {
				Vertex v{};
				v.position = glm::vec3(globalTransform * glm::vec4(positions[i], 1.0f));
				v.normal = glm::normalize(normalMatrix * normals[i]);
				v.tangent = glm::normalize(normalMatrix * tangents[i]);
				v.texCoord = texcoords[i];
				result.vertices.push_back(v);
			}

			for (uint32_t idx : localIndices) {
				result.indices.push_back(idx); //+ sub.vertexOffset);
			}

			// === MATERIALS ===
			if (primitive.materialIndex.has_value()) {
				const auto& material = asset.materials[primitive.materialIndex.value()];
				auto& pbr = material.pbrData;
				auto baseColor = pbr.baseColorFactor;

				sub.material.baseColorFactor = glm::vec4(baseColor.x(), baseColor.y(), baseColor.z(), baseColor.w());
				sub.material.metallicFactor = pbr.metallicFactor;
				sub.material.roughnessFactor = pbr.roughnessFactor;

				// 1. Base Color Texture
				if (pbr.baseColorTexture.has_value() && asset.textures[pbr.baseColorTexture.value().textureIndex].imageIndex.has_value()) {
					size_t imgIdx = asset.textures[pbr.baseColorTexture.value().textureIndex].imageIndex.value();
					std::string key = "base:" + std::to_string(imgIdx);

					if (textureCache.find(key) != textureCache.end()) {
						sub.material.baseColorTextureIndex = textureCache[key];
					}
					else {
						// LOAD TEXTURE HERE
						TextureData tex = prepareTextureInfo(asset, asset.images[imgIdx], path);
						if (!tex.path.empty() || tex.encodedData != nullptr) {
							int newIdx = (int)result.textureData.size();
							result.textureData.push_back(tex);
							textureCache[key] = newIdx;
							sub.material.baseColorTextureIndex = newIdx;
						}
					}
				}

				// 2. Normal Texture
				if (material.normalTexture.has_value() && asset.textures[material.normalTexture.value().textureIndex].imageIndex.has_value()) {
					size_t imgIdx = asset.textures[material.normalTexture.value().textureIndex].imageIndex.value();
					std::string key = "norm:" + std::to_string(imgIdx);

					if (textureCache.find(key) != textureCache.end()) {
						sub.material.normalTextureIndex = textureCache[key];
					}
					else {
						// LOAD TEXTURE HERE
						TextureData tex = prepareTextureInfo(asset, asset.images[imgIdx], path);
						if (!tex.path.empty() || tex.encodedData != nullptr) {
							int newIdx = (int)result.textureData.size();
							result.textureData.push_back(tex);
							textureCache[key] = newIdx;
							sub.material.normalTextureIndex = newIdx;
						}
					}
				}

				// 3. Metallic-Roughness (NEW)
				if (pbr.metallicRoughnessTexture.has_value()) {
					size_t texIndex = pbr.metallicRoughnessTexture.value().textureIndex;
					if (asset.textures[texIndex].imageIndex.has_value()) {
						size_t imgIdx = asset.textures[texIndex].imageIndex.value();
						std::string key = "mr:" + std::to_string(imgIdx);

						if (textureCache.find(key) != textureCache.end()) {
							sub.material.metallicRoughnessTextureIndex = textureCache[key];
						}
						else {
							TextureData tex = prepareTextureInfo(asset, asset.images[imgIdx], path);
							if (!tex.path.empty() || tex.encodedData != nullptr) {
								int newIdx = (int)result.textureData.size();
								result.textureData.push_back(tex);
								textureCache[key] = newIdx;
								sub.material.metallicRoughnessTextureIndex = newIdx;
							}
						}
					}
				}
			}
			result.submeshes.push_back(sub);
		}
	}

	for (size_t childIndex : node.children) {
		processFastGltfNode(asset, childIndex, globalTransform, result, textureCache, path);
	}
}

size_t calculateTotalVertices(const fastgltf::Asset& asset) {
	size_t totalVertices = 0;
	for (const auto& mesh : asset.meshes) {
		for (const auto& primitive : mesh.primitives) {
			auto it = primitive.findAttribute("POSITION");
			if (it != primitive.attributes.end()) {
				totalVertices += asset.accessors[it->accessorIndex].count;
			}
		}
	}
	return totalVertices;
}

// =============================================================
// Main Load Function
// =============================================================
Mesh loadWithFastGltf(const std::string& path) {
	Mesh result;
	fastgltf::Parser parser;
	auto gltfFile = fastgltf::MappedGltfFile::FromPath(path);
	if (!gltfFile) throw std::runtime_error("Failed to load glTF file: " + path);

	auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

	auto assetRet = parser.loadGltf(gltfFile.get(), std::filesystem::path(path).parent_path(), gltfOptions);
	if (auto error = assetRet.error(); error != fastgltf::Error::None) {
		throw std::runtime_error("Failed to parse: " + std::string(fastgltf::getErrorMessage(error)));
	}
	auto& asset = assetRet.get();
	size_t totalVerts = calculateTotalVertices(asset);
	result.vertices.reserve(totalVerts);
	result.indices.reserve(totalVerts);
	std::unordered_map<std::string, int> textureCache;

	size_t sceneIndex = asset.defaultScene.value_or(0);
	if (asset.scenes.empty()) return result;

	const auto& scene = asset.scenes[sceneIndex];
	glm::mat4 rootTransform(1.0f);

	for (size_t nodeIndex : scene.nodeIndices) {
		processFastGltfNode(asset, nodeIndex, rootTransform, result, textureCache, path);
	}

	// 3. Parallel Texture Loading
	// Now that result.textureData is fully populated and stable, we process it in parallel.
	if (!result.textureData.empty()) {
		std::cout << "Decoding " << result.textureData.size() << " textures in parallel...\n";

		std::vector<std::future<void>> futures;
		futures.reserve(result.textureData.size());

		for (auto& tex : result.textureData) {
			// Launch async job for each texture
			futures.push_back(std::async(std::launch::async, [&tex]() {
				decodeTextureParallel(tex);
				}));
		}

		// Wait for all threads to finish
		for (auto& f : futures) {
			f.wait();
		}
	}

	std::cout << "Loaded " << result.vertices.size() << " vertices, " << result.textureData.size() << " textures.\n";
	return result;
}

int main()
{
	// ------------------------
	// 1. Initialize SDL3
	// ------------------------
	if (!SDL_Init(SDL_INIT_VIDEO)) {
		const char* err = SDL_GetError();
		std::cerr << "SDL_Init failed: " << err << "\n";
		return -1;
	}

	std::cout << "Hello CMake." << std::endl;

	SDL_Window* window = SDL_CreateWindow(
		"Vulkan SDL3",
		1280, 720,
		SDL_WINDOW_VULKAN
	);

	if (!window) {
		std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
		SDL_Quit();
		return -1;
	}

	Mesh model;
	try {
		// model = loadWithAssimp("models/sponza-palace/source/scene.glb");
		// model = loadWithFastGltf("models/sponza-palace/source/scene.glb");
		// model = loadWithAssimp("models/main_sponza/NewSponza_Main_glTF_003.gltf");
		model = loadWithFastGltf("models/main_sponza/NewSponza_Main_glTF_003.gltf");
		// model = loadWithAssimp("models/main_sponza/NewSponza_Main_Yup_003.fbx");
		// model = loadWithAssimp("models/london-city/source/traffic_slam_2_map.glb");
		// model = loadWithAssimp("models/dae-diorama-grandmas-house/source/Dae_diorama_upload/Dae_diorama_upload.fbx");
		// model = loadWithAssimp("models/polygon-mini-free/source/model.obj");
		// model = loadWithAssimp("models/figure-embodying-the-element-silver/Ag_rechte_f Figur_lowres.obj");
	}
	catch (const std::exception& e) {
		std::cerr << "Error loading GLB: " << e.what() << "\n";
		return -1;
	}

	// ------------------------
	// 2. Initialize volk
	// ------------------------
	if (volkInitialize() != VK_SUCCESS) {
		std::cerr << "volkInitialize failed\n";
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}
	vk::detail::defaultDispatchLoaderDynamic.init(vkGetInstanceProcAddr);

	// ------------------------
	// 3. Get SDL Vulkan extensions
	// ------------------------
	Uint32 count_instance_extensions;
	const char* const* instance_extensions = SDL_Vulkan_GetInstanceExtensions(&count_instance_extensions);
	if (instance_extensions == NULL) {
		std::cerr << "SDL_Vulkan_GetInstanceExtensions failed\n";
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	std::vector<const char*> extensions(instance_extensions, instance_extensions + count_instance_extensions);
	for (auto ext : extensions)
		std::cout << ext << "\n";

	// ------------------------
	// 4. Create Vulkan instance using vk-bootstrap
	// ------------------------
	vkb::InstanceBuilder builder;
	auto instRet = builder
		.set_app_name("SDL3 Vulkan App")
		.require_api_version(1, 4, 0)
		.set_minimum_instance_version(1, 4)
		// .use_default_debug_messenger()
		.enable_extensions(extensions)
		// .enable_extension(VK_EXT_SHADER_OBJECT_EXTENSION_NAME)
		.build();

	if (!instRet) {
		std::cerr << "Failed to create instance: " << instRet.error().message() << "\n";
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	vkb::Instance vkbInstance = instRet.value();
	VkInstance vkInstance = vkbInstance.instance;
	volkLoadInstance(vkInstance);
	vk::Instance instance(vkInstance);
	vk::detail::defaultDispatchLoaderDynamic.init(instance);

	// ------------------------
	// 6. Create Vulkan surface with SDL3
	// ------------------------
	VkSurfaceKHR surfaceVk;
	if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surfaceVk)) {
		std::cerr << "Failed to create Vulkan surface\n";
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}
	vk::SurfaceKHR surface(surfaceVk);

	// ------------------------
	// 5. Select physical device and create logical device
	// ------------------------
	VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
	shaderObjectFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
	shaderObjectFeatures.shaderObject = VK_TRUE;

	VkPhysicalDeviceVulkan13Features features13{};
	features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	features13.synchronization2 = VK_TRUE;
	features13.dynamicRendering = VK_TRUE;
	features13.pNext = &shaderObjectFeatures;

	vkb::PhysicalDeviceSelector selector{ vkbInstance };
	auto physRet = selector
		.set_minimum_version(1, 4)
		.add_required_extension(vk::EXTShaderObjectExtensionName)
		.add_required_extension_features(shaderObjectFeatures)
		.set_required_features_13(features13)
		.set_surface(surface)
		// .add_required_extension_features(VkPhysicalDeviceFeatures{ .samplerAnisotropy = VK_TRUE })
		.select();

	if (!physRet) {
		std::cerr << "Failed to select physical device: " << physRet.error().message() << "\n";
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}


	vkb::PhysicalDevice vkbPhys = physRet.value();

	if (!vkbPhys.are_extension_features_present(shaderObjectFeatures)) {
		std::cout << "no" << "\n";
	}

	vkb::DeviceBuilder deviceBuilder{ vkbPhys };
	auto deviceRet = deviceBuilder
		.build();

	if (!deviceRet) {
		std::cerr << "Failed to create device: " << deviceRet.error().message() << "\n";
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	if (!vkbPhys.is_extension_present(VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) {
		throw std::runtime_error("Shader Object not supported");
	}
	vk::PhysicalDevice physicalDevice(vkbPhys.physical_device);

	vkb::Device vkbDevice = deviceRet.value();
	VkDevice vkDevice = vkbDevice.device;
	volkLoadDevice(vkDevice);
	vk::Device device(vkDevice);
	vk::detail::defaultDispatchLoaderDynamic.init(device);

	// ------------------------
	// 7. Setup VMA (minimal)
	// ------------------------
	VmaVulkanFunctions vulkanFunctions{};
	vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.physicalDevice = physicalDevice;
	allocatorInfo.device = device;
	allocatorInfo.instance = instance;
	allocatorInfo.pVulkanFunctions = &vulkanFunctions;
	allocatorInfo.vulkanApiVersion = vk::ApiVersion14;

	VmaAllocator allocator;
	if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
		std::cerr << "Failed to create VMA allocator\n";
		vkb::destroy_device(vkbDevice);
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	std::cout << "SDL3 + Vulkan instance, device, and VMA initialized successfully!\n";

	std::unique_ptr<VertexBuffer> vertexBuffer;
	std::unique_ptr<IndexBuffer> indexBuffer;
	std::unique_ptr<InstanceBuffer> instanceBuffer;

	try {
		vertexBuffer = std::make_unique<VertexBuffer>(allocator, device, model.vertices);
		indexBuffer = std::make_unique<IndexBuffer>(allocator, device, model.indices);
		instanceBuffer = std::make_unique<InstanceBuffer>(allocator, device, instances);
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create buffers: " << e.what() << "\n";
		return -1;
	}
	// ------------------------
	// 10. Create Swapchain
	// ------------------------
	vkb::SwapchainBuilder swapchain_builder{ vkbDevice };
	auto swap_ret = swapchain_builder
		.set_desired_extent(1280, 720)
		.set_desired_format(VkSurfaceFormatKHR{
		VK_FORMAT_B8G8R8A8_SRGB,
		VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
			})
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.build();
	if (!swap_ret) {
		std::cerr << "Failed to create Swapchain\n";
		vmaDestroyAllocator(allocator);
		vkb::destroy_device(vkbDevice);
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}
	vkb::Swapchain vkbSwapchain = swap_ret.value();	

	// ------------------------
	// 11a. Create Depth Image & View
	// ------------------------
	vk::Format depthFormat = vk::Format::eD32Sfloat;

	// Image
	vk::ImageCreateInfo depthImageInfo{};
	depthImageInfo.imageType = vk::ImageType::e2D;
	depthImageInfo.extent = vk::Extent3D{ 1280, 720, 1 };
	depthImageInfo.mipLevels = 1;
	depthImageInfo.arrayLayers = 1;
	depthImageInfo.format = depthFormat;
	depthImageInfo.tiling = vk::ImageTiling::eOptimal;
	depthImageInfo.initialLayout = vk::ImageLayout::eUndefined;
	depthImageInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
	depthImageInfo.samples = vk::SampleCountFlagBits::e1;
	depthImageInfo.sharingMode = vk::SharingMode::eExclusive;

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VkImage depthImageRaw;
	VmaAllocation depthAlloc;
	if (vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&depthImageInfo), &allocInfo, &depthImageRaw, &depthAlloc, nullptr) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create depth image");
	}
	vk::Image depthImage(depthImageRaw);

	// Image view
	vk::ImageViewCreateInfo depthViewInfo{};
	depthViewInfo.image = depthImage;
	depthViewInfo.viewType = vk::ImageViewType::e2D;
	depthViewInfo.format = depthFormat;
	depthViewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
	depthViewInfo.subresourceRange.baseMipLevel = 0;
	depthViewInfo.subresourceRange.levelCount = 1;
	depthViewInfo.subresourceRange.baseArrayLayer = 0;
	depthViewInfo.subresourceRange.layerCount = 1;

	vk::ImageView depthImageView = device.createImageView(depthViewInfo).value;
	// ------------------------
	// 11. Get Swapchain Images and ImageViews
	// ------------------------
	auto imagesRet = vkbSwapchain.get_images();
	if (!imagesRet) {
		std::cerr << "Failed to get swapchain images\n";
		vmaDestroyAllocator(allocator);
		vkb::destroy_swapchain(vkbSwapchain);
		vkb::destroy_device(vkbDevice);
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}

	std::vector<VkImage> swapchainImages = imagesRet.value();
	uint32_t swapchainImageCount = static_cast<uint32_t>(swapchainImages.size());
	const int MAX_FRAMES_IN_FLIGHT = swapchainImageCount;
	// const int MAX_FRAMES_IN_FLIGHT = 2;
	auto viewsRet = vkbSwapchain.get_image_views();
	if (!viewsRet) {
		std::cerr << "Failed to get swapchain image views\n";
		vmaDestroyAllocator(allocator);
		vkb::destroy_swapchain(vkbSwapchain);
		vkb::destroy_device(vkbDevice);
		vkb::destroy_surface(vkbInstance, surface);
		vkb::destroy_instance(vkbInstance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return -1;
	}
	std::vector<VkImageView> swapchainImageViews = viewsRet.value();
	{
		// Queues
		vk::Queue graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
		vk::Queue presentQueue = vkbDevice.get_queue(vkb::QueueType::present).value();
		uint32_t graphicsQueueIndex = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
		// ------------------------
		// 12. Command Pool
		// ------------------------
		vk::CommandPoolCreateInfo commandPoolInfo{};
		commandPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		commandPoolInfo.queueFamilyIndex = graphicsQueueIndex;

		vk::UniqueCommandPool commandPool = device.createCommandPoolUnique(commandPoolInfo).value;
		// ------------------------
		// 13. Command Buffer
		// ------------------------
		vk::CommandBufferAllocateInfo commandAllocInfo{};
		commandAllocInfo.commandPool = commandPool.get();
		commandAllocInfo.level = vk::CommandBufferLevel::ePrimary;
		commandAllocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

		auto allocatedCommandBuffers = device.allocateCommandBuffersUnique(commandAllocInfo);
		std::vector<vk::UniqueCommandBuffer> commandBuffers = std::move(allocatedCommandBuffers.value);

		// ------------------------
		// 13. Upload Textures
		// ------------------------
		std::vector<std::unique_ptr<TextureImage>> gpuTextures;

		std::cout << "Uploading " << model.textureData.size() << " textures...\n";
		for (auto& cpuTex : model.textureData) {
			vk::Format fmt = cpuTex.isLinear ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;
			gpuTextures.push_back(std::make_unique<TextureImage>(
				allocator, device, commandPool.get(), graphicsQueue, cpuTex, fmt
			));
			// Free CPU memory now that it's on GPU
			cpuTex.free();
		}
		// 1. Create Texture Sampler (Common for all textures)
		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
		// samplerInfo.anisotropyEnable = VK_TRUE;
		// samplerInfo.maxAnisotropy = 16.0f; // Ensure your physical device supports this!
		samplerInfo.anisotropyEnable = VK_FALSE;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

		vk::UniqueSampler textureSampler = device.createSamplerUnique(samplerInfo).value;

		// 2. Create Descriptor Set Layout
		// Binding 0: Combined Image Sampler (Fragment Shader)
		vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 0;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &samplerLayoutBinding;

		vk::UniqueDescriptorSetLayout descriptorSetLayout = device.createDescriptorSetLayoutUnique(layoutInfo).value;

		// 1. Create a 1x1 White Default Texture (Fallback)
		TextureData whiteTexData;
		whiteTexData.width = 1; whiteTexData.height = 1; whiteTexData.channels = 4;
		unsigned char whitePixels[] = { 255, 255, 255, 255 };
		whiteTexData.pixels = whitePixels; // No need to free this specifically

		auto defaultTexture = std::make_unique<TextureImage>(
			allocator, device, commandPool.get(), graphicsQueue, whiteTexData, vk::Format::eR8G8B8A8Srgb
		);

		TextureData normalTexData;
		normalTexData.width = 1; normalTexData.height = 1; normalTexData.channels = 4;
		unsigned char normalPixels[] = { 128, 128, 255, 255 };
		normalTexData.pixels = normalPixels;

		auto defaultNormalTexture = std::make_unique<TextureImage>(
			allocator, device, commandPool.get(), graphicsQueue, normalTexData, vk::Format::eR8G8B8A8Unorm
		);

		TextureData mrTexData;
		mrTexData.width = 1; mrTexData.height = 1; mrTexData.channels = 4;
		unsigned char mrPixels[] = { 0, 255, 0, 255 };
		mrTexData.pixels = mrPixels;

		auto defaultMrTexture = std::make_unique<TextureImage>(
			allocator, device, commandPool.get(), graphicsQueue, mrTexData, vk::Format::eR8G8B8A8Unorm
		);

		// 2. Create Descriptor Pool
		// We need 1 set for the default texture + 1 set per loaded texture
		uint32_t totalTextures = 1 + (uint32_t)gpuTextures.size();

		vk::DescriptorPoolSize poolSize{};
		poolSize.type = vk::DescriptorType::eCombinedImageSampler;
		poolSize.descriptorCount = (totalTextures + 2) * 3;

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		poolInfo.maxSets = (totalTextures + 2) * 3;

		vk::UniqueDescriptorPool descriptorPool = device.createDescriptorPoolUnique(poolInfo).value;

		// 3. Allocate Descriptor Sets
		std::vector<vk::DescriptorSetLayout> layoutsVector(totalTextures, descriptorSetLayout.get());
		vk::DescriptorSetAllocateInfo defaultBaseColorAllocInfo{};
		defaultBaseColorAllocInfo.descriptorPool = descriptorPool.get();
		defaultBaseColorAllocInfo.descriptorSetCount = totalTextures;
		defaultBaseColorAllocInfo.pSetLayouts = layoutsVector.data();

		std::vector<vk::DescriptorSet> textureDescriptorSets = device.allocateDescriptorSets(defaultBaseColorAllocInfo).value;

		// 4. Update Descriptor Sets
		// Function to write texture to a specific set index
		auto updateDescriptorSet = [&](vk::DescriptorSet set, vk::ImageView view) {
			vk::DescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			imageInfo.imageView = view;
			imageInfo.sampler = textureSampler.get();

			vk::WriteDescriptorSet descriptorWrite{};
			descriptorWrite.dstSet = set;
			descriptorWrite.dstBinding = 0;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pImageInfo = &imageInfo;

			device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
			};

		// Update Default Set (Index 0)
		updateDescriptorSet(textureDescriptorSets[0], defaultTexture->getView());

		vk::DescriptorSetAllocateInfo defaultNormalAllocInfo{};
		defaultNormalAllocInfo.descriptorPool = descriptorPool.get();
		defaultNormalAllocInfo.descriptorSetCount = 1;
		defaultNormalAllocInfo.pSetLayouts = &descriptorSetLayout.get();

		vk::DescriptorSet defaultNormalSet = device.allocateDescriptorSets(defaultNormalAllocInfo).value[0];
		updateDescriptorSet(defaultNormalSet, defaultNormalTexture->getView());

		// Update Loaded Textures (Indices 1 to N)
		for (size_t i = 0; i < gpuTextures.size(); i++) {
			updateDescriptorSet(textureDescriptorSets[i + 1], gpuTextures[i]->getView());
		}

		vk::DescriptorSetAllocateInfo defaultMrAllocInfo{};
		defaultMrAllocInfo.descriptorPool = descriptorPool.get();
		defaultMrAllocInfo.descriptorSetCount = 1;
		defaultMrAllocInfo.pSetLayouts = &descriptorSetLayout.get();

		vk::DescriptorSet defaultMrSet = device.allocateDescriptorSets(defaultMrAllocInfo).value[0];
		updateDescriptorSet(defaultMrSet, defaultMrTexture->getView());

		// ------------------------
		// 12. Semaphores and Fences
		// ------------------------
		std::vector<vk::UniqueSemaphore> imageAvailableSemaphores(MAX_FRAMES_IN_FLIGHT);
		std::vector<vk::UniqueSemaphore> renderFinishedSemaphores(MAX_FRAMES_IN_FLIGHT);
		
		vk::FenceCreateInfo fenceInfo{};
		fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
		std::vector<vk::UniqueFence> inFlightFences(MAX_FRAMES_IN_FLIGHT);

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			imageAvailableSemaphores[i] = device.createSemaphoreUnique({}).value;
			renderFinishedSemaphores[i] = device.createSemaphoreUnique({}).value;
			inFlightFences[i] = device.createFenceUnique(fenceInfo).value;
		}

		// ------------------------
		// 14. Load Shaders
		// ------------------------
		std::vector<uint32_t> vertCode, fragCode;
		try {
			vertCode = loadSpirv("shaders/triangle.vert.spv");
			fragCode = loadSpirv("shaders/triangle.frag.spv");

			std::cout << "Vertex SPIR-V words: " << vertCode.size() << "\n";
			std::cout << "Fragment SPIR-V words: " << fragCode.size() << "\n";
		} catch (const std::exception& e) {
			std::cerr << "Failed to load shaders: " << e.what() << "\n";
			vmaDestroyAllocator(allocator);
			vkb::destroy_swapchain(vkbSwapchain);
			vkb::destroy_device(vkbDevice);
			vkb::destroy_surface(vkbInstance, surface);
			vkb::destroy_instance(vkbInstance);
			SDL_DestroyWindow(window);
			SDL_Quit();
			return -1;
		}
		// ------------------------
		// 16. Create Pipeline Layout
		// ------------------------
		vk::PushConstantRange pcRange{};
		pcRange.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
		pcRange.offset = 0;
		pcRange.size = sizeof(MeshPushConstants);

		vk::PushConstantRange allRanges[] = { pcRange };

		vk::DescriptorSetLayout layouts[] = { 
			descriptorSetLayout.get()
			, descriptorSetLayout.get()
			, descriptorSetLayout.get() };

		// shader create info
		vk::ShaderCreateInfoEXT vertInfo{};
		vertInfo.setStage(vk::ShaderStageFlagBits::eVertex)
			.setNextStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(vertCode.size() * sizeof(vertCode.front()))
			.setPCode(vertCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(allRanges)
			.setSetLayoutCount(3)
			.setPSetLayouts(layouts);

		vk::ShaderCreateInfoEXT fragInfo{};
		fragInfo.setStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(fragCode.size() * sizeof(fragCode.front()))
			.setPCode(fragCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(allRanges)
			.setSetLayoutCount(3)
			.setPSetLayouts(layouts);

		vk::ShaderEXT vertShader, fragShader;
		vertShader = device.createShaderEXT(vertInfo).value;
		fragShader = device.createShaderEXT(fragInfo).value;


		std::vector<vk::PushConstantRange> pushRanges = { pcRange };
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.setPushConstantRanges(pushRanges);
		pipelineLayoutInfo.setSetLayoutCount(3);
		pipelineLayoutInfo.setPSetLayouts(layouts);
		// arrays for binding shaders
		vk::PipelineLayout pipelineLayout;
		try {
			pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo).value;
		}
		catch (const std::exception& e) {
			std::cerr << "Failed to create pipeline layout: " << e.what() << "\n";
			device.destroyShaderEXT(vertShader);
			device.destroyShaderEXT(fragShader);
			vmaDestroyAllocator(allocator);
			vkb::destroy_swapchain(vkbSwapchain);
			vkb::destroy_device(vkbDevice);
			vkb::destroy_surface(vkbInstance, surface);
			vkb::destroy_instance(vkbInstance);
			SDL_DestroyWindow(window);
			SDL_Quit();
			return -1;
		}

		// ------------------------
		// 8. Main loop
		// ------------------------
		Camera camera;
		bool mouseEnabled = true;
		SDL_SetWindowRelativeMouseMode(window, mouseEnabled);
		bool running = true;
		SDL_Event event;
		uint32_t currentFrame = 0;
		MeshPushConstants pc{};
		float cameraAmpilfier = 1.0f;
		// Update push constant for animation
		static float time = 0.0f;
		// Push constant data
		uint32_t lastTime = SDL_GetTicks();
		while (running) {
			uint32_t currentTime = SDL_GetTicks();
			float dt = (currentTime - lastTime) / 1000.0f; // convert ms to seconds
			lastTime = currentTime;
			while (SDL_PollEvent(&event)) {
				bool shiftHeld =
					(event.key.mod & SDL_KMOD_LSHIFT) ||
					(event.key.mod & SDL_KMOD_RSHIFT);
				const bool* keys = SDL_GetKeyboardState(nullptr);
				if (event.type == SDL_EVENT_QUIT || keys[SDL_SCANCODE_ESCAPE])
					running = false;

				if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
					if (event.key.scancode == SDL_SCANCODE_ESCAPE)
						running = false;


					if (shiftHeld) {
						cameraAmpilfier = 4.0f;
						if(event.key.scancode == SDL_SCANCODE_GRAVE) {
							mouseEnabled = !mouseEnabled;
							SDL_SetWindowRelativeMouseMode(window, mouseEnabled);
						}
					}
				}
				if (event.type == SDL_EVENT_KEY_UP && !event.key.repeat) {
					if (!shiftHeld)
						cameraAmpilfier = 1.0f;
				}
				if (mouseEnabled && event.type == SDL_EVENT_MOUSE_MOTION) {
					camera.yaw += event.motion.xrel * camera.sensitivity;
					camera.pitch -= event.motion.yrel * camera.sensitivity;
					camera.pitch = glm::clamp(camera.pitch, -89.0f, 89.0f);
				}
			}
			const bool* keys = SDL_GetKeyboardState(nullptr);
			glm::vec3 front{
				cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch)),
				sin(glm::radians(camera.pitch)),
				sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch))
			};
			front = glm::normalize(front);

			glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0, 1, 0)));

			if (keys[SDL_SCANCODE_W]) camera.position += front * camera.speed * dt * cameraAmpilfier;
			if (keys[SDL_SCANCODE_A]) camera.position -= right * camera.speed * dt * cameraAmpilfier;
			if (keys[SDL_SCANCODE_D]) camera.position += right * camera.speed * dt * cameraAmpilfier;
			if (keys[SDL_SCANCODE_S]) camera.position -= front * camera.speed * dt * cameraAmpilfier;
				
	

			vk::SwapchainKHR swapchainHPP(vkbSwapchain.swapchain);

			(void)device.waitForFences(inFlightFences[currentFrame].get(), VK_TRUE, UINT64_MAX);
			(void)device.resetFences(inFlightFences[currentFrame].get());

			uint32_t imageIndex = 0;
			
			vk::Result result = device.acquireNextImageKHR(
				swapchainHPP,             // swapchain
				UINT64_MAX,            // timeout, UINT64_MAX = infinite
				imageAvailableSemaphores[currentFrame].get(), // semaphore to signal
				{},                    // fence (none in this example)
				&imageIndex            // output: index of the acquired image
			);

			if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
				std::cerr << "Failed to acquireNextImageKHR\n";
				return -1;
			}

			// Record command buffer to clear blue
			vk::CommandBuffer cmd = commandBuffers[currentFrame].get();
			(void)cmd.reset();
			(void)cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

			// Transition the swapchain image from UNDEFINED to COLOR_ATTACHMENT_OPTIMAL
			vk::ImageMemoryBarrier2 layoutBarrier;
			layoutBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setSrcAccessMask(vk::AccessFlagBits2::eNone)
				.setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setOldLayout(vk::ImageLayout::eUndefined) // We don't care what was there before
				.setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setImage(swapchainImages[imageIndex])
				.setSubresourceRange(vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

			vk::DependencyInfo depInfo;
			depInfo.setImageMemoryBarriers(layoutBarrier);
			cmd.pipelineBarrier2(depInfo);

			// Transition depth image
			vk::ImageMemoryBarrier2 depthBarrier{};
			depthBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
				.setSrcAccessMask(vk::AccessFlagBits2::eNone)
				.setDstStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests)
				.setDstAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
				.setOldLayout(vk::ImageLayout::eUndefined)
				.setNewLayout(vk::ImageLayout::eDepthAttachmentOptimal)
				.setImage(depthImage)
				.setSubresourceRange({ vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 });

			vk::DependencyInfo depInfoDepth{};
			depInfoDepth.setImageMemoryBarriers(depthBarrier);
			cmd.pipelineBarrier2(depInfoDepth);

			vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.2f, 1.0f, 1.0f}));
			vk::ClearValue clearDepth(vk::ClearDepthStencilValue{ 1.0f, 0 });

			vk::RenderingAttachmentInfo colorAttachment{};
			colorAttachment.setImageView(swapchainImageViews[imageIndex])
				.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setLoadOp(vk::AttachmentLoadOp::eClear)
				.setStoreOp(vk::AttachmentStoreOp::eStore)
				.setClearValue(clearColor);

			vk::RenderingAttachmentInfo depthAttachment{};

			depthAttachment.setImageView(depthImageView)
				.setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
				.setLoadOp(vk::AttachmentLoadOp::eClear)
				.setStoreOp(vk::AttachmentStoreOp::eDontCare)
				.setClearValue(clearDepth);


			vk::RenderingInfo renderInfo{};
			renderInfo.setRenderArea({ {0,0},{1280,720} })
				.setLayerCount(1)
				.setColorAttachments(colorAttachment)
				.setPDepthAttachment(&depthAttachment);
			cmd.beginRendering(renderInfo);
			vk::ShaderStageFlagBits stages[] = {
				vk::ShaderStageFlagBits::eVertex,
				vk::ShaderStageFlagBits::eFragment
			};

			vk::ShaderEXT shaders[] = {
				vertShader,
				fragShader
			};
			// Bind shaders and draw
			cmd.bindShadersEXT(2, stages, shaders);
			// Begin rendering
			cmd.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);

			// cmd.bindVertexBuffers(0, 1, &vertexBuffer->getBuffer(), &offset);
			
			


			const vk::Viewport viewport{ 0, 0, 1280.f, 720.f, 0.f, 1.f };
			const vk::Rect2D rect{ {0,0},{1280,720} };
			cmd.setViewport(0, viewport);
			cmd.setScissor(0, rect);
			cmd.setRasterizerDiscardEnable(false);
			cmd.setCullMode(vk::CullModeFlagBits::eBack);
			cmd.setFrontFace(vk::FrontFace::eCounterClockwise);
			cmd.setStencilTestEnable(false);
			cmd.setDepthTestEnable(true);
			cmd.setDepthWriteEnable(true);
			cmd.setDepthCompareOp(vk::CompareOp::eLess);
			cmd.setDepthBiasEnable(false);
			cmd.setPolygonModeEXT(vk::PolygonMode::eFill);
			cmd.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
			vk::SampleMask mask = ~0u;  // enable all samples
			cmd.setSampleMaskEXT(vk::SampleCountFlagBits::e1, &mask);
			cmd.setAlphaToCoverageEnableEXT(VK_FALSE);
			cmd.setColorBlendEnableEXT(0, VK_FALSE);
			cmd.setColorWriteMaskEXT(0, vk::ColorComponentFlags(0xF));
			cmd.setViewportWithCount(1, &viewport);
			cmd.setScissorWithCount(1, &rect);
			cmd.setPrimitiveRestartEnable(VK_FALSE);

			auto attributesArray = Vertex::getAttributeDescriptions(0);

			// If using instance buffer as second binding, keep it:
			vk::VertexInputBindingDescription2EXT instanceBinding{};
			instanceBinding.binding = 1;
			instanceBinding.stride = sizeof(InstanceData);
			instanceBinding.inputRate = vk::VertexInputRate::eInstance;
			instanceBinding.divisor = 1;
			vk::VertexInputBindingDescription2EXT bindingDescs[2] = {
				Vertex::getBindingDescription(0), // Main geometry
				// If you use instances:
				instanceBinding
			};
			// vk::VertexInputBindingDescription2EXT bindings[2]{ binding, instanceBinding };
			cmd.setVertexInputEXT(2, bindingDescs, static_cast<uint32_t>(attributesArray.size()), attributesArray.data());

			// Bind buffers
			vk::DeviceSize offsets[2] = { 0, 0 };
			vk::Buffer buffers[2] = {
				vertexBuffer->getBuffer(),
				instanceBuffer->getBuffer()
			};
			vk::DeviceSize sizes[2] = { sizeof(Vertex) * model.vertices.size(), sizeof(InstanceData) * instances.size() };
			vk::DeviceSize strides[2] = { sizeof(Vertex), sizeof(InstanceData) };
			//vk::DeviceSize stride = sizeof(Vertex); // Make sure this >= sum of attribute sizes
			//vk::DeviceSize size = sizeof(Vertex) * vertices.size(); // Make sure this >= sum of attribute sizes

			cmd.bindVertexBuffers2(
				0
				, 2
				, buffers
				, offsets
				, sizes
				, strides);

			cmd.bindIndexBuffer(
				indexBuffer->getBuffer(),
				0,
				vk::IndexType::eUint32
			);
			pc.view = getView(camera);
			pc.proj = getProjection(1280.0f, 720.0f);
			pc.time = time;
			//int submeshCounter = 0;
			for (const auto& sub : model.submeshes) {
				//++submeshCounter;
				//if (submeshCounter % 2) continue;
				pc.baseColor = sub.material.baseColorFactor;
				pc.metallic = sub.material.metallicFactor;
				pc.roughness = sub.material.roughnessFactor;
				//pc.baseColor = glm::vec4(1.0f);
				//pc.metallic = 0.0f;             // Non-metal (easier to see initially)
				//pc.roughness = 0.5f;
				cmd.pushConstants(
					pipelineLayout,
					vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
					0,
					sizeof(MeshPushConstants),
					&pc
				);

				// base color
				vk::DescriptorSet baseColorSet;
				if (sub.material.baseColorTextureIndex >= 0) {
					// Index + 1 because set[0] is the default white texture
					baseColorSet = textureDescriptorSets[sub.material.baseColorTextureIndex + 1];
				}
				else {
					baseColorSet = textureDescriptorSets[0];
				}

				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipelineLayout, // This must be the layout that includes the descriptorSetLayout
					0,
					1, &baseColorSet,
					0, nullptr
				);

				//normal
				vk::DescriptorSet normalSet;
				if (sub.material.normalTextureIndex >= 0) {
					normalSet = textureDescriptorSets[sub.material.normalTextureIndex + 1];
				}
				else {
					normalSet = defaultNormalSet;
				}
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics
					, pipelineLayout
					, 1
					, 1
					, &normalSet
					, 0
					, nullptr);

				vk::DescriptorSet mrSet;
				if (sub.material.metallicRoughnessTextureIndex >= 0) {
					// Offset by +1 because textureDescriptorSets[0] is default white
					mrSet = textureDescriptorSets[sub.material.metallicRoughnessTextureIndex + 1];
				}
				else {
					mrSet = defaultMrSet;
				}

				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipelineLayout,
					2, // Set 2
					1, &mrSet,
					0, nullptr
				);

				cmd.drawIndexed(
					sub.indexCount,      // number of indices
					instanceBuffer->getInstanceCount(),  // number of instances
					sub.indexOffset,     // first index
					sub.vertexOffset,    // vertex offset (baseVertex)
					0                    // first instance
				);
			}

			cmd.endRendering();


			// Transition from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR
			layoutBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
				.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe)
				.setDstAccessMask(vk::AccessFlagBits2::eNone)
				.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setNewLayout(vk::ImageLayout::ePresentSrcKHR);

			cmd.pipelineBarrier2(depInfo);

			(void)cmd.end();

			// Submit
			vk::CommandBufferSubmitInfo cmdInfo{};
			cmdInfo.setCommandBuffer(cmd);

			vk::SemaphoreSubmitInfo waitInfo{
				imageAvailableSemaphores[currentFrame].get(),
				0,
				vk::PipelineStageFlagBits2::eColorAttachmentOutput
			};

			vk::SemaphoreSubmitInfo signalInfo{
				renderFinishedSemaphores[imageIndex].get(),
				0,
				vk::PipelineStageFlagBits2::eAllCommands
			};

			vk::SubmitInfo2 submit{};
			submit.setCommandBufferInfos(cmdInfo)
				.setWaitSemaphoreInfos(waitInfo)
				.setSignalSemaphoreInfos(signalInfo);

			(void)graphicsQueue.submit2(submit, inFlightFences[currentFrame].get());

			// Present
			vk::PresentInfoKHR present{};
			present.setWaitSemaphores(renderFinishedSemaphores[imageIndex].get())
				.setSwapchains(swapchainHPP)
				.setImageIndices(imageIndex);

			vk::Result presentResult = presentQueue.presentKHR(present);
			if (presentResult != vk::Result::eSuccess && presentResult != vk::Result::eSuboptimalKHR) {
				std::cerr << "Failed to present\n";
			}

			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

			
			time += 0.001f;
		}
		(void)device.waitIdle();

		if (device)
		{
			device.destroyShaderEXT(vertShader);
			device.destroyShaderEXT(fragShader);
			device.destroyPipelineLayout(pipelineLayout);
			device.destroyImageView(depthImageView);
		}
	};

	vertexBuffer.reset();
	indexBuffer.reset();
	instanceBuffer.reset();
	vmaDestroyImage(allocator, depthImage, depthAlloc);
	vmaDestroyAllocator(allocator);
	vkbSwapchain.destroy_image_views(swapchainImageViews);
	vkb::destroy_swapchain(vkbSwapchain);
	vkb::destroy_device(vkbDevice);
	vkb::destroy_surface(vkbInstance, surface);
	vkb::destroy_instance(vkbInstance);
	SDL_DestroyWindow(window);
	SDL_Quit();


	return 0;
}
