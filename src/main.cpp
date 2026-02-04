#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
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
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"

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

struct FrameUBO {
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 lightSpaceMatrix;
	glm::vec4 cameraPos;
	glm::vec4 lightDir;
	float time;
	float shadowBias;
	float padding[2];  // Pad to 16-byte alignment
};

struct MeshPushConstants {
	glm::vec4 baseColor{ 1.0f, 1.0f, 1.0f, 1.0f };  // 16 bytes
	float metallic{ 0.0f };                          // 4 bytes
	float roughness{ 0.5f };                         // 4 bytes
	float alphaCutoff{ 0.5f };                       // 4 bytes
	int alphaMode{ 0 };                              // 4 bytes
};

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texCoord;
	glm::vec4 tangent;

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
		attributes[3].format = vk::Format::eR32G32B32A32Sfloat;
		attributes[3].offset = offsetof(Vertex, tangent);

		// Instance buffer attribute
		attributes[4].location = locationOffset + 4;             // matches shader
		attributes[4].binding = 1;              // instance buffer binding
		attributes[4].format = vk::Format::eR32G32B32Sfloat; // vec3
		attributes[4].offset = 0;               // offset inside InstanceData struct

		return attributes;
	}
};

constexpr uint32_t SHADOW_MAP_SIZE = 2048;

struct ShadowMapResources {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	vk::ImageView view;
	vk::Sampler sampler;
};

struct DirectionalLight {
	glm::vec3 direction{ -0.5f, -1.0f, -0.3f };
	glm::vec3 color{ 1.0f, 1.0f, 1.0f };
	float intensity{ 1.0f };
};

struct ShadowPushConstants {
	float alphaCutoff;      // 4 bytes
	int alphaMode;          // 4 bytes
	float padding[2];       // 8 bytes for alignment
};

glm::mat4 calculateLightSpaceMatrix(const DirectionalLight& light, const glm::vec3& sceneCenter, float sceneRadius) {
	// Normalize light direction
	glm::vec3 lightDir = glm::normalize(light.direction);

	// Position the light "camera" far enough to see the whole scene
	glm::vec3 lightPos = sceneCenter - lightDir * sceneRadius * 2.0f;

	// Light view matrix (looking at scene center)
	glm::mat4 lightView = glm::lookAt(lightPos, sceneCenter, glm::vec3(0.0f, 1.0f, 0.0f));

	// Orthographic projection for directional light
	// Size based on scene radius
	float orthoSize = sceneRadius * 1.5f;
	glm::mat4 lightProj = glm::ortho(
		-orthoSize, orthoSize,    // left, right
		-orthoSize, orthoSize,    // bottom, top
		0.1f, sceneRadius * 4.0f  // near, far
	);

	// Vulkan clip space fix (Y flip)
	lightProj[1][1] *= -1;

	return lightProj * lightView;
}

struct SceneBounds {
	glm::vec3 center{ 0.0f };
	float radius{ 100.0f };
};

SceneBounds calculateSceneBounds(const std::vector<Vertex>& vertices) {
	if (vertices.empty()) {
		return {};
	}

	glm::vec3 minBounds{ FLT_MAX };
	glm::vec3 maxBounds{ -FLT_MAX };

	for (const auto& v : vertices) {
		minBounds = glm::min(minBounds, v.position);
		maxBounds = glm::max(maxBounds, v.position);
	}

	SceneBounds bounds;
	bounds.center = (minBounds + maxBounds) * 0.5f;
	bounds.radius = glm::length(maxBounds - minBounds) * 0.5f;

	return bounds;
}

ShadowMapResources createShadowMap(VmaAllocator allocator, vk::Device device) {
	ShadowMapResources shadow{};

	// 1. Create Depth Image
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_D32_SFLOAT;
	imageInfo.extent = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 };
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	if (vmaCreateImage(allocator, &imageInfo, &allocInfo,
		&shadow.image, &shadow.allocation, nullptr) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create shadow map image");
	}

	// 2. Create Image View
	vk::ImageViewCreateInfo viewInfo{};
	viewInfo.image = vk::Image(shadow.image);
	viewInfo.viewType = vk::ImageViewType::e2D;
	viewInfo.format = vk::Format::eD32Sfloat;
	viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	shadow.view = device.createImageView(viewInfo).value;

	// 3. Create Shadow Sampler (with depth comparison)
	vk::SamplerCreateInfo samplerInfo{};
	samplerInfo.magFilter = vk::Filter::eLinear;
	samplerInfo.minFilter = vk::Filter::eLinear;
	samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToBorder;
	samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToBorder;
	samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToBorder;
	samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
	samplerInfo.compareEnable = VK_TRUE;
	samplerInfo.compareOp = vk::CompareOp::eLessOrEqual;
	samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;

	shadow.sampler = device.createSampler(samplerInfo).value;

	return shadow;
}

void destroyShadowMap(ShadowMapResources& shadow, VmaAllocator allocator, vk::Device device) {
	if (shadow.sampler) device.destroySampler(shadow.sampler);
	if (shadow.view) device.destroyImageView(shadow.view);
	if (shadow.image) vmaDestroyImage(allocator, shadow.image, shadow.allocation);
}

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

struct UBOBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
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
	// Skip if already loaded
	if (tex.pixels != nullptr) {
		return;
	}

	if (tex.encodedData != nullptr && tex.encodedSize > 0) {
		// Load from Memory (Embedded - for FBX/glTF)
		tex.pixels = stbi_load_from_memory(
			tex.encodedData,
			static_cast<int>(tex.encodedSize),
			&tex.width, &tex.height, &tex.channels, 4);
	}
	else if (!tex.path.empty()) {
		// Load from File (OBJ and external references)

		// Normalize path separators
		std::string normalizedPath = tex.path;
		std::replace(normalizedPath.begin(), normalizedPath.end(), '\\', '/');

		tex.pixels = stbi_load(
			normalizedPath.c_str(),
			&tex.width, &tex.height, &tex.channels, 4);

		// If failed, try some fallback paths
		if (!tex.pixels) {
			// Try lowercase extension
			std::filesystem::path p(normalizedPath);
			std::string ext = p.extension().string();
			std::string lowerExt = ext;
			std::transform(lowerExt.begin(), lowerExt.end(), lowerExt.begin(), ::tolower);

			if (ext != lowerExt) {
				std::string altPath = p.parent_path().string() + "/" +
					p.stem().string() + lowerExt;
				tex.pixels = stbi_load(altPath.c_str(), &tex.width, &tex.height, &tex.channels, 4);
			}

			// Try common image formats
			if (!tex.pixels) {
				std::vector<std::string> extensions = { ".png", ".jpg", ".jpeg", ".tga", ".bmp" };
				std::string basePath = p.parent_path().string() + "/" + p.stem().string();

				for (const auto& tryExt : extensions) {
					std::string altPath = basePath + tryExt;
					tex.pixels = stbi_load(altPath.c_str(), &tex.width, &tex.height, &tex.channels, 4);
					if (tex.pixels) {
						std::cout << "  Found texture at: " << altPath << "\n";
						break;
					}
				}
			}
		}
	}

	if (!tex.pixels) {
		std::cerr << "Texture failed to load: " << (tex.path.empty() ? "Embedded" : tex.path) << "\n";
		// Create a 1x1 magenta fallback
		tex.width = 1;
		tex.height = 1;
		tex.channels = 4;
		tex.pixels = (unsigned char*)malloc(4);
		if (tex.pixels) {
			tex.pixels[0] = 255;  // R
			tex.pixels[1] = 0;    // G
			tex.pixels[2] = 255;  // B
			tex.pixels[3] = 255;  // A
		}
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

enum class AlphaMode : int {
	OPAQUE = 0,    // Fully opaque, ignore alpha
	MASK = 1,      // Alpha testing with cutoff
	BLEND = 2      // Alpha blending (transparency)
};

struct Material {
	glm::vec4 baseColorFactor{ 1.0f };
	float metallicFactor{ 1.0f };
	float roughnessFactor{ 1.0f };
	int baseColorTextureIndex = -1;
	int normalTextureIndex = -1;
	int metallicRoughnessTextureIndex = -1;
	float alphaCutoff{ 0.5f };
	AlphaMode alphaMode{ AlphaMode::OPAQUE };
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

TextureData prepareAssimpTextureInfo(const aiScene* scene, const aiMaterial* mat, const std::string& modelPath, aiTextureType type) {
	TextureData texture{};
	aiString path;

	if (mat->GetTexture(type, 0, &path) == AI_SUCCESS) {
		// 1. Check for Embedded Texture (marked with *)
		const aiTexture* embeddedTex = scene->GetEmbeddedTexture(path.C_Str());
		if (embeddedTex) {
			texture.encodedData = reinterpret_cast<const unsigned char*>(embeddedTex->pcData);

			if (embeddedTex->mHeight == 0) {
				// Compressed (png/jpg) inside the binary
				texture.encodedSize = embeddedTex->mWidth;
			}
			else {
				// Raw ARGB data (less common in modern formats like GLB/FBX but possible)
				// Note: stbi_load_from_memory generally expects file headers (PNG/JPG). 
				// If Assimp returns raw texels, this might need manual copying, 
				// but we will stick to the existing logic's buffer size assumption.
				texture.encodedSize = embeddedTex->mWidth * embeddedTex->mHeight * 4;
			}
		}
		// 2. Load from File
		else {
			std::string fullPath = path.C_Str();
			// Fix path relative to model directory
			std::filesystem::path mPath(modelPath);
			std::filesystem::path tPath(fullPath);

			// If path is not absolute, append it to model directory
			if (!tPath.is_absolute()) {
				texture.path = (mPath.parent_path() / tPath).string();
			}
			else {
				texture.path = fullPath;
			}
		}
	}
	return texture;
}

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
	std::unordered_map<std::string, int>& textureCache, const std::string& path,
	std::mutex& textureCacheMutex) {  // NEW: Added mutex parameter

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
			if (mesh->HasTangentsAndBitangents()) {
				glm::vec3 T = glm::normalize(normalMatrix * glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z));
				glm::vec3 B = glm::normalize(normalMatrix * glm::vec3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z));
				glm::vec3 N = vertex.normal;
				float handedness = (glm::dot(glm::cross(N, T), B) < 0.0f) ? -1.0f : 1.0f;
				vertex.tangent = glm::vec4(T, handedness);
			}
			else {
				vertex.tangent = glm::vec4(0.0f);
			}
			result.vertices.push_back(vertex);
		}

		// 3. Indices
		for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
			const aiFace& face = mesh->mFaces[i];
			if (face.mNumIndices != 3) continue;
			result.indices.push_back(face.mIndices[0]);
			result.indices.push_back(face.mIndices[1]);
			result.indices.push_back(face.mIndices[2]);
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

			float opacity = 1.0f;
			aiGetMaterialFloat(material, AI_MATKEY_OPACITY, &opacity);
			info.material.baseColorFactor.a *= opacity;

			// Check for transparent color
			aiColor4D transparentColor;
			if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_TRANSPARENT, &transparentColor)) {
				float avgTransparency = (transparentColor.r + transparentColor.g + transparentColor.b) / 3.0f;
				if (avgTransparency > 0.01f) {
					info.material.baseColorFactor.a *= (1.0f - avgTransparency);
				}
			}

			// Determine alpha mode
			if (info.material.baseColorFactor.a < 0.99f) {
				info.material.alphaMode = AlphaMode::BLEND;
			}
			else {
				info.material.alphaMode = AlphaMode::OPAQUE;
			}

			// Default to non-metal for FBX
			float metallic = 0.0f;
			float roughness = 0.5f;

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

				// THREAD-SAFE: Lock mutex before accessing textureCache
				std::lock_guard<std::mutex> lock(textureCacheMutex);

				if (textureCache.find(key) != textureCache.end()) {
					info.material.baseColorTextureIndex = textureCache[key];
				}
				else {
					TextureData tex = prepareAssimpTextureInfo(scene, material, path, type);
					if (!tex.path.empty() || tex.encodedData != nullptr) {
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

				// THREAD-SAFE: Lock mutex before accessing textureCache
				std::lock_guard<std::mutex> lock(textureCacheMutex);

				if (textureCache.find(key) != textureCache.end()) {
					info.material.normalTextureIndex = textureCache[key];
				}
				else {
					TextureData tex = prepareAssimpTextureInfo(scene, material, path, aiTextureType_NORMALS);
					if (!tex.path.empty() || tex.encodedData != nullptr) {
						int newIdx = (int)result.textureData.size();
						tex.isLinear = true;
						result.textureData.push_back(tex);
						textureCache[key] = newIdx;
						info.material.normalTextureIndex = newIdx;
					}
				}
			}

			// D. Metallic-Roughness
			if (material->GetTexture(aiTextureType_UNKNOWN, 0, &texPath) == AI_SUCCESS) {
				std::string key = texPath.C_Str();

				// THREAD-SAFE: Lock mutex before accessing textureCache
				std::lock_guard<std::mutex> lock(textureCacheMutex);

				if (textureCache.find(key) != textureCache.end()) {
					info.material.metallicRoughnessTextureIndex = textureCache[key];
				}
				else {
					TextureData tex = prepareAssimpTextureInfo(scene, material, path, aiTextureType_UNKNOWN);
					if (!tex.path.empty() || tex.encodedData != nullptr) {
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

	// Recurse into children - pass the mutex along
	for (unsigned int i = 0; i < node->mNumChildren; ++i) {
		processNode(node->mChildren[i], scene, globalTransform, result, textureCache, path, textureCacheMutex);
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

TextureData prepareObjTextureInfo(const aiMaterial* mat, const std::string& modelPath, aiTextureType type) {
	TextureData texture{};
	aiString texPath;

	if (mat->GetTexture(type, 0, &texPath) == AI_SUCCESS) {
		std::string rawPath = texPath.C_Str();

		// OBJ/MTL files always reference external files, never embedded
		std::filesystem::path modelDir = std::filesystem::path(modelPath).parent_path();
		std::filesystem::path texturePath(rawPath);

		// Handle different path formats in MTL files
		if (texturePath.is_absolute()) {
			texture.path = texturePath.string();
		}
		else {
			// Try direct path first
			std::filesystem::path fullPath = modelDir / texturePath;

			// If not found, try just the filename (common in OBJ exports)
			if (!std::filesystem::exists(fullPath)) {
				fullPath = modelDir / texturePath.filename();
			}

			// Also check common texture subdirectories
			if (!std::filesystem::exists(fullPath)) {
				for (const auto& subdir : { "textures", "Textures", "tex", "maps", "Materials" }) {
					auto tryPath = modelDir / subdir / texturePath.filename();
					if (std::filesystem::exists(tryPath)) {
						fullPath = tryPath;
						break;
					}
				}
			}

			texture.path = fullPath.string();
		}

		// Debug output
		if (!std::filesystem::exists(texture.path)) {
			std::cerr << "Warning: Texture not found: " << texture.path << "\n";
		}
	}

	return texture;
}

void processNodeForObj(aiNode* node, const aiScene* scene, glm::mat4 parentTransform, Mesh& result,
	std::unordered_map<std::string, int>& textureCache, const std::string& path,
	std::mutex& textureCacheMutex) {

	// 1. Transform
	glm::mat4 nodeTransform = aiMatrix4x4ToGlm(node->mTransformation);
	glm::mat4 globalTransform = parentTransform * nodeTransform;
	glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(globalTransform)));

	for (unsigned int m = 0; m < node->mNumMeshes; ++m) {
		const aiMesh* mesh = scene->mMeshes[node->mMeshes[m]];

		uint32_t vertexOffset = static_cast<uint32_t>(result.vertices.size());
		uint32_t indexOffset = static_cast<uint32_t>(result.indices.size());

		SubmeshInfo info{};
		info.vertexOffset = vertexOffset;
		info.indexOffset = indexOffset;
		info.indexCount = mesh->mNumFaces * 3;

		// 2. Vertices
		for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
			Vertex vertex{};
			glm::vec4 pos = globalTransform * glm::vec4(
				mesh->mVertices[i].x,
				mesh->mVertices[i].y,
				mesh->mVertices[i].z,
				1.0f
			);
			vertex.position = glm::vec3(pos);

			if (mesh->HasNormals()) {
				vertex.normal = glm::normalize(normalMatrix * glm::vec3(
					mesh->mNormals[i].x,
					mesh->mNormals[i].y,
					mesh->mNormals[i].z
				));
			}
			else {
				vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
			}

			if (mesh->HasTextureCoords(0)) {
				vertex.texCoord = glm::vec2(
					mesh->mTextureCoords[0][i].x,
					mesh->mTextureCoords[0][i].y
				);
			}
			else {
				vertex.texCoord = glm::vec2(0.0f);
			}

			if (mesh->HasTangentsAndBitangents()) {
				glm::vec3 T = glm::normalize(normalMatrix * glm::vec3(
					mesh->mTangents[i].x,
					mesh->mTangents[i].y,
					mesh->mTangents[i].z
				));
				glm::vec3 B = glm::normalize(normalMatrix * glm::vec3(
					mesh->mBitangents[i].x,
					mesh->mBitangents[i].y,
					mesh->mBitangents[i].z
				));
				glm::vec3 N = vertex.normal;
				float handedness = (glm::dot(glm::cross(N, T), B) < 0.0f) ? -1.0f : 1.0f;
				vertex.tangent = glm::vec4(T, handedness);
			}
			else {
				vertex.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
			}

			result.vertices.push_back(vertex);
		}

		// 3. Indices - Use correct winding order for OBJ
		for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
			const aiFace& face = mesh->mFaces[i];
			if (face.mNumIndices != 3) continue;

			// Standard winding order (CCW)
			result.indices.push_back(face.mIndices[0]);
			result.indices.push_back(face.mIndices[1]);
			result.indices.push_back(face.mIndices[2]);
		}

		// 4. Materials - OBJ/MTL specific handling
		if (mesh->mMaterialIndex < scene->mNumMaterials) {
			aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

			// Get material name for debugging
			aiString matName;
			material->Get(AI_MATKEY_NAME, matName);

			// A. Base Color/Diffuse Color
			aiColor4D diffuseColor(1.0f, 1.0f, 1.0f, 1.0f);
			if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diffuseColor)) {
				info.material.baseColorFactor = glm::vec4(
					diffuseColor.r,
					diffuseColor.g,
					diffuseColor.b,
					1.0f  // FORCE alpha to 1.0, we'll handle transparency separately
				);
			}

			// B. Shininess to Roughness conversion
			float shininess = 0.0f;
			if (AI_SUCCESS == aiGetMaterialFloat(material, AI_MATKEY_SHININESS, &shininess)) {
				info.material.roughnessFactor = 1.0f - glm::clamp(shininess / 1000.0f, 0.0f, 1.0f);
			}
			else {
				info.material.roughnessFactor = 0.5f;
			}

			// C. OBJ doesn't have metallic
			info.material.metallicFactor = 0.0f;

			// D. Diffuse/Albedo Texture
			{
				aiString texPath;
				if (material->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS) {
					std::string key = std::string("diffuse:") + texPath.C_Str();

					std::lock_guard<std::mutex> lock(textureCacheMutex);
					auto it = textureCache.find(key);
					if (it != textureCache.end()) {
						info.material.baseColorTextureIndex = it->second;
					}
					else {
						TextureData tex = prepareObjTextureInfo(material, path, aiTextureType_DIFFUSE);
						if (!tex.path.empty()) {
							int newIdx = static_cast<int>(result.textureData.size());
							tex.isLinear = false;
							result.textureData.push_back(tex);
							textureCache[key] = newIdx;
							info.material.baseColorTextureIndex = newIdx;
						}
					}
				}
			}

			// E. Handle Opacity/Transparency CORRECTLY for OBJ
			float finalOpacity = 1.0f;

			// Method 1: Check 'd' (dissolve/opacity) - this is the correct way
			float opacity = 1.0f;
			if (AI_SUCCESS == aiGetMaterialFloat(material, AI_MATKEY_OPACITY, &opacity)) {
				finalOpacity = opacity;
			}

			// Method 2: Check 'Tr' (transparency) - BUT be careful!
			// Many exporters incorrectly use Tr=1.0 to mean "opaque"
			float transparency = 0.0f;
			if (AI_SUCCESS == aiGetMaterialFloat(material, AI_MATKEY_TRANSPARENCYFACTOR, &transparency)) {
				// Only treat as transparent if Tr is between 0 and 1 (exclusive)
				// Tr=0.0 means opaque, Tr=1.0 is often ALSO used to mean opaque (incorrectly)
				if (transparency > 0.001f && transparency < 0.999f) {
					finalOpacity = 1.0f - transparency;
				}
				// If Tr == 1.0, assume it's the buggy "opaque" usage, keep finalOpacity as is
			}

			// Method 3: Check transparent color
			aiColor4D transparentColor;
			if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_TRANSPARENT, &transparentColor)) {
				float avgTransparency = (transparentColor.r + transparentColor.g + transparentColor.b) / 3.0f;
				if (avgTransparency > 0.01f && avgTransparency < 0.99f) {
					finalOpacity *= (1.0f - avgTransparency);
				}
			}

			// Apply final opacity
			info.material.baseColorFactor.a = finalOpacity;

			// Determine alpha mode
			if (finalOpacity < 0.99f) {
				info.material.alphaMode = AlphaMode::BLEND;
			}
			else {
				info.material.alphaMode = AlphaMode::OPAQUE;
			}

			// F. Normal Map
			{
				aiString texPath;
				aiTextureType normalType = aiTextureType_NORMALS;
				if (material->GetTextureCount(aiTextureType_NORMALS) == 0) {
					normalType = aiTextureType_HEIGHT;
				}

				if (material->GetTexture(normalType, 0, &texPath) == AI_SUCCESS) {
					std::string key = std::string("normal:") + texPath.C_Str();

					std::lock_guard<std::mutex> lock(textureCacheMutex);
					auto it = textureCache.find(key);
					if (it != textureCache.end()) {
						info.material.normalTextureIndex = it->second;
					}
					else {
						TextureData tex = prepareObjTextureInfo(material, path, normalType);
						if (!tex.path.empty()) {
							int newIdx = static_cast<int>(result.textureData.size());
							tex.isLinear = true;
							result.textureData.push_back(tex);
							textureCache[key] = newIdx;
							info.material.normalTextureIndex = newIdx;
						}
					}
				}
			}

			// G. Specular Map (as metallic-roughness approximation)
			{
				aiString texPath;
				if (material->GetTexture(aiTextureType_SPECULAR, 0, &texPath) == AI_SUCCESS) {
					std::string key = std::string("specular:") + texPath.C_Str();

					std::lock_guard<std::mutex> lock(textureCacheMutex);
					auto it = textureCache.find(key);
					if (it != textureCache.end()) {
						info.material.metallicRoughnessTextureIndex = it->second;
					}
					else {
						TextureData tex = prepareObjTextureInfo(material, path, aiTextureType_SPECULAR);
						if (!tex.path.empty()) {
							int newIdx = static_cast<int>(result.textureData.size());
							tex.isLinear = true;
							result.textureData.push_back(tex);
							textureCache[key] = newIdx;
							info.material.metallicRoughnessTextureIndex = newIdx;
						}
					}
				}
			}
		}

		result.submeshes.push_back(info);
	}

	// Recurse into children
	for (unsigned int i = 0; i < node->mNumChildren; ++i) {
		processNodeForObj(node->mChildren[i], scene, globalTransform, result, textureCache, path, textureCacheMutex);
	}
}

Mesh loadWithAssimp(const std::string& path) {
	Assimp::Importer importer;

	// Determine file extension
	std::filesystem::path filePath(path);
	std::string ext = filePath.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	bool isObjFile = (ext == ".obj");

	// Configure import flags
	unsigned int importFlags =
		aiProcess_Triangulate |
		aiProcess_CalcTangentSpace |
		aiProcess_GenSmoothNormals |
		aiProcess_JoinIdenticalVertices;

	// OBJ-specific flags
	if (isObjFile) {
		// Don't flip UVs for OBJ - they typically have correct orientation
		// Add optimization for large OBJ files
		importFlags |= aiProcess_OptimizeMeshes;
		importFlags |= aiProcess_OptimizeGraph;
	}
	else {
		// For other formats (FBX, glTF via Assimp)
		importFlags |= aiProcess_FlipUVs;
		importFlags |= aiProcess_GlobalScale;
	}

	const aiScene* scene = importer.ReadFile(path, importFlags);

	if (!scene || !scene->HasMeshes()) {
		throw std::runtime_error("Failed to load model: " + path + "\nAssimp error: " + importer.GetErrorString());
	}

	std::cout << "Loading with Assimp: " << path << "\n";
	std::cout << "  Meshes: " << scene->mNumMeshes << "\n";
	std::cout << "  Materials: " << scene->mNumMaterials << "\n";
	std::cout << "  Textures (embedded): " << scene->mNumTextures << "\n";

	Mesh result;

	// Pre-calculate total vertices/indices for reservation
	size_t totalVerts = 0;
	size_t totalIndices = 0;
	calculateTotalAssimpVertices(scene->mRootNode, scene, totalVerts, totalIndices);

	result.vertices.reserve(totalVerts);
	result.indices.reserve(totalIndices);

	std::cout << "  Expected vertices: " << totalVerts << ", indices: " << totalIndices << "\n";

	std::unordered_map<std::string, int> textureCache;
	std::mutex textureCacheMutex;

	// Process nodes based on file type
	if (isObjFile) {
		processNodeForObj(scene->mRootNode, scene, glm::mat4(1.0f), result, textureCache, path, textureCacheMutex);
	}
	else {
		// Use original processNode for FBX/glTF
		processNode(scene->mRootNode, scene, glm::mat4(1.0f), result, textureCache, path, textureCacheMutex);
	}

	std::cout << "  Loaded vertices: " << result.vertices.size() << ", indices: " << result.indices.size() << "\n";
	std::cout << "  Submeshes: " << result.submeshes.size() << "\n";
	std::cout << "  Textures to load: " << result.textureData.size() << "\n";

	// Parallel texture decoding
	if (!result.textureData.empty()) {
		std::cout << "Decoding " << result.textureData.size() << " textures in parallel...\n";

		// Use hardware concurrency for thread count
		unsigned int numThreads = std::thread::hardware_concurrency();
		if (numThreads == 0) numThreads = 4;

		std::vector<std::future<void>> futures;
		futures.reserve(result.textureData.size());

		// Track progress
		std::atomic<int> loadedCount{ 0 };
		int totalTextures = static_cast<int>(result.textureData.size());

		for (size_t i = 0; i < result.textureData.size(); ++i) {
			futures.push_back(std::async(std::launch::async, [&result, i, &loadedCount, totalTextures]() {
				TextureData& tex = result.textureData[i];
				decodeTextureParallel(tex);

				int loaded = ++loadedCount;
				if (loaded % 10 == 0 || loaded == totalTextures) {
					std::cout << "  Texture progress: " << loaded << "/" << totalTextures << "\n";
				}
				}));
		}

		// Wait for all texture loads to complete
		for (auto& f : futures) {
			f.wait();
		}

		std::cout << "All textures decoded.\n";
	}

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
			std::vector<glm::vec4> tangents(vCount, glm::vec4(0.0f));

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
						tangents[i] = glm::vec4(v.x(), v.y(), v.z(), v.w());
					});
			}

			// === TRANSFORM AND PUSH ===
			for (size_t i = 0; i < vCount; ++i) {
				Vertex v{};
				v.position = glm::vec3(globalTransform * glm::vec4(positions[i], 1.0f));
				v.normal = glm::normalize(normalMatrix * normals[i]);
				glm::vec3 tXYZ = glm::vec3(tangents[i]);
				glm::vec3 transformedTangent = glm::normalize(normalMatrix * tXYZ);
				v.tangent = glm::vec4(transformedTangent, tangents[i].w);
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

				switch (material.alphaMode) {
				case fastgltf::AlphaMode::Opaque:
					sub.material.alphaMode = AlphaMode::OPAQUE;
					break;
				case fastgltf::AlphaMode::Mask:
					sub.material.alphaMode = AlphaMode::MASK;
					sub.material.alphaCutoff = material.alphaCutoff;
					break;
				case fastgltf::AlphaMode::Blend:
					sub.material.alphaMode = AlphaMode::BLEND;
					break;
				}

				// If base color has alpha < 1.0 and mode is opaque, 
				// consider it as blend mode
				if (sub.material.alphaMode == AlphaMode::OPAQUE &&
					sub.material.baseColorFactor.a < 0.99f) {
					sub.material.alphaMode = AlphaMode::BLEND;
				}

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
						tex.isLinear = true;
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
							tex.isLinear = true;
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

struct RenderSubmesh {
	size_t submeshIndex;
	float distanceToCamera;
	bool isTransparent;
};

// Function to sort submeshes for proper transparency rendering
std::vector<RenderSubmesh> sortSubmeshesForRendering(
	const std::vector<SubmeshInfo>& submeshes,
	const std::vector<Vertex>& vertices,
	const glm::vec3& cameraPos)
{
	std::vector<RenderSubmesh> renderList;
	renderList.reserve(submeshes.size());

	for (size_t i = 0; i < submeshes.size(); ++i) {
		const auto& sub = submeshes[i];

		// Calculate center of submesh (approximate using first vertex)
		glm::vec3 center(0.0f);
		if (sub.vertexOffset < vertices.size()) {
			center = vertices[sub.vertexOffset].position;
		}

		RenderSubmesh rs;
		rs.submeshIndex = i;
		rs.distanceToCamera = glm::length(center - cameraPos);
		rs.isTransparent = (static_cast<int>(sub.material.alphaMode) == 2); // BLEND mode

		renderList.push_back(rs);
	}

	// Sort: opaque first (front-to-back), then transparent (back-to-front)
	std::sort(renderList.begin(), renderList.end(),
		[](const RenderSubmesh& a, const RenderSubmesh& b) {
			// Opaque objects first
			if (a.isTransparent != b.isTransparent) {
				return !a.isTransparent; // opaque comes first
			}
			// For opaque: front-to-back (smaller distance first)
			// For transparent: back-to-front (larger distance first)
			if (a.isTransparent) {
				return a.distanceToCamera > b.distanceToCamera;
			}
			else {
				return a.distanceToCamera < b.distanceToCamera;
			}
		});

	return renderList;
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
		// model = loadWithFastGltf("models/main_sponza/NewSponza_Main_glTF_003.gltf");
		model = loadWithAssimp("models/tomsk_school/tomsk_school.obj");
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

	SceneBounds sceneBounds = calculateSceneBounds(model.vertices);
	std::cout << "Scene center: " << sceneBounds.center.x << ", "
		<< sceneBounds.center.y << ", " << sceneBounds.center.z
		<< " radius: " << sceneBounds.radius << "\n";

	std::cout << "First 5 submesh materials:\n";
	for (int i = 0; i < std::min(5, (int)model.submeshes.size()); i++) {
		auto& m = model.submeshes[i].material;
		std::cout << "  Sub " << i << ": baseColor=("
			<< m.baseColorFactor.r << ", "
			<< m.baseColorFactor.g << ", "
			<< m.baseColorFactor.b << ", "
			<< m.baseColorFactor.a << ") "
			<< "alphaMode=" << static_cast<int>(m.alphaMode)
			<< " cutoff=" << m.alphaCutoff << "\n";
	}

	// Initialize directional light
	DirectionalLight sunLight;
	sunLight.direction = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f));
	sunLight.color = glm::vec3(1.0f, 0.98f, 0.95f);
	sunLight.intensity = 1.0f;

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

	VkPhysicalDeviceVulkan12Features features12{};
	features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	features12.timelineSemaphore = VK_TRUE;
	features12.vulkanMemoryModel = VK_TRUE;
	features12.vulkanMemoryModelDeviceScope = VK_TRUE;
	features12.bufferDeviceAddress = VK_TRUE;
	features12.scalarBlockLayout = VK_TRUE;
	features12.storageBuffer8BitAccess = VK_TRUE;

	VkPhysicalDeviceFeatures coreFeatures{};
	coreFeatures.fragmentStoresAndAtomics = VK_TRUE;
	coreFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;
	coreFeatures.shaderInt64 = VK_TRUE;

	vkb::PhysicalDeviceSelector selector{ vkbInstance };
	auto physRet = selector
		.set_minimum_version(1, 4)
		.add_required_extension(vk::EXTShaderObjectExtensionName)
		.add_required_extension_features(shaderObjectFeatures)
		.set_required_features(coreFeatures)
		.set_required_features_12(features12)
		.set_required_features_13(features13)
		.set_surface(surface)
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

	ShadowMapResources shadowMap;
	try {
		shadowMap = createShadowMap(allocator, device);
		std::cout << "Shadow map created: " << SHADOW_MAP_SIZE << "x" << SHADOW_MAP_SIZE << "\n";
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create shadow map: " << e.what() << "\n";
		return -1;
	}

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

		std::vector<UBOBuffer> frameUBOs(MAX_FRAMES_IN_FLIGHT);

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = sizeof(FrameUBO);
			bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VmaAllocationCreateInfo uboAllocInfo{};
			uboAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
			uboAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VmaAllocationInfo allocationInfo{};
			if (vmaCreateBuffer(allocator, &bufferInfo, &uboAllocInfo,
				&frameUBOs[i].buffer, &frameUBOs[i].allocation,
				&allocationInfo) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create UBO buffer");
			}
			frameUBOs[i].mapped = allocationInfo.pMappedData;
		}
		std::cout << "Created " << MAX_FRAMES_IN_FLIGHT << " UBO buffers\n";

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

		vk::DescriptorSetLayoutBinding uboBinding{};
		uboBinding.binding = 0;
		uboBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
		uboBinding.descriptorCount = 1;
		uboBinding.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
		uboBinding.pImmutableSamplers = nullptr;

		vk::DescriptorSetLayoutCreateInfo uboLayoutInfo{};
		uboLayoutInfo.bindingCount = 1;
		uboLayoutInfo.pBindings = &uboBinding;

		vk::UniqueDescriptorSetLayout uboDescriptorSetLayout = device.createDescriptorSetLayoutUnique(uboLayoutInfo).value;

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
		unsigned char mrPixels[] = { 0, 128, 0, 255 };
		mrTexData.pixels = mrPixels;

		auto defaultMrTexture = std::make_unique<TextureImage>(
			allocator, device, commandPool.get(), graphicsQueue, mrTexData, vk::Format::eR8G8B8A8Unorm
		);

		// 2. Create Descriptor Pool
		// We need 1 set for the default texture + 1 set per loaded texture
		uint32_t totalTextures = 1 + (uint32_t)gpuTextures.size();

		std::vector<vk::DescriptorPoolSize> poolSizes = {
		{ vk::DescriptorType::eCombinedImageSampler, totalTextures + 4 },
		{ vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) }
		};

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = totalTextures + 4 + MAX_FRAMES_IN_FLIGHT;

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

		vk::DescriptorSetAllocateInfo shadowMapAllocInfo{};
		shadowMapAllocInfo.descriptorPool = descriptorPool.get();
		shadowMapAllocInfo.descriptorSetCount = 1;
		shadowMapAllocInfo.pSetLayouts = &descriptorSetLayout.get();

		vk::DescriptorSet shadowMapDescriptorSet = device.allocateDescriptorSets(shadowMapAllocInfo).value[0];

		// Update shadow map descriptor (uses the shadow sampler with depth comparison)
		{
			vk::DescriptorImageInfo shadowImageInfo{};
			shadowImageInfo.imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
			shadowImageInfo.imageView = shadowMap.view;
			shadowImageInfo.sampler = shadowMap.sampler;

			vk::WriteDescriptorSet shadowWrite{};
			shadowWrite.dstSet = shadowMapDescriptorSet;
			shadowWrite.dstBinding = 0;
			shadowWrite.dstArrayElement = 0;
			shadowWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
			shadowWrite.descriptorCount = 1;
			shadowWrite.pImageInfo = &shadowImageInfo;

			device.updateDescriptorSets(1, &shadowWrite, 0, nullptr);
		}

		std::vector<vk::DescriptorSet> uboDescriptorSets(MAX_FRAMES_IN_FLIGHT);

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vk::DescriptorSetAllocateInfo uboAllocInfo{};
			uboAllocInfo.descriptorPool = descriptorPool.get();
			uboAllocInfo.descriptorSetCount = 1;
			uboAllocInfo.pSetLayouts = &uboDescriptorSetLayout.get();

			uboDescriptorSets[i] = device.allocateDescriptorSets(uboAllocInfo).value[0];

			vk::DescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = vk::Buffer(frameUBOs[i].buffer);
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(FrameUBO);

			vk::WriteDescriptorSet descriptorWrite{};
			descriptorWrite.dstSet = uboDescriptorSets[i];
			descriptorWrite.dstBinding = 0;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pBufferInfo = &bufferInfo;

			device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
		}
		std::cout << "UBO descriptor sets created\n";

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
		std::vector<uint32_t> shadowVertCode, shadowFragCode;
		try {
			shadowVertCode = loadSpirv("shaders/shadow.vert.spv");
			shadowFragCode = loadSpirv("shaders/shadow.frag.spv");
			std::cout << "Shadow shaders loaded successfully\n";
		}
		catch (const std::exception& e) {
			std::cerr << "Failed to load shadow shaders: " << e.what() << "\n";
			return -1;
		}
		// ------------------------
		// 16. Create Pipeline Layout
		// ------------------------
		// Push constant range - now only 32 bytes
		vk::PushConstantRange pcRange{};
		pcRange.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
		pcRange.offset = 0;
		pcRange.size = sizeof(MeshPushConstants);

		// Layouts: Set 0 = UBO, Sets 1-4 = Textures
		vk::DescriptorSetLayout layouts[] = {
			uboDescriptorSetLayout.get(),     // Set 0: Frame UBO
			descriptorSetLayout.get(),         // Set 1: Base Color
			descriptorSetLayout.get(),         // Set 2: Normal Map
			descriptorSetLayout.get(),         // Set 3: Metallic-Roughness
			descriptorSetLayout.get()          // Set 4: Shadow Map
		};

		// Main shader create info
		vk::ShaderCreateInfoEXT vertInfo{};
		vertInfo.setStage(vk::ShaderStageFlagBits::eVertex)
			.setNextStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(vertCode.size() * sizeof(vertCode.front()))
			.setPCode(vertCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(&pcRange)
			.setSetLayoutCount(5)
			.setPSetLayouts(layouts);

		vk::ShaderCreateInfoEXT fragInfo{};
		fragInfo.setStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(fragCode.size() * sizeof(fragCode.front()))
			.setPCode(fragCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(&pcRange)
			.setSetLayoutCount(5)
			.setPSetLayouts(layouts);

		vk::ShaderEXT vertShader = device.createShaderEXT(vertInfo).value;
		vk::ShaderEXT fragShader = device.createShaderEXT(fragInfo).value;

		// Shadow push constant range - MUST include both VERTEX and FRAGMENT stages
		vk::PushConstantRange shadowPcRange{};
		shadowPcRange.stageFlags = vk::ShaderStageFlagBits::eFragment;
		shadowPcRange.offset = 0;
		shadowPcRange.size = sizeof(ShadowPushConstants);

		// Shadow layouts: Set 0 = UBO (for lightSpaceMatrix), Set 1 = Base Color (for alpha testing)
		vk::DescriptorSetLayout shadowLayouts[] = {
			uboDescriptorSetLayout.get(),
			descriptorSetLayout.get()
		};

		vk::ShaderCreateInfoEXT shadowVertInfo{};
		shadowVertInfo.setStage(vk::ShaderStageFlagBits::eVertex)
			.setNextStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(shadowVertCode.size() * sizeof(shadowVertCode.front()))
			.setPCode(shadowVertCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(&shadowPcRange)
			.setSetLayoutCount(2)
			.setPSetLayouts(shadowLayouts);

		vk::ShaderCreateInfoEXT shadowFragInfo{};
		shadowFragInfo.setStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(shadowFragCode.size() * sizeof(shadowFragCode.front()))
			.setPCode(shadowFragCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(&shadowPcRange)
			.setSetLayoutCount(2)
			.setPSetLayouts(shadowLayouts);

		vk::ShaderEXT shadowVertShader = device.createShaderEXT(shadowVertInfo).value;
		vk::ShaderEXT shadowFragShader = device.createShaderEXT(shadowFragInfo).value;

		// Shadow pipeline layout
		vk::PipelineLayoutCreateInfo shadowLayoutInfo{};
		shadowLayoutInfo.setPushConstantRangeCount(1);
		shadowLayoutInfo.setPPushConstantRanges(&shadowPcRange);
		shadowLayoutInfo.setSetLayoutCount(2);
		shadowLayoutInfo.setPSetLayouts(shadowLayouts);

		vk::PipelineLayout shadowPipelineLayout = device.createPipelineLayout(shadowLayoutInfo).value;

		// Main pipeline layout
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.setPushConstantRangeCount(1);
		pipelineLayoutInfo.setPPushConstantRanges(&pcRange);
		pipelineLayoutInfo.setSetLayoutCount(5);
		pipelineLayoutInfo.setPSetLayouts(layouts);

		vk::PipelineLayout pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo).value;

		std::cout << "Pipeline layouts created successfully\n";

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

			FrameUBO frameData{};
			frameData.view = getView(camera);
			frameData.proj = getProjection(1280.0f, 720.0f);
			frameData.lightSpaceMatrix = calculateLightSpaceMatrix(sunLight, sceneBounds.center, sceneBounds.radius);
			frameData.cameraPos = glm::vec4(camera.position, 0.0f);
			frameData.lightDir = glm::vec4(sunLight.direction, 0.0f);
			frameData.time = time;
			frameData.shadowBias = 0.005f;

			memcpy(frameUBOs[currentFrame].mapped, &frameData, sizeof(FrameUBO));

			// Record command buffer to clear blue
			vk::CommandBuffer cmd = commandBuffers[currentFrame].get();
			(void)cmd.reset();
			(void)cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

			// Shadow pass
			{
				glm::mat4 lightSpaceMatrix = calculateLightSpaceMatrix(sunLight, sceneBounds.center, sceneBounds.radius);

				// Transition shadow map to depth attachment
				vk::ImageMemoryBarrier2 shadowBarrier{};
				shadowBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
					.setSrcAccessMask(vk::AccessFlagBits2::eShaderRead)
					.setDstStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests)
					.setDstAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
					.setOldLayout(vk::ImageLayout::eUndefined)
					.setNewLayout(vk::ImageLayout::eDepthAttachmentOptimal)
					.setImage(vk::Image(shadowMap.image))
					.setSubresourceRange({ vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 });

				vk::DependencyInfo shadowDepInfo{};
				shadowDepInfo.setImageMemoryBarriers(shadowBarrier);
				cmd.pipelineBarrier2(shadowDepInfo);

				// Shadow depth attachment
				vk::RenderingAttachmentInfo shadowDepthAttachment{};
				shadowDepthAttachment.setImageView(shadowMap.view)
					.setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
					.setLoadOp(vk::AttachmentLoadOp::eClear)
					.setStoreOp(vk::AttachmentStoreOp::eStore)
					.setClearValue(vk::ClearValue(vk::ClearDepthStencilValue{ 1.0f, 0 }));

				vk::RenderingInfo shadowRenderInfo{};
				shadowRenderInfo.setRenderArea({ {0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE} })
					.setLayerCount(1)
					.setColorAttachmentCount(0)
					.setPDepthAttachment(&shadowDepthAttachment);

				cmd.beginRendering(shadowRenderInfo);

				// Bind shadow shaders
				vk::ShaderStageFlagBits shadowStages[] = {
					vk::ShaderStageFlagBits::eVertex,
					vk::ShaderStageFlagBits::eFragment
				};
				vk::ShaderEXT shadowShaders[] = { shadowVertShader, shadowFragShader };
				cmd.bindShadersEXT(2, shadowStages, shadowShaders);

				// Set shadow pass state
				const vk::Viewport shadowViewport{ 0, 0, (float)SHADOW_MAP_SIZE, (float)SHADOW_MAP_SIZE, 0.f, 1.f };
				const vk::Rect2D shadowRect{ {0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE} };

				cmd.setViewportWithCount(1, &shadowViewport);
				cmd.setScissorWithCount(1, &shadowRect);
				cmd.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
				cmd.setRasterizerDiscardEnable(false);
				cmd.setCullMode(vk::CullModeFlagBits::eFront);
				cmd.setFrontFace(vk::FrontFace::eCounterClockwise);
				cmd.setDepthTestEnable(true);
				cmd.setDepthWriteEnable(true);
				cmd.setDepthCompareOp(vk::CompareOp::eLessOrEqual);
				cmd.setDepthBiasEnable(true);
				cmd.setDepthBias(1.25f, 0.0f, 1.75f);
				cmd.setStencilTestEnable(false);
				cmd.setPolygonModeEXT(vk::PolygonMode::eFill);
				cmd.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
				vk::SampleMask shadowMask = ~0u;
				cmd.setSampleMaskEXT(vk::SampleCountFlagBits::e1, &shadowMask);
				cmd.setAlphaToCoverageEnableEXT(VK_FALSE);
				cmd.setPrimitiveRestartEnable(VK_FALSE);

				// Set vertex input
				auto attributesArray = Vertex::getAttributeDescriptions(0);
				vk::VertexInputBindingDescription2EXT instanceBinding{};
				instanceBinding.binding = 1;
				instanceBinding.stride = sizeof(InstanceData);
				instanceBinding.inputRate = vk::VertexInputRate::eInstance;
				instanceBinding.divisor = 1;
				vk::VertexInputBindingDescription2EXT bindingDescs[2] = {
					Vertex::getBindingDescription(0),
					instanceBinding
				};
				cmd.setVertexInputEXT(2, bindingDescs, static_cast<uint32_t>(attributesArray.size()), attributesArray.data());

				// Bind buffers
				vk::DeviceSize offsets[2] = { 0, 0 };
				vk::Buffer buffers[2] = { vertexBuffer->getBuffer(), instanceBuffer->getBuffer() };
				vk::DeviceSize sizes[2] = { sizeof(Vertex) * model.vertices.size(), sizeof(InstanceData) * instances.size() };
				vk::DeviceSize strides[2] = { sizeof(Vertex), sizeof(InstanceData) };
				cmd.bindVertexBuffers2(0, 2, buffers, offsets, sizes, strides);
				cmd.bindIndexBuffer(indexBuffer->getBuffer(), 0, vk::IndexType::eUint32);

				// Bind UBO at set 0
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, shadowPipelineLayout, 0, 1,
					&uboDescriptorSets[currentFrame], 0, nullptr);

				// Draw all non-blend submeshes
				for (const auto& sub : model.submeshes) {
					if (sub.material.alphaMode == AlphaMode::BLEND)
						continue;

					ShadowPushConstants shadowPc{};
					shadowPc.alphaCutoff = sub.material.alphaCutoff;
					shadowPc.alphaMode = static_cast<int>(sub.material.alphaMode);

					cmd.pushConstants(
						shadowPipelineLayout,
						vk::ShaderStageFlagBits::eFragment,
						0,
						sizeof(ShadowPushConstants),
						&shadowPc
					);

					// Bind base color texture for alpha testing
					if (sub.material.alphaMode == AlphaMode::MASK && sub.material.baseColorTextureIndex >= 0) {
						vk::DescriptorSet baseColorSet = textureDescriptorSets[sub.material.baseColorTextureIndex + 1];
						cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, shadowPipelineLayout, 1, 1, &baseColorSet, 0, nullptr);
					}
					else {
						cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, shadowPipelineLayout, 1, 1, &textureDescriptorSets[0], 0, nullptr);
					}

					cmd.drawIndexed(
						sub.indexCount,
						instanceBuffer->getInstanceCount(),
						sub.indexOffset,
						sub.vertexOffset,
						0
					);
				}

				cmd.endRendering();

				// Transition shadow map for shader reading
				shadowBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eLateFragmentTests)
					.setSrcAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
					.setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
					.setDstAccessMask(vk::AccessFlagBits2::eShaderRead)
					.setOldLayout(vk::ImageLayout::eDepthAttachmentOptimal)
					.setNewLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal);

				shadowDepInfo.setImageMemoryBarriers(shadowBarrier);
				cmd.pipelineBarrier2(shadowDepInfo);
			}

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
			glm::mat4 lightSpaceMatrix = calculateLightSpaceMatrix(sunLight, sceneBounds.center, sceneBounds.radius);

			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
				&uboDescriptorSets[currentFrame], 0, nullptr);

			// Sort submeshes for proper transparency rendering
			auto sortedSubmeshes = sortSubmeshesForRendering(model.submeshes, model.vertices, camera.position);
			bool currentlyBlending = false;

			for (const auto& renderSub : sortedSubmeshes) {
				const auto& sub = model.submeshes[renderSub.submeshIndex];

				bool needsBlending = (sub.material.alphaMode == AlphaMode::BLEND);

				if (needsBlending != currentlyBlending) {
					currentlyBlending = needsBlending;

					if (needsBlending) {
						cmd.setColorBlendEnableEXT(0, VK_TRUE);

						vk::ColorBlendEquationEXT blendEquation{};
						blendEquation.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
						blendEquation.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
						blendEquation.colorBlendOp = vk::BlendOp::eAdd;
						blendEquation.srcAlphaBlendFactor = vk::BlendFactor::eOne;
						blendEquation.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
						blendEquation.alphaBlendOp = vk::BlendOp::eAdd;
						cmd.setColorBlendEquationEXT(0, 1, &blendEquation);

						cmd.setDepthWriteEnable(VK_FALSE);
					}
					else {
						cmd.setColorBlendEnableEXT(0, VK_FALSE);
						cmd.setDepthWriteEnable(VK_TRUE);
					}
				}

				MeshPushConstants pc{};
				pc.baseColor = sub.material.baseColorFactor;
				pc.metallic = sub.material.metallicFactor;
				pc.roughness = sub.material.roughnessFactor;
				pc.alphaCutoff = sub.material.alphaCutoff;
				pc.alphaMode = static_cast<int>(sub.material.alphaMode);

				cmd.pushConstants(
					pipelineLayout,
					vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
					0,
					sizeof(MeshPushConstants),
					&pc
				);

				// Bind textures at sets 1-4
				vk::DescriptorSet baseColorSet = (sub.material.baseColorTextureIndex >= 0)
					? textureDescriptorSets[sub.material.baseColorTextureIndex + 1]
					: textureDescriptorSets[0];
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, 1, &baseColorSet, 0, nullptr);

				vk::DescriptorSet normalSet = (sub.material.normalTextureIndex >= 0)
					? textureDescriptorSets[sub.material.normalTextureIndex + 1]
					: defaultNormalSet;
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 2, 1, &normalSet, 0, nullptr);

				vk::DescriptorSet mrSet = (sub.material.metallicRoughnessTextureIndex >= 0)
					? textureDescriptorSets[sub.material.metallicRoughnessTextureIndex + 1]
					: defaultMrSet;
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 3, 1, &mrSet, 0, nullptr);

				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 4, 1, &shadowMapDescriptorSet, 0, nullptr);

				cmd.drawIndexed(
					sub.indexCount,
					instanceBuffer->getInstanceCount(),
					sub.indexOffset,
					sub.vertexOffset,
					0
				);
			}

			if (currentlyBlending) {
				cmd.setColorBlendEnableEXT(0, VK_FALSE);
				cmd.setDepthWriteEnable(VK_TRUE);
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
			device.destroyShaderEXT(shadowVertShader);
			device.destroyShaderEXT(shadowFragShader);
			device.destroyPipelineLayout(shadowPipelineLayout);
			device.destroyPipelineLayout(pipelineLayout);
			device.destroyImageView(depthImageView);
		}
		for (auto& ubo : frameUBOs) {
			if (ubo.buffer) {
				vmaDestroyBuffer(allocator, ubo.buffer, ubo.allocation);
			}
		}
	};

	vertexBuffer.reset();
	indexBuffer.reset();
	instanceBuffer.reset();
	vmaDestroyImage(allocator, depthImage, depthAlloc);
	destroyShadowMap(shadowMap, allocator, device);
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
