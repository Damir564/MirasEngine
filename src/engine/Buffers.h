#pragma once
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <string>
#include <vector>
#include "Vertex.h"
#include "ModelInstance.h"

// GPU-only (VRAM) buffer. Contents are filled through a staging upload.
class DeviceBuffer {
public:
	DeviceBuffer(VmaAllocator allocator, vk::DeviceSize size, vk::BufferUsageFlags usage)
		: m_allocator(allocator), m_size(size)
	{
		VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bufferInfo.size = size;
		bufferInfo.usage = static_cast<VkBufferUsageFlags>(usage | vk::BufferUsageFlagBits::eTransferDst);
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkBuffer rawBuffer;
		if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &rawBuffer, &m_allocation, nullptr) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create device buffer");
		}
		m_buffer = vk::Buffer(rawBuffer);
	}

	~DeviceBuffer() {
		if (m_buffer && m_allocation)
			vmaDestroyBuffer(m_allocator, VkBuffer(m_buffer), m_allocation);
	}

	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	vk::Buffer getBuffer() const { return m_buffer; }
	vk::DeviceSize getSize() const { return m_size; }

private:
	VmaAllocator m_allocator;
	vk::Buffer m_buffer;
	VmaAllocation m_allocation = nullptr;
	vk::DeviceSize m_size;
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
