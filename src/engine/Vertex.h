#pragma once

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