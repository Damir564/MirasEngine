#include "Types.h"
#include "stb_image.h" // Needed for free()

glm::mat4 Camera::getView() const {
    glm::vec3 front{
        cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
        sin(glm::radians(pitch)),
        sin(glm::radians(yaw)) * cos(glm::radians(pitch))
    };
    return glm::lookAt(position, position + glm::normalize(front), glm::vec3(0, 1, 0));
}

glm::mat4 Camera::getProjection(float width, float height) const {
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), width / height, 0.1f, 10000.0f);
    proj[1][1] *= -1; // Vulkan clip space fix
    return proj;
}

vk::VertexInputBindingDescription2EXT Vertex::getBindingDescription(uint32_t binding) {
    vk::VertexInputBindingDescription2EXT desc{};
    desc.binding = binding;
    desc.stride = sizeof(Vertex);
    desc.inputRate = vk::VertexInputRate::eVertex;
    desc.divisor = 1;
    return desc;
}

std::array<vk::VertexInputAttributeDescription2EXT, 5> Vertex::getAttributeDescriptions(uint32_t locationOffset) {
    std::array<vk::VertexInputAttributeDescription2EXT, 5> attributes{};
    // Position
    attributes[0].location = locationOffset + 0; attributes[0].binding = 0;
    attributes[0].format = vk::Format::eR32G32B32Sfloat; attributes[0].offset = offsetof(Vertex, position);
    // Normal
    attributes[1].location = locationOffset + 1; attributes[1].binding = 0;
    attributes[1].format = vk::Format::eR32G32B32Sfloat; attributes[1].offset = offsetof(Vertex, normal);
    // TexCoord
    attributes[2].location = locationOffset + 2; attributes[2].binding = 0;
    attributes[2].format = vk::Format::eR32G32Sfloat; attributes[2].offset = offsetof(Vertex, texCoord);
    // Tangent
    attributes[3].location = locationOffset + 3; attributes[3].binding = 0;
    attributes[3].format = vk::Format::eR32G32B32A32Sfloat; attributes[3].offset = offsetof(Vertex, tangent);
    // Instance Data
    attributes[4].location = locationOffset + 4; attributes[4].binding = 1;
    attributes[4].format = vk::Format::eR32G32B32Sfloat; attributes[4].offset = 0;
    return attributes;
}

void TextureData::free() {
    if (pixels) {
        stbi_image_free(pixels);
        pixels = nullptr;
    }
}