#pragma once
#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

// Throws std::runtime_error if the file is missing or not a whole number of 32-bit words.
std::vector<uint32_t> loadSpirv(const std::filesystem::path& path);

struct ShaderPair {
    vk::ShaderEXT vert;
    vk::ShaderEXT frag;
};

// Creates a vertex + fragment shader object pair. Throws std::runtime_error on failure.
// Linked pairs may optimize across stages but must always be bound together; an unlinked vertex
// shader can also be bound without a fragment shader (see bindVertexShaderOnly).
ShaderPair createShaderPair(vk::Device device,
    const std::filesystem::path& vertSpv,
    const std::filesystem::path& fragSpv,
    std::span<const vk::DescriptorSetLayout> setLayouts,
    std::span<const vk::PushConstantRange> pushConstants = {},
    const vk::SpecializationInfo* fragSpecialization = nullptr,
    bool linked = true);

// Creates a compute shader object. Throws std::runtime_error on failure.
vk::ShaderEXT createComputeShader(vk::Device device,
    const std::filesystem::path& compSpv,
    std::span<const vk::DescriptorSetLayout> setLayouts,
    std::span<const vk::PushConstantRange> pushConstants = {});

void destroyShaderPair(vk::Device device, ShaderPair& pair);

inline void bindShaderPair(vk::CommandBuffer cmd, const ShaderPair& pair) {
    const vk::ShaderStageFlagBits stages[] = { vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment };
    const vk::ShaderEXT shaders[] = { pair.vert, pair.frag };
    cmd.bindShadersEXT(2, stages, shaders);
}

// Depth-only rendering: with no fragment shader bound, fragments are only depth tested and written.
inline void bindVertexShaderOnly(vk::CommandBuffer cmd, vk::ShaderEXT vert) {
    const vk::ShaderStageFlagBits stages[] = { vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment };
    const vk::ShaderEXT shaders[] = { vert, nullptr };
    cmd.bindShadersEXT(2, stages, shaders);
}

inline void bindComputeShader(vk::CommandBuffer cmd, vk::ShaderEXT shader) {
    const vk::ShaderStageFlagBits stage = vk::ShaderStageFlagBits::eCompute;
    cmd.bindShadersEXT(1, &stage, &shader);
}
