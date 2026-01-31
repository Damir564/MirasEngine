#pragma once

#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <memory>
#include <optional>
#include <filesystem>
#include <fstream>
#include <future>
#include <algorithm>

// Vulkan & Windowing
#include <volk.h> // Must be included before vulkan.hpp
#include <vulkan/vulkan.hpp>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vk_mem_alloc.h>
#include <VkBootstrap.h>

// Math
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/quaternion.hpp>

// Helper to load SPIR-V
inline std::vector<uint32_t> loadSpirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open SPIR-V file: " + path.string());
    size_t size = file.tellg();
    if (size % 4 != 0) throw std::runtime_error("Invalid SPIR-V size");
    std::vector<uint32_t> code(size / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);
    return code;
}