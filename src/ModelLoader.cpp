#include "ModelLoader.h"
#include "VulkanContext.h" // Required for VulkanBuffer and VulkanTexture

// Library implementations
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Define this only in one .cpp file (usually main or here if main doesn't have it)
// If main.cpp already defines this, remove the line below.
// #define FASTGLTF_USE_STD_MODULE 0 

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <glm/gtc/matrix_inverse.hpp> // For normal matrix
#include <future>
#include <iostream>

// =============================================================
// Helper Functions (Internal)
// =============================================================

// Helper: Convert glTF transform to GLM Matrix
glm::mat4 getTransformMatrix(const fastgltf::Node& node, const glm::mat4& parentMatrix) {
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

void decodeTextureParallel(TextureData& tex) {
    // If we have raw bytes from memory (GLB or Embedded)
    if (tex.encodedData != nullptr && tex.encodedSize > 0) {
        tex.pixels = stbi_load_from_memory(
            tex.encodedData,
            static_cast<int>(tex.encodedSize),
            &tex.width, &tex.height, &tex.channels, 4);
    }
    // If we have a file path
    else if (!tex.path.empty()) {
        tex.pixels = stbi_load(
            tex.path.c_str(),
            &tex.width, &tex.height, &tex.channels, 4);
    }

    // Fallback if load failed
    if (!tex.pixels) {
        std::cerr << "Texture failed to load: " << (tex.path.empty() ? "Embedded" : tex.path) << "\n";
        // Create a 1x1 magenta fallback
        tex.width = 1; tex.height = 1; tex.channels = 4;
        tex.pixels = (unsigned char*)malloc(4);
        tex.pixels[0] = 255; tex.pixels[1] = 0; tex.pixels[2] = 255; tex.pixels[3] = 255;
    }
}

TextureData prepareTextureInfo(const fastgltf::Asset& asset, const fastgltf::Image& image, const std::string& modelPath) {
    TextureData texData{};

    std::visit(fastgltf::visitor{
        [&](const fastgltf::sources::URI& uri) {
            // Store Path only, decode later
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

// =============================================================
// Node Processing Logic
// =============================================================

void processNode(fastgltf::Asset& asset, size_t nodeIndex, const glm::mat4& parentTransform,
    MeshData& result, std::unordered_map<std::string, int>& textureCache, const std::string& path)
{
    auto& node = asset.nodes[nodeIndex];
    glm::mat4 globalTransform = getTransformMatrix(node, parentTransform);

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

            // === DATA ARRAYS ===
            size_t vCount = posAccessor.count;
            std::vector<glm::vec3> positions(vCount);
            std::vector<glm::vec3> normals(vCount, glm::vec3(0.0f));
            std::vector<glm::vec2> texcoords(vCount, glm::vec2(0.0f));
            std::vector<glm::vec4> tangents(vCount, glm::vec4(0.0f));

            // Read Positions
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, posAccessor,
                [&](fastgltf::math::fvec3 v, size_t i) {
                    positions[i] = glm::vec3(v.x(), v.y(), v.z());
                });

            // Read Normals
            if (auto it = primitive.findAttribute("NORMAL"); it != primitive.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, asset.accessors[it->accessorIndex],
                    [&](fastgltf::math::fvec3 v, size_t i) {
                        normals[i] = glm::vec3(v.x(), v.y(), v.z());
                    });
            }
            // Read UVs
            if (auto it = primitive.findAttribute("TEXCOORD_0"); it != primitive.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(asset, asset.accessors[it->accessorIndex],
                    [&](fastgltf::math::fvec2 v, size_t i) {
                        texcoords[i] = glm::vec2(v.x(), v.y());
                    });
            }
            // Read Tangents
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
                result.indices.push_back(idx);
            }

            // === MATERIALS ===
            if (primitive.materialIndex.has_value()) {
                const auto& material = asset.materials[primitive.materialIndex.value()];
                auto& pbr = material.pbrData;

                sub.material.baseColorFactor = glm::vec4(pbr.baseColorFactor.x(), pbr.baseColorFactor.y(), pbr.baseColorFactor.z(), pbr.baseColorFactor.w());
                sub.material.metallicFactor = pbr.metallicFactor;
                sub.material.roughnessFactor = pbr.roughnessFactor;

                // Helper lambda to load texture
                auto loadTex = [&](const auto& texInfo, std::string prefix, bool isLinear) -> int {
                    if (!texInfo.has_value()) return -1;

                    // access textureIndex generically
                    size_t textureIndex = texInfo.value().textureIndex;

                    if (!asset.textures[textureIndex].imageIndex.has_value()) return -1;

                    size_t imgIdx = asset.textures[textureIndex].imageIndex.value();
                    std::string key = prefix + ":" + std::to_string(imgIdx);

                    if (textureCache.find(key) != textureCache.end()) {
                        return textureCache[key];
                    }

                    TextureData tex = prepareTextureInfo(asset, asset.images[imgIdx], path);
                    tex.isLinear = isLinear;
                    if (!tex.path.empty() || tex.encodedData != nullptr) {
                        int newIdx = (int)result.textureData.size();
                        result.textureData.push_back(tex);
                        textureCache[key] = newIdx;
                        return newIdx;
                    }
                    return -1;
                    };

                sub.material.baseColorTextureIndex = loadTex(pbr.baseColorTexture, "base", false);
                sub.material.normalTextureIndex = loadTex(material.normalTexture, "norm", true);
                sub.material.metallicRoughnessTextureIndex = loadTex(pbr.metallicRoughnessTexture, "mr", true);
            }

            result.submeshes.push_back(sub);
        }
    }

    for (size_t childIndex : node.children) {
        processNode(asset, childIndex, globalTransform, result, textureCache, path);
    }
}

// =============================================================
// Model Class Implementation
// =============================================================

void Model::loadFromFile(const std::string& path) {
    fastgltf::Parser parser;
    auto data = fastgltf::MappedGltfFile::FromPath(path);
    if (!data) throw std::runtime_error("Failed to open glTF: " + path);

    auto assetRet = parser.loadGltf(data.get(), std::filesystem::path(path).parent_path(),
        fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers);

    if (auto error = assetRet.error(); error != fastgltf::Error::None) {
        throw std::runtime_error("Failed to parse glTF: " + std::string(fastgltf::getErrorMessage(error)));
    }

    auto& asset = assetRet.get();

    // Reserve estimation (Optional, but good for performance)
    size_t totalVerts = 0;
    for (const auto& mesh : asset.meshes) {
        for (const auto& prim : mesh.primitives) {
            auto it = prim.findAttribute("POSITION");
            if (it != prim.attributes.end()) totalVerts += asset.accessors[it->accessorIndex].count;
        }
    }
    meshData.vertices.reserve(totalVerts);
    meshData.indices.reserve(totalVerts); // Rough estimate

    // Process Scene
    size_t sceneIndex = asset.defaultScene.value_or(0);
    if (!asset.scenes.empty()) {
        const auto& scene = asset.scenes[sceneIndex];
        glm::mat4 rootTransform(1.0f);
        std::unordered_map<std::string, int> texCache;

        for (size_t nodeIndex : scene.nodeIndices) {
            processNode(asset, nodeIndex, rootTransform, meshData, texCache, path);
        }
    }

    // Parallel Texture Decoding
    if (!meshData.textureData.empty()) {
        std::cout << "Decoding " << meshData.textureData.size() << " textures in parallel...\n";
        std::vector<std::future<void>> futures;
        futures.reserve(meshData.textureData.size());

        for (auto& tex : meshData.textureData) {
            futures.push_back(std::async(std::launch::async, [&tex]() {
                decodeTextureParallel(tex);
                }));
        }

        for (auto& f : futures) {
            f.wait();
        }
    }
}

void Model::uploadToGPU(VulkanContext& ctx) {
    if (meshData.vertices.empty()) return;

    // Upload Geometry
    vertexBuffer = std::make_unique<VulkanBuffer>(
        ctx,
        sizeof(Vertex) * meshData.vertices.size(),
        vk::BufferUsageFlagBits::eVertexBuffer,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    vertexBuffer->upload(meshData.vertices.data(), sizeof(Vertex) * meshData.vertices.size());
    vertexBuffer->setCount((uint32_t)meshData.vertices.size());

    indexBuffer = std::make_unique<VulkanBuffer>(
        ctx,
        sizeof(uint32_t) * meshData.indices.size(),
        vk::BufferUsageFlagBits::eIndexBuffer,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    indexBuffer->upload(meshData.indices.data(), sizeof(uint32_t) * meshData.indices.size());
    indexBuffer->setCount((uint32_t)meshData.indices.size());

    // Upload Textures
    std::cout << "Uploading " << meshData.textureData.size() << " textures to GPU...\n";
    textures.clear();
    for (auto& tex : meshData.textureData) {
        vk::Format fmt = tex.isLinear ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;

        // This assumes VulkanTexture constructor handles:
        // Staging buffer creation -> Map -> Copy -> Unmap -> Image Creation -> Cmd Copy -> Submit -> Wait
        textures.push_back(std::make_unique<VulkanTexture>(ctx, tex, fmt));

        // Free CPU pixels now that they are on GPU
        tex.free();
    }
}