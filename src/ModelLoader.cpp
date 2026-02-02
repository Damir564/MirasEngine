#include "ModelLoader.h"
#include "VulkanContext.h"

// Library implementations
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <future>
#include <iostream>
#include <algorithm>

// =============================================================
// Helper Functions (Common)
// =============================================================

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

// =============================================================
// FastGLTF Helper Functions
// =============================================================

namespace GLTF {

    glm::mat4 getTransformMatrix(const fastgltf::Node& node, const glm::mat4& parentMatrix) {
        glm::mat4 localMatrix(1.0f);

        std::visit(fastgltf::visitor{
            [&](const fastgltf::math::fmat4x4& matrix) {
                memcpy(&localMatrix, matrix.data(), sizeof(float) * 16);
            },
            [&](const fastgltf::TRS& trs) {
                glm::vec3 translation(trs.translation[0], trs.translation[1], trs.translation[2]);
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

    TextureData prepareTextureInfo(const fastgltf::Asset& asset, const fastgltf::Image& image, const std::string& modelPath) {
        TextureData texData{};

        std::visit(fastgltf::visitor{
            [&](const fastgltf::sources::URI& uri) {
                std::string pathString = std::string(uri.uri.path());
                texData.path = (std::filesystem::path(modelPath).parent_path() / pathString).string();
            },
            [&](const fastgltf::sources::Array& array) {
                texData.encodedData = reinterpret_cast<const unsigned char*>(array.bytes.data());
                texData.encodedSize = array.bytes.size();
            },
            [&](const fastgltf::sources::BufferView& view) {
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

                auto posAttrIt = primitive.findAttribute("POSITION");
                if (posAttrIt == primitive.attributes.end()) continue;
                auto& posAccessor = asset.accessors[posAttrIt->accessorIndex];

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

                size_t vCount = posAccessor.count;
                std::vector<glm::vec3> positions(vCount);
                std::vector<glm::vec3> normals(vCount, glm::vec3(0.0f));
                std::vector<glm::vec2> texcoords(vCount, glm::vec2(0.0f));
                std::vector<glm::vec4> tangents(vCount, glm::vec4(0.0f));

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

                if (primitive.materialIndex.has_value()) {
                    const auto& material = asset.materials[primitive.materialIndex.value()];
                    auto& pbr = material.pbrData;

                    sub.material.baseColorFactor = glm::vec4(pbr.baseColorFactor.x(), pbr.baseColorFactor.y(), pbr.baseColorFactor.z(), pbr.baseColorFactor.w());
                    sub.material.metallicFactor = pbr.metallicFactor;
                    sub.material.roughnessFactor = pbr.roughnessFactor;

                    auto loadTex = [&](const auto& texInfo, std::string prefix, bool isLinear) -> int {
                        if (!texInfo.has_value()) return -1;

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

} // namespace GLTF

// =============================================================
// Assimp Helper Functions
// =============================================================

namespace OBJ {

    glm::mat4 aiMatrixToGlm(const aiMatrix4x4& from) {
        glm::mat4 to;
        to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
        to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
        to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
        to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
        return to;
    }

    glm::vec3 aiVec3ToGlm(const aiVector3D& v) {
        return glm::vec3(v.x, v.y, v.z);
    }

    glm::vec4 aiColor4ToGlm(const aiColor4D& c) {
        return glm::vec4(c.r, c.g, c.b, c.a);
    }

    TextureData prepareTextureInfo(const std::string& texturePath, bool isLinear = false) {
        TextureData texData{};
        texData.path = texturePath;
        texData.isLinear = isLinear;
        return texData;
    }

    void processMesh(aiMesh* mesh, const aiScene* scene, const glm::mat4& transform,
        MeshData& result, std::unordered_map<std::string, int>& textureCache,
        const std::string& modelDirectory)
    {
        SubmeshInfo sub{};
        sub.vertexOffset = static_cast<uint32_t>(result.vertices.size());
        sub.indexOffset = static_cast<uint32_t>(result.indices.size());

        glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform)));

        for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
            Vertex vertex{};

            glm::vec3 position = aiVec3ToGlm(mesh->mVertices[i]);
            vertex.position = glm::vec3(transform * glm::vec4(position, 1.0f));

            if (mesh->HasNormals()) {
                glm::vec3 normal = aiVec3ToGlm(mesh->mNormals[i]);
                vertex.normal = glm::normalize(normalMatrix * normal);
            }
            else {
                vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            if (mesh->mTextureCoords[0]) {
                vertex.texCoord = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            }
            else {
                vertex.texCoord = glm::vec2(0.0f, 0.0f);
            }

            if (mesh->HasTangentsAndBitangents()) {
                glm::vec3 tangent = aiVec3ToGlm(mesh->mTangents[i]);
                glm::vec3 transformedTangent = glm::normalize(normalMatrix * tangent);

                glm::vec3 bitangent = aiVec3ToGlm(mesh->mBitangents[i]);
                glm::vec3 transformedBitangent = glm::normalize(normalMatrix * bitangent);
                float handedness = glm::dot(glm::cross(vertex.normal, transformedTangent), transformedBitangent) < 0.0f ? -1.0f : 1.0f;

                vertex.tangent = glm::vec4(transformedTangent, handedness);
            }
            else {
                vertex.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }

            result.vertices.push_back(vertex);
        }

        for (uint32_t i = 0; i < mesh->mNumFaces; ++i) {
            aiFace face = mesh->mFaces[i];
            for (uint32_t j = 0; j < face.mNumIndices; ++j) {
                result.indices.push_back(face.mIndices[j]);
            }
        }

        sub.indexCount = static_cast<uint32_t>(mesh->mNumFaces * 3);

        if (mesh->mMaterialIndex >= 0) {
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

            aiColor4D baseColor(1.0f);
            if (AI_SUCCESS == material->Get(AI_MATKEY_COLOR_DIFFUSE, baseColor)) {
                sub.material.baseColorFactor = aiColor4ToGlm(baseColor);
            }
            else {
                sub.material.baseColorFactor = glm::vec4(1.0f);
            }

            float metallic = 0.0f;
            float roughness = 0.5f;

            material->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
            material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);

            sub.material.metallicFactor = metallic;
            sub.material.roughnessFactor = roughness;

            auto loadTexture = [&](aiTextureType type, const std::string& prefix, bool isLinear) -> int {
                if (material->GetTextureCount(type) > 0) {
                    aiString texPath;
                    if (AI_SUCCESS == material->GetTexture(type, 0, &texPath)) {
                        std::string fullPath = (std::filesystem::path(modelDirectory) / texPath.C_Str()).string();
                        std::string key = prefix + ":" + fullPath;

                        if (textureCache.find(key) != textureCache.end()) {
                            return textureCache[key];
                        }

                        TextureData tex = prepareTextureInfo(fullPath, isLinear);
                        int newIdx = (int)result.textureData.size();
                        result.textureData.push_back(tex);
                        textureCache[key] = newIdx;
                        return newIdx;
                    }
                }
                return -1;
                };

            sub.material.baseColorTextureIndex = loadTexture(aiTextureType_DIFFUSE, "base", false);
            sub.material.normalTextureIndex = loadTexture(aiTextureType_NORMALS, "norm", true);

            sub.material.metallicRoughnessTextureIndex = loadTexture(aiTextureType_METALNESS, "mr", true);
            if (sub.material.metallicRoughnessTextureIndex == -1) {
                sub.material.metallicRoughnessTextureIndex = loadTexture(aiTextureType_SHININESS, "mr", true);
            }
        }

        result.submeshes.push_back(sub);
    }

    void processNode(aiNode* node, const aiScene* scene, const glm::mat4& parentTransform,
        MeshData& result, std::unordered_map<std::string, int>& textureCache,
        const std::string& modelDirectory)
    {
        glm::mat4 nodeTransform = aiMatrixToGlm(node->mTransformation);
        glm::mat4 globalTransform = parentTransform * nodeTransform;

        for (uint32_t i = 0; i < node->mNumMeshes; ++i) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            processMesh(mesh, scene, globalTransform, result, textureCache, modelDirectory);
        }

        for (uint32_t i = 0; i < node->mNumChildren; ++i) {
            processNode(node->mChildren[i], scene, globalTransform, result, textureCache, modelDirectory);
        }
    }

} // namespace OBJ

// =============================================================
// Model Class Implementation
// =============================================================

void Model::loadFromFile(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == ".gltf" || extension == ".glb") {
        loadGLTF(path);
    }
    else if (extension == ".obj") {
        loadOBJ(path);
    }
    else {
        throw std::runtime_error("Unsupported file format: " + extension + ". Supported formats: .gltf, .glb, .obj");
    }
}

void Model::loadGLTF(const std::string& path) {
    std::cout << "Loading GLTF model with fastgltf: " << path << "\n";

    fastgltf::Parser parser;
    auto data = fastgltf::MappedGltfFile::FromPath(path);
    if (!data) throw std::runtime_error("Failed to open glTF: " + path);

    auto assetRet = parser.loadGltf(data.get(), std::filesystem::path(path).parent_path(),
        fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers);

    if (auto error = assetRet.error(); error != fastgltf::Error::None) {
        throw std::runtime_error("Failed to parse glTF: " + std::string(fastgltf::getErrorMessage(error)));
    }

    auto& asset = assetRet.get();

    size_t totalVerts = 0;
    for (const auto& mesh : asset.meshes) {
        for (const auto& prim : mesh.primitives) {
            auto it = prim.findAttribute("POSITION");
            if (it != prim.attributes.end()) totalVerts += asset.accessors[it->accessorIndex].count;
        }
    }
    meshData.vertices.reserve(totalVerts);
    meshData.indices.reserve(totalVerts);

    size_t sceneIndex = asset.defaultScene.value_or(0);
    if (!asset.scenes.empty()) {
        const auto& scene = asset.scenes[sceneIndex];
        glm::mat4 rootTransform(1.0f);
        std::unordered_map<std::string, int> texCache;

        for (size_t nodeIndex : scene.nodeIndices) {
            GLTF::processNode(asset, nodeIndex, rootTransform, meshData, texCache, path);
        }
    }

    std::cout << "Loaded GLTF model: " << path << "\n";
    std::cout << "  Vertices: " << meshData.vertices.size() << "\n";
    std::cout << "  Indices: " << meshData.indices.size() << "\n";
    std::cout << "  Submeshes: " << meshData.submeshes.size() << "\n";
    std::cout << "  Textures: " << meshData.textureData.size() << "\n";

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

void Model::loadOBJ(const std::string& path) {
    std::cout << "Loading OBJ model with Assimp: " << path << "\n";

    Assimp::Importer importer;

    unsigned int flags =
        aiProcess_Triangulate |
        aiProcess_GenNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_FlipUVs |
        aiProcess_JoinIdenticalVertices |
        aiProcess_OptimizeMeshes |
        aiProcess_ValidateDataStructure;

    const aiScene* scene = importer.ReadFile(path, flags);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("Failed to load model with Assimp: " + std::string(importer.GetErrorString()));
    }

    std::string modelDirectory = std::filesystem::path(path).parent_path().string();

    size_t totalVerts = 0;
    size_t totalIndices = 0;
    for (uint32_t i = 0; i < scene->mNumMeshes; ++i) {
        totalVerts += scene->mMeshes[i]->mNumVertices;
        totalIndices += scene->mMeshes[i]->mNumFaces * 3;
    }
    meshData.vertices.reserve(totalVerts);
    meshData.indices.reserve(totalIndices);

    glm::mat4 rootTransform(1.0f);
    std::unordered_map<std::string, int> texCache;

    OBJ::processNode(scene->mRootNode, scene, rootTransform, meshData, texCache, modelDirectory);

    std::cout << "Loaded OBJ model: " << path << "\n";
    std::cout << "  Vertices: " << meshData.vertices.size() << "\n";
    std::cout << "  Indices: " << meshData.indices.size() << "\n";
    std::cout << "  Submeshes: " << meshData.submeshes.size() << "\n";
    std::cout << "  Textures: " << meshData.textureData.size() << "\n";

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

    std::cout << "Uploading " << meshData.textureData.size() << " textures to GPU...\n";
    textures.clear();
    for (auto& tex : meshData.textureData) {
        vk::Format fmt = tex.isLinear ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;

        textures.push_back(std::make_unique<VulkanTexture>(ctx, tex, fmt));

        tex.free();
    }
}