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
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <future>
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"
#include "engine/Buffers.h"
#include "engine/Camera.h"
#include "engine/Shadow.h"
#include "engine/ModelTypes.h"
#include "engine/ModelManager.h"
#include "engine/Gizmo.h"
#include "engine/SceneSerializer.h"
#include "engine/CameraAnimation.h"
#include "engine/FileDialog.h"
#include "engine/IfcConverter.h"
#include "engine/IfcLayerInfo.h"
#include "engine/IfcNodeParser.h"
#include "engine/IfcScene.h"
#include "engine/IfcSceneLoader.h"

#define VULKAN_API_VERSION_MAJOR 1
#define VULKAN_API_VERSION_MINOR 3
#define VULKAN_API_VERSION_VK vk::ApiVersion13
#define VULKAN_API_VERSION_CURRENT VK_API_VERSION_1_3

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
	glm::mat4 modelMatrix{ 1.0f };					// 64 bytes
	glm::vec4 baseColor{ 1.0f, 1.0f, 1.0f, 1.0f };  // 16 bytes
	float metallic{ 0.0f };                          // 4 bytes
	float roughness{ 0.5f };                         // 4 bytes
	float alphaCutoff{ 0.5f };                       // 4 bytes
	int alphaMode{ 0 };                              // 4 bytes
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
				[&](const fastgltf::sources::Vector& bufferVector) {
					texData.encodedData = reinterpret_cast<const unsigned char*>(bufferVector.bytes.data() + bufferView.byteOffset);
					texData.encodedSize = bufferView.byteLength;
				},
				// Handle "Array" just in case (older versions or specific configs)
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

struct ModelCacheHeader {
	uint32_t magic = 0x564B4D44; // "VKMD"
	uint32_t version = 2;        // Version 2 (includes textures)
	uint64_t vertexCount;
	uint64_t indexCount;
	uint64_t submeshCount;
	uint64_t textureCount;
	//uint8_t isIfc = 0;
	//uint8_t  reserved[7] = {};
};

class ModelSerializer {
public:
	static bool IsCacheValid(const std::string& sourcePath, const std::string& cachePath) {
		namespace fs = std::filesystem;
		if (!fs::exists(cachePath)) return false;
		if (!fs::exists(sourcePath)) return false;
		return fs::last_write_time(cachePath) > fs::last_write_time(sourcePath);
	}

	static bool SaveToCache(const std::string& cachePath, const Mesh& model) {
		std::ofstream file(cachePath, std::ios::binary);
		if (!file.is_open()) return false;

		ModelCacheHeader header{};
		header.vertexCount = model.vertices.size();
		header.indexCount = model.indices.size();
		header.submeshCount = model.submeshes.size();
		header.textureCount = model.textureData.size();
	/*	if (model.ifcScene)
			header.isIfc = 1;*/

		file.write(reinterpret_cast<const char*>(&header), sizeof(header));

		if (header.vertexCount > 0)
			file.write(reinterpret_cast<const char*>(model.vertices.data()), header.vertexCount * sizeof(Vertex));
		if (header.indexCount > 0)
			file.write(reinterpret_cast<const char*>(model.indices.data()), header.indexCount * sizeof(uint32_t));
		if (header.submeshCount > 0)
			file.write(reinterpret_cast<const char*>(model.submeshes.data()), header.submeshCount * sizeof(SubmeshInfo));

		// Save Textures (Raw Decoded Data)
		// This makes the cache file large, but load time is essentially disk read speed (0 cpu decode)
		for (const auto& tex : model.textureData) {
			file.write(reinterpret_cast<const char*>(&tex.width), sizeof(int));
			file.write(reinterpret_cast<const char*>(&tex.height), sizeof(int));
			file.write(reinterpret_cast<const char*>(&tex.channels), sizeof(int));

			bool linear = tex.isLinear;
			file.write(reinterpret_cast<const char*>(&linear), sizeof(bool));

			// Save raw pixels
			size_t dataSize = tex.width * tex.height * 4; // We force 4 channels in loader
			if (tex.pixels) {
				file.write(reinterpret_cast<const char*>(tex.pixels), dataSize);
			}
			else {
				// Should not happen if decodeTextureParallel ran, but handle gracefully
				std::vector<unsigned char> dummy(dataSize, 255);
				file.write(reinterpret_cast<const char*>(dummy.data()), dataSize);
			}
		}

		file.close();
		return true;
	}

	static bool LoadFromCache(const std::string& cachePath, Mesh& outModel) {
		std::ifstream file(cachePath, std::ios::binary);
		if (!file.is_open()) return false;

		ModelCacheHeader header{};
		file.read(reinterpret_cast<char*>(&header), sizeof(header));

		if (header.magic != 0x564B4D44 || header.version != 2) return false;

		outModel.vertices.resize(header.vertexCount);
		outModel.indices.resize(header.indexCount);
		outModel.submeshes.resize(header.submeshCount);
		outModel.textureData.resize(header.textureCount);

		if (header.vertexCount > 0)
			file.read(reinterpret_cast<char*>(outModel.vertices.data()), header.vertexCount * sizeof(Vertex));
		if (header.indexCount > 0)
			file.read(reinterpret_cast<char*>(outModel.indices.data()), header.indexCount * sizeof(uint32_t));
		if (header.submeshCount > 0)
			file.read(reinterpret_cast<char*>(outModel.submeshes.data()), header.submeshCount * sizeof(SubmeshInfo));

		// Load Textures
		for (size_t i = 0; i < header.textureCount; i++) {
			TextureData& tex = outModel.textureData[i];
			file.read(reinterpret_cast<char*>(&tex.width), sizeof(int));
			file.read(reinterpret_cast<char*>(&tex.height), sizeof(int));
			file.read(reinterpret_cast<char*>(&tex.channels), sizeof(int));

			bool linear;
			file.read(reinterpret_cast<char*>(&linear), sizeof(bool));
			tex.isLinear = linear;

			size_t dataSize = tex.width * tex.height * 4;
			tex.pixels = (unsigned char*)malloc(dataSize); // standard malloc, free later
			tex.fromCache = true;
			file.read(reinterpret_cast<char*>(tex.pixels), dataSize);

			// We don't restore 'path' or 'encodedData' because we have the raw pixels now
		}

		//if (header.isIfc > 0)
		//	outModel.ifcScene = {};

		return true;
	}
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
				reinterpret_cast<const unsigned char*>(array.bytes.data()),
				static_cast<int>(array.bytes.size()),
				&texData.width, &texData.height, &texData.channels, 4);
		},
		[&](const fastgltf::sources::BufferView& view) {
			// 3. Load from BufferView (GLB binary chunk)
			auto& bufferView = asset.bufferViews[view.bufferViewIndex];
			auto& buffer = asset.buffers[bufferView.bufferIndex];

			std::visit(fastgltf::visitor{
				[&](const fastgltf::sources::Array& bufferArray) {
					const unsigned char* data = reinterpret_cast<const unsigned char*>(bufferArray.bytes.data() + bufferView.byteOffset);
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

			// === DATA ARRAYS ===
			size_t vCount = posAccessor.count;
			std::vector<glm::vec3> positions(vCount);
			// Initialize with 0
			std::vector<glm::vec3> normals(vCount, glm::vec3(0.0f));
			std::vector<glm::vec2> texcoords(vCount, glm::vec2(0.0f));
			// Initialize tangents to valid vector to avoid NaN in normalization later
			std::vector<glm::vec4> tangents(vCount, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

			// Load Positions
			fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, posAccessor,
				[&](fastgltf::math::fvec3 v, size_t i) {
					positions[i] = glm::vec3(v.x(), v.y(), v.z());
				});

			// Load Normals (if they exist)
			bool hasNormals = false;
			if (auto it = primitive.findAttribute("NORMAL"); it != primitive.attributes.end()) {
				hasNormals = true;
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, asset.accessors[it->accessorIndex],
					[&](fastgltf::math::fvec3 v, size_t i) {
						normals[i] = glm::vec3(v.x(), v.y(), v.z());
					});
			}

			// === GENERATE NORMALS IF MISSING ===
			if (!hasNormals) {
				// Iterate over triangles to calculate face normals
				for (size_t i = 0; i < localIndices.size(); i += 3) {
					// Protect against out of bounds if index count isn't multiple of 3
					if (i + 2 >= localIndices.size()) break;

					uint32_t i0 = localIndices[i];
					uint32_t i1 = localIndices[i + 1];
					uint32_t i2 = localIndices[i + 2];

					glm::vec3 p0 = positions[i0];
					glm::vec3 p1 = positions[i1];
					glm::vec3 p2 = positions[i2];

					glm::vec3 edge1 = p1 - p0;
					glm::vec3 edge2 = p2 - p0;
					// Cross product gives the normal perpendicular to the face
					glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

					// Accumulate normals (smooth shading approximation)
					normals[i0] += faceNormal;
					normals[i1] += faceNormal;
					normals[i2] += faceNormal;
				}

				// Normalize results
				for (auto& n : normals) {
					if (glm::length(n) > 0.0001f)
						n = glm::normalize(n);
					else
						n = glm::vec3(0.0f, 1.0f, 0.0f); // Fallback
				}
			}

			// Load UVs
			if (auto it = primitive.findAttribute("TEXCOORD_0"); it != primitive.attributes.end()) {
				fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(asset, asset.accessors[it->accessorIndex],
					[&](fastgltf::math::fvec2 v, size_t i) {
						texcoords[i] = glm::vec2(v.x(), v.y());
					});
			}

			// Load Tangents
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

				// Apply Normal Matrix
				v.normal = glm::normalize(normalMatrix * normals[i]);

				// Handle Tangents safely
				glm::vec3 tXYZ = glm::vec3(tangents[i]);
				// Only normalize if length is valid to prevent NaN
				if (glm::length(tXYZ) > 0.0001f) {
					glm::vec3 transformedTangent = glm::normalize(normalMatrix * tXYZ);
					v.tangent = glm::vec4(transformedTangent, tangents[i].w);
				}
				else {
					v.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
				}

				v.texCoord = texcoords[i];
				result.vertices.push_back(v);
			}

			for (uint32_t idx : localIndices) {
				result.indices.push_back(idx);
			}

			// === MATERIALS (Existing logic) ===
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

				if (sub.material.alphaMode == AlphaMode::OPAQUE &&
					sub.material.baseColorFactor.a < 0.99f) {
					sub.material.alphaMode = AlphaMode::BLEND;
				}

				// Texture loading logic (same as your previous code)
				if (pbr.baseColorTexture.has_value() && asset.textures[pbr.baseColorTexture.value().textureIndex].imageIndex.has_value()) {
					size_t imgIdx = asset.textures[pbr.baseColorTexture.value().textureIndex].imageIndex.value();
					std::string key = "base:" + std::to_string(imgIdx);
					if (textureCache.find(key) != textureCache.end()) {
						sub.material.baseColorTextureIndex = textureCache[key];
					}
					else {
						TextureData tex = prepareTextureInfo(asset, asset.images[imgIdx], path);
						if (!tex.path.empty() || tex.encodedData != nullptr) {
							int newIdx = (int)result.textureData.size();
							result.textureData.push_back(tex);
							textureCache[key] = newIdx;
							sub.material.baseColorTextureIndex = newIdx;
						}
					}
				}

				if (material.normalTexture.has_value() && asset.textures[material.normalTexture.value().textureIndex].imageIndex.has_value()) {
					size_t imgIdx = asset.textures[material.normalTexture.value().textureIndex].imageIndex.value();
					std::string key = "norm:" + std::to_string(imgIdx);
					if (textureCache.find(key) != textureCache.end()) {
						sub.material.normalTextureIndex = textureCache[key];
					}
					else {
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
//void processFastGltfNodeTracked(
//	fastgltf::Asset& asset,
//	size_t nodeIndex,
//	const glm::mat4& parentTransform,
//	Mesh& result,
//	std::unordered_map<std::string, int>& textureCache,
//	const std::string& path,
//	std::vector<size_t>& submeshNodeMap)
//{
//	const auto& node = asset.nodes[nodeIndex];
//
//	glm::mat4 localTransform(1.0f);
//	if (auto* trs = std::get_if<fastgltf::TRS>(&node.transform)) {
//		glm::vec3 t(trs->translation[0], trs->translation[1], trs->translation[2]);
//		glm::quat r(trs->rotation[3], trs->rotation[0], trs->rotation[1], trs->rotation[2]);
//		glm::vec3 s(trs->scale[0], trs->scale[1], trs->scale[2]);
//		localTransform = glm::translate(glm::mat4(1.0f), t)
//			* glm::mat4_cast(r)
//			* glm::scale(glm::mat4(1.0f), s);
//	}
//	else if (auto* mat = std::get_if<fastgltf::math::fmat4x4>(&node.transform)) {
//		std::memcpy(&localTransform, mat, sizeof(glm::mat4));
//	}
//
//	glm::mat4 worldTransform = parentTransform * localTransform;
//
//	if (node.meshIndex.has_value()) {
//		size_t before = result.submeshes.size();
//
//		processFastGltfNode(asset, *node.meshIndex, worldTransform, result, textureCache, path);
//
//		for (size_t s = before; s < result.submeshes.size(); ++s) {
//			submeshNodeMap.push_back(nodeIndex);
//		}
//	}
//
//	for (size_t child : node.children) {
//		processFastGltfNodeTracked(asset, child, worldTransform, result,
//			textureCache, path, submeshNodeMap);
//	}
//}

IfcInfo buildIfcLayers(const fastgltf::Asset& asset,
	const std::vector<size_t>& submeshNodeMap) {
	IfcInfo info;
	info.isIfc = true;

	// Group submeshes by IFC type
	std::unordered_map<std::string, std::vector<int>> typeMap;

	for (int si = 0; si < static_cast<int>(submeshNodeMap.size()); ++si) {
		size_t nodeIdx = submeshNodeMap[si];
		const auto& node = asset.nodes[nodeIdx];
		std::string name(node.name.begin(), node.name.end());
		std::string ifcType = extractIfcType(name);
		typeMap[ifcType].push_back(si);
	}

	// Convert to sorted layer list
	for (auto& [typeName, indices] : typeMap) {
		IfcTypeLayer layer;
		layer.typeName = typeName;
		layer.submeshIndices = std::move(indices);
		layer.visible = true;
		info.layers.push_back(std::move(layer));
	}

	// Sort alphabetically
	std::sort(info.layers.begin(), info.layers.end(),
		[](const IfcTypeLayer& a, const IfcTypeLayer& b) {
			return a.typeName < b.typeName;
		});

	std::cout << "[IFC] " << info.layers.size() << " type layers found:\n";
	for (const auto& l : info.layers) {
		std::cout << "  " << l.typeName << " (" << l.submeshIndices.size() << " meshes)\n";
	}

	return info;
}

void buildIfcScene(
	IfcScene& scene,
	const fastgltf::Asset& asset,
	const std::vector<size_t>& submeshNodeMap)
{
	scene.submeshVisibilityCache.assign(submeshNodeMap.size(), false);

	for (std::size_t si = 0; si < submeshNodeMap.size(); ++si) {
		std::size_t nodeIdx = submeshNodeMap[si];
		const auto& node = asset.nodes[nodeIdx];

		std::string guid(node.name.begin(), node.name.end());

		auto it = scene.elements.find(guid);
		if (it != scene.elements.end()) {
			it->second.submeshIndex = si;
			scene.submeshToGuid[si] = guid;

			scene.submeshVisibilityCache[si] = it->second.visible;
		}
	}

	std::vector<std::string> toRemove;
	for (auto& [guid, elem] : scene.elements) {
		if (elem.submeshIndex == std::numeric_limits<std::size_t>::max())
			toRemove.push_back(guid);
	}

	for (auto& guid : toRemove) {
		scene.elements.erase(guid);

		for (auto& [sg, sn] : scene.spatial) {
			auto& ev = sn.elementGuids;
			ev.erase(
				std::remove(ev.begin(), ev.end(), guid),
				ev.end());
		}
	}

	std::size_t matched = scene.submeshToGuid.size();

	std::cout << "[IFC] " << scene.elements.size() << " elements with geometry, "
		<< toRemove.size() << " without geometry removed, "
		<< matched << " submeshes matched\n";
}


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

IfcScene beginLoadIfcScene(const std::string& ifcPath, const std::string& glbPath, const std::string jsonPath) {
	IfcScene ifcScene = {};

	
	if (!loadIfcScene(jsonPath, ifcScene)) {
		std::cerr << "[IFC] Warning: failed to load metadata from " << jsonPath << "\n";
	}
	fastgltf::Parser parser;
	auto gltfFile = fastgltf::MappedGltfFile::FromPath(glbPath);
	if (!gltfFile)
		return ifcScene;

	auto assetRet = parser.loadGltf(gltfFile.get(),
		std::filesystem::path(glbPath).parent_path(), fastgltf::Options::None);

	if (assetRet.error() != fastgltf::Error::None)
		return ifcScene;

	auto& asset = assetRet.get();
	std::vector<size_t> submeshNodeMap;

	std::size_t scn = asset.defaultScene.value_or(0);
	if (!asset.scenes.empty()) {
		std::function<void(size_t)> walk = [&](size_t ni) {
			const auto& n = asset.nodes[ni];
			if (n.meshIndex.has_value()) {
				std::size_t primCount = asset.meshes[*n.meshIndex].primitives.size();
				for (std::size_t p = 0; p < primCount; ++p) {
					submeshNodeMap.push_back(ni);
				}
			}
			for (size_t c : n.children) walk(c);
			};
		for (size_t ni : asset.scenes[scn].nodeIndices)
			walk(ni);
	}

	buildIfcScene(ifcScene, asset, submeshNodeMap);
	

	return ifcScene;
}

Mesh loadIfcModel(const std::string& ifcPath) {
	auto glbOpt = IfcConverter::toGlb(ifcPath);
	if (!glbOpt) throw std::runtime_error("IFC conversion failed to GLB: " + ifcPath);
	std::string glbPath = *glbOpt;

	Mesh result = loadWithFastGltf(glbPath);

	auto jsonOpt = IfcConverter::toJson(ifcPath);
	if (!jsonOpt) throw std::runtime_error("IFC conversion failed to JSON: " + ifcPath);
	std::string jsonPath = *jsonOpt;
	result.ifcScene = beginLoadIfcScene(ifcPath, glbPath, jsonPath);


	std::cout << "[IFC] " << result.vertices.size() << " verts, "
		<< result.ifcScene->elements.size() << " elements, "
		<< result.ifcScene->spatial.size() << " spatial nodes\n";

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
	const glm::vec3& cameraPos,
	const IfcScene* ifcScene = nullptr)
{
	std::vector<RenderSubmesh> renderList;
	renderList.reserve(submeshes.size());

	for (size_t i = 0; i < submeshes.size(); ++i) {
		if (ifcScene && !ifcScene->isSubmeshVisible(i)) 
			continue;

		const auto& sub = submeshes[i];

		RenderSubmesh rs;
		rs.submeshIndex = i;

		rs.distanceToCamera = 0.0;

		rs.isTransparent = (static_cast<int>(sub.material.alphaMode) == 2);

		renderList.push_back(rs);
	}

	auto transparentStart = std::partition(renderList.begin(), renderList.end(),
		[](const RenderSubmesh& rs) {
			return !rs.isTransparent;
		});

	std::sort(transparentStart, renderList.end(),
		[](const RenderSubmesh& a, const RenderSubmesh& b) {
			return a.distanceToCamera > b.distanceToCamera;
		});

	return renderList;
}

Mesh loadModelSmart(const std::string& path) {
	std::string cachePath = path + ".cache";
	Mesh result;
	bool loadedFromCache = false;

	namespace fs = std::filesystem;
	fs::path fsPath = fs::path(path);
	std::string ext = pathToUtf8(fsPath.extension());
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	bool isIfc = (ext == ".ifc");

	// 1. Try Cache
	if (ModelSerializer::IsCacheValid(path, cachePath)) {
		std::cout << "[CACHE] Found valid cache for: " << path << ". Loading... ";
		auto start = std::chrono::high_resolution_clock::now();

		if (ModelSerializer::LoadFromCache(cachePath, result)) {
			if (isIfc) {
				fs::path fileStem = fsPath.stem();
				std::string jsonPath = pathToUtf8((fsPath.parent_path() / "converted" / fileStem)) + ".json";
				std::string glbPath = pathToUtf8((fsPath.parent_path() / "converted" / fileStem)) + ".glb";
				result.ifcScene = beginLoadIfcScene(path, glbPath, jsonPath);
			}
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float, std::milli> duration = end - start;
			std::cout << "Done (" << duration.count() << "ms)\n";
			loadedFromCache = true;
		}
		else {
			std::cout << "Failed (Corruption?)\n";
		}
	}

	if (!loadedFromCache) {
		if (isIfc) {
			result = loadIfcModel(path);
		}
		else if (ext == ".gltf" || ext == ".glb") {
			result = loadWithFastGltf(path);
		}
		else {
			result = loadWithAssimp(path);
		}

		ModelSerializer::SaveToCache(cachePath, result);
	}

	return result;
}

#define SCREEN_WIDTH 1600.0
#define SCREEN_HEIGHT 900.0

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
		SCREEN_WIDTH, SCREEN_HEIGHT,
		SDL_WINDOW_VULKAN
	);

	if (!window) {
		std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
		SDL_Quit();
		return -1;
	}

	// Initialize directional light
	DirectionalLight sunLight;
	sunLight.direction = glm::normalize(glm::vec3(-0.8f, -0.3f, -0.3f));
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
		.require_api_version(VULKAN_API_VERSION_MAJOR, VULKAN_API_VERSION_MINOR, 0)
		.set_minimum_instance_version(VULKAN_API_VERSION_MAJOR, VULKAN_API_VERSION_MINOR)
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
	coreFeatures.wideLines = VK_TRUE;

	vkb::PhysicalDeviceSelector selector{ vkbInstance };
	auto physRet = selector
		.set_minimum_version(VULKAN_API_VERSION_MAJOR, VULKAN_API_VERSION_MINOR)
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
	allocatorInfo.vulkanApiVersion = VULKAN_API_VERSION_VK;

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

	// ------------------------
	// 10. Create Swapchain
	// ------------------------
	vkb::SwapchainBuilder swapchain_builder{ vkbDevice };
	auto swap_ret = swapchain_builder
		.set_desired_extent(SCREEN_WIDTH, SCREEN_HEIGHT)
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
	depthImageInfo.extent = vk::Extent3D{ (int)SCREEN_WIDTH, (int)SCREEN_HEIGHT, 1 };
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

		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
		samplerInfo.anisotropyEnable = VK_FALSE;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

		vk::UniqueSampler textureSampler = device.createSamplerUnique(samplerInfo).value;

		// 2. Texture Descriptor Set Layout
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

		// 3. UBO Descriptor Set Layout
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

		// 4. Descriptor Pool
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			{ vk::DescriptorType::eCombinedImageSampler, 1000 },
			{ vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) + 10 }
		};

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = 1100;

		vk::UniqueDescriptorPool descriptorPool = device.createDescriptorPoolUnique(poolInfo).value;

		// 5. Shadow Map Descriptor Set
		vk::DescriptorSetAllocateInfo shadowMapAllocInfo{};
		shadowMapAllocInfo.descriptorPool = descriptorPool.get();
		shadowMapAllocInfo.descriptorSetCount = 1;
		shadowMapAllocInfo.pSetLayouts = &descriptorSetLayout.get();

		vk::DescriptorSet shadowMapDescriptorSet = device.allocateDescriptorSets(shadowMapAllocInfo).value[0];

		// Update shadow map descriptor
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

		// 6. UBO Descriptor Sets
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
		shadowPcRange.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
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
		std::unique_ptr<ModelManager> modelManager;
		try {
			modelManager = std::make_unique<ModelManager>(
				allocator, device, commandPool.get(), graphicsQueue,
				descriptorPool.get(), descriptorSetLayout.get(), textureSampler.get()
			);
			std::cout << "ModelManager created successfully\n";
		}
		catch (const std::exception& e) {
			std::cerr << "Failed to create ModelManager: " << e.what() << "\n";
			return -1;
		}

		// Track current scene bounds (updated when models change)
		SceneBounds sceneBounds;
		sceneBounds.center = glm::vec3(0.0f);
		sceneBounds.radius = 100.0f;  // Default radius

		// ========================
// GIZMO SETUP
// ========================
		Gizmo gizmo;

		// Load gizmo shaders
		std::vector<uint32_t> gizmoVertCode = loadSpirv("shaders/gizmo.vert.spv");
		std::vector<uint32_t> gizmoFragCode = loadSpirv("shaders/gizmo.frag.spv");

		// Gizmo push constant (just model matrix)
		vk::PushConstantRange gizmoPcRange{};
		gizmoPcRange.stageFlags = vk::ShaderStageFlagBits::eVertex;
		gizmoPcRange.offset = 0;
		gizmoPcRange.size = sizeof(glm::mat4);

		vk::DescriptorSetLayout gizmoLayouts[] = { uboDescriptorSetLayout.get() };

		vk::ShaderCreateInfoEXT gizmoVertInfo{};
		gizmoVertInfo.setStage(vk::ShaderStageFlagBits::eVertex)
			.setNextStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(gizmoVertCode.size() * sizeof(uint32_t))
			.setPCode(gizmoVertCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(&gizmoPcRange)
			.setSetLayoutCount(1)
			.setPSetLayouts(gizmoLayouts);

		vk::ShaderCreateInfoEXT gizmoFragInfo{};
		gizmoFragInfo.setStage(vk::ShaderStageFlagBits::eFragment)
			.setFlags(vk::ShaderCreateFlagBitsEXT::eLinkStage)
			.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
			.setCodeSize(gizmoFragCode.size() * sizeof(uint32_t))
			.setPCode(gizmoFragCode.data())
			.setPName("main")
			.setPushConstantRangeCount(1)
			.setPPushConstantRanges(&gizmoPcRange)
			.setSetLayoutCount(1)
			.setPSetLayouts(gizmoLayouts);

		vk::ShaderEXT gizmoVertShader = device.createShaderEXT(gizmoVertInfo).value;
		vk::ShaderEXT gizmoFragShader = device.createShaderEXT(gizmoFragInfo).value;

		vk::PipelineLayoutCreateInfo gizmoLayoutInfo{};
		gizmoLayoutInfo.setPushConstantRangeCount(1);
		gizmoLayoutInfo.setPPushConstantRanges(&gizmoPcRange);
		gizmoLayoutInfo.setSetLayoutCount(1);
		gizmoLayoutInfo.setPSetLayouts(gizmoLayouts);

		vk::PipelineLayout gizmoPipelineLayout = device.createPipelineLayout(gizmoLayoutInfo).value;

		// Create gizmo vertex buffers
		auto translateLines = generateTranslateGizmoLines(2.0f);
		auto rotateLines = generateRotateGizmoLines(32, 1.5f);
		auto scaleLines = generateScaleGizmoLines(2.0f);

		auto createGizmoBuffer = [&](const std::vector<GizmoVertex>& data) -> std::pair<vk::Buffer, VmaAllocation> {
			VkBufferCreateInfo bufInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
			bufInfo.size = sizeof(GizmoVertex) * data.size();
			bufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
			VmaAllocationCreateInfo allocCI{};
			allocCI.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
			allocCI.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

			VkBuffer raw;
			VmaAllocation alloc;
			vmaCreateBuffer(allocator, &bufInfo, &allocCI, &raw, &alloc, nullptr);
			void* mapped;
			vmaMapMemory(allocator, alloc, &mapped);
			memcpy(mapped, data.data(), bufInfo.size);
			vmaUnmapMemory(allocator, alloc);
			return { vk::Buffer(raw), alloc };
			};

		auto [translateBuffer, translateAlloc] = createGizmoBuffer(translateLines);
		auto [rotateBuffer, rotateAlloc] = createGizmoBuffer(rotateLines);
		auto [scaleBuffer, scaleAlloc] = createGizmoBuffer(scaleLines);

		std::cout << "Gizmo system initialized\n";

		// ========================
// CAMERA ANIMATION SETUP
// ========================
		CameraAnimator cameraAnimator;

		// Path visualization buffer (reused each frame when path is visible)
		VkBuffer pathVisualizationBuffer = VK_NULL_HANDLE;
		VmaAllocation pathVisualizationAlloc = VK_NULL_HANDLE;
		uint32_t pathVisualizationVertexCount = 0;
		bool showAnimationPath = true;

		// Helper to rebuild path visualization buffer
		auto rebuildPathVisualization = [&]() {
			// Destroy old buffer
			if (pathVisualizationBuffer != VK_NULL_HANDLE) {
				vmaDestroyBuffer(allocator, pathVisualizationBuffer, pathVisualizationAlloc);
				pathVisualizationBuffer = VK_NULL_HANDLE;
				pathVisualizationAlloc = VK_NULL_HANDLE;
				pathVisualizationVertexCount = 0;
			}

			auto pathPoints = cameraAnimator.generatePathPoints(30);
			if (pathPoints.size() < 2) return;

			// Convert to line segments (pairs of vertices)
			std::vector<GizmoVertex> lineVerts;
			lineVerts.reserve((pathPoints.size() - 1) * 2 + cameraAnimator.getPath().keyframes.size() * 6);

			// Path lines (yellow)
			for (size_t i = 0; i < pathPoints.size() - 1; ++i) {
				lineVerts.push_back({ pathPoints[i], glm::vec3(1.0f, 1.0f, 0.0f) });
				lineVerts.push_back({ pathPoints[i + 1], glm::vec3(1.0f, 1.0f, 0.0f) });
			}

			// Keyframe markers (small crosses, cyan for straight, magenta for curved)
			const auto& keyframes = cameraAnimator.getPath().keyframes;
			float crossSize = 0.3f;
			for (const auto& kf : keyframes) {
				glm::vec3 color = kf.useCurve ? glm::vec3(1.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 1.0f);
				// X cross
				lineVerts.push_back({ kf.position - glm::vec3(crossSize, 0, 0), color });
				lineVerts.push_back({ kf.position + glm::vec3(crossSize, 0, 0), color });
				// Y cross
				lineVerts.push_back({ kf.position - glm::vec3(0, crossSize, 0), color });
				lineVerts.push_back({ kf.position + glm::vec3(0, crossSize, 0), color });
				// Z cross
				lineVerts.push_back({ kf.position - glm::vec3(0, 0, crossSize), color });
				lineVerts.push_back({ kf.position + glm::vec3(0, 0, crossSize), color });

				// Look direction line (green, short)
				glm::vec3 lookDir = glm::normalize(kf.lookTarget - kf.position);
				lineVerts.push_back({ kf.position, glm::vec3(0.0f, 1.0f, 0.0f) });
				lineVerts.push_back({ kf.position + lookDir * 1.0f, glm::vec3(0.0f, 1.0f, 0.0f) });
			}

			if (lineVerts.empty()) return;

			VkBufferCreateInfo bufInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
			bufInfo.size = sizeof(GizmoVertex) * lineVerts.size();
			bufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
			VmaAllocationCreateInfo allocCI{};
			allocCI.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
			allocCI.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

			vmaCreateBuffer(allocator, &bufInfo, &allocCI, &pathVisualizationBuffer, &pathVisualizationAlloc, nullptr);
			void* mapped;
			vmaMapMemory(allocator, pathVisualizationAlloc, &mapped);
			memcpy(mapped, lineVerts.data(), bufInfo.size);
			vmaUnmapMemory(allocator, pathVisualizationAlloc);

			pathVisualizationVertexCount = static_cast<uint32_t>(lineVerts.size());
			};

		std::cout << "Camera animation system initialized\n";

		// ============================================
		// IMGUI SETUP - For ImGui v1.92.5-docking
		// ============================================

		// Create descriptor pool for ImGui
		VkDescriptorPoolSize imguiPoolSizes[] = {
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }
		};

		VkDescriptorPoolCreateInfo imguiPoolInfo = {};
		imguiPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		imguiPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		imguiPoolInfo.maxSets = 1;
		imguiPoolInfo.poolSizeCount = 1;
		imguiPoolInfo.pPoolSizes = imguiPoolSizes;

		VkDescriptorPool imguiDescriptorPool;
		if (vkCreateDescriptorPool(vkDevice, &imguiPoolInfo, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create ImGui descriptor pool");
		}

		// Initialize ImGui
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.Fonts->AddFontFromFileTTF(
			"C:/Windows/Fonts/arial.ttf",
			18.0f,
			nullptr,
			io.Fonts->GetGlyphRangesCyrillic()
		);
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

		ImGui::StyleColorsDark();

		// Setup Platform/Renderer backends
		ImGui_ImplSDL3_InitForVulkan(window);

		// Setup PipelineRenderingCreateInfo for dynamic rendering
		VkFormat colorFormat = VK_FORMAT_B8G8R8A8_SRGB;

		VkPipelineRenderingCreateInfoKHR pipelineRenderingInfo = {};
		pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
		pipelineRenderingInfo.colorAttachmentCount = 1;
		pipelineRenderingInfo.pColorAttachmentFormats = &colorFormat;
		pipelineRenderingInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;  // No depth for ImGui overlay
		pipelineRenderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		// Zero-initialize as required
		ImGui_ImplVulkan_InitInfo initInfo = {};
		initInfo.ApiVersion = VULKAN_API_VERSION_CURRENT;  // Or VK_API_VERSION_1_3
		initInfo.Instance = vkInstance;
		initInfo.PhysicalDevice = vkbPhys.physical_device;
		initInfo.Device = vkDevice;
		initInfo.QueueFamily = graphicsQueueIndex;
		initInfo.Queue = VkQueue(graphicsQueue);
		initInfo.DescriptorPool = imguiDescriptorPool;
		initInfo.MinImageCount = swapchainImageCount;
		initInfo.ImageCount = swapchainImageCount;

		// Dynamic rendering setup
		initInfo.UseDynamicRendering = true;
		initInfo.PipelineInfoMain.RenderPass = VK_NULL_HANDLE;
		initInfo.PipelineInfoMain.Subpass = 0;
		initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineRenderingInfo;

		// VkInstance inst = vkInstance;
		auto testFn = vkGetInstanceProcAddr(vkInstance, "vkCmdBeginRendering");
		std::cout << "vkCmdBeginRendering address: " << (void*)testFn << std::endl;

		if (testFn == nullptr) {
			std::cerr << "ERROR: vkGetInstanceProcAddr returns null for vkCmdBeginRendering!\n";
		}

		ImGui_ImplVulkan_LoadFunctions(VULKAN_API_VERSION_CURRENT, [](const char* function_name, void* user_data) {
			return vkGetInstanceProcAddr((VkInstance)user_data, function_name);
			}, vkInstance);

		ImGui_ImplVulkan_Init(&initInfo);

		std::cout << "ImGui v1.92.5-docking initialized successfully!\n";

		// ------------------------
		// 8. Main loop
		// ------------------------
		Camera camera;
		bool mouseEnabled = false;
		bool rightMouseHeld = false;
		SDL_SetWindowRelativeMouseMode(window, mouseEnabled);
		bool running = true;
		static char sceneSavePath[512] = "tomsk_school1.scn";
		static char sceneLoadPath[512] = "tomsk_school1.scn";
		bool pendingSceneLoad = false;
		SceneSerializer::LoadedScene pendingScene;
		SDL_Event event;
		uint32_t currentFrame = 0;
		MeshPushConstants pc{};
		float cameraAmpilfier = 1.0f;
		// Update push constant for animation
		static float time = 0.0f;
		// Push constant data
		uint32_t lastTime = SDL_GetTicks();

		bool showPropertiesWindow = false;
		enum class IfcSelectionKind {
			kNone,
			kElement,
			kSpatial
		};
		IfcSelectionKind selectedIfcKind = IfcSelectionKind::kNone;
		std::string selectedIfcGuid;
		int selectedInstanceForProperties = -1;
		while (running) {
			uint32_t currentTime = SDL_GetTicks();
			float dt = (currentTime - lastTime) / 1000.0f; // convert ms to seconds
			lastTime = currentTime;
			while (SDL_PollEvent(&event)) {
				// Let ImGui process events FIRST
				if (!mouseEnabled)
					ImGui_ImplSDL3_ProcessEvent(&event);

				// Get IO to check if ImGui wants input
				ImGuiIO& imguiIO = ImGui::GetIO();

				bool shiftHeld =
					(event.key.mod & SDL_KMOD_LSHIFT) ||
					(event.key.mod & SDL_KMOD_RSHIFT);
				const bool* keys = SDL_GetKeyboardState(nullptr);

				if (event.type == SDL_EVENT_QUIT)
					running = false;

				// Only handle keyboard if ImGui doesn't want it
				if (mouseEnabled || !imguiIO.WantCaptureKeyboard) {
					if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
						if (event.key.scancode == SDL_SCANCODE_ESCAPE)
							running = false;

						if (shiftHeld) {
							cameraAmpilfier = 4.0f;
							if (event.key.scancode == SDL_SCANCODE_GRAVE) {
								mouseEnabled = !mouseEnabled;
								SDL_SetWindowRelativeMouseMode(window, mouseEnabled);
							}
						}
						if (gizmo.selectedInstance >= 0) {
							const auto& insts = modelManager->getInstances();
							if (gizmo.selectedInstance >= static_cast<int>(insts.size())) {
								gizmo.deselect();
							}
							else {
								// Also check if the model is still valid
								GPUModel* model = modelManager->getModel(insts[gizmo.selectedInstance].modelIndex);
								if (!model || !model->isValid()) {
									gizmo.deselect();
								}
							}
						}
						// ========== GIZMO MODE KEYS ==========
						if (!mouseEnabled && gizmo.selectedInstance >= 0) {
							if (event.key.scancode == SDL_SCANCODE_1) {
								gizmo.mode = GizmoMode::Translate;
							}
							if (event.key.scancode == SDL_SCANCODE_2) {
								gizmo.mode = GizmoMode::Rotate;
							}
							if (event.key.scancode == SDL_SCANCODE_3) {
								gizmo.mode = GizmoMode::Scale;
							}
							if (event.key.scancode == SDL_SCANCODE_DELETE) {
								int toDelete = gizmo.selectedInstance;
								gizmo.deselect();
								modelManager->removeInstance(gizmo.selectedInstance);
							}
						}
						// Deselect with Escape (when GUI visible)
						if (!mouseEnabled && event.key.scancode == SDL_SCANCODE_ESCAPE) {
							if (gizmo.selectedInstance >= 0) {
								gizmo.deselect();
								running = true; // Override the quit from escape
							}
						}
					}
					if (event.type == SDL_EVENT_KEY_UP && !event.key.repeat) {
						if (!shiftHeld)
							cameraAmpilfier = 1.0f;
					}
				}
				if (!mouseEnabled) {
					if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT 
						&& !ImGui::GetIO().WantCaptureMouse) {
						float mx = event.button.x;
						float my = event.button.y;

						glm::mat4 viewMat = getView(camera);
						glm::mat4 projMat = getProjection(SCREEN_WIDTH, SCREEN_HEIGHT);
						Ray ray = screenToWorldRay(mx, my, SCREEN_WIDTH, SCREEN_HEIGHT, viewMat, projMat);

						bool clickedOnGizmo = false;

						// First: try clicking on gizmo axis
						if (gizmo.selectedInstance >= 0 && gizmo.mode != GizmoMode::None && gizmo.selectedInstance < static_cast<int>(modelManager->getInstances().size())) {

							auto& inst = modelManager->getInstances()[gizmo.selectedInstance];
							float gizmoScaleVal = getGizmoScale(inst.position, camera.position,
								0.15f, projMat);

							GizmoAxis hitAxis = GizmoAxis::None;

							if (gizmo.mode == GizmoMode::Rotate) {
								hitAxis = pickRotateGizmoAxis(
									glm::vec2(mx, my),
									inst.position,
									gizmoScaleVal,
									15.0f,
									viewMat, projMat,
									SCREEN_WIDTH, SCREEN_HEIGHT);
							}
							else {
								hitAxis = pickGizmoAxis(
									glm::vec2(mx, my),
									inst.position,
									gizmoScaleVal,
									20.0f,
									viewMat, projMat,
									SCREEN_WIDTH, SCREEN_HEIGHT);
							}

							if (hitAxis != GizmoAxis::None) {
								gizmo.activeAxis = hitAxis;
								gizmo.isDragging = true;
								gizmo.dragStart = glm::vec2(mx, my);
								gizmo.originalPosition = inst.position;
								gizmo.originalRotation = inst.rotation;
								gizmo.originalScale = inst.scale;
								clickedOnGizmo = true;
							}
						}

						// Second: if not clicking gizmo, try picking a model
						if (!clickedOnGizmo) {
							float bestT = 1e30f;
							int bestInstance = -1;
							const auto& instances = modelManager->getInstances();

							for (size_t i = 0; i < instances.size(); ++i) {
								if (!instances[i].visible) continue;
								GPUModel* model = modelManager->getModel(instances[i].modelIndex);
								if (!model) continue;

								glm::mat4 instanceTransform = instances[i].getTransformMatrix();

								float t;
								if (rayIntersectsTransformedAABB(ray,
									model->boundsMin, model->boundsMax,
									instanceTransform, t))
								{
									if (t < bestT) {
										bestT = t;
										bestInstance = static_cast<int>(i);
									}
								}
							}

							if (bestInstance >= 0) {
								gizmo.select(bestInstance);
							}
							else {
								gizmo.deselect();
							}
						}
					}

					if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_LEFT) {
						gizmo.isDragging = false;
						gizmo.activeAxis = GizmoAxis::None;
					}

					if (event.type == SDL_EVENT_MOUSE_MOTION && gizmo.isDragging &&
						gizmo.selectedInstance >= 0 &&
						gizmo.selectedInstance < static_cast<int>(modelManager->getInstances().size())) {

						auto& inst = modelManager->getInstances()[gizmo.selectedInstance];
						glm::vec2 currentMouse(event.motion.x, event.motion.y);
						glm::vec2 delta = currentMouse - gizmo.dragStart;

						glm::mat4 viewMat = getView(camera);
						glm::mat4 projMat = getProjection(SCREEN_WIDTH, SCREEN_HEIGHT);
						glm::mat4 vp = projMat * viewMat;

						// Determine axis direction
						glm::vec3 axisMask(0.0f);
						if (gizmo.activeAxis == GizmoAxis::X) axisMask = glm::vec3(1, 0, 0);
						if (gizmo.activeAxis == GizmoAxis::Y) axisMask = glm::vec3(0, 1, 0);
						if (gizmo.activeAxis == GizmoAxis::Z) axisMask = glm::vec3(0, 0, 1);

						// Project the axis direction to screen space to find the best mouse direction
						glm::vec2 pixelCenter = worldToScreen(gizmo.originalPosition, vp, SCREEN_WIDTH, SCREEN_HEIGHT);
						glm::vec2 pixelAxisEnd = worldToScreen(gizmo.originalPosition + axisMask, vp, SCREEN_WIDTH, SCREEN_HEIGHT);

						glm::vec2 screenAxisDir = pixelAxisEnd - pixelCenter;
						float screenAxisLen = glm::length(screenAxisDir);

						if (screenAxisLen < 1.0f) {
							// Axis is pointing directly at camera, use fallback
							screenAxisDir = glm::vec2(1.0f, 0.0f);
							screenAxisLen = 1.0f;
						}
						screenAxisDir /= screenAxisLen;

						// Project mouse delta onto screen-space axis direction
						float mouseDotAxis = glm::dot(delta, screenAxisDir);

						// Convert pixel movement to world units:
						// screenAxisLen pixels = 1 world unit at the object's depth
						float worldUnitsPerPixel = 1.0f / screenAxisLen;
						float worldMovement = mouseDotAxis * worldUnitsPerPixel;

						switch (gizmo.mode) {
						case GizmoMode::Translate: {
							inst.position = gizmo.originalPosition + axisMask * worldMovement;
							break;
						}
						case GizmoMode::Rotate: {
							// Rotation: use a fixed angular speed per pixel
							float degreesPerPixel = 0.5f;
							float rotationDegrees = mouseDotAxis * degreesPerPixel;
							inst.rotation = gizmo.originalRotation + axisMask * rotationDegrees;
							break;
						}
						case GizmoMode::Scale: {
							// Scale: proportional to mouse movement
							float scalePerPixel = 0.01f;
							float scaleChange = mouseDotAxis * scalePerPixel;
							inst.scale = glm::max(gizmo.originalScale + axisMask * scaleChange, glm::vec3(0.01f));
							break;
						}
						default: break;
						}
					}
				}

				// Only handle mouse if ImGui doesn't want it
				if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_RIGHT) {
					if (!mouseEnabled && !imguiIO.WantCaptureMouse) {
						rightMouseHeld = true;
					}
				}
				if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_RIGHT) {
					rightMouseHeld = false;
				}

				// Camera rotation
				// if (event.type == SDL_EVENT_MOUSE_MOTION) {
				if (event.type == SDL_EVENT_MOUSE_MOTION && !cameraAnimator.isPlaying()) {
					if (mouseEnabled) {
						// Camera mode: always rotate
						camera.yaw += event.motion.xrel * camera.sensitivity;
						camera.pitch -= event.motion.yrel * camera.sensitivity;
						camera.pitch = glm::clamp(camera.pitch, -89.0f, 89.0f);
					}
					else if (rightMouseHeld) {
						// GUI mode: rotate while holding right mouse button
						camera.yaw += event.motion.xrel * camera.sensitivity;
						camera.pitch -= event.motion.yrel * camera.sensitivity;
						camera.pitch = glm::clamp(camera.pitch, -89.0f, 89.0f);
					}
				}
			}

			// Camera movement - only if ImGui doesn't want keyboard
			ImGuiIO& imguiIO = ImGui::GetIO();
			if ((mouseEnabled || !imguiIO.WantCaptureKeyboard) && !cameraAnimator.isPlaying()) {
			// if (mouseEnabled || !imguiIO.WantCaptureKeyboard) {
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
			}
			// ImGUI
			ImDrawData* drawData = nullptr;
			if (!mouseEnabled) {
				auto startGUI = std::chrono::high_resolution_clock::now();
				ImGui_ImplVulkan_NewFrame();
				ImGui_ImplSDL3_NewFrame();
				ImGui::NewFrame();

				modelManager->update();
				if (pendingSceneLoad) {
					const auto& loadingTasks = modelManager->getLoadingTasks();
					const auto& loadedModels = modelManager->getModels();

					// Check if all models finished loading
					bool allDone = loadingTasks.empty() && !loadedModels.empty();

					// Also check if expected model count matches
					if (allDone && loadedModels.size() >= pendingScene.models.size()) {
						// Build map from file model index to manager model index by matching paths
						std::vector<int> fileToManager(pendingScene.models.size(), -1);

						for (size_t fi = 0; fi < pendingScene.models.size(); ++fi) {
							for (size_t mi = 0; mi < loadedModels.size(); ++mi) {
								if (loadedModels[mi] && loadedModels[mi]->isValid() &&
									loadedModels[mi]->sourcePath == pendingScene.models[fi].path) {
									fileToManager[fi] = static_cast<int>(mi);
									break;
								}
							}
						}

						// Create all instances
						for (const auto& inst : pendingScene.instances) {
							if (inst.fileModelIndex >= fileToManager.size()) continue;
							int managerIdx = fileToManager[inst.fileModelIndex];
							if (managerIdx < 0) continue;

							size_t newIdx = modelManager->createInstance(
								static_cast<size_t>(managerIdx), inst.position);

							auto& newInst = modelManager->getInstances()[newIdx];
							newInst.name = inst.name;
							newInst.rotation = inst.rotation;
							newInst.scale = inst.scale;
							newInst.visible = inst.visible;
						}

						pendingSceneLoad = false;
						std::cout << "[SCENE] All instances created\n";
					}
				}
				// ============================================
				// MODEL MANAGER UI
				// ============================================
				ImGui::Begin("Model Manager");

				ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
				ImGui::Text("Camera: (%.2f, %.2f, %.2f)",
					camera.position.x, camera.position.y, camera.position.z);
				ImGui::Separator();

				// ============================================
				// SCENE SAVE / LOAD
				// ============================================
				static const std::string SCENES_ROOT = ".";
				static const std::string MODELS_ROOT = "models";
				static const std::string ANIMS_ROOT = ".";

				ImGui::Text("Scene File:");
				ImGui::InputText("##scenepath", sceneSavePath, sizeof(sceneSavePath));

				// --- Browse for scene file (to load) ---
				ImGui::SameLine();
				if (ImGui::Button("Browse##scene")) {
					namespace fs = std::filesystem;
					fs::create_directories(SCENES_ROOT);
					std::string absRoot = fs::weakly_canonical(fs::absolute(SCENES_ROOT)).string();

					IGFD::FileDialogConfig config;
					config.path = absRoot;
					config.countSelectionMax = 1;
					config.flags = ImGuiFileDialogFlags_Modal;

					ImGuiFileDialog::Instance()->OpenDialog(
						"BrowseSceneDlg",      // unique key
						"Select Scene File",   // title
						".scn",                // filter: ONLY .scn
						config
					);
				}

				// --- Save Scene (uses current path in text field) ---
				if (ImGui::Button("Save Scene")) {
					namespace fs = std::filesystem;
					fs::path savePath(sceneSavePath);

					// Auto-append .scn if no extension
					if (!savePath.has_extension()) {
						savePath += ".scn";
						strncpy(sceneSavePath, savePath.string().c_str(), sizeof(sceneSavePath) - 1);
						sceneSavePath[sizeof(sceneSavePath) - 1] = '\0';
					}

					std::string ext = savePath.extension().string();
					if (ext != ".scn") {
						std::cerr << "[SCENE] Invalid extension: " << ext << ". Use .scn\n";
					}
					else {
						if (SceneSerializer::Save(sceneSavePath, *modelManager)) {
							std::cout << "[SCENE] Scene saved successfully\n";
						}
						else {
							std::cerr << "[SCENE] Failed to save scene\n";
						}
					}
				}

				ImGui::SameLine();

				// --- Save As (opens dialog to pick save location) ---
				if (ImGui::Button("Save As...")) {
					namespace fs = std::filesystem;
					fs::create_directories(SCENES_ROOT);
					std::string absRoot = fs::weakly_canonical(fs::absolute(SCENES_ROOT)).string();

					IGFD::FileDialogConfig config;
					config.path = absRoot;
					config.fileName = "untitled.scn";
					config.countSelectionMax = 1;
					config.flags = ImGuiFileDialogFlags_ConfirmOverwrite | ImGuiFileDialogFlags_Modal;

					ImGuiFileDialog::Instance()->OpenDialog(
						"SaveAsSceneDlg",
						"Save Scene As",
						".scn",
						config
					);
				}

				ImGui::SameLine();

				// --- Load Scene (uses current path in text field) ---
				if (ImGui::Button("Load Scene")) {
					auto loaded = SceneSerializer::Load(sceneSavePath);
					if (loaded.valid) {
						gizmo.deselect();
						device.waitIdle();

						auto& currentInstances = modelManager->getInstances();
						while (!currentInstances.empty()) {
							modelManager->removeInstance(currentInstances.size() - 1);
						}
						while (!modelManager->getModels().empty()) {
							modelManager->unloadModel(modelManager->getModels().size() - 1);
						}

						std::vector<int> fileToManagerIndex(loaded.models.size(), -1);
						for (size_t i = 0; i < loaded.models.size(); ++i) {
							std::ifstream testFile(loaded.models[i].path);
							if (!testFile.good()) {
								std::cerr << "[SCENE] Model file not found: " << loaded.models[i].path << "\n";
								continue;
							}
							testFile.close();
							modelManager->loadModelAsync(loaded.models[i].path, loaded.models[i].name);
							fileToManagerIndex[i] = static_cast<int>(i);
						}

						pendingSceneLoad = true;
						pendingScene = loaded;
						std::cout << "[SCENE] Loading " << loaded.models.size() << " models...\n";
					}
				}

				ImGui::SameLine();

				if (ImGui::Button("Clear Scene")) {
					gizmo.deselect();
					device.waitIdle();
					auto& currentInstances = modelManager->getInstances();
					while (!currentInstances.empty()) {
						modelManager->removeInstance(currentInstances.size() - 1);
					}
					while (!modelManager->getModels().empty()) {
						modelManager->unloadModel(modelManager->getModels().size() - 1);
					}
				}

				ImGui::Separator();

				static char modelPath[512] = "models/";
				ImGui::InputText("Model Path", modelPath, sizeof(modelPath));

				if (ImGui::Button("Load Custom")) {
					modelManager->loadModelAsync(modelPath);
				}

				ImGui::SameLine();

				// --- Browse for 3D model ---
				if (ImGui::Button("Browse##model")) {
					namespace fs = std::filesystem;
					fs::create_directories(MODELS_ROOT);
					std::string absRoot = fs::weakly_canonical(fs::absolute(MODELS_ROOT)).string();

					IGFD::FileDialogConfig config;
					config.path = absRoot;
					config.countSelectionMax = 1;
					config.flags = ImGuiFileDialogFlags_Modal;

					ImGuiFileDialog::Instance()->OpenDialog(
						"BrowseModelDlg",
						"Select 3D Model",
						"{.gltf,.glb,.obj,.ifc}, .gltf,.glb,.obj,.ifc",   // allowed model formats
						config
					);
				}

				ImGui::Separator();

				ImVec2 dialogSize = ImVec2(600, 400);

				if (ImGuiFileDialog::Instance()->Display("BrowseSceneDlg",
					ImGuiWindowFlags_NoCollapse, dialogSize))
				{
					if (ImGuiFileDialog::Instance()->IsOk()) {
						std::string selectedPath = ImGuiFileDialog::Instance()->GetFilePathName();

						auto rel = makeRelativeIfInside(selectedPath, SCENES_ROOT);
						if (rel.has_value()) {
							strncpy(sceneSavePath, rel.value().c_str(), sizeof(sceneSavePath) - 1);
							sceneSavePath[sizeof(sceneSavePath) - 1] = '\0';
						}
						else {
							std::cerr << "[SCENE] File must be inside \"" << SCENES_ROOT << "/\"\n";
						}
					}
					ImGuiFileDialog::Instance()->Close();
				}

				// --- Dialog: Save As Scene ---
				if (ImGuiFileDialog::Instance()->Display("SaveAsSceneDlg",
					ImGuiWindowFlags_NoCollapse, dialogSize))
				{
					if (ImGuiFileDialog::Instance()->IsOk()) {
						std::string selectedPath = ImGuiFileDialog::Instance()->GetFilePathName();

						// Enforce .scn extension
						namespace fs = std::filesystem;
						fs::path p(selectedPath);
						if (!p.has_extension() || p.extension() != ".scn") {
							p.replace_extension(".scn");
						}

						auto rel = makeRelativeIfInside(p.string(), SCENES_ROOT);
						if (rel.has_value()) {
							strncpy(sceneSavePath, rel.value().c_str(), sizeof(sceneSavePath) - 1);
							sceneSavePath[sizeof(sceneSavePath) - 1] = '\0';

							if (SceneSerializer::Save(sceneSavePath, *modelManager)) {
								std::cout << "[SCENE] Scene saved to: " << sceneSavePath << "\n";
							}
							else {
								std::cerr << "[SCENE] Failed to save scene\n";
							}
						}
						else {
							std::cerr << "[SCENE] Save location must be inside \"" << SCENES_ROOT << "/\"\n";
						}
					}
					ImGuiFileDialog::Instance()->Close();
				}

				// --- Dialog: Browse Model ---
				if (ImGuiFileDialog::Instance()->Display("BrowseModelDlg",
					ImGuiWindowFlags_NoCollapse, dialogSize))
				{
					if (ImGuiFileDialog::Instance()->IsOk()) {
						std::string selectedPath = ImGuiFileDialog::Instance()->GetFilePathName();

						auto rel = makeRelativeIfInside(selectedPath, MODELS_ROOT);
						if (rel.has_value()) {
							strncpy(modelPath, rel.value().c_str(), sizeof(modelPath) - 1);
							modelPath[sizeof(modelPath) - 1] = '\0';

							namespace fs = std::filesystem;
							std::string name = fs::path(modelPath).stem().string();
							modelManager->loadModelAsync(modelPath, name);
						}
						else {
							std::cerr << "[MODEL] File must be inside \"" << MODELS_ROOT << "/\"\n";
						}
					}
					ImGuiFileDialog::Instance()->Close();
				}

				// Loading tasks
				const auto& tasks = modelManager->getLoadingTasks();
				if (!tasks.empty()) {
					ImGui::Text("Loading:");
					for (const auto& task : tasks) {
						const char* stateStr = "Unknown";
						switch (task.state) {
						case LoadingState::LoadingCPU: stateStr = "Parsing..."; break;
						case LoadingState::UploadingGPU: stateStr = "Uploading..."; break;
						case LoadingState::Failed: stateStr = "FAILED"; break;
						default: break;
						}
						ImGui::BulletText("%s - %s", task.name.c_str(), stateStr);
					}
					ImGui::Separator();
				}

				// Loaded Models
				ImGui::Text("Loaded Models: %zu", modelManager->getModels().size());
				const auto& models = modelManager->getModels();
				static int selectedModel = -1;
				size_t vertexCount = 0;
				for (size_t i = 0; i < models.size(); ++i) {
					const auto& model = models[i];
					ImGui::PushID(static_cast<int>(i));

					bool isSelected = (selectedModel == static_cast<int>(i));
					if (ImGui::Selectable(model->name.c_str(), isSelected)) {
						selectedModel = static_cast<int>(i);
					}

					// Right-click context menu
					if (ImGui::BeginPopupContextItem()) {
						if (ImGui::MenuItem("Create Instance")) {
							modelManager->createInstance(i, camera.position + glm::vec3(0, 0, 0));
						}
						if (ImGui::MenuItem("Create at Origin")) {
							modelManager->createInstance(i, glm::vec3(0.0f));
						}
						ImGui::Separator();
						if (ImGui::MenuItem("Unload")) {
							if (gizmo.selectedInstance >= 0) {
								const auto& insts = modelManager->getInstances();
								if (gizmo.selectedInstance >= static_cast<int>(insts.size())) {
									gizmo.deselect();
								}
								else {
									GPUModel* model = modelManager->getModel(insts[gizmo.selectedInstance].modelIndex);
									if (!model || !model->isValid()) {
										gizmo.deselect();  // <-- THIS might be firing!
									}
								}
							}
							modelManager->unloadModel(i);
							selectedModel = -1;
							gizmo.deselect();
						}
						ImGui::EndPopup();
					}

					
					ImGui::SameLine();
					ImGui::TextDisabled("(%zu verts, %zu tex)",
						model->vertexCount, model->textures.size());
					ImGui::PopID();
				}
				ImGui::Separator();

				// Instances
				ImGui::Text("Scene Instances: %zu", modelManager->getInstances().size());
				auto& instances = modelManager->getInstances();

				for (size_t i = 0; i < instances.size(); ++i) {
					auto& inst = instances[i];
					ImGui::PushID(static_cast<int>(i) + 10000);

					ImGui::Checkbox("##vis", &inst.visible);
					ImGui::SameLine();

					// Use gizmo.selectedInstance as single source of truth
					bool isSelected = (gizmo.selectedInstance == static_cast<int>(i));
					if (ImGui::Selectable(inst.name.c_str(), isSelected)) {
						gizmo.select(static_cast<int>(i));
					}

					if (ImGui::BeginPopupContextItem()) {
						if (ImGui::MenuItem("Delete")) {
							if (gizmo.selectedInstance == static_cast<int>(i)) {
								gizmo.deselect();
							}
							else if (gizmo.selectedInstance > static_cast<int>(i)) {
								gizmo.selectedInstance--;
							}
							modelManager->removeInstance(i);
						}
						if (ImGui::MenuItem("Copy")) {
							if (gizmo.selectedInstance == static_cast<int>(i)) {
								gizmo.deselect();
							}
							else if (gizmo.selectedInstance > static_cast<int>(i)) {
								gizmo.selectedInstance--;
							}
							modelManager->createInstance(instances[i].modelIndex, instances[i].position, instances[i].rotation, instances[i].scale);
						}
						ImGui::EndPopup();
					}
					// vertexCount += modelManager->getModel(inst.modelIndex)->vertexCount;
					ImGui::PopID();
				}
				// ImGui::TextDisabled("(Verts in scene %zu)",
//					vertexCount);
				// Single unified inspector
				ImGui::Separator();
				if (gizmo.selectedInstance >= 0 &&
					gizmo.selectedInstance < static_cast<int>(instances.size())) {

					/*GPUModel* gpuModel = modelManager->getModel(
						instances[gizmo.selectedInstance].modelIndex);*/

					if (instances[gizmo.selectedInstance].ifcScene) {
						IfcScene& scene = instances[gizmo.selectedInstance].ifcScene.value();

						ImGui::Separator();
						ImGui::Text("IFC Hierarchy (%zu elements, %zu spatial)",
							scene.elements.size(), scene.spatial.size());

						// show all / hide all
						if (ImGui::Button("Show All##ifc")) {
							for (auto& [guid, elem] : scene.elements)
								elem.visible = true;
							for (auto& [guid, node] : scene.spatial)
								node.visible = true;

							scene.syncVisibilityCache();
						}
						ImGui::SameLine();
						if (ImGui::Button("Hide All##ifc")) {
							for (auto& [guid, elem] : scene.elements)
								elem.visible = false;
							for (auto& [guid, node] : scene.spatial)
								node.visible = false;

							scene.syncVisibilityCache();
						}

						ImGui::Separator();

						auto clearPreviousSelection = [&]() {
							if (selectedIfcKind == IfcSelectionKind::kElement) {
								auto it = scene.elements.find(selectedIfcGuid);
								if (it != scene.elements.end()) it->second.selected = false;
							}
							else if (selectedIfcKind == IfcSelectionKind::kSpatial) {
								auto it = scene.spatial.find(selectedIfcGuid);
								if (it != scene.spatial.end()) it->second.selected = false;
							}
						};

						// recursive spatial tree drawing
						std::function<void(const std::string&)> drawSpatialNode =
							[&](const std::string& spatialGuid)
							{
								auto it = scene.spatial.find(spatialGuid);
								if (it == scene.spatial.end()) return;

								IfcSpatialNode& node = it->second;

								bool hasElements = !node.elementGuids.empty();
								bool hasChildren = false;
								for (const auto& childGuid : node.childSpatialGuids) {
									auto cit = scene.spatial.find(childGuid);
									if (cit != scene.spatial.end()) {
										if (!cit->second.elementGuids.empty() ||
											!cit->second.childSpatialGuids.empty()) {
											hasChildren = true;
											break;
										}
									}
								}

								if (!hasElements && !hasChildren)
									return;

								ImGui::PushID(node.guid.c_str());

								// checkbox for whole branch visibility
								bool branchVis = node.visible;
								if (ImGui::Checkbox("##sv", &branchVis)) {
									scene.setVisibilityRecursive(node.guid, branchVis);
									scene.syncVisibilityCache();
								}
								ImGui::SameLine();

								// count elements under this branch
								std::size_t elemAndSpatialCount = node.elementGuids.size() 
									+ node.childSpatialGuids.size();

								ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
								if (node.childSpatialGuids.empty() && node.elementGuids.empty()) flags |= ImGuiTreeNodeFlags_Leaf;
								if (node.selected) flags |= ImGuiTreeNodeFlags_Selected;

								bool open = ImGui::TreeNodeEx(node.guid.c_str(), flags, "%s: %s (%zu)",
									node.type.c_str(), node.name.c_str(), elemAndSpatialCount);

								if (ImGui::IsItemClicked(ImGuiMouseButton_Left) ||
									(ImGui::IsItemFocused() && ImGui::IsKeyPressed(ImGuiKey_Enter))) {
									// deselect previous
									for (auto& [g, e] : scene.elements)
										e.selected = false;
									for (auto& [g, s] : scene.spatial)
										s.selected = false;
									node.selected = true;
									selectedIfcGuid = node.guid;
									selectedIfcKind = IfcSelectionKind::kSpatial;
									selectedInstanceForProperties = gizmo.selectedInstance;
								}

								if (ImGui::BeginPopupContextItem("SpatialContextMenu")) {
									if (ImGui::MenuItem("Properties")) {
										selectedIfcGuid = node.guid;
										selectedIfcKind = IfcSelectionKind::kSpatial;
										selectedInstanceForProperties = gizmo.selectedInstance;
										showPropertiesWindow = true;
									}

									ImGui::EndPopup();
								}

								// tooltip with details
								if (ImGui::IsItemHovered() && !ImGui::IsPopupOpen("ElementContextMenu")) {
									ImGui::BeginTooltip();
									ImGui::Text("GUID: %s", node.guid.c_str());
									ImGui::Text("Type: %s", node.type.c_str());
									ImGui::Text("Children: %zu", node.childSpatialGuids.size());
									ImGui::Text("Elements: %zu", node.elementGuids.size());
									ImGui::EndTooltip();
								}

								if (open) {
									// draw child spatial nodes
									for (const auto& childGuid : node.childSpatialGuids) {
										drawSpatialNode(childGuid);
									}

									// draw contained elements
									ImGuiListClipper clipper;
									clipper.Begin(static_cast<int>(node.elementGuids.size()));
									while (clipper.Step()) {
										for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
											const auto& elemGuid = node.elementGuids[i];

											auto eit = scene.elements.find(elemGuid);
											if (eit == scene.elements.end()) continue;

											IfcElement& elem = eit->second;

											ImGui::PushID(elem.guid.c_str());

											// element visibility checkbox
											if (ImGui::Checkbox("##ev", &elem.visible)) {
												scene.syncVisibilityCache();
											}
											ImGui::SameLine();

											// leaf node
											ImGuiTreeNodeFlags elemFlags =
												ImGuiTreeNodeFlags_Leaf |
												ImGuiTreeNodeFlags_NoTreePushOnOpen;

											if (elem.selected)
												elemFlags |= ImGuiTreeNodeFlags_Selected;

											const char* displayName = elem.name.empty() ? elem.type.c_str() : elem.name.c_str();
											ImGui::TreeNodeEx(elem.guid.c_str(), elemFlags, "%s", displayName);

											// click to select
											if (ImGui::IsItemClicked(ImGuiMouseButton_Left) ||
												(ImGui::IsItemFocused() && ImGui::IsKeyPressed(ImGuiKey_Enter))) {
												// deselect previous
												for (auto& [g, e] : scene.elements)
													e.selected = false;
												for (auto& [g, s] : scene.spatial)
													s.selected = false;
												elem.selected = true;
												selectedIfcGuid = elem.guid;
												selectedIfcKind = IfcSelectionKind::kElement;
												selectedInstanceForProperties = gizmo.selectedInstance;
											}

											if (ImGui::BeginPopupContextItem("ElementContextMenu")) {
												if (ImGui::MenuItem("Properties")) {
													selectedIfcGuid = elem.guid;
													selectedIfcKind = IfcSelectionKind::kElement;
													selectedInstanceForProperties = gizmo.selectedInstance;
													showPropertiesWindow = true;
												}

												ImGui::EndPopup();
											}

											// tooltip with full details
											if (ImGui::IsItemHovered() && !ImGui::IsPopupOpen("ElementContextMenu")) {
												ImGui::BeginTooltip();
												ImGui::Text("GUID: %s", elem.guid.c_str());
												ImGui::Text("Type: %s", elem.type.c_str());
												ImGui::Text("Name: %s", elem.name.c_str());
												ImGui::Text("Tag: %s", elem.tag.c_str());
												ImGui::Text("Storey: %s", elem.storey.c_str());
												ImGui::Text("Parent spatial GUID: %s", elem.parentSpatialGuid.c_str());

												if (!elem.objectType.empty())
													ImGui::Text("ObjectType: %s", elem.objectType.c_str());

												if (elem.typeInfo) {
													ImGui::Separator();
													ImGui::Text("TypeInfo:");
													ImGui::Text("  Type: %s", elem.typeInfo->type.c_str());
													ImGui::Text("  Name: %s", elem.typeInfo->name.c_str());
												}

												/*if (!elem.data.empty()) {
													ImGui::Separator();
													ImGui::Text("Properties:");
													for (auto& [k, v] : elem.data) {
														ImGui::Text("  %s = %s", k.c_str(), v.c_str());
													}
												}*/

												ImGui::EndTooltip();
											}

											ImGui::PopID();
										}
									}

									ImGui::TreePop();
								}

								ImGui::PopID();
							};

						// draw from roots
						for (const auto& rootGuid : scene.roots) {
							drawSpatialNode(rootGuid);
						}

						/*static int cachedOrphanInstance = -1;
						static std::vector<std::string> cachedOrphanGuids;

						if (gizmo.selectedInstance != cachedOrphanInstance) {
							cachedOrphanInstance = gizmo.selectedInstance;
							cachedOrphanGuids.clear();
							for (const auto& [guid, elem] : scene.elements) {
								if (elem.parentSpatialGuid.empty() ||
									scene.spatial.find(elem.parentSpatialGuid) == scene.spatial.end()) {
									cachedOrphanGuids.push_back(guid);
								}
							}
						}

						if (!cachedOrphanGuids.empty()) {
							ImGui::Separator();
							if (ImGui::TreeNode("Unassigned Elements")) {
								for (const auto& guid : cachedOrphanGuids) {
									auto eit = scene.elements.find(guid);
									if (eit == scene.elements.end()) continue;

									IfcElement& elem = eit->second;

									ImGui::PushID(elem.guid.c_str());
									if (ImGui::Checkbox("##ev", &elem.visible)) {
										scene.syncVisibilityCache();
									}
									ImGui::SameLine();

									ImGui::TreeNodeEx(elem.guid.c_str(), ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen,
										"%s: %s", elem.type.c_str(), elem.name.c_str());

									ImGui::PopID();
								}
								ImGui::TreePop();
							}
						}*/
					}

					auto& sel = instances[gizmo.selectedInstance];
					ImGui::Text("Selected: %s", sel.name.c_str());

					const char* modeStr = "None";
					if (gizmo.mode == GizmoMode::Translate) modeStr = "Translate (1)";
					if (gizmo.mode == GizmoMode::Rotate) modeStr = "Rotate (2)";
					if (gizmo.mode == GizmoMode::Scale) modeStr = "Scale (3)";
					ImGui::Text("Gizmo: %s", modeStr);

					if (ImGui::Button("Translate")) gizmo.mode = GizmoMode::Translate;
					ImGui::SameLine();
					if (ImGui::Button("Rotate")) gizmo.mode = GizmoMode::Rotate;
					ImGui::SameLine();
					if (ImGui::Button("Scale")) gizmo.mode = GizmoMode::Scale;

					ImGui::DragFloat3("Position", &sel.position[0], 0.1f);
					ImGui::DragFloat3("Rotation", &sel.rotation[0], 1.0f);
					ImGui::DragFloat3("Scale", &sel.scale[0], 0.01f, 0.01f, 100.0f);

					if (ImGui::Button("Deselect")) {
						gizmo.deselect();
					}
				}
				else {
					ImGui::Text("Click on model to select (in viewport)");
					ImGui::Text("Or click instance in list above");
					ImGui::Text("Keys 1/2/3 for Translate/Rotate/Scale");
				}

				if (showPropertiesWindow &&
					selectedInstanceForProperties >= 0 &&
					selectedInstanceForProperties < static_cast<int>(instances.size()) &&
					instances[selectedInstanceForProperties].ifcScene &&
					selectedIfcKind != IfcSelectionKind::kNone) {
			
					IfcScene& scene = instances[selectedInstanceForProperties].ifcScene.value();

					ImGui::SetNextWindowSize(ImVec2(450, 550), ImGuiCond_FirstUseEver);
					static ImGuiTextFilter propertyFilter;
					if (selectedIfcKind == IfcSelectionKind::kElement) {
						auto it = scene.elements.find(selectedIfcGuid);
						if (it != scene.elements.end()) {
							IfcElement& elem = it->second;

							std::string windowTitle = "Properties: " + elem.type + "###PropertiesWindow";

							if (ImGui::Begin(windowTitle.c_str(), &showPropertiesWindow)) {

								// Basic Info Section
								if (ImGui::CollapsingHeader("Basic Information", ImGuiTreeNodeFlags_DefaultOpen)) {
									ImGui::Text("GUID:"); ImGui::SameLine(120);
									ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "%s", elem.guid.c_str());

									ImGui::Text("Type:"); ImGui::SameLine(120);
									ImGui::Text("%s", elem.type.c_str());

									ImGui::Text("Name:"); ImGui::SameLine(120);
									ImGui::Text("%s", elem.name.c_str());

									ImGui::Text("Tag:"); ImGui::SameLine(120);
									ImGui::Text("%s", elem.tag.c_str());

									ImGui::Text("Storey:"); ImGui::SameLine(120);
									ImGui::Text("%s", elem.storey.c_str());

									ImGui::Text("Parent GUID:"); ImGui::SameLine(120);
									ImGui::TextWrapped("%s", elem.parentSpatialGuid.c_str());

									if (!elem.objectType.empty()) {
										ImGui::Text("ObjectType:"); ImGui::SameLine(120);
										ImGui::Text("%s", elem.objectType.c_str());
									}
								}

								// Type Info Section
								if (elem.typeInfo && ImGui::CollapsingHeader("Type Information", ImGuiTreeNodeFlags_DefaultOpen)) {
									ImGui::Text("Type:"); ImGui::SameLine(120);
									ImGui::Text("%s", elem.typeInfo->type.c_str());

									ImGui::Text("Name:"); ImGui::SameLine(120);
									ImGui::Text("%s", elem.typeInfo->name.c_str());
								}

								// Properties Section (elem.data)
								if (!elem.data.empty() && ImGui::CollapsingHeader("Properties", ImGuiTreeNodeFlags_DefaultOpen)) {

									propertyFilter.Draw("Filter properties...");

									ImGui::BeginChild("PropertiesList", ImVec2(0, 300), true);
									if (ImGui::BeginTable("PropsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY)) {
										ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthStretch, 0.4f);
										ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 0.6f);
										ImGui::TableHeadersRow();

										for (const auto& [k, v] : elem.data) {
											if (propertyFilter.PassFilter(k.c_str()) || propertyFilter.PassFilter(v.c_str())) {
												ImGui::TableNextRow();
												ImGui::TableNextColumn(); ImGui::TextWrapped("%s", k.c_str());
												ImGui::TableNextColumn(); ImGui::TextWrapped("%s", v.c_str());
											}
										}
										ImGui::EndTable();
									}

									ImGui::EndChild();
								}

								ImGui::Separator();
								if (ImGui::Button("Close")) showPropertiesWindow = false;
							}
							ImGui::End();
						}
					}
					else if (selectedIfcKind == IfcSelectionKind::kSpatial) {
						auto it = scene.spatial.find(selectedIfcGuid);
						if (it != scene.spatial.end()) {
							IfcSpatialNode& node = it->second;

							ImGui::SetNextWindowSize(ImVec2(450, 550), ImGuiCond_FirstUseEver);

							std::string windowTitle = "Properties: " + node.type;
							if (!node.name.empty()) windowTitle += " - " + node.name;
							windowTitle += "###PropertiesWindow";

							if (ImGui::Begin(windowTitle.c_str(), &showPropertiesWindow)) {
								if (ImGui::CollapsingHeader("Basic Information", ImGuiTreeNodeFlags_DefaultOpen)) {
									ImGui::Text("GUID:"); ImGui::SameLine(120);
									ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "%s", node.guid.c_str());

									ImGui::Text("Type:"); ImGui::SameLine(120);
									ImGui::Text("%s", node.type.c_str());

									ImGui::Text("Name:"); ImGui::SameLine(120);
									ImGui::Text("%s", node.name.c_str());

									ImGui::Text("LongName:"); ImGui::SameLine(120);
									ImGui::Text("%s", node.longName.c_str());

									ImGui::Text("Parent GUID:"); ImGui::SameLine(120);
									ImGui::TextWrapped("%s", node.parentGuid.c_str());

									ImGui::Text("Children:"); ImGui::SameLine(120);
									ImGui::Text("%zu", node.childSpatialGuids.size());

									ImGui::Text("Elements:"); ImGui::SameLine(120);
									ImGui::Text("%zu", node.elementGuids.size());
								}

								if (!node.data.empty() && ImGui::CollapsingHeader("Properties", ImGuiTreeNodeFlags_DefaultOpen)) {

									// We re-use the exact same filter from the Element block
									propertyFilter.Draw("Filter properties...");

									ImGui::BeginChild("SpatialPropertiesList", ImVec2(0, 300), true);

									if (ImGui::BeginTable("SpatialPropsTable", 2,
										ImGuiTableFlags_Borders |
										ImGuiTableFlags_RowBg |
										ImGuiTableFlags_Resizable |
										ImGuiTableFlags_ScrollY)) {

										ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthStretch, 0.4f);
										ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 0.6f);
										ImGui::TableHeadersRow();

										for (const auto& [k, v] : node.data) {
											// Replaced std::transform loops with lightning-fast PassFilter
											if (propertyFilter.PassFilter(k.c_str()) || propertyFilter.PassFilter(v.c_str())) {
												ImGui::TableNextRow();
												ImGui::TableNextColumn();
												ImGui::TextWrapped("%s", k.c_str());
												ImGui::TableNextColumn();
												ImGui::TextWrapped("%s", v.c_str());
											}
										}

										ImGui::EndTable();
									}

									ImGui::EndChild();
									ImGui::Text("Total: %zu properties", node.data.size());
								}

								ImGui::Separator();
								if (ImGui::Button("Close"))
									showPropertiesWindow = false;
							}
							ImGui::End();
						}
					}
				}


				ImGui::End();
				// ============================================
				// COUNT ELEMENTS UI
				// ============================================
				static std::string lastSelectedGuid = "";
				static int lastSelectedInstance = -1;
				static std::string cachedElementType = "";
				static size_t cachedCountInModel = 0;
				static size_t cachedCountInScene = 0;

				if (selectedIfcKind != IfcSelectionKind::kNone && selectedInstanceForProperties != -1) {

					// 2. Only recalculate if the user clicked on a DIFFERENT element
					if (selectedIfcGuid != lastSelectedGuid || selectedInstanceForProperties != lastSelectedInstance) {

						// Update the trackers
						lastSelectedGuid = selectedIfcGuid;
						lastSelectedInstance = selectedInstanceForProperties;

						// Reset counts
						cachedCountInModel = 0;
						cachedCountInScene = 0;
						cachedElementType = "";

						IfcScene& scene = instances[selectedInstanceForProperties].ifcScene.value();

						if (selectedIfcKind == IfcSelectionKind::kElement) {
							auto it = scene.elements.find(selectedIfcGuid);
							if (it != scene.elements.end()) {
								cachedElementType = it->second.name;
							}

							if (!cachedElementType.empty()) {
								for (int sceneIterator = 0; sceneIterator != instances.size(); ++sceneIterator) {
									if (!instances[sceneIterator].ifcScene) continue;

									const auto& instScene = instances[sceneIterator].ifcScene.value();

									// 3. CRITICAL: Use const auto& to prevent copying the map pair every loop!
									for (const auto& el : instScene.elements) {
										if (el.second.name == cachedElementType) {
											++cachedCountInScene;
											if (sceneIterator == selectedInstanceForProperties) {
												++cachedCountInModel;
											}
										}
									}
								}
							}
						}
						else if (selectedIfcKind == IfcSelectionKind::kSpatial) {
							auto it = scene.spatial.find(selectedIfcGuid); // Fixed this from scene.elements
							if (it != scene.spatial.end()) {
								cachedElementType = it->second.name;
							}

							if (!cachedElementType.empty()) {
								for (int sceneIterator = 0; sceneIterator != instances.size(); ++sceneIterator) {
									if (!instances[sceneIterator].ifcScene) continue;

									const auto& instScene = instances[sceneIterator].ifcScene.value();

									// 3. CRITICAL: Use const auto& here as well
									for (const auto& el : instScene.spatial) {
										if (el.second.name == cachedElementType) {
											++cachedCountInScene;
											if (sceneIterator == selectedInstanceForProperties) {
												++cachedCountInModel;
											}
										}
									}
								}
							}
						}
					}

					// 4. Draw ImGui instantly using the cached data (Zero performance hit)
					ImGui::Begin("Element Counter");
					ImGui::Text("Type: %s", cachedElementType.c_str());
					ImGui::Separator();
					ImGui::Text("Count:");
					ImGui::Text("In model: %zu", cachedCountInModel);
					ImGui::Text("In scene: %zu", cachedCountInScene);
					ImGui::End();

				}
				else {
					// Reset cache trigger if nothing is selected
					lastSelectedGuid = "";
					lastSelectedInstance = -1;
				}

				// ============================================
				// CAMERA ANIMATION UI
				// ============================================
				ImGui::Begin("Camera Animation");

				auto& animPath = cameraAnimator.getPath();

				// Path name
				static char pathName[128] = "CameraPath1";
				ImGui::InputText("Path Name", pathName, sizeof(pathName));
				animPath.name = pathName;

				ImGui::Separator();

				// === PLAYBACK CONTROLS ===
				ImGui::Text("Playback");

				AnimationPlayState playState = cameraAnimator.getPlayState();
				const char* stateText = "Stopped";
				if (playState == AnimationPlayState::Playing) stateText = "Playing";
				if (playState == AnimationPlayState::Paused) stateText = "Paused";
				ImGui::Text("State: %s", stateText);

				if (ImGui::Button("Play")) {
					cameraAnimator.play();
				}
				ImGui::SameLine();
				if (ImGui::Button("Pause")) {
					cameraAnimator.pause();
				}
				ImGui::SameLine();
				if (ImGui::Button("Stop")) {
					cameraAnimator.stop();
				}

				float speed = cameraAnimator.getPlaybackSpeed();
				if (ImGui::SliderFloat("Speed", &speed, 0.1f, 5.0f, "%.1fx")) {
					cameraAnimator.setPlaybackSpeed(speed);
				}

				// Timeline scrubber
				float currentTime = cameraAnimator.getCurrentTime();
				float duration = cameraAnimator.getDuration();
				if (duration > 0.0f) {
					if (ImGui::SliderFloat("Timeline", &currentTime, 0.0f, duration, "%.2fs")) {
						// Allow scrubbing when not playing
						if (!cameraAnimator.isPlaying()) {
							// Manually set time for preview
							// We need a setter for this - using a small hack
							cameraAnimator.stop();
							// Force time through play+pause at the right moment
							// Better: add a setCurrentTime method
						}
					}

					// Progress bar
					float progress = duration > 0.0f ? currentTime / duration : 0.0f;
					ImGui::ProgressBar(progress, ImVec2(-1, 0),
						(std::to_string((int)currentTime) + "s / " + std::to_string((int)duration) + "s").c_str());
				}

				ImGui::Checkbox("Loop", &animPath.loop);

				ImGui::Separator();

				// === ADD KEYFRAME ===
				ImGui::Text("Add Keyframes");

				static float newKeyframeTime = 0.0f;
				static bool newKeyframeCurved = true;
				ImGui::DragFloat("Time (s)", &newKeyframeTime, 0.1f, 0.0f, 600.0f, "%.1f s");
				ImGui::Checkbox("Curved (Catmull-Rom)", &newKeyframeCurved);

				if (ImGui::Button("Add Keyframe at Camera Position")) {
					CameraKeyframe kf = CameraAnimator::makeKeyframe(
						camera.position, camera.yaw, camera.pitch,
						newKeyframeTime, newKeyframeCurved
					);
					cameraAnimator.addKeyframe(kf);

					// Auto-increment time for next keyframe
					newKeyframeTime += 2.0f;

					rebuildPathVisualization();
				}

				if (ImGui::Button("Add Keyframe at Origin")) {
					CameraKeyframe kf;
					kf.position = glm::vec3(0.0f);
					kf.lookTarget = glm::vec3(0.0f, 0.0f, -10.0f);
					kf.timestamp = newKeyframeTime;
					kf.useCurve = newKeyframeCurved;
					cameraAnimator.addKeyframe(kf);
					newKeyframeTime += 2.0f;
					rebuildPathVisualization();
				}

				ImGui::Separator();

				// === KEYFRAME LIST ===
				ImGui::Text("Keyframes: %zu", animPath.keyframes.size());

				bool pathModified = false;

				for (size_t i = 0; i < animPath.keyframes.size(); ++i) {
					auto& kf = animPath.keyframes[i];
					ImGui::PushID(static_cast<int>(i));

					bool nodeOpen = ImGui::TreeNode("", "KF %zu - %.1fs %s",
						i, kf.timestamp, kf.useCurve ? "(Curve)" : "(Line)");

					if (nodeOpen) {
						if (ImGui::DragFloat3("Position", &kf.position[0], 0.1f)) pathModified = true;
						if (ImGui::DragFloat3("Look At", &kf.lookTarget[0], 0.1f)) pathModified = true;
						if (ImGui::DragFloat("Time", &kf.timestamp, 0.1f, 0.0f, 600.0f)) pathModified = true;
						if (ImGui::Checkbox("Curved", &kf.useCurve)) pathModified = true;

						if (ImGui::Button("Set to Camera")) {
							kf = CameraAnimator::makeKeyframe(
								camera.position, camera.yaw, camera.pitch,
								kf.timestamp, kf.useCurve
							);
							pathModified = true;
						}
						ImGui::SameLine();
						if (ImGui::Button("Go to")) {
							// Move camera to this keyframe position
							camera.position = kf.position;
							float yaw, pitch;
							CameraAnimator::makeKeyframe(kf.position, 0, 0, 0); // dummy
							glm::vec3 dir = glm::normalize(kf.lookTarget - kf.position);
							camera.yaw = glm::degrees(atan2(dir.z, dir.x));
							camera.pitch = glm::degrees(asin(glm::clamp(dir.y, -1.0f, 1.0f)));
						}
						ImGui::SameLine();
						if (ImGui::Button("Delete")) {
							cameraAnimator.removeKeyframe(i);
							pathModified = true;
							ImGui::TreePop();
							ImGui::PopID();
							break; // List changed, exit loop
						}

						ImGui::TreePop();
					}

					ImGui::PopID();
				}

				if (pathModified) {
					animPath.recalculateDuration();
					rebuildPathVisualization();
				}

				ImGui::Separator();

				// === VISUALIZATION ===
				ImGui::Checkbox("Show Path", &showAnimationPath);

				ImGui::Separator();

				// === SAVE/LOAD ===
				static char animSavePath[256] = "camera_path.cmap";
				ImGui::InputText("Anim File", animSavePath, sizeof(animSavePath));


				// --- Browse for Camera Animation ---
				if (ImGui::Button("Browse##animation")) {
					namespace fs = std::filesystem;
					fs::create_directories(ANIMS_ROOT);
					std::string absRoot = fs::weakly_canonical(fs::absolute(ANIMS_ROOT)).string();

					IGFD::FileDialogConfig config;
					config.path = absRoot;
					config.countSelectionMax = 1;
					config.flags = ImGuiFileDialogFlags_Modal;

					ImGuiFileDialog::Instance()->OpenDialog(
						"BrowseAnimDlg",
						"Select camera animation",
						".cmap",
						config
					);
				}

				if (ImGuiFileDialog::Instance()->Display("BrowseAnimDlg",
					ImGuiWindowFlags_NoCollapse, dialogSize))
				{
					if (ImGuiFileDialog::Instance()->IsOk()) {
						std::string selectedPath = ImGuiFileDialog::Instance()->GetFilePathName();

						auto rel = makeRelativeIfInside(selectedPath, ANIMS_ROOT);
						if (rel.has_value()) {
							strncpy(animSavePath, rel.value().c_str(), sizeof(animSavePath) - 1);
							animSavePath[sizeof(animSavePath) - 1] = '\0';
						}
						else {
							std::cerr << "[ANIMS] File must be inside \"" << ANIMS_ROOT << "/\"\n";
						}
					}
					ImGuiFileDialog::Instance()->Close();
				}

				if (ImGui::Button("Save Path")) {
					cameraAnimator.savePath(animSavePath);
				}
				ImGui::SameLine();
				if (ImGui::Button("Load Path")) {
					if (cameraAnimator.loadPath(animSavePath)) {
						strncpy(pathName, cameraAnimator.getPath().name.c_str(), sizeof(pathName) - 1);
						rebuildPathVisualization();
					}
				}
				ImGui::SameLine();
				if (ImGui::Button("Clear All")) {
					cameraAnimator.clearKeyframes();
					rebuildPathVisualization();
				}

				ImGui::End();

				// Finalize ImGui frame (must call before RenderDrawData)
				ImGui::Render();
				drawData = ImGui::GetDrawData();
				auto endGUI = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> cpuTime = endGUI - startGUI;
				// std::cout << "GUI Frame Time: " << cpuTime.count() << " ms\n";
			}
			else {
				auto startModelsUpdate = std::chrono::high_resolution_clock::now();
				modelManager->update();

				// Handle pending scene load even when GUI is hidden
				if (pendingSceneLoad) {
					const auto& loadingTasks = modelManager->getLoadingTasks();
					const auto& loadedModels = modelManager->getModels();

					bool allDone = loadingTasks.empty() && !loadedModels.empty();
					if (allDone && loadedModels.size() >= pendingScene.models.size()) {
						std::vector<int> fileToManager(pendingScene.models.size(), -1);
						for (size_t fi = 0; fi < pendingScene.models.size(); ++fi) {
							for (size_t mi = 0; mi < loadedModels.size(); ++mi) {
								if (loadedModels[mi] && loadedModels[mi]->isValid() &&
									loadedModels[mi]->sourcePath == pendingScene.models[fi].path) {
									fileToManager[fi] = static_cast<int>(mi);
									break;
								}
							}
						}
						for (const auto& inst : pendingScene.instances) {
							if (inst.fileModelIndex >= fileToManager.size()) continue;
							int managerIdx = fileToManager[inst.fileModelIndex];
							if (managerIdx < 0) continue;
							size_t newIdx = modelManager->createInstance(
								static_cast<size_t>(managerIdx), inst.position);
							auto& newInst = modelManager->getInstances()[newIdx];
							newInst.name = inst.name;
							newInst.rotation = inst.rotation;
							newInst.scale = inst.scale;
							newInst.visible = inst.visible;
						}
						pendingSceneLoad = false;
						std::cout << "[SCENE] All instances created\n";
					}
				}

				drawData = nullptr;
				auto endModelsUpdate = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> cpuTime = endModelsUpdate - startModelsUpdate;
				// std::cout << "Models Update Frame Time: " << cpuTime.count() << " ms\n";
			}
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
			// ============================================
			// CAMERA ANIMATION UPDATE
			// ============================================
			bool animationControllingCamera = false;
			if (cameraAnimator.update(dt)) {
				CameraState camState = cameraAnimator.getCurrentState();
				camera.position = camState.position;
				camera.yaw = camState.yaw;
				camera.pitch = camState.pitch;
				animationControllingCamera = true;
			}

			FrameUBO frameData{};
			frameData.view = getView(camera);
			frameData.proj = getProjection(SCREEN_WIDTH, SCREEN_HEIGHT);
			frameData.lightSpaceMatrix = calculateLightSpaceMatrix(sunLight, sceneBounds.center, sceneBounds.radius);
			frameData.cameraPos = glm::vec4(camera.position, 0.0f);
			frameData.lightDir = glm::vec4(sunLight.direction, 0.0f);
			frameData.time = time;
			frameData.shadowBias = 0.005f;
			memcpy(frameUBOs[currentFrame].mapped, &frameData, sizeof(FrameUBO));

			auto startShadowPass = std::chrono::high_resolution_clock::now();
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

				static auto shadowAttribs = Vertex::getVertexOnlyAttributes(0);
				static vk::VertexInputBindingDescription2EXT shadowBinding = Vertex::getBindingDescription(0);
				cmd.setVertexInputEXT(1, &shadowBinding,
					static_cast<uint32_t>(shadowAttribs.size()), shadowAttribs.data());

				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, shadowPipelineLayout, 0, 1,
					&uboDescriptorSets[currentFrame], 0, nullptr);

				// Draw shadow pass for all model instances
				const auto& shadowInstances = modelManager->getInstances();

				int lastAlphaMode = -1;
				float lastAlphaCutoff = -1.0f;

				vk::DescriptorSet boundShadowSet = nullptr;

				auto startShadowPassLoop = std::chrono::high_resolution_clock::now();
				for (const auto& inst : shadowInstances) {
					if (!inst.visible) continue;

					GPUModel* gpuModel = modelManager->getModel(inst.modelIndex);
					if (!gpuModel || !gpuModel->isValid()) continue;
					/*if (inst.ifcScene) {
						const auto& scene = inst.ifcScene.value();
						if (!scene.roots.empty()) {
							auto it = scene.spatial.find(scene.roots.front());
							if (it != scene.spatial.end() && !it->second.visible) {
								continue;
							}
						}
					}*/

					// Bind vertex/index buffers for this model
					vk::Buffer modelBuffers[1] = { gpuModel->vertexBuffer->getBuffer() };
					vk::DeviceSize modelOffsets[1] = { 0 };
					vk::DeviceSize modelSizes[1] = { sizeof(Vertex) * gpuModel->vertexCount };
					vk::DeviceSize modelStrides[1] = { sizeof(Vertex) };

					cmd.bindVertexBuffers2(0, 1, modelBuffers, modelOffsets, modelSizes, modelStrides);
					cmd.bindIndexBuffer(gpuModel->indexBuffer->getBuffer(), 0, vk::IndexType::eUint32);

					glm::mat4 instanceTransform = inst.getTransformMatrix();

					glm::mat4 MVP = lightSpaceMatrix * instanceTransform;
					if (!isAABBVisible(gpuModel->boundsMin, gpuModel->boundsMax, MVP)) {
						continue; // Skip this entire model, it's not on screen!
					}

					bool instanceMatrixChanged = true;

					// Draw all non-blend submeshes for shadow
					for (std::size_t si = 0; si < gpuModel->submeshes.size(); ++si) {
						const auto& sub = gpuModel->submeshes[si];
						if (inst.ifcScene && !inst.ifcScene->isSubmeshVisible(si))
							continue;

						if (sub.material.alphaMode == AlphaMode::BLEND)
							continue;

						bool alphaChanged = (static_cast<int>(sub.material.alphaMode) != lastAlphaMode) ||
							(sub.material.alphaCutoff != lastAlphaCutoff);

						ShadowPushConstants shadowPc{};
						shadowPc.modelMatrix = instanceTransform;
						shadowPc.alphaCutoff = sub.material.alphaCutoff;
						shadowPc.alphaMode = static_cast<int>(sub.material.alphaMode);

						if (instanceMatrixChanged || alphaChanged) {
							shadowPc.alphaCutoff = sub.material.alphaCutoff;
							shadowPc.alphaMode = static_cast<int>(sub.material.alphaMode);

							cmd.pushConstants(
								shadowPipelineLayout,
								vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
								0,
								sizeof(ShadowPushConstants),
								&shadowPc
							);

							lastAlphaMode = shadowPc.alphaMode;
							lastAlphaCutoff = shadowPc.alphaCutoff;
							instanceMatrixChanged = false;
						}

						vk::DescriptorSet targetSet = nullptr;

						if (sub.material.alphaMode == AlphaMode::MASK && sub.material.baseColorTextureIndex >= 0 &&
							sub.material.baseColorTextureIndex < static_cast<int>(gpuModel->textureDescriptorSets.size())) {
							targetSet = gpuModel->textureDescriptorSets[sub.material.baseColorTextureIndex];
						}
						else {
							targetSet = modelManager->getDefaultBaseColorSet();
						}

						if (targetSet != boundShadowSet) {
							cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, shadowPipelineLayout, 1, 1, &targetSet, 0, nullptr);
							boundShadowSet = targetSet; // Update our tracker
						}

						// Draw
						cmd.drawIndexed(sub.indexCount, 1, sub.indexOffset, sub.vertexOffset, 0);
					}
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
				auto endShadowPass = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> timeShadowPass = endShadowPass - startShadowPass;
				std::cout << "Shadow Pass Time: " << timeShadowPass.count() << " ms\n";
				auto endShadowPassLoop = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> timeShadowPassLoop = endShadowPassLoop - startShadowPassLoop;
				std::cout << "\t Shadow Pass Loop Time: " << timeShadowPassLoop.count() << " ms\n";
			}

			auto startMainPass = std::chrono::high_resolution_clock::now();
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
			depthBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests)
				.setSrcAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
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
			renderInfo.setRenderArea({ {0,0},{(int)SCREEN_WIDTH, (int)SCREEN_HEIGHT} })
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
			cmd.bindShadersEXT(2, stages, shaders);
			cmd.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
			const vk::Viewport viewport{ 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, 0.f, 1.f };
			const vk::Rect2D rect{ {0,0},{(int)SCREEN_WIDTH, (int)SCREEN_HEIGHT} };
			cmd.setViewport(0, viewport);
			cmd.setScissor(0, rect);
			cmd.setRasterizerDiscardEnable(false);
			cmd.setCullMode(vk::CullModeFlagBits::eNone);
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

			static auto mainAttribs = Vertex::getVertexOnlyAttributes(0);
			static vk::VertexInputBindingDescription2EXT mainBinding = Vertex::getBindingDescription(0);
			cmd.setVertexInputEXT(1, &mainBinding,
				static_cast<uint32_t>(mainAttribs.size()), mainAttribs.data());


			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
				&uboDescriptorSets[currentFrame], 0, nullptr);

			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 4, 1,
				&shadowMapDescriptorSet, 0, nullptr);

			const auto& modelInstances = modelManager->getInstances();

			static const std::vector<Vertex> emptyVertices;

			auto startMainPassLoop = std::chrono::high_resolution_clock::now();		
			for (const auto& inst : modelInstances) {
				if (!inst.visible) continue;

				GPUModel* gpuModel = modelManager->getModel(inst.modelIndex);
				if (!gpuModel || !gpuModel->isValid()) continue;
				/*if (inst.ifcScene) {
					const auto& scene = inst.ifcScene.value();
					if (!scene.roots.empty()) {
						auto it = scene.spatial.find(scene.roots.front());
						if (it != scene.spatial.end() && !it->second.visible) {
							continue;
						}
					}
				}*/

				glm::mat4 instanceTransform = inst.getTransformMatrix();

				glm::mat4 MVP = frameData.proj * frameData.view * instanceTransform;
				if (!isAABBVisible(gpuModel->boundsMin, gpuModel->boundsMax, MVP)) {
					continue;
				}

				// Bind vertex/index buffers for this model
				vk::Buffer modelBuffers[1] = { gpuModel->vertexBuffer->getBuffer() };
				vk::DeviceSize modelOffsets[1] = { 0 };
				vk::DeviceSize modelSizes[1] = { sizeof(Vertex) * gpuModel->vertexCount };
				vk::DeviceSize modelStrides[1] = { sizeof(Vertex) };

				cmd.bindVertexBuffers2(0, 1, modelBuffers, modelOffsets, modelSizes, modelStrides);
				cmd.bindIndexBuffer(gpuModel->indexBuffer->getBuffer(), 0, vk::IndexType::eUint32);
				static const std::vector<Vertex> emptyVertices;
				// Sort submeshes
				auto sortedSubmeshes = sortSubmeshesForRendering(
					gpuModel->submeshes,
					std::vector<Vertex>(), 
					camera.position,
					(inst.ifcScene.has_value() ? &inst.ifcScene.value() : nullptr)
				);

				bool currentlyBlending = false;
				vk::DescriptorSet boundMaterialSets[3] = { nullptr, nullptr, nullptr };
				for (const auto& renderSub : sortedSubmeshes) {
					const auto& sub = gpuModel->submeshes[renderSub.submeshIndex];

					if (inst.ifcScene && !inst.ifcScene->isSubmeshVisible(renderSub.submeshIndex))
						continue;
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

					// Push constants
					MeshPushConstants pc{};
					pc.modelMatrix = instanceTransform;
					pc.baseColor = sub.material.baseColorFactor;
					pc.metallic = sub.material.metallicFactor;
					pc.roughness = sub.material.roughnessFactor;
					pc.alphaCutoff = sub.material.alphaCutoff;
					pc.alphaMode = static_cast<int>(sub.material.alphaMode);

					cmd.pushConstants(
						pipelineLayout,
						vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
						0, sizeof(MeshPushConstants), &pc
					);

					// Bind textures
					vk::DescriptorSet baseColorSet = (sub.material.baseColorTextureIndex >= 0 &&
						sub.material.baseColorTextureIndex < static_cast<int>(gpuModel->textureDescriptorSets.size()))
						? gpuModel->textureDescriptorSets[sub.material.baseColorTextureIndex]
						: modelManager->getDefaultBaseColorSet();
					// cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, 1, &baseColorSet, 0, nullptr);

					vk::DescriptorSet normalSet = (sub.material.normalTextureIndex >= 0 &&
						sub.material.normalTextureIndex < static_cast<int>(gpuModel->textureDescriptorSets.size()))
						? gpuModel->textureDescriptorSets[sub.material.normalTextureIndex]
						: modelManager->getDefaultNormalSet();
					// cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 2, 1, &normalSet, 0, nullptr);

					vk::DescriptorSet mrSet = (sub.material.metallicRoughnessTextureIndex >= 0 &&
						sub.material.metallicRoughnessTextureIndex < static_cast<int>(gpuModel->textureDescriptorSets.size()))
						? gpuModel->textureDescriptorSets[sub.material.metallicRoughnessTextureIndex]
						: modelManager->getDefaultMRSet();
					// cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 3, 1, &mrSet, 0, nullptr);

					if (baseColorSet != boundMaterialSets[0] || normalSet != boundMaterialSets[1] || mrSet != boundMaterialSets[2]) {
						boundMaterialSets[0] = baseColorSet;
						boundMaterialSets[1] = normalSet;
						boundMaterialSets[2] = mrSet;

						cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, 3, boundMaterialSets, 0, nullptr);
					}

					// Draw
					cmd.drawIndexed(sub.indexCount, 1, sub.indexOffset, sub.vertexOffset, 0);
				}

				if (currentlyBlending) {
					cmd.setColorBlendEnableEXT(0, VK_FALSE);
					cmd.setDepthWriteEnable(VK_TRUE);
				}
			}

			auto endMainPass = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> timeMainPass = endMainPass - startMainPass;
			std::cout << "Main Pass Time: " << timeMainPass.count() << " ms\n";
			auto endMainPassLoop = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> timeMainPassLoop = endMainPassLoop - startMainPassLoop;
			std::cout << "\t Main Pass Loop Time: " << timeMainPassLoop.count() << " ms\n";
			auto startGuiPass = std::chrono::high_resolution_clock::now();
			bool drawGizmo = (!mouseEnabled &&
				gizmo.selectedInstance >= 0 &&
				gizmo.selectedInstance < static_cast<int>(modelManager->getInstances().size()) &&
				gizmo.mode != GizmoMode::None &&
				modelManager->getInstances()[gizmo.selectedInstance].visible);

			bool drawPath = (!mouseEnabled &&
				showAnimationPath &&
				pathVisualizationBuffer != VK_NULL_HANDLE &&
				pathVisualizationVertexCount > 0);

			// ============================================
			// GIZMO & PATH VISUALIZATION (SHARED SETUP)
			// ============================================
			if (drawGizmo || drawPath) {
				// OPTIMIZATION 1: Bind shared shaders and state ONCE
				vk::ShaderStageFlagBits lineStages[] = { vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment };
				vk::ShaderEXT lineShaders[] = { gizmoVertShader, gizmoFragShader };
				cmd.bindShadersEXT(2, lineStages, lineShaders);

				cmd.setPrimitiveTopology(vk::PrimitiveTopology::eLineList);
				cmd.setCullMode(vk::CullModeFlagBits::eNone);
				cmd.setColorBlendEnableEXT(0, VK_FALSE);

				// OPTIMIZATION 2: Use static variables to prevent rebuilding the Vertex Layout every frame
				static const vk::VertexInputBindingDescription2EXT gizmoBinding{
					0, sizeof(GizmoVertex), vk::VertexInputRate::eVertex, 1
				};

				static const std::array<vk::VertexInputAttributeDescription2EXT, 2> gizmoAttribs = { {
					{ 0, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(GizmoVertex, position)) },
					{ 1, 0, vk::Format::eR32G32B32Sfloat, static_cast<uint32_t>(offsetof(GizmoVertex, color)) }
				} };

				cmd.setVertexInputEXT(1, &gizmoBinding, static_cast<uint32_t>(gizmoAttribs.size()), gizmoAttribs.data());

				// Bind shared UBO
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, gizmoPipelineLayout, 0, 1,
					&uboDescriptorSets[currentFrame], 0, nullptr);

				// --- DRAW GIZMO ---
				if (drawGizmo) {
					auto& gizmoInst = modelManager->getInstances()[gizmo.selectedInstance];

					cmd.setLineWidth(3.0f);
					cmd.setDepthTestEnable(false);  // Draw on top of everything
					cmd.setDepthWriteEnable(false);

					float gizmoScale = getGizmoScale(gizmoInst.position, camera.position, 0.15f, frameData.proj);
					glm::mat4 gizmoTransform = glm::translate(glm::mat4(1.0f), gizmoInst.position) * glm::scale(glm::mat4(1.0f), glm::vec3(gizmoScale));

					cmd.pushConstants(gizmoPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4), &gizmoTransform);

					vk::Buffer gizmoVB;
					uint32_t gizmoVertexCount = 0;

					switch (gizmo.mode) {
					case GizmoMode::Translate: gizmoVB = translateBuffer; gizmoVertexCount = static_cast<uint32_t>(translateLines.size()); break;
					case GizmoMode::Rotate:    gizmoVB = rotateBuffer;    gizmoVertexCount = static_cast<uint32_t>(rotateLines.size()); break;
					case GizmoMode::Scale:     gizmoVB = scaleBuffer;     gizmoVertexCount = static_cast<uint32_t>(scaleLines.size()); break;
					default: break;
					}

					if (gizmoVertexCount > 0) {
						vk::DeviceSize offset = 0, size = sizeof(GizmoVertex) * gizmoVertexCount, stride = sizeof(GizmoVertex);
						cmd.bindVertexBuffers2(0, 1, &gizmoVB, &offset, &size, &stride);
						cmd.draw(gizmoVertexCount, 1, 0, 0);
					}
				}

				// --- DRAW CAMERA PATH ---
				if (drawPath) {
					cmd.setLineWidth(2.0f);
					cmd.setDepthTestEnable(true);   // Path goes behind objects
					cmd.setDepthWriteEnable(false);

					glm::mat4 identity(1.0f);
					cmd.pushConstants(gizmoPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4), &identity);

					vk::Buffer pathVB = vk::Buffer(pathVisualizationBuffer);
					vk::DeviceSize pathOffset = 0, pathSize = sizeof(GizmoVertex) * pathVisualizationVertexCount, pathStride = sizeof(GizmoVertex);

					cmd.bindVertexBuffers2(0, 1, &pathVB, &pathOffset, &pathSize, &pathStride);
					cmd.draw(pathVisualizationVertexCount, 1, 0, 0);
				}

				// Restore state for ImGui / Next Frame
				cmd.setDepthTestEnable(true);
				cmd.setDepthWriteEnable(true);
				cmd.setPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
			}

			// ============================================
			// IMGUI OVERLAY
			// ============================================
			if (drawData && drawData->TotalVtxCount > 0) {
				// OPTIMIZATION 3: We DO NOT begin a new render pass!
				// ImGui natively supports being injected at the end of an active Render Pass.
				ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
			}

			// FINALLY end the render pass once everything (Models, Gizmos, UI) is drawn
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
			auto endGuiPass = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> timeGuiPass = endGuiPass - startGuiPass;
			std::cout << "GUI Pass Time: " << timeGuiPass.count() << " ms\n\n";
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

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplSDL3_Shutdown();
		ImGui::DestroyContext();
		vkDestroyDescriptorPool(vkDevice, imguiDescriptorPool, nullptr);

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
		modelManager.reset();
		device.destroyShaderEXT(gizmoVertShader);
		device.destroyShaderEXT(gizmoFragShader);
		device.destroyPipelineLayout(gizmoPipelineLayout);
		vmaDestroyBuffer(allocator, VkBuffer(translateBuffer), translateAlloc);
		vmaDestroyBuffer(allocator, VkBuffer(rotateBuffer), rotateAlloc);
		vmaDestroyBuffer(allocator, VkBuffer(scaleBuffer), scaleAlloc);
		if (pathVisualizationBuffer != VK_NULL_HANDLE) {
			vmaDestroyBuffer(allocator, pathVisualizationBuffer, pathVisualizationAlloc);
		}
	};


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
