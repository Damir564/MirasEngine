#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <cstddef>
#include <limits>
#include <iostream>
#include <algorithm>

struct IfcTypeInfo {
    std::string guid;
    std::string type;
    std::string name;
    std::string predefinedType;
};

struct IfcElement {
    std::string guid;
    std::string type;
    std::string name;
    std::string objectType;
    std::string tag;
    std::string storey;
    std::string parentSpatialGuid;

    std::optional<IfcTypeInfo> typeInfo;

    std::unordered_map<std::string, std::string> data;

    std::size_t submeshIndex = std::numeric_limits<std::size_t>::max();
    bool visible = true;
    bool selected = false;
};

struct IfcSpatialNode {
    std::string guid;
    std::string type;
    std::string name;
    std::string longName;
    std::string parentGuid;

    std::unordered_map<std::string, std::string> data;

    std::vector<std::string> childSpatialGuids;
    std::vector<std::string> elementGuids;

    bool visible = true;
    bool expanded = true;
    bool selected = false;
};

struct IfcScene {
    std::string schema;

    std::unordered_map<std::string, IfcElement> elements;
    std::unordered_map<std::string, IfcSpatialNode> spatial;
    std::vector<std::string> roots;

    std::unordered_map<std::size_t, std::string> submeshToGuid;
    std::unordered_map<std::string, std::size_t> guidToSubmesh;

    std::vector<bool> submeshVisibilityCache;

    bool isSubmeshVisible(std::size_t submeshIdx) const {
        if (submeshIdx < submeshVisibilityCache.size()) {
            return submeshVisibilityCache[submeshIdx];
        }
        return false;
    }

    void syncVisibilityCache() {
        for (const auto& [submeshIdx, guid] : submeshToGuid) {
            auto eit = elements.find(guid);
            if (eit != elements.end()) {
                if (submeshIdx < submeshVisibilityCache.size()) {
                    submeshVisibilityCache[submeshIdx] = eit->second.visible;
                }
            }
        }
    }

    static int findSubmeshByGuid(const IfcScene& scene, const std::string& guid) {
        auto it = scene.guidToSubmesh.find(guid);
        if (it != scene.guidToSubmesh.end()) {
            return static_cast<int>(it->second);
        }
        return -1;
    }

    //bool isSubmeshVisible(std::size_t submeshIdx) const {
    //    auto it = submeshToGuid.find(submeshIdx);
    //    if (it == submeshToGuid.end())
    //        return false;

    //    auto eit = elements.find(it->second);
    //    if (eit == elements.end())
    //        return false;

    //    return eit->second.visible;
    //}

    void setVisibilityRecursive(const std::string& spatialGuid, bool vis) {
        auto it = spatial.find(spatialGuid);
        if (it == spatial.end()) return;

        it->second.visible = vis;

        for (const auto& eg : it->second.elementGuids) {
            auto eit = elements.find(eg);
            if (eit != elements.end()) {
                eit->second.visible = vis;

                if (eit->second.submeshIndex != std::numeric_limits<std::size_t>::max() &&
                    eit->second.submeshIndex < submeshVisibilityCache.size()) {
                    submeshVisibilityCache[eit->second.submeshIndex] = vis;
                }
            }
        }

        for (auto& cg : it->second.childSpatialGuids)
            setVisibilityRecursive(cg, vis);
    }

    static void printSpatialTree(const IfcScene& scene, const std::string& spatialGuid, int depth = 0);
    static void printIfcScene(const IfcScene& scene);
};

