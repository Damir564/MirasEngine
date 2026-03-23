#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct IfcTypeLayer {
    std::string typeName;
    std::vector<int> submeshIndices;
    bool visible = true;
};

struct IfcInfo {
    bool isIfc = false;
    std::vector<IfcTypeLayer> layers;

    bool isSubmeshVisible(int submeshIdx) const {
        if (!isIfc) return true;
        for (const auto& layer : layers) {
            for (int si : layer.submeshIndices) {
                if (si == submeshIdx) return layer.visible;
            }
        }
        return true;
    }
};