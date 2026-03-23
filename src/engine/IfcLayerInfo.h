#pragma once
#include <string>
#include <vector>
#include <unordered_map>

enum class IfcViewMode {
    Hierarchy = 0,
    Flat = 1
};

struct IfcTypeLayer {
    std::string typeName;
    std::vector<int> submeshIndices;
    bool visible = true;
};

struct IfcTreeNode {
    std::string name;
    std::string fullPath;
    int parent = -1;
    std::vector<int> children;
    std::vector<int> submeshIndices;
    bool visible = true;
};

struct IfcInfo {
    bool isIfc = false;

    std::vector<IfcTypeLayer> layers;

    std::vector<IfcTreeNode> tree;
    std::vector<int> rootIndices;
    std::vector<int> submeshToNode;

    IfcViewMode viewMode = IfcViewMode::Hierarchy;

    bool isSubmeshVisibleFlat(int submeshIdx) const {
        if (!isIfc) return true;
        for (const auto& layer : layers) {
            for (int si : layer.submeshIndices) {
                if (si == submeshIdx) return layer.visible;
            }
        }
        return true;
    }

    bool isSubmeshVisibleHierarchy(int submeshIdx) const {
        if (!isIfc) return true;
        if (submeshIdx < 0 || submeshIdx >= static_cast<int>(submeshToNode.size()))
            return true;

        int nodeIdx = submeshToNode[submeshIdx];
        while (nodeIdx >= 0) {
            if (!tree[nodeIdx].visible) return false;
            nodeIdx = tree[nodeIdx].parent;
        }
        return true;
    }

    bool isSubmeshVisible(int submeshIdx) const {
        if (!isIfc) return true;

        if (viewMode == IfcViewMode::Flat)
            return isSubmeshVisibleFlat(submeshIdx);

        return isSubmeshVisibleHierarchy(submeshIdx);
    }

    void setNodeVisibleRecursive(int nodeIdx, bool v) {
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(tree.size())) return;
        tree[nodeIdx].visible = v;
        for (int child : tree[nodeIdx].children) {
            setNodeVisibleRecursive(child, v);
        }
    }

    int getSubtreeSubmeshCount(int nodeIdx) const {
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(tree.size())) return 0;
        int count = static_cast<int>(tree[nodeIdx].submeshIndices.size());
        for (int child : tree[nodeIdx].children) {
            count += getSubtreeSubmeshCount(child);
        }
        return count;
    }
};