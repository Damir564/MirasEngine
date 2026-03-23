#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct IfcTypeLayer {
    std::string typeName;
    std::vector<int> submeshIndices;
    bool visible = true;
};

struct IfcTreeNode {
    std::string name;
    int parent = -1;
    std::vector<int> children;
    std::vector<int> submeshIndices;
    bool visible = true;
};

struct IfcInfo {
    bool isIfc = false;
    std::vector<IfcTypeLayer> layers;

    std::vector<IfcTreeNode> treeRoots;
    std::vector<IfcTreeNode> tree;
    std::vector<int> rootIndices;
    std::vector<int> submeshToNode;

    bool isSubmeshVisible(int submeshIdx) const {
        if (!isIfc) return true;

        if (submeshIdx >= 0 && submeshIdx < static_cast<int>(submeshToNode.size())) {
            int nodeIdx = submeshToNode[submeshIdx];
            while (nodeIdx >= 0) {
                if (!tree[nodeIdx].visible) return false;
                nodeIdx = tree[nodeIdx].parent;
            }
        }

        for (const auto& layer : layers) {
            for (int si : layer.submeshIndices) {
                if (si == submeshIdx) return layer.visible;
            }
        }
        return true;
    }

    void setNodeVisibleRecursive(int nodeIdx, bool v) {
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(tree.size())) return;
        tree[nodeIdx].visible = v;
        for (int child : tree[nodeIdx].children) {
            setNodeVisibleRecursive(child, v);
        }
    }
};