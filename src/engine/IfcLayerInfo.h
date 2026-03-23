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
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(tree.size()))
            return true;

        return tree[nodeIdx].visible;
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

    void setNodeVisibleWithParents(int nodeIdx, bool v) {
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(tree.size())) return;

        tree[nodeIdx].visible = v;

        if (v) {
            int p = tree[nodeIdx].parent;
            while (p >= 0) {
                tree[p].visible = true;
                p = tree[p].parent;
            }
        }
        else {
            for (int child : tree[nodeIdx].children) {
                setNodeVisibleRecursive(child, false);
            }
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

    bool hasAnyVisibleInSubtree(int nodeIdx) const {
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(tree.size())) return false;

        if (tree[nodeIdx].visible)
            return true;

        for (int child : tree[nodeIdx].children) {
            if (hasAnyVisibleInSubtree(child))
                return true;
        }
        return false;
    }

    void refreshParentVisibilityUpwards(int nodeIdx) {
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(tree.size())) return;

        int parentIdx = tree[nodeIdx].parent;
        while (parentIdx >= 0) {
            bool anyVisible = false;

            for (int childIdx : tree[parentIdx].children) {
                if (hasAnyVisibleInSubtree(childIdx)) {
                    anyVisible = true;
                    break;
                }
            }

            tree[parentIdx].visible = anyVisible;
            parentIdx = tree[parentIdx].parent;
        }
    }

    void setNodeVisibility(int nodeIdx, bool v) {
        if (nodeIdx < 0 || nodeIdx >= static_cast<int>(tree.size())) return;

        // If parent clicked: affect full subtree
        setNodeVisibleRecursive(nodeIdx, v);

        if (v) {
            // turn on all parents too
            int parentIdx = tree[nodeIdx].parent;
            while (parentIdx >= 0) {
                tree[parentIdx].visible = true;
                parentIdx = tree[parentIdx].parent;
            }
        }
        else {
            // if disabled, parents may need to turn off too
            refreshParentVisibilityUpwards(nodeIdx);
        }
    }

    void rebuildParentVisibilityFromChildren() {
        if (tree.empty()) return;

        for (int i = static_cast<int>(tree.size()) - 1; i >= 0; --i) {
            if (tree[i].children.empty()) continue;

            bool anyVisible = false;
            for (int childIdx : tree[i].children) {
                if (hasAnyVisibleInSubtree(childIdx)) {
                    anyVisible = true;
                    break;
                }
            }

            tree[i].visible = anyVisible;
        }
    }

};