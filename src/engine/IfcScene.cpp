#include "IfcScene.h"

void IfcScene::printSpatialTree(
    const IfcScene& scene,
    const std::string& spatialGuid,
    int depth)
{
    auto it = scene.spatial.find(spatialGuid);
    if (it == scene.spatial.end()) return;

    const auto& node = it->second;
    std::string indent(depth * 2, ' ');

    // print spatial node
    std::cout << indent
        << "[" << node.type << "] "
        << "\"" << node.name << "\" "
        << "guid=" << node.guid
        << "\n";

    // print elements contained in this spatial node
    for (const auto& elemGuid : node.elementGuids) {
        auto eit = scene.elements.find(elemGuid);
        if (eit == scene.elements.end()) continue;

        const auto& elem = eit->second;
        std::cout << indent << "  "
            << "- [" << elem.type << "] "
            << "\"" << elem.name << "\" "
            << "tag=" << elem.tag << " "
            << "guid=" << elem.guid;

        if (elem.typeInfo) {
            std::cout << " typeInfo={"
                << elem.typeInfo->type << " "
                << "\"" << elem.typeInfo->name << "\""
                << "}";
        }

        if (!elem.data.empty()) {
            std::cout << " data={";
            bool first = true;
            for (const auto& [k, v] : elem.data) {
                if (!first) std::cout << ", ";
                std::cout << k << "=" << v;
                first = false;
            }
            std::cout << "}";
        }

        std::cout << "\n";
    }

    // recurse into child spatial nodes
    for (const auto& childGuid : node.childSpatialGuids) {
        printSpatialTree(scene, childGuid, depth + 1);
    }
}

void IfcScene::printIfcScene(const IfcScene& scene)
{
    std::cout << "=== IFC Scene ===\n";
    std::cout << "Schema: " << scene.schema << "\n";
    std::cout << "Elements: " << scene.elements.size() << "\n";
    std::cout << "Spatial nodes: " << scene.spatial.size() << "\n";
    std::cout << "Roots: " << scene.roots.size() << "\n\n";

    // print full hierarchy
    std::cout << "=== Hierarchy ===\n";
    for (const auto& rootGuid : scene.roots) {
        printSpatialTree(scene, rootGuid);
    }

    // print orphan elements (not in any spatial node)
    std::cout << "\n=== Orphan elements (no spatial parent) ===\n";
    for (const auto& [guid, elem] : scene.elements) {
        if (elem.parentSpatialGuid.empty() ||
            scene.spatial.find(elem.parentSpatialGuid) == scene.spatial.end())
        {
            std::cout << "  - [" << elem.type << "] "
                << "\"" << elem.name << "\" "
                << "guid=" << elem.guid << "\n";
        }
    }
}