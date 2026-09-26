#include "IfcSceneLoader.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

static std::string jstr(const json& j, const std::string& key)
{
    if (j.contains(key) && j[key].is_string())
        return j[key].get<std::string>();
    return {};
}

bool loadIfcScene(const std::string& jsonPath, IfcScene& scene)
{
    scene = {};

    std::ifstream f(jsonPath, std::ios::binary | std::ios::ate);
    if (!f.is_open())
    {
        std::cerr << "[IFC] Cannot open " << jsonPath << "\n";
        return false;
    }

    // Parsing from a contiguous buffer is much faster than from an istream.
    std::string content(static_cast<size_t>(f.tellg()), '\0');
    f.seekg(0);
    f.read(content.data(), static_cast<std::streamsize>(content.size()));

    json root;
    try
    {
        root = json::parse(content.data(), content.data() + content.size());
    }
    catch (const std::exception& e)
    {
        std::cerr << "[IFC] JSON parse error: " << e.what() << "\n";
        return false;
    }

    scene.schema = jstr(root, "schema");

    // roots
    if (root.contains("roots") && root["roots"].is_array())
    {
        for (auto& r : root["roots"])
        {
            if (r.is_string())
                scene.roots.push_back(r.get<std::string>());
        }
    }

    if (root.contains("spatial") && root["spatial"].is_object())
    {
        scene.spatial.reserve(root["spatial"].size());
        for (auto& [guid, sj] : root["spatial"].items())
        {
            IfcSpatialNode node;
            node.guid = guid;
            node.type = jstr(sj, "type");
            node.name = jstr(sj, "name");
            node.longName = jstr(sj, "longName");
            node.parentGuid = jstr(sj, "parentGuid");

            if (sj.contains("children") && sj["children"].is_array())
            {
                for (auto& c : sj["children"])
                {
                    if (c.is_string())
                        node.childSpatialGuids.push_back(c.get<std::string>());
                }
            }

            if (sj.contains("elements") && sj["elements"].is_array())
            {
                for (auto& e : sj["elements"])
                {
                    if (e.is_string())
                        node.elementGuids.push_back(e.get<std::string>());
                }
            }

            if (sj.contains("data") && sj["data"].is_object())
            {
                for (auto& [k, v] : sj["data"].items())
                {
                    if (v.is_string())
                        node.data[k] = v.get<std::string>();
                    else if (v.is_number_integer())
                        node.data[k] = std::to_string(v.get<int64_t>());
                    else if (v.is_number_float())
                        node.data[k] = std::to_string(v.get<double>());
                    else if (v.is_boolean())
                        node.data[k] = v.get<bool>() ? "true" : "false";
                    else if (!v.is_null())
                        node.data[k] = v.dump();
                }
            }

            scene.spatial[guid] = std::move(node);
        }
    }

    if (root.contains("elements") && root["elements"].is_object())
    {
        scene.elements.reserve(root["elements"].size());
        for (auto& [guid, ej] : root["elements"].items())
        {
            IfcElement elem;
            elem.guid = guid;
            elem.type = jstr(ej, "type");
            elem.name = jstr(ej, "name");
            elem.objectType = jstr(ej, "objectType");
            elem.tag = jstr(ej, "tag");
            elem.storey = jstr(ej, "storey");
            elem.parentSpatialGuid = jstr(ej, "parentSpatialGuid");

            if (ej.contains("typeInfo") && ej["typeInfo"].is_object())
            {
                auto& tj = ej["typeInfo"];
                IfcTypeInfo ti;
                ti.guid = jstr(tj, "guid");
                ti.type = jstr(tj, "type");
                ti.name = jstr(tj, "name");
                ti.predefinedType = jstr(tj, "predefinedType");
                elem.typeInfo = std::move(ti);
            }

            if (ej.contains("data") && ej["data"].is_object())
            {
                for (auto& [k, v] : ej["data"].items())
                {
                    if (v.is_string())
                        elem.data[k] = v.get<std::string>();
                    else if (v.is_number_integer())
                        elem.data[k] = std::to_string(v.get<int64_t>());
                    else if (v.is_number_float())
                        elem.data[k] = std::to_string(v.get<double>());
                    else if (v.is_boolean())
                        elem.data[k] = v.get<bool>() ? "true" : "false";
                    else if (!v.is_null())
                        elem.data[k] = v.dump();
                }
            }

            scene.elements[guid] = std::move(elem);
        }
    }

    std::cout << "[IFC] Loaded: "
        << scene.elements.size() << " elements, "
        << scene.spatial.size() << " spatial nodes, "
        << scene.roots.size() << " roots\n";

    return true;
}