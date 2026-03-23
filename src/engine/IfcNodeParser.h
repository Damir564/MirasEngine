#pragma once
#include <string>
#include <set>

inline std::string extractIfcType(const std::string& nodeName) {
    if (nodeName.empty()) return "Other";

    // ===== Format 1: Starts with "Ifc" =====
    // e.g. "IfcWall #123", "IfcPipeSegment", "IfcWall_0AB2"
    if (nodeName.size() >= 3 && nodeName.substr(0, 3) == "Ifc") {
        size_t end = nodeName.find_first_of(" #/_");
        if (end != std::string::npos)
            return nodeName.substr(0, end);
        return nodeName;
    }

    // ===== Format 2: Contains "Ifc" somewhere =====
    size_t ifcPos = nodeName.find("Ifc");
    if (ifcPos != std::string::npos) {
        size_t end = nodeName.find_first_of(" #/_:", ifcPos);
        if (end != std::string::npos)
            return nodeName.substr(ifcPos, end - ifcPos);
        return nodeName.substr(ifcPos);
    }

    // ===== Format 3: Revit element names "FamilyName:TypeName:ID" =====
    // Group by the first segment before ':'
    size_t firstColon = nodeName.find(':');
    if (firstColon != std::string::npos && firstColon > 0) {
        std::string family = nodeName.substr(0, firstColon);

        // Clean up common prefixes
        if (family.substr(0, 2) == "M_") {
            family = family.substr(2);
        }

        // Trim trailing spaces
        while (!family.empty() && family.back() == ' ')
            family.pop_back();

        // Remove " - Generic", " - Hosted" etc. suffixes for cleaner grouping
        for (const char* suffix : { " - Generic", " - Hosted", " - Steel" }) {
            size_t pos = family.find(suffix);
            if (pos != std::string::npos) {
                family = family.substr(0, pos);
            }
        }

        if (!family.empty())
            return family;
    }

    return "Other";
}