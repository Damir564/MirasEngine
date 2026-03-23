#pragma once
#include <string>
#include <set>

inline std::string extractIfcType(const std::string& nodeName) {
    size_t slash = nodeName.find('/');
    std::string candidate = (slash != std::string::npos)
        ? nodeName.substr(0, slash)
        : nodeName;

    if (candidate.size() >= 3 && candidate.substr(0, 3) == "Ifc") {
        return candidate;
    }

    if (nodeName.substr(0, 3) == "Ifc") {
        size_t end = nodeName.find_first_of(" /#_:");
        return (end != std::string::npos) ? nodeName.substr(0, end) : nodeName;
    }

    return "Other";
}