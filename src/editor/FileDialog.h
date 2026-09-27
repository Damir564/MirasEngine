#pragma once

#include "../external/ImGuiFileDialog.h"
#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>

// Returns the path relative to the working directory if filePath is inside rootDir, otherwise nullopt.
inline std::optional<std::string> makeRelativeIfInside(
    const std::string& filePath,
    const std::string& rootDir)
{
    namespace fs = std::filesystem;
    fs::path absFile = fs::weakly_canonical(fs::absolute(filePath));
    fs::path absRoot = fs::weakly_canonical(fs::absolute(rootDir));

    std::string fileStr = absFile.string();
    std::string rootStr = absRoot.string();

    if (fileStr.rfind(rootStr, 0) != 0)
        return std::nullopt;

    std::string rel = fs::relative(absFile, fs::current_path()).string();
    std::replace(rel.begin(), rel.end(), '\\', '/');
    return rel;
}
