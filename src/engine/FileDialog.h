#pragma once

//#include "../external/portable-file-dialogs.h"
//#ifdef OPAQUE
//#undef OPAQUE
//#endif
//#ifdef TRANSPARENT
//#undef TRANSPARENT
//#endif
//#ifdef near
//#undef near
//#endif
//#ifdef far
//#undef far
//#endif
//#ifdef min
//#undef min
//#endif
//#ifdef max
//#undef max
//#endif

#include "../external/ImGuiFileDialog.h"
#include <filesystem>

// Returns relative path if file is inside rootDir, otherwise nullopt
static std::optional<std::string> makeRelativeIfInside(
    const std::string& filePath,
    const std::string& rootDir)
{
    namespace fs = std::filesystem;
    fs::path absFile = fs::weakly_canonical(fs::absolute(filePath));
    fs::path absRoot = fs::weakly_canonical(fs::absolute(rootDir));

    std::string fileStr = absFile.string();
    std::string rootStr = absRoot.string();

    // Check file is under root
    if (fileStr.rfind(rootStr, 0) != 0)
        return std::nullopt;

    std::string rel = fs::relative(absFile, fs::current_path()).string();
    std::replace(rel.begin(), rel.end(), '\\', '/');
    return rel;
}
