#pragma once

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

struct SDL_Window;
struct FileDialogState;

// Native OS open/save dialogs (SDL3), one at a time. The dialog runs on its own thread; the main
// thread collects the result with take().
class FileDialog {
public:
    // patterns: extensions without dots, separated by ';' ("gltf;glb"). root is created if missing;
    // save dialogs start at root/defaultFileName and ask before overwriting.
    void open(SDL_Window* window, const char* key, const char* title, const char* filterName, const char* patterns,
        const std::string& root, const char* defaultFileName, bool save);
    bool isOpen() const;
    // True once the dialog opened under key has closed; path is empty when it was cancelled.
    bool take(const char* key, std::string& path);

private:
    std::shared_ptr<FileDialogState> m_state;
};

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

// Paths under the working directory stay relative so scenes remain portable; anything else is kept absolute.
inline std::string toStoredPath(const std::string& filePath)
{
    namespace fs = std::filesystem;
    fs::path absFile = fs::weakly_canonical(fs::absolute(filePath));
    std::string path = makeRelativeIfInside(absFile.string(), fs::current_path().string()).value_or(absFile.string());
    std::replace(path.begin(), path.end(), '\\', '/');
    return path;
}
