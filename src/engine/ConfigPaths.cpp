#include "ConfigPaths.h"
#include <cstdlib>
#include <fstream>
#include <iterator>
#include "Log.h"

namespace {

std::filesystem::path findConfigDir()
{
    wchar_t* appData = nullptr;
    size_t length = 0;
    std::filesystem::path base;
    if (_wdupenv_s(&appData, &length, L"APPDATA") == 0 && appData) {
        base = appData;
        free(appData);
    }
    if (base.empty())
        return {};
    const std::filesystem::path dir = base / "MirasEngine";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec || !std::filesystem::is_directory(dir, ec)) {
        LOG_ERROR("[CONFIG] Can't create " << dir.string() << "; settings stay in the working directory\n");
        return {};
    }
    return dir;
}

} // namespace

const std::filesystem::path& configDir()
{
    static const std::filesystem::path dir = findConfigDir();
    return dir;
}

std::string configPath(std::string_view name)
{
    return (configDir() / std::filesystem::path(name)).string();
}

void migrateLocalConfig()
{
    namespace fs = std::filesystem;
    if (configDir().empty())
        return;
    constexpr const char* kNames[] = { "settings.json", "editor_prefs.json", "editor_keys.json", "imgui.ini", "layouts" };
    std::error_code ec;
    for (const char* name : kNames) {
        const fs::path local = fs::current_path(ec) / name;
        const fs::path shared = configDir() / name;
        if (!fs::exists(local, ec) || fs::exists(shared, ec))
            continue;
        fs::copy(local, shared, fs::copy_options::recursive, ec);
        if (ec)
            LOG_ERROR("[CONFIG] Can't copy " << local.string() << " to " << shared.string() << ": " << ec.message() << "\n");
        else
            LOG_INFO("[CONFIG] Copied " << name << " to " << configDir().string() << "\n");
    }
}

namespace {

constexpr const char* kDataDir = "data";

bool writeText(const std::filesystem::path& path, const std::string& text)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream file(path, std::ios::binary);
    if (!(file << text)) {
        LOG_ERROR("[CONFIG] Can't write " << path.string() << "\n");
        return false;
    }
    return true;
}

} // namespace

std::string dataPath(std::string_view name)
{
    return (std::filesystem::path(kDataDir) / std::filesystem::path(name)).string();
}

bool writeDataFile(const std::string& path, const std::string& text)
{
    const std::filesystem::path target(path);
    if (!writeText(target, text))
        return false;
#ifdef MIRAS_SOURCE_ASSETS_DIR
    std::error_code ec;
    const std::filesystem::path sourceDir = std::filesystem::path(MIRAS_SOURCE_ASSETS_DIR) / kDataDir;
    if (target.parent_path() == kDataDir && std::filesystem::is_directory(sourceDir.parent_path(), ec))
        writeText(sourceDir / target.filename(), text);
#endif
    return true;
}

void migrateDataFiles()
{
    namespace fs = std::filesystem;
    if (configDir().empty())
        return;
    constexpr const char* kNames[] = { "weapon_tuning.json", "arms_poses.json", "entity_presets.json" };
    std::error_code ec;
    for (const char* name : kNames) {
        const fs::path old = configDir() / name;
        const std::string current = dataPath(name);
        if (fs::exists(current, ec) || !fs::exists(old, ec))
            continue;
        std::ifstream file(old, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (file.bad() || !writeDataFile(current, text))
            continue;
        fs::remove(old, ec);
        LOG_INFO("[CONFIG] Moved " << name << " from " << configDir().string() << " to " << current << "\n");
    }
}
