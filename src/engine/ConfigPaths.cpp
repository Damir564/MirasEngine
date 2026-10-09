#include "ConfigPaths.h"
#include <cstdlib>
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
