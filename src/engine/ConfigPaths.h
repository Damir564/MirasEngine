#pragma once
#include <filesystem>
#include <string>
#include <string_view>

// Folder for user settings shared by every build (debug, release, packaged): %APPDATA%\MirasEngine,
// created on first use. Falls back to the working directory when it can't be created.
const std::filesystem::path& configDir();
// `name` inside configDir(), in the form the narrow file APIs (std::ifstream, ImGui) take.
std::string configPath(std::string_view name);
// Copies settings files (and saved layouts) that older builds kept in the working directory into
// configDir(), unless it already has them. Call once at start-up, before anything reads them.
void migrateLocalConfig();

// Game data that ships with the game (weapon tuning, arms poses, entity presets): data/<name> next to
// engine.exe, the working directory. Source: assets/data/, copied there by every build.
std::string dataPath(std::string_view name);
// Writes `text` to `path`. A file in data/ is also written to the source tree's assets/data/ when the exe
// runs from a build of it, so the edit survives the next build (which copies assets/ over data/) and can be
// committed.
bool writeDataFile(const std::string& path, const std::string& text);
// Moves game data files that older builds kept in configDir() into data/, unless data/ already has them.
void migrateDataFiles();
