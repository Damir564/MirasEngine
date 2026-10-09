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
