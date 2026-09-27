#pragma once
#include <string>

enum class LaunchMode {
    Editor,
    Game,
};

struct AppOptions {
    LaunchMode mode = LaunchMode::Editor;
    bool validation = false;
    int exitAfterFrames = -1; // -1 = run until the window is closed
    std::string scenePath;
    bool showHelp = false;
};

// Unknown or malformed arguments print a warning to stderr and are ignored.
AppOptions parseAppOptions(int argc, char** argv);
void printUsage(const char* executableName);
