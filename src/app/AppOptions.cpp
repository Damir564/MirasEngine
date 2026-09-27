#include "AppOptions.h"
#include <charconv>
#include <iostream>
#include <string_view>

namespace {

// Returns the argument after argv[i] (advancing i), or nullptr with a warning when it is missing.
const char* takeValue(int argc, char** argv, int& i, std::string_view flag, const char* what)
{
    if (i + 1 >= argc) {
        std::cerr << "Warning: " << flag << " needs " << what << ", ignoring\n";
        return nullptr;
    }
    return argv[++i];
}

void parseFrameCount(std::string_view value, AppOptions& options)
{
    int frames = 0;
    auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), frames);
    if (ec != std::errc() || end != value.data() + value.size() || frames <= 0)
        std::cerr << "Warning: invalid --exit-after-frames value '" << value << "', ignoring\n";
    else
        options.exitAfterFrames = frames;
}

} // namespace

AppOptions parseAppOptions(int argc, char** argv)
{
    AppOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--editor") {
            options.mode = LaunchMode::Editor;
        }
        else if (arg == "--game") {
            options.mode = LaunchMode::Game;
        }
        else if (arg == "--validation") {
            options.validation = true;
        }
        else if (arg == "--exit-after-frames") {
            if (const char* value = takeValue(argc, argv, i, arg, "a frame count"))
                parseFrameCount(value, options);
        }
        else if (arg == "--scene") {
            if (const char* value = takeValue(argc, argv, i, arg, "a scene path"))
                options.scenePath = value;
        }
        else if (arg == "--help" || arg == "-h") {
            options.showHelp = true;
        }
        else {
            std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
        }
    }
    return options;
}

void printUsage(const char* executableName)
{
    std::cout
        << "Usage: " << executableName << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --editor                 Start the editor (default)\n"
        << "  --game                   Start the game (main menu, plays level1.scn)\n"
        << "  --scene <path>           Editor: open this .scn at startup; game: use it as the level\n"
        << "  --validation             Enable the Vulkan validation layers\n"
        << "  --exit-after-frames <N>  Quit after N rendered frames (for automated runs)\n"
        << "  --help, -h               Print this help and exit\n";
}
