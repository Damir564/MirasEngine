#pragma once
#include <string>
#include <optional>
#include <filesystem>
#include <iostream>

class IfcConverter {
public:
    static void setExecutablePath(const std::string& path) { s_path = path; }

    static bool isAvailable() {
        return std::system((s_path + " --version > /dev/null 2>&1").c_str()) == 0;
    }

    static std::optional<std::string> toGlb(const std::string& ifcPath) {
        namespace fs = std::filesystem;

        if (!fs::exists(ifcPath)) return std::nullopt;

        fs::path input(ifcPath);
        fs::path outputDir = input.parent_path() / "converted";
        fs::create_directories(outputDir);
        std::string glbPath = (outputDir / input.stem()).string() + ".glb";

        // Skip if cached GLB is newer than IFC
        if (fs::exists(glbPath) && fs::last_write_time(glbPath) >= fs::last_write_time(ifcPath)) {
            std::cout << "[IFC] Using cached: " << glbPath << "\n";
            return glbPath;
        }

        std::string cmd = s_path
            + " --use-world-coords"
            // + " --weld-vertices"
            + " --use-element-names"
            + " --y-up"
            // + " --exclude entities IfcOpeningElemen;IfcSpace"
            + " " + ifcPath
            + " " + glbPath;

        std::cout << "[IFC] Converting: " << cmd << "\n";
        int ret = std::system(cmd.c_str());

        if (ret != 0 || !fs::exists(glbPath)) {
            std::cerr << "[IFC] Conversion failed\n";
            return std::nullopt;
        }

        std::cout << "[IFC] Converted to: " << glbPath << "\n";
        return glbPath;
    }

private:
    static inline std::string s_path = "IfcConvert";
};