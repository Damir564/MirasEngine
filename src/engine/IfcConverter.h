#pragma once
#include <string>
#include <optional>
#include <filesystem>
#include <iostream>

#define IFC_CONVERT_PARAMS " --weld-vertices --use-element-names --y-up" /*  --weld-vertices --use-element-types*/

static std::string loadIfcParams() {
    std::string params = IFC_CONVERT_PARAMS;  // defaults

    std::ifstream file("ifcparse.config");
    if (file.is_open()) {
        std::string line;
        std::getline(file, line);
        if (!line.empty()) {
            params = " " + line;
            std::cout << "[IFC] Config params: " << params << "\n";
        }
    }

    return params;
}

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

        if (fs::exists(glbPath) && fs::last_write_time(glbPath) >= fs::last_write_time(ifcPath)) {
            std::cout << "[IFC] Using cached: " << glbPath << "\n";
            return glbPath;
        }

        unsigned int threads = std::thread::hardware_concurrency();
        if (threads == 0) threads = 4;

        std::string ifcParams = loadIfcParams();

        const std::string cmd = s_path
            + " -j " + std::to_string(threads)
            + " " + ifcPath
            + " " + glbPath
            + ifcParams;

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