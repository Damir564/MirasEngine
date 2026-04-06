#pragma once
#include <string>
#include <optional>
#include <filesystem>
#include <iostream>
#include <thread>
#include <fstream>

#define IFC_CONVERT_PARAMS " --weld-vertices --use-element-names --y-up" /*  --weld-vertices --use-element-types*/

static std::string pathToUtf8(const std::filesystem::path& p)
{
    auto u8 = p.u8string();
    return std::string(u8.begin(), u8.end());
}
class IfcConverter {
public:

    static std::string loadIfcParams() {
        std::string params = IFC_CONVERT_PARAMS;  // defaults

        std::ifstream file(s_Pathconfig);
        if (file.is_open()) {
            std::string line;
            std::getline(file, line);
            if (line.size() >= 3 &&
                (unsigned char)line[0] == 0xEF &&
                (unsigned char)line[1] == 0xBB &&
                (unsigned char)line[2] == 0xBF) {
                line.erase(0, 3);
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty()) {
                params = "\ " + line;
                std::cout << "[IFC] Config params: " << params << "\n";
            }
        }

        return params;
    }

    static std::optional<std::string> toGlb(const std::string& ifcPath) {
        namespace fs = std::filesystem;

        if (!fs::exists(fs::u8path(ifcPath))) return std::nullopt;

        fs::path input(ifcPath);
        fs::path outputDir = input.parent_path() / "converted";
        fs::create_directories(outputDir);
        std::string glbPath = pathToUtf8((outputDir / input.stem())) + ".glb";

        if (fs::exists(glbPath) && fs::last_write_time(glbPath) >= fs::last_write_time(ifcPath)) {
            std::cout << "[IFC] Using cached: " << glbPath << "\n";
            return glbPath;
        }

        unsigned int threads = std::thread::hardware_concurrency();
        if (threads == 0) threads = 4;

        std::string ifcParams = loadIfcParams();

        const std::string cmd = s_pathExe
            + " -j " + std::to_string(threads)
            + " " + " \"" + ifcPath + "\""
            + " " + " \"" + glbPath + "\""
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

    static std::optional<std::string> toJson(const std::string& ifcPath) {
        namespace fs = std::filesystem;

        if (!fs::exists(ifcPath)) return std::nullopt;

        fs::path input(ifcPath);
        fs::path outputDir = input.parent_path() / "converted";
        fs::create_directories(outputDir);
        std::string jsonPath = (outputDir / input.stem()).string() + ".json";

        if (fs::exists(jsonPath) && fs::last_write_time(jsonPath) >= fs::last_write_time(ifcPath)) {
            std::cout << "[IFC] Using cached metadata: " << jsonPath << "\n";
            return jsonPath;
        }

        const std::string cmd = s_PathExeJson
            + " \"" + ifcPath + "\""
            + " \"" + jsonPath + "\"";

        std::cout << "[IFC] Extracting metadata: " << cmd << "\n";
        int ret = std::system(cmd.c_str());

        if (ret != 0 || !fs::exists(jsonPath)) {
            std::cerr << "[IFC] Metadata extraction failed\n";
            return std::nullopt;
        }

        std::cout << "[IFC] Metadata extracted to: " << jsonPath << "\n";
        return jsonPath;
    }

private:
    static inline std::string s_pathExe = (std::filesystem::path("external") / "IfcConvert.exe").string();
    static inline std::string s_Pathconfig = (std::filesystem::path("external") / "IfcConvert.config").string();
    static inline std::string s_PathExeJson = (std::filesystem::path("external") / "extract_ifc_metadata.exe").string();
};