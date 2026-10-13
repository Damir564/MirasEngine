// Serves the command table (EditorCommands.cpp) to MCP clients and command scripts: tool schemas, the MCP
// server, script execution, and the undo batch around each call.
#include "Editor.h"
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <nlohmann/json.hpp>
#include <SDL3/SDL_mouse.h>
#include "FileDialog.h"
#include "McpServer.h"
#include "engine/EntityTypes.h"
#include "engine/Log.h"
#include "engine/ModelLoader.h"
#include "engine/ModelManager.h"
#include "engine/Prefab.h"
#include "engine/SceneManager.h"
#include "EditorCommandApi.h"

namespace {

constexpr const char* kInstructions =
    "Edits the scene open in the MirasEngine editor. Units are meters, +Y is up. Objects are referred to "
    "by id (number) or name (string); create commands return the new object's description including its "
    "id. Rotations are Euler angles in degrees. Level shapes are built in object space resting on y = 0 and "
    "centered on X/Z, so `position` is the bottom center. Each tool call is one undo step in the editor. "
    "Use get_scene to see what exists and list_assets for prefab, material, model and scene files. "
    "Scenes are not saved automatically: call save_scene.";

// One line of a command script: `command key=value key=value ...`, values in JSON ("text", 1.5,
// [x, y, z], true); a bare word without quotes is taken as a string. Empty lines and # comments are
// skipped (empty command). Throws with a message on malformed lines.
struct ScriptLine {
    int number = 0;
    std::string command;
    json args = json::object();
};

ScriptLine parseScriptLine(const std::string& line, int number)
{
    ScriptLine out;
    out.number = number;
    size_t i = 0;
    const auto skipSpace = [&] {
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])))
            ++i;
    };
    skipSpace();
    if (i == line.size() || line[i] == '#')
        return out;
    const size_t nameStart = i;
    while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i])))
        ++i;
    out.command = line.substr(nameStart, i - nameStart);
    for (;;) {
        skipSpace();
        if (i == line.size() || line[i] == '#')
            break;
        const size_t keyStart = i;
        while (i < line.size() && line[i] != '=' && !std::isspace(static_cast<unsigned char>(line[i])))
            ++i;
        const std::string key = line.substr(keyStart, i - keyStart);
        if (i == line.size() || line[i] != '=' || key.empty())
            fail("expected key=value, got '" + key + "'");
        ++i;
        // A string runs to its closing quote, an array or object to its matching bracket (spaces
        // allowed inside), anything else to the next space.
        const size_t valueStart = i;
        int depth = 0;
        bool inString = false;
        for (; i < line.size(); ++i) {
            const char c = line[i];
            if (inString) {
                if (c == '\\')
                    ++i;
                else if (c == '"')
                    inString = false;
            }
            else if (c == '"')
                inString = true;
            else if (c == '[' || c == '{')
                ++depth;
            else if (c == ']' || c == '}')
                --depth;
            else if (depth == 0 && std::isspace(static_cast<unsigned char>(c)))
                break;
        }
        if (inString || depth != 0)
            fail("unterminated value for " + key);
        const std::string text = line.substr(valueStart, i - valueStart);
        if (text.empty())
            fail("missing value for " + key);
        json value = json::parse(text, nullptr, false);
        if (value.is_discarded()) {
            if (text.find_first_of("\"[]{}") != std::string::npos)
                fail("invalid value for " + key + ": " + text);
            value = text;
        }
        if (out.args.contains(key))
            fail(key + " given twice");
        out.args[key] = std::move(value);
    }
    return out;
}

} // namespace

const Editor::CommandApi::Command* Editor::CommandApi::find(const std::string& name)
{
    for (const Command& command : table()) {
        if (name == command.name)
            return &command;
    }
    return nullptr;
}

json Editor::CommandApi::schema(const Param& param)
{
    const std::string type = param.type;
    const json number = { { "type", "number" } };
    const auto numbers = [&](int count) {
        return json{ { "type", "array" }, { "items", number }, { "minItems", count }, { "maxItems", count } };
    };
    json out;
    if (type == "vec3")
        out = numbers(3);
    else if (type == "vec2or3")
        out = { { "type", "array" }, { "items", number }, { "minItems", 2 }, { "maxItems", 3 } };
    else if (type == "object")
        out = { { "type", json::array({ "integer", "string" }) } };
    else if (type == "points2")
        out = { { "type", "array" }, { "items", numbers(2) }, { "minItems", 3 } };
    else if (type == "points3")
        out = { { "type", "array" }, { "items", numbers(3) } };
    else if (type == "faces")
        out = { { "type", "array" }, { "items", { { "type", "array" }, { "items", { { "type", "integer" } } }, { "minItems", 3 } } } };
    else if (type == "materials") {
        const json inlineSlot = { { "type", "object" }, { "properties", {
            { "material", { { "type", "string" } } }, { "color", numbers(3) }, { "roughness", number },
            { "metallic", number }, { "texture", { { "type", "string" } } } } } };
        out = { { "type", "array" }, { "items", { { "anyOf", json::array({ { { "type", "string" } }, inlineSlot }) } } } };
    }
    else if (type == "strings")
        out = { { "type", "array" }, { "items", { { "type", "string" } } } };
    else if (type == "dict")
        out = { { "type", "object" } };
    else if (type == "objects")
        out = { { "type", "array" }, { "items", { { "type", json::array({ "integer", "string" }) } } } };
    else if (type == "mouse")
        out = { { "type", "array" }, { "items", { { "type", "object" }, { "properties", {
            { "action", { { "type", "string" }, { "enum", json::array({ "move", "down", "up", "click", "dblclick" }) } } },
            { "x", number }, { "y", number },
            { "button", { { "type", "string" }, { "enum", json::array({ "left", "right" }) } } },
            { "mods", { { "type", "string" } } } } },
            { "required", json::array({ "action" }) } } } };
    else if (type == "slots")
        out = { { "type", "array" }, { "items", { { "type", json::array({ "integer", "null" }) } } } };
    else if (type.rfind("enum:", 0) == 0) {
        json values = json::array();
        size_t start = 5;
        while (start <= type.size()) {
            const size_t bar = std::min(type.find('|', start), type.size());
            values.push_back(type.substr(start, bar - start));
            start = bar + 1;
        }
        out = { { "type", "string" }, { "enum", values } };
    }
    else
        out = { { "type", type } };
    out["description"] = param.description;
    return out;
}

json Editor::CommandApi::toolDescriptions()
{
    json tools = json::array();
    for (const Command& command : table()) {
        json properties = json::object();
        json required = json::array();
        for (const Param& param : command.params) {
            properties[param.name] = schema(param);
            if (param.required)
                required.push_back(param.name);
        }
        json inputSchema = { { "type", "object" }, { "properties", properties } };
        if (!required.empty())
            inputSchema["required"] = required;
        tools.push_back({ { "name", command.name }, { "description", command.description }, { "inputSchema", inputSchema } });
    }
    return tools;
}

// ---------------------------------------------------------------------------------------------
// Editor side
// ---------------------------------------------------------------------------------------------

Editor::~Editor()
{
    detachIniHandler();
    // beforeUiFrame() saves changes on a later frame, which a change made right before quitting never gets.
    m_prefs.cameraSpeed = m_camera.speed;
    if (m_prefs != m_savedPrefs)
        saveEditorPrefs(m_prefs);
}

void Editor::startMcpServer(uint16_t port)
{
    m_mcp = std::make_unique<McpServer>(port, CommandApi::toolDescriptions(), kInstructions);
    if (!m_mcp->running()) {
        LOG_ERROR("MCP server disabled: " << m_mcp->error() << "\n");
        setStatus("MCP server disabled: " + m_mcp->error(), true);
        m_mcp.reset();
        return;
    }
    LOG_INFO("MCP server listening on http://127.0.0.1:" << port << "/mcp\n");
    setStatus("MCP server listening on http://127.0.0.1:" + std::to_string(port) + "/mcp");
}

void Editor::pollMcpServer()
{
    if (!m_mcp || !m_mcp->hasPendingCalls())
        return;
    // Each call is one undo step, so drags and text edits finish first and earlier changes are recorded.
    if (editInProgress() || m_scenes.isLoading())
        return;
    updateObjectHistory();
    m_mcp->poll([this](const std::string& name, const json& arguments) -> McpServer::ToolResult {
        const CommandApi::Command* command = CommandApi::find(name);
        if (!command)
            return { "Unknown tool: " + name, true };
        try {
            json result;
            if (!command->scriptable) {
                result = command->run(*this, Args(arguments));
            }
            else {
                beginCommandBatch();
                try {
                    result = command->run(*this, Args(arguments));
                }
                catch (...) {
                    endCommandBatch(name);
                    throw;
                }
                endCommandBatch(name);
            }
            return { result.dump(-1, ' ', false, json::error_handler_t::replace), false };
        }
        catch (const std::exception& ex) {
            setStatus("MCP " + name + ": " + ex.what(), true);
            return { ex.what(), true };
        }
    });
}

void Editor::runScript(const std::string& path)
{
    m_pendingScript = path;
}

void Editor::pollScript()
{
    if (m_pendingScript.empty() || editInProgress() || m_scenes.isLoading())
        return;
    const std::string path = std::exchange(m_pendingScript, std::string());
    const std::string fileName = std::filesystem::path(path).filename().string();
    const auto report = [&](const std::string& message) {
        LOG_ERROR("[SCRIPT] " << fileName << ": " << message << "\n");
        setStatus(fileName + ": " + message, true);
    };

    // Everything is parsed and checked before the first command runs, so a typo changes nothing.
    std::ifstream file(path);
    if (!file) {
        report("cannot open the file");
        return;
    }
    std::vector<ScriptLine> lines;
    std::string text;
    for (int number = 1; std::getline(file, text); ++number) {
        try {
            ScriptLine line = parseScriptLine(text, number);
            if (line.command.empty())
                continue;
            const CommandApi::Command* command = CommandApi::find(line.command);
            if (!command)
                fail("unknown command " + line.command);
            if (!command->scriptable)
                fail(line.command + " can't be used in scripts");
            lines.push_back(std::move(line));
        }
        catch (const std::exception& e) {
            report("line " + std::to_string(number) + ": " + e.what());
            return;
        }
    }

    // One undo step for the whole script; on an error, what ran so far stays (and undoes together).
    updateObjectHistory();
    beginCommandBatch();
    size_t done = 0;
    std::string error;
    for (const ScriptLine& line : lines) {
        try {
            CommandApi::find(line.command)->run(*this, Args(line.args));
            ++done;
        }
        catch (const std::exception& e) {
            error = "line " + std::to_string(line.number) + ": " + line.command + ": " + e.what();
            break;
        }
    }
    endCommandBatch("script " + fileName);
    if (!error.empty()) {
        report(error + " (" + std::to_string(done) + " commands ran before it; Ctrl+Z undoes them)");
        return;
    }
    LOG_INFO("[SCRIPT] " << fileName << ": ran " << done << " commands\n");
    setStatus("Ran " + fileName + ": " + std::to_string(done) + " commands");
}

void Editor::beginCommandBatch()
{
    m_batch = {};
    m_batch.active = true;
    for (const auto& model : m_models.getModels()) {
        if (model && model->polyMesh)
            m_batch.existingModels.push_back(model->sourcePath);
    }
}

void Editor::endCommandBatch(const std::string& action)
{
    m_batch.active = false;
    // Rebuilt once at the end rather than after every edit; each rebuild waits for the GPU.
    for (const std::string& path : m_batch.dirty) {
        const auto found = m_models.findModelByPath(path);
        if (!found)
            continue;
        try {
            m_models.rebuildPolyMesh(*found);
        }
        catch (const std::exception& e) {
            setStatus("Failed to rebuild " + m_models.getModel(*found)->name + ": " + e.what(), true);
        }
    }
    HistoryEntry entry;
    entry.action = action;
    for (LevelEdit& edit : m_batch.edits) {
        // Deleted shapes come back through the object part of the entry, with their old mesh.
        const auto found = m_models.findModelByPath(edit.modelPath);
        if (!found || !m_models.getModel(*found)->polyMesh)
            continue;
        edit.after = *m_models.getModel(*found)->polyMesh;
        if (edit.modelPath == m_levelBaselinePath)
            m_levelBaseline = edit.after;
        entry.levels.push_back(std::move(edit));
    }
    if (!entry.levels.empty()) {
        pushHistory(std::move(entry));
        m_levelEntryPushed = true;
        // Face and vertex picks may refer to indices the edit moved.
        m_levelSelectionInstance = -1;
    }
    m_batch = {};
    markSceneChanged();
    updateObjectHistory();
}
