// Scene commands for the MCP server and command scripts (File > Run Script, --script): each has a name, a description and typed
// parameters, from which the MCP tool list is generated, plus a handler taking JSON arguments.
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
#include "FileDialog.h"
#include "McpServer.h"
#include "engine/Log.h"
#include "engine/ModelLoader.h"
#include "engine/ModelManager.h"
#include "engine/Prefab.h"
#include "engine/SceneManager.h"

using nlohmann::json;

namespace {

constexpr const char* kInstructions =
    "Edits the scene open in the MirasEngine editor. Units are meters, +Y is up. Objects are referred to "
    "by id (number) or name (string); create commands return the new object's description including its "
    "id. Rotations are Euler angles in degrees. Level shapes are built in object space resting on y = 0 and "
    "centered on X/Z, so `position` is the bottom center. Each tool call is one undo step in the editor. "
    "Use get_scene to see what exists and list_assets for prefab, material, model and scene files. "
    "Scenes are not saved automatically: call save_scene.";

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error(message);
}

// Rounded so results stay readable: float -> double conversion adds digits of noise.
double num(float v)
{
    return std::round(static_cast<double>(v) * 10000.0) / 10000.0;
}

json vecJson(const glm::vec3& v)
{
    return json::array({ num(v.x), num(v.y), num(v.z) });
}

float numberIn(const json& v, const std::string& what)
{
    if (!v.is_number())
        fail(what + " must be a number");
    const float f = v.get<float>();
    if (!std::isfinite(f))
        fail(what + " must be finite");
    return f;
}

// [x, y, z]; a single number sets all three.
glm::vec3 vec3In(const json& v, const std::string& what)
{
    if (v.is_number())
        return glm::vec3(numberIn(v, what));
    if (!v.is_array() || v.size() != 3)
        fail(what + " must be [x, y, z]");
    return { numberIn(v[0], what), numberIn(v[1], what), numberIn(v[2], what) };
}

// Typed access to a command's arguments; throws with a message naming the argument.
class Args {
public:
    explicit Args(const json& values) : m_values(values) {}

    bool has(const char* key) const
    {
        if (!m_values.is_object())
            return false;
        const auto it = m_values.find(key);
        return it != m_values.end() && !it->is_null();
    }
    const json& at(const char* key) const
    {
        if (!has(key))
            fail(std::string("missing argument '") + key + "'");
        return m_values.at(key);
    }
    float number(const char* key) const { return numberIn(at(key), key); }
    float number(const char* key, float fallback) const { return has(key) ? number(key) : fallback; }
    int integer(const char* key) const
    {
        const float f = number(key);
        if (std::floor(f) != f)
            fail(std::string(key) + " must be a whole number");
        return static_cast<int>(f);
    }
    int integer(const char* key, int fallback) const { return has(key) ? integer(key) : fallback; }
    std::string string(const char* key) const
    {
        const json& v = at(key);
        if (!v.is_string())
            fail(std::string(key) + " must be a string");
        return v.get<std::string>();
    }
    std::string string(const char* key, const std::string& fallback) const { return has(key) ? string(key) : fallback; }
    bool boolean(const char* key, bool fallback) const
    {
        if (!has(key))
            return fallback;
        const json& v = at(key);
        if (!v.is_boolean())
            fail(std::string(key) + " must be true or false");
        return v.get<bool>();
    }
    glm::vec3 vec3(const char* key) const { return vec3In(at(key), key); }
    glm::vec3 vec3(const char* key, const glm::vec3& fallback) const { return has(key) ? vec3(key) : fallback; }

private:
    const json& m_values;
};

std::string extensionOf(const std::filesystem::path& path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

// Files with one of the extensions under root, as paths relative to the working directory.
json listFiles(const std::string& root, std::initializer_list<const char*> extensions, bool recursive)
{
    namespace fs = std::filesystem;
    std::vector<std::string> paths;
    std::error_code ec;
    const auto consider = [&](const fs::directory_entry& entry) {
        std::error_code fileEc;
        if (!entry.is_regular_file(fileEc))
            return;
        const std::string ext = extensionOf(entry.path());
        for (const char* wanted : extensions) {
            if (ext == wanted)
                paths.push_back(entry.path().lexically_normal().generic_string());
        }
    };
    if (recursive) {
        for (auto it = fs::recursive_directory_iterator(root, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec))
            consider(*it);
    }
    else {
        for (auto it = fs::directory_iterator(root, ec); !ec && it != fs::directory_iterator(); it.increment(ec))
            consider(*it);
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// Footprint (x, z) extruded from y = 0 up to height.
PolyMesh makePrism(const std::vector<glm::vec2>& points, float height)
{
    const size_t n = points.size();
    float area = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const glm::vec2& a = points[i];
        const glm::vec2& b = points[(i + 1) % n];
        area += a.x * b.y - b.x * a.y;
    }
    if (std::abs(area) < 1e-6f)
        fail("points must enclose an area");
    PolyMesh mesh;
    std::vector<uint32_t> bottom, top;
    for (size_t i = 0; i < n; ++i) {
        mesh.positions.push_back({ points[i].x, 0.0f, points[i].y });
        mesh.positions.push_back({ points[i].x, height, points[i].y });
        bottom.push_back(static_cast<uint32_t>(2 * i));
        top.push_back(static_cast<uint32_t>(2 * i + 1));
    }
    mesh.addFace(bottom, { 0.0f, -1.0f, 0.0f });
    mesh.addFace(top, { 0.0f, 1.0f, 0.0f });
    for (size_t i = 0; i < n; ++i) {
        const size_t j = (i + 1) % n;
        const glm::vec2 d = points[j] - points[i];
        // Outward in the footprint plane; which side that is depends on the points' winding.
        const glm::vec2 out = area > 0.0f ? glm::vec2(d.y, -d.x) : glm::vec2(-d.y, d.x);
        const uint32_t a = static_cast<uint32_t>(2 * i), b = static_cast<uint32_t>(2 * j);
        mesh.addFace({ a, b, b + 1, a + 1 }, { out.x, 0.0f, out.y });
    }
    return mesh;
}

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

struct Editor::CommandApi {
    using Handler = json (*)(Editor&, const Args&);
    struct Param {
        const char* name;
        // vec3, vec2or3, number, integer, string, boolean, object, points2, points3, faces, materials, slots,
        // or enum:a|b|c
        const char* type;
        const char* description;
        bool required = false;
    };
    struct Command {
        const char* name;
        const char* description;
        std::vector<Param> params;
        Handler run;
        // False for commands that replace the scene or step the history (open_scene, undo): MCP only,
        // and not wrapped in an undo step.
        bool scriptable = true;
    };

    static const std::vector<Command>& table();
    static const Command* find(const std::string& name);
    static json schema(const Param& param);
    static json toolDescriptions();

    static int instanceIndex(Editor& e, const json& ref);
    static int instanceArg(Editor& e, const Args& a) { return instanceIndex(e, a.at("object")); }
    static json describe(Editor& e, size_t index);
    static json describeMesh(const PolyMesh& mesh);
    static void checkNameFree(Editor& e, const std::string& name, int except = -1);
    static std::string newName(Editor& e, const Args& a, const std::string& base);
    static json placeInstance(Editor& e, size_t modelIndex, const std::string& name, const Args& a);
    static void applyMaterial(Editor& e, PolyMesh& mesh, const std::string& materialPath);
    static PolyMaterial materialSlot(Editor& e, const json& value);
    static json addLevelObject(Editor& e, PolyMesh mesh, const std::string& base, const Args& a);
    static json createShape(Editor& e, PolyShape shape, const glm::vec3& size, const Args& a);
};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

int Editor::CommandApi::instanceIndex(Editor& e, const json& ref)
{
    const auto& instances = e.m_models.getInstances();
    // An object description (as returned by the create commands) works too.
    const json& key = ref.is_object() && ref.contains("id") ? ref["id"] : ref;
    if (key.is_number()) {
        const uint64_t id = key.get<uint64_t>();
        for (size_t i = 0; i < instances.size(); ++i) {
            if (instances[i].id == id)
                return static_cast<int>(i);
        }
        fail("no object with id " + std::to_string(id));
    }
    if (key.is_string()) {
        const std::string name = key.get<std::string>();
        for (size_t i = 0; i < instances.size(); ++i) {
            if (instances[i].name == name)
                return static_cast<int>(i);
        }
        fail("no object named '" + name + "'");
    }
    fail("object must be an id (number) or a name (string)");
}

json Editor::CommandApi::describe(Editor& e, size_t index)
{
    const ModelInstance& instance = e.m_models.getInstances()[index];
    json out = {
        { "id", instance.id },
        { "name", instance.name },
        { "position", vecJson(instance.position) },
        { "rotation", vecJson(instance.rotation) },
        { "scale", vecJson(instance.scale) },
        { "color", vecJson(instance.color) },
        { "visible", instance.visible },
        { "locked", instance.locked },
    };
    const GPUModel* model = e.m_models.getModel(instance.modelIndex);
    if (!model)
        return out;
    out["type"] = model->polyMesh ? (model->prefabPath.empty() ? "shape" : "prefab") : "model";
    if (!model->prefabPath.empty())
        out["prefab"] = model->prefabPath;
    if (!model->polyMesh)
        out["model"] = model->sourcePath;

    // World-space bounds; level shapes from their mesh, which is current even before a deferred rebuild.
    const glm::mat4 transform = instance.getTransformMatrix();
    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    const auto add = [&](const glm::vec3& p) {
        const glm::vec3 world(transform * glm::vec4(p, 1.0f));
        lo = glm::min(lo, world);
        hi = glm::max(hi, world);
    };
    if (model->polyMesh) {
        for (const glm::vec3& p : model->polyMesh->positions)
            add(p);
        out["faceCount"] = model->polyMesh->faces.size();
    }
    else {
        for (int i = 0; i < 8; ++i)
            add({ (i & 1) ? model->boundsMax.x : model->boundsMin.x, (i & 2) ? model->boundsMax.y : model->boundsMin.y,
                (i & 4) ? model->boundsMax.z : model->boundsMin.z });
    }
    if (lo.x <= hi.x)
        out["bounds"] = { { "min", vecJson(lo) }, { "max", vecJson(hi) } };
    return out;
}

json Editor::CommandApi::describeMesh(const PolyMesh& mesh)
{
    json vertices = json::array();
    for (const glm::vec3& p : mesh.positions)
        vertices.push_back(vecJson(p));
    json faces = json::array();
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        const PolyFace& face = mesh.faces[i];
        faces.push_back({
            { "index", i },
            { "verts", face.verts },
            { "normal", vecJson(mesh.faceNormal(face)) },
            { "center", vecJson(mesh.faceCenter(face)) },
            { "material", face.material < mesh.materials.size() ? json(face.material) : json(nullptr) },
        });
    }
    json materials = json::array();
    for (const PolyMaterial& slot : mesh.materials) {
        if (!slot.materialPath.empty())
            materials.push_back({ { "material", slot.materialPath } });
        else
            materials.push_back({ { "color", vecJson(glm::vec3(slot.color)) }, { "texture", slot.texturePath } });
    }
    return { { "gridSize", num(mesh.gridSize) }, { "vertices", vertices }, { "faces", faces }, { "materials", materials } };
}

void Editor::CommandApi::checkNameFree(Editor& e, const std::string& name, int except)
{
    if (name.empty())
        fail("name must not be empty");
    const auto& instances = e.m_models.getInstances();
    for (size_t i = 0; i < instances.size(); ++i) {
        if (static_cast<int>(i) != except && instances[i].name == name)
            fail("an object named '" + name + "' already exists");
    }
}

std::string Editor::CommandApi::newName(Editor& e, const Args& a, const std::string& base)
{
    if (!a.has("name"))
        return e.uniqueInstanceName(base);
    const std::string name = a.string("name");
    checkNameFree(e, name);
    return name;
}

json Editor::CommandApi::placeInstance(Editor& e, size_t modelIndex, const std::string& name, const Args& a)
{
    const size_t index = e.m_models.createInstance(modelIndex, a.vec3("position", glm::vec3(0.0f)),
        a.vec3("rotation", glm::vec3(0.0f)), a.vec3("scale", glm::vec3(1.0f)));
    ModelInstance& instance = e.m_models.getInstances()[index];
    instance.name = name;
    instance.color = a.vec3("color", glm::vec3(1.0f));
    e.markSceneChanged();
    return describe(e, index);
}

void Editor::CommandApi::applyMaterial(Editor& e, PolyMesh& mesh, const std::string& materialPath)
{
    if (!e.m_materials.find(materialPath))
        fail("no material " + materialPath + " (list_assets shows the .mat files)");
    const uint32_t slot = levelSlotFor(mesh, materialPath);
    if (slot == kNoPolyMaterial)
        fail("too many materials on one shape");
    for (PolyFace& face : mesh.faces)
        face.material = slot;
    pruneLinkedSlots(mesh);
}

// A .mat path, or an object as get_object lists slots: { material } or { color, roughness, metallic, texture }.
PolyMaterial Editor::CommandApi::materialSlot(Editor& e, const json& value)
{
    PolyMaterial slot;
    const json& path = value.is_object() && value.contains("material") ? value["material"] : value;
    if (path.is_string()) {
        slot.materialPath = path.get<std::string>();
        if (!e.m_materials.find(slot.materialPath))
            fail("no material " + slot.materialPath + " (list_assets shows the .mat files)");
        return slot;
    }
    if (!value.is_object())
        fail("each materials entry must be a .mat path or an object");
    if (value.contains("color"))
        slot.color = glm::vec4(vec3In(value["color"], "materials color"), 1.0f);
    if (value.contains("roughness"))
        slot.roughness = std::clamp(numberIn(value["roughness"], "materials roughness"), 0.0f, 1.0f);
    if (value.contains("metallic"))
        slot.metallic = std::clamp(numberIn(value["metallic"], "materials metallic"), 0.0f, 1.0f);
    if (value.contains("texture") && !value["texture"].is_null()) {
        if (!value["texture"].is_string())
            fail("materials texture must be a string");
        slot.texturePath = value["texture"].get<std::string>();
    }
    return slot;
}

json Editor::CommandApi::addLevelObject(Editor& e, PolyMesh mesh, const std::string& base, const Args& a)
{
    mesh.gridSize = e.m_gridSize;
    if (a.has("material"))
        applyMaterial(e, mesh, a.string("material"));
    const std::string name = newName(e, a, base);
    const auto modelIndex = e.createLevelModel(std::move(mesh), name);
    if (!modelIndex)
        fail(e.m_statusMessage);
    return placeInstance(e, *modelIndex, name, a);
}

json Editor::CommandApi::createShape(Editor& e, PolyShape shape, const glm::vec3& size, const Args& a)
{
    if (glm::any(glm::lessThanEqual(size, glm::vec3(0.0f))) && shape != PolyShape::Plane)
        fail("size must be positive");
    PolyShapeParams params;
    params.shape = shape;
    params.size = size;
    params.segments = a.integer("segments", shape == PolyShape::Plane ? 1 : 16);
    params.steps = a.integer("steps", 6);
    params.thickness = a.number("thickness", 0.5f);
    return addLevelObject(e, makePolyShape(params), kPolyShapeNames[static_cast<int>(shape)], a);
}

// ---------------------------------------------------------------------------------------------
// Command table
// ---------------------------------------------------------------------------------------------

const std::vector<Editor::CommandApi::Command>& Editor::CommandApi::table()
{
    using P = Param;
    const P object{ "object", "object", "Object id (number) or name (string)", true };
    const P position{ "position", "vec3", "World position [x, y, z]; for level shapes the bottom center. Default [0, 0, 0]" };
    const P rotation{ "rotation", "vec3", "Euler angles in degrees [x, y, z]" };
    const P scale{ "scale", "vec3", "Scale [x, y, z], default [1, 1, 1]" };
    const P name{ "name", "string", "Object name; must be unused. Default: a unique name" };
    const P material{ "material", "string", "Shared material for every face, e.g. materials/Brick.mat (see list_assets)" };
    const P color{ "color", "vec3", "Tint multiplied into the material color, RGB 0-1" };
    const P size{ "size", "vec3", "Size [x, y, z]: width, height, depth", true };

    static const std::vector<Command> commands = {
        // ---- Scene ----
        { "get_scene", "Lists every object in the scene with id, name, type (shape, prefab, model), transform, color and "
            "world bounds, plus the scene file path and the selected object.", {},
            [](Editor& e, const Args&) -> json {
                json objects = json::array();
                for (size_t i = 0; i < e.m_models.getInstances().size(); ++i)
                    objects.push_back(describe(e, i));
                const auto& instances = e.m_models.getInstances();
                return {
                    { "scenePath", e.m_scenes.currentPath() },
                    { "loading", e.m_scenes.isLoading() },
                    { "selected", e.hasSelection() ? json(instances[e.m_gizmo.selectedInstance].id) : json(nullptr) },
                    { "objects", objects },
                };
            } },
        { "get_object", "Describes one object. For level shapes and prefabs also returns the geometry in object space: "
            "vertices, faces (index, vertex indices, normal, center, material slot) and material slots.",
            { object, { "geometry", "boolean", "Include the mesh of level shapes (default true)" } },
            [](Editor& e, const Args& a) -> json {
                const int index = instanceArg(e, a);
                json out = describe(e, index);
                const GPUModel* model = e.m_models.getModel(e.m_models.getInstances()[index].modelIndex);
                if (model && model->polyMesh && a.boolean("geometry", true))
                    out["mesh"] = describeMesh(*model->polyMesh);
                return out;
            } },
        { "list_assets", "Lists files usable in the scene: prefabs (prefabs/*.prefab), shared materials (materials/*.mat), "
            "glTF models (models/), scenes (*.scn) and level shape kinds.", {},
            [](Editor& e, const Args&) -> json {
                return {
                    { "prefabs", listFiles(kPrefabsRoot, { ".prefab" }, true) },
                    { "materials", listFiles(e.m_materials.root(), { ".mat" }, true) },
                    { "models", listFiles(kModelsRoot, { ".gltf", ".glb" }, true) },
                    { "scenes", listFiles(kScenesRoot, { ".scn" }, false) },
                    { "shapes", std::vector<std::string>(std::begin(kPolyShapeNames), std::end(kPolyShapeNames)) },
                };
            } },
        { "new_scene", "Clears the scene and the undo history. Unsaved changes are lost.", {},
            [](Editor& e, const Args&) -> json {
                e.newScene();
                return { { "ok", true } };
            }, false },
        { "open_scene", "Opens a .scn file, replacing the scene and clearing the undo history. Models load in the "
            "background; get_scene reports loading until they are in.",
            { { "path", "string", "Scene file, relative to the editor's working directory", true } },
            [](Editor& e, const Args& a) -> json {
                const std::string path = a.string("path");
                if (!std::filesystem::exists(path))
                    fail("no such file: " + path);
                // As openScene(), but reporting the outcome.
                const SceneManager::OpenResult opened = e.m_scenes.open(path);
                if (!opened.ok)
                    fail("failed to open scene " + path);
                e.deselectAll();
                e.clearHistory();
                e.setStatus("Opening " + path + " (" + std::to_string(opened.queuedModels) + " models)...");
                return { { "ok", true }, { "path", path }, { "missingFiles", opened.missingFiles } };
            }, false },
        { "save_scene", "Saves the scene to its file, or to `path` (which then becomes the scene's file).",
            { { "path", "string", "Target .scn file; default: the current scene file" } },
            [](Editor& e, const Args& a) -> json {
                std::string path = a.string("path", e.m_scenes.currentPath());
                if (path.empty())
                    fail("the scene has no file yet; pass a path");
                if (extensionOf(path) != ".scn")
                    path += ".scn";
                if (!e.m_scenes.save(path))
                    fail("failed to save " + path);
                e.m_scenes.setCurrentPath(path);
                e.setStatus("Scene saved: " + path);
                return { { "ok", true }, { "path", path } };
            } },
        { "undo", "Undoes the last editor step (an MCP call is one step).", {},
            [](Editor& e, const Args&) -> json {
                const bool had = !e.m_undoStack.empty();
                e.undo();
                return { { "ok", had }, { "message", e.m_statusMessage } };
            }, false },
        { "redo", "Redoes the last undone step.", {},
            [](Editor& e, const Args&) -> json {
                const bool had = !e.m_redoStack.empty();
                e.redo();
                return { { "ok", had }, { "message", e.m_statusMessage } };
            }, false },

        // ---- Level shapes ----
        { "create_box", "Adds a box level shape (editable geometry).",
            { position, size, rotation, name, material, color },
            [](Editor& e, const Args& a) { return createShape(e, PolyShape::Box, a.vec3("size"), a); } },
        { "create_plane", "Adds a flat, single-sided plane level shape facing up (a floor).",
            { position, { "size", "vec2or3", "[width, depth], or [x, y, z] with y ignored", true },
                { "segments", "integer", "Grid cells per side (default 1)" }, rotation, name, material, color },
            [](Editor& e, const Args& a) {
                const json& s = a.at("size");
                const glm::vec3 size = s.is_array() && s.size() == 2
                    ? glm::vec3(numberIn(s[0], "size"), 0.0f, numberIn(s[1], "size")) : a.vec3("size");
                if (size.x <= 0.0f || size.z <= 0.0f)
                    fail("size must be positive");
                return createShape(e, PolyShape::Plane, size, a);
            } },
        { "create_cylinder", "Adds a cylinder level shape standing on its base.",
            { position, { "size", "vec3", "[diameter x, height, diameter z]", true },
                { "segments", "integer", "Number of sides (default 16)" }, rotation, name, material, color },
            [](Editor& e, const Args& a) { return createShape(e, PolyShape::Cylinder, a.vec3("size"), a); } },
        { "create_wedge", "Adds a wedge (ramp) level shape: full height along the back (-Z), sloping down towards +Z.",
            { position, size, rotation, name, material, color },
            [](Editor& e, const Args& a) { return createShape(e, PolyShape::Wedge, a.vec3("size"), a); } },
        { "create_stairs", "Adds a staircase level shape climbing towards -Z.",
            { position, size, { "steps", "integer", "Number of steps (default 6)" }, rotation, name, material, color },
            [](Editor& e, const Args& a) { return createShape(e, PolyShape::Stairs, a.vec3("size"), a); } },
        { "create_arch", "Adds an arch level shape: a half-ellipse in the XY plane, passable along Z.",
            { position, size, { "segments", "integer", "Segments along the curve (default 16)" },
                { "thickness", "number", "Thickness of the arch band (default 0.5)" }, rotation, name, material, color },
            [](Editor& e, const Args& a) { return createShape(e, PolyShape::Arch, a.vec3("size"), a); } },
        { "create_prism", "Adds a level shape from a floor outline: the polygon [[x, z], ...] (object space, any "
            "winding, may be concave) extruded from y = 0 up to `height`. Good for walls, rooms and platforms of any shape.",
            { { "points", "points2", "Outline corners [[x, z], ...], at least 3", true },
                { "height", "number", "Height of the extrusion", true }, position, rotation, name, material, color },
            [](Editor& e, const Args& a) {
                const json& list = a.at("points");
                if (!list.is_array() || list.size() < 3)
                    fail("points needs at least 3 [x, z] pairs");
                std::vector<glm::vec2> points;
                for (const json& p : list) {
                    if (!p.is_array() || p.size() != 2)
                        fail("each point must be [x, z]");
                    const glm::vec2 point(numberIn(p[0], "points"), numberIn(p[1], "points"));
                    if (!points.empty() && glm::distance(point, points.back()) < 1e-4f)
                        continue;
                    points.push_back(point);
                }
                if (points.size() > 3 && glm::distance(points.front(), points.back()) < 1e-4f)
                    points.pop_back();
                if (points.size() < 3)
                    fail("points needs at least 3 distinct corners");
                const float height = a.number("height");
                if (height <= 0.0f)
                    fail("height must be positive");
                return addLevelObject(e, makePrism(points, height), "Prism", a);
            } },
        { "create_mesh", "Adds a level shape from raw geometry in object space. Faces list vertex indices "
            "counter-clockwise seen from outside; each face must be planar (it may be concave). For several "
            "materials on one shape, list the slots in `materials` and pick one per face with `faceMaterials`.",
            { { "vertices", "points3", "Positions [[x, y, z], ...]", true },
                { "faces", "faces", "Faces [[i0, i1, i2, ...], ...], at least 3 indices each", true },
                { "materials", "materials", "Material slots: a .mat path, or { color: [r, g, b], roughness, metallic, "
                    "texture } for an inline material. Not with `material`" },
                { "faceMaterials", "slots", "Slot index into `materials` for each face, in face order (null: no "
                    "material). Default: every face uses slot 0" },
                position, rotation, scale, name, material, color },
            [](Editor& e, const Args& a) {
                PolyMesh mesh;
                const json& vertices = a.at("vertices");
                const json& faces = a.at("faces");
                if (!vertices.is_array() || !faces.is_array() || faces.empty())
                    fail("vertices and faces must be non-empty arrays");
                for (const json& v : vertices)
                    mesh.positions.push_back(vec3In(v, "vertices"));
                for (const json& f : faces) {
                    if (!f.is_array() || f.size() < 3)
                        fail("each face needs at least 3 vertex indices");
                    PolyFace face;
                    for (const json& i : f) {
                        if (!i.is_number_integer() && !(i.is_number() && std::floor(i.get<double>()) == i.get<double>()))
                            fail("face entries must be vertex indices");
                        const int64_t index = i.get<int64_t>();
                        if (index < 0 || index >= static_cast<int64_t>(mesh.positions.size()))
                            fail("face index " + std::to_string(index) + " is out of range");
                        if (std::find(face.verts.begin(), face.verts.end(), static_cast<uint32_t>(index)) != face.verts.end())
                            fail("a face uses vertex " + std::to_string(index) + " twice");
                        face.verts.push_back(static_cast<uint32_t>(index));
                    }
                    mesh.faces.push_back(std::move(face));
                }
                if (a.has("material") && (a.has("materials") || a.has("faceMaterials")))
                    fail("pass either material or materials/faceMaterials");
                if (a.has("materials")) {
                    const json& slots = a.at("materials");
                    if (!slots.is_array() || slots.size() > kMaxPolyMaterialSlots)
                        fail("materials must be an array of at most " + std::to_string(kMaxPolyMaterialSlots) + " slots");
                    for (const json& slot : slots)
                        mesh.materials.push_back(materialSlot(e, slot));
                }
                if (a.has("faceMaterials")) {
                    const json& slots = a.at("faceMaterials");
                    if (!slots.is_array() || slots.size() != mesh.faces.size())
                        fail("faceMaterials needs one entry per face (" + std::to_string(mesh.faces.size()) + ")");
                    for (size_t i = 0; i < slots.size(); ++i) {
                        if (slots[i].is_null()) {
                            mesh.faces[i].material = kNoPolyMaterial;
                            continue;
                        }
                        const float slot = numberIn(slots[i], "faceMaterials");
                        if (std::floor(slot) != slot || slot < 0.0f || slot >= static_cast<float>(mesh.materials.size()))
                            fail("faceMaterials[" + std::to_string(i) + "] is not a slot index into materials");
                        mesh.faces[i].material = static_cast<uint32_t>(slot);
                    }
                }
                mesh.removeUnusedVertices();
                return addLevelObject(e, std::move(mesh), "Mesh", a);
            } },

        // ---- Materials ----
        { "create_material", "Creates a shared material file (materials/<name>.mat) that level shapes and models can "
            "use; returns its path. Texture paths may be absolute or relative to the editor's working directory.",
            { { "name", "string", "Material name, used as the file name", true },
                { "texture", "string", "Base color texture file" },
                { "normalTexture", "string", "Normal map file" },
                { "color", "vec3", "Color multiplied into the texture, RGB 0-1 (default white)" },
                { "roughness", "number", "0-1, default 0.8" },
                { "metallic", "number", "0-1, default 0" },
                { "texelSize", "number", "World units one texture repeat covers on level shapes (default 1)" },
                { "replace", "boolean", "Overwrite a material of the same name instead of adding a numbered one" } },
            [](Editor& e, const Args& a) -> json {
                MaterialAsset material;
                const auto texture = [&](const char* key) {
                    if (!a.has(key))
                        return std::string();
                    const std::string path = a.string(key);
                    if (!std::filesystem::is_regular_file(path))
                        fail(std::string(key) + ": no such file " + path);
                    return toStoredPath(path);
                };
                material.baseColorTexture = texture("texture");
                material.normalTexture = texture("normalTexture");
                material.color = glm::vec4(glm::clamp(a.vec3("color", glm::vec3(1.0f)), 0.0f, 1.0f), 1.0f);
                material.roughness = std::clamp(a.number("roughness", 0.8f), 0.0f, 1.0f);
                material.metallic = std::clamp(a.number("metallic", 0.0f), 0.0f, 1.0f);
                material.texelSize = std::max(a.number("texelSize", 1.0f), 0.01f);
                const std::string name = a.string("name");
                if (name.empty() || name.find_first_of("<>:\"/\\|?*") != std::string::npos)
                    fail("name must be non-empty and usable as a file name");
                std::string path = (std::filesystem::path(e.m_materials.root()) / (name + ".mat")).generic_string();
                if (a.boolean("replace", false) && e.m_materials.find(path)) {
                    material.path = path;
                    material.name = name;
                    if (!e.m_materials.save(material))
                        fail("failed to save " + path);
                }
                else if (const auto created = e.m_materials.create(name, material)) {
                    path = *created;
                }
                else {
                    fail("failed to create material " + name);
                }
                return { { "ok", true }, { "path", path }, { "texture", material.baseColorTexture },
                    { "normalTexture", material.normalTexture } };
            } },

        // ---- Prefabs and models ----
        { "add_prefab", "Adds an instance of a .prefab file. Instances of one prefab share their geometry and start locked.",
            { { "path", "string", "Prefab file, e.g. prefabs/Pillar.prefab (see list_assets)", true },
                position, rotation, scale, name, color },
            [](Editor& e, const Args& a) -> json {
                const std::string path = a.string("path");
                const auto modelIndex = e.loadPrefabModel(path);
                if (!modelIndex)
                    fail(e.m_statusMessage);
                const std::string instanceName = newName(e, a, e.m_models.getModel(*modelIndex)->name);
                json out = placeInstance(e, *modelIndex, instanceName, a);
                e.m_models.getInstances().back().locked = true;
                out["locked"] = true;
                return out;
            } },
        { "save_prefab", "Saves a level shape as a .prefab file and links the object to it (locked). Other instances "
            "of that file in the scene follow the new geometry.",
            { object, { "path", "string", "Target file; default prefabs/<object name>.prefab" } },
            [](Editor& e, const Args& a) -> json {
                const int index = instanceArg(e, a);
                const ModelInstance instance = e.m_models.getInstances()[index];
                const GPUModel* model = e.m_models.getModel(instance.modelIndex);
                if (!model || !model->polyMesh)
                    fail(instance.name + " is not a level shape");
                std::filesystem::path path = a.string("path", std::string(kPrefabsRoot) + "/" + instance.name + kPrefabExtension);
                if (extensionOf(path) != kPrefabExtension)
                    path += kPrefabExtension;
                std::error_code ec;
                if (path.has_parent_path())
                    std::filesystem::create_directories(path.parent_path(), ec);
                const std::string stored = toStoredPath(path.string());
                e.saveAsPrefab(index, stored);
                const int after = instanceIndex(e, json(instance.id));
                const GPUModel* saved = e.m_models.getModel(e.m_models.getInstances()[after].modelIndex);
                if (!saved || saved->prefabPath != stored)
                    fail(e.m_statusMessage);
                return describe(e, after);
            } },
        { "add_model", "Adds an instance of a glTF/GLB model file, or of the built-in cube (path builtin:cube).",
            { { "path", "string", "Model file, e.g. models/tree.glb (see list_assets)", true },
                position, rotation, scale, name, color },
            [](Editor& e, const Args& a) -> json {
                const std::string path = a.string("path");
                std::optional<size_t> modelIndex = e.m_models.findModelByPath(path);
                if (!modelIndex) {
                    if (!isBuiltinModelPath(path) && !std::filesystem::exists(path))
                        fail("no such file: " + path);
                    try {
                        modelIndex = e.m_models.loadModelSync(path, std::filesystem::path(path).stem().string());
                    }
                    catch (const std::exception& ex) {
                        fail("failed to load " + path + ": " + ex.what());
                    }
                }
                const std::string instanceName = newName(e, a, e.m_models.getModel(*modelIndex)->name);
                return placeInstance(e, *modelIndex, instanceName, a);
            } },

        // ---- Objects ----
        { "set_object", "Changes an object's properties; only the given ones change. `locked` protects a level shape's "
            "geometry from edits (prefab instances start locked).",
            { object, { "name", "string", "New name (must be unused)" }, position, rotation, scale, color,
                { "visible", "boolean", "Shown in the scene" }, { "locked", "boolean", "Geometry protected from edits" } },
            [](Editor& e, const Args& a) -> json {
                const int index = instanceArg(e, a);
                if (a.has("name"))
                    checkNameFree(e, a.string("name"), index);
                ModelInstance& instance = e.m_models.getInstances()[index];
                if (a.has("name")) instance.name = a.string("name");
                instance.position = a.vec3("position", instance.position);
                instance.rotation = a.vec3("rotation", instance.rotation);
                instance.scale = a.vec3("scale", instance.scale);
                instance.color = a.vec3("color", instance.color);
                instance.visible = a.boolean("visible", instance.visible);
                instance.locked = a.boolean("locked", instance.locked);
                e.markSceneChanged();
                return describe(e, index);
            } },
        { "delete_object", "Removes an object from the scene.", { object },
            [](Editor& e, const Args& a) -> json {
                const int index = instanceArg(e, a);
                const std::string deleted = e.m_models.getInstances()[index].name;
                e.deleteInstance(index);
                return { { "deleted", deleted } };
            } },
        { "duplicate_object", "Copies an object. Level shapes get their own copy of the geometry; prefab instances keep "
            "sharing theirs.",
            { object, { "offset", "vec3", "Moves the copy by this much; default: beside the original along X" }, name },
            [](Editor& e, const Args& a) -> json {
                const int index = instanceArg(e, a);
                const ModelInstance source = e.m_models.getInstances()[index];
                const std::string copyName = newName(e, a, source.name);
                const size_t count = e.m_models.getInstances().size();
                e.duplicateInstance(index);
                if (e.m_models.getInstances().size() == count)
                    fail(e.m_statusMessage);
                ModelInstance& copy = e.m_models.getInstances().back();
                copy.name = copyName;
                if (a.has("offset"))
                    copy.position = source.position + a.vec3("offset");
                return describe(e, e.m_models.getInstances().size() - 1);
            } },
        { "select_object", "Selects an object in the editor, or clears the selection when `object` is omitted.",
            { { "object", "object", "Object id or name" } },
            [](Editor& e, const Args& a) -> json {
                if (!a.has("object")) {
                    e.deselectAll();
                    return { { "selected", nullptr } };
                }
                const int index = instanceArg(e, a);
                e.selectInstance(index);
                return { { "selected", e.m_models.getInstances()[index].id } };
            } },
        { "focus_object", "Moves the editor camera so the object fills the view.", { object },
            [](Editor& e, const Args& a) -> json {
                e.focusOnInstance(instanceArg(e, a));
                return { { "position", vecJson(e.m_camera.position) } };
            } },
        { "get_camera", "Returns the editor camera: position, yaw and pitch in degrees, and the view direction.", {},
            [](Editor& e, const Args&) -> json {
                return { { "position", vecJson(e.m_camera.position) }, { "yaw", num(e.m_camera.yaw) },
                    { "pitch", num(e.m_camera.pitch) }, { "forward", vecJson(getFront(e.m_camera)) } };
            } },
        { "set_camera", "Moves the editor camera. Yaw -90 looks along -Z, 0 along +X; pitch is up/down. `look_at` "
            "turns the camera towards a point instead.",
            { { "position", "vec3", "Camera position" }, { "yaw", "number", "Degrees" }, { "pitch", "number", "Degrees, -89 to 89" },
                { "look_at", "vec3", "Point to look at" } },
            [](Editor& e, const Args& a) -> json {
                Camera& camera = e.m_camera;
                camera.position = a.vec3("position", camera.position);
                camera.yaw = a.number("yaw", camera.yaw);
                camera.pitch = a.number("pitch", camera.pitch);
                if (a.has("look_at")) {
                    const glm::vec3 to = a.vec3("look_at") - camera.position;
                    if (glm::length(to) > 1e-4f) {
                        const glm::vec3 dir = glm::normalize(to);
                        camera.pitch = glm::degrees(std::asin(std::clamp(dir.y, -1.0f, 1.0f)));
                        camera.yaw = glm::degrees(std::atan2(dir.z, dir.x));
                    }
                }
                camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);
                return { { "position", vecJson(camera.position) }, { "yaw", num(camera.yaw) }, { "pitch", num(camera.pitch) } };
            } },
    };
    return commands;
}

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

Editor::~Editor() = default;

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

PolyMesh& Editor::batchEditMesh(size_t modelIndex)
{
    GPUModel* model = m_models.getModel(modelIndex);
    const std::string& path = model->sourcePath;
    const bool existed = std::find(m_batch.existingModels.begin(), m_batch.existingModels.end(), path) != m_batch.existingModels.end();
    const bool recorded = std::any_of(m_batch.edits.begin(), m_batch.edits.end(), [&](const LevelEdit& edit) { return edit.modelPath == path; });
    // Shapes created during the batch need no geometry entry: their creation records the final mesh.
    if (existed && !recorded)
        m_batch.edits.push_back({ path, *model->polyMesh, {} });
    if (std::find(m_batch.dirty.begin(), m_batch.dirty.end(), path) == m_batch.dirty.end())
        m_batch.dirty.push_back(path);
    return *model->polyMesh;
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
