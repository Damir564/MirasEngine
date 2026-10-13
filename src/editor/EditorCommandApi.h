#pragma once

// Editor::CommandApi and its argument helpers, shared by EditorCommands.cpp (the command table) and
// EditorMcp.cpp (MCP server, scripts); the helpers are private to each (anonymous namespace).
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Editor.h"

using nlohmann::json;

namespace {

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

} // namespace

struct Editor::CommandApi {
    using Handler = json (*)(Editor&, const Args&);
    struct Param {
        const char* name;
        // vec3, vec2or3, number, integer, string, strings, boolean, object, points2, points3, faces, materials,
        // slots, mouse (view_mouse steps), connections (entity outputs), or enum:a|b|c
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
