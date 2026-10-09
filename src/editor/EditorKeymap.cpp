#include "EditorKeymap.h"
#include <SDL3/SDL.h>
#include <nlohmann/json.hpp>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "engine/Log.h"

// Defined (not static) in imgui_impl_sdl3.cpp: the SDL -> ImGuiKey mapping ImGui itself uses, so held()
// and matches() see the same keys as pressed().
ImGuiKey ImGui_ImplSDL3_KeyEventToImGuiKey(SDL_Keycode keycode, SDL_Scancode scancode);

namespace {

constexpr ImGuiKeyChord kCtrl = ImGuiMod_Ctrl;
constexpr ImGuiKeyChord kShift = ImGuiMod_Shift;
constexpr ImGuiKeyChord kAlt = ImGuiMod_Alt;

constexpr const char* kFile = "File";
constexpr const char* kEdit = "Edit";
constexpr const char* kTools = "Tools";
constexpr const char* kLevel = "Level shapes";
// Keys used only while drawing (a shape, or on a face); they take priority then, so never conflict.
constexpr const char* kFaceDrawing = "Drawing";
constexpr const char* kView = "View";
constexpr const char* kCamera = "Camera";

// In EditorAction order.
const EditorActionInfo kActions[] = {
    // New scene drops unsaved changes, so it has no default shortcut.
    { "newScene", "New scene", kFile, {} },
    { "openScene", "Open scene", kFile, { kCtrl | ImGuiKey_O } },
    { "saveScene", "Save scene", kFile, { kCtrl | ImGuiKey_S } },
    { "saveSceneAs", "Save scene as", kFile, { kCtrl | kShift | ImGuiKey_S } },
    { "importModel", "Import model", kFile, { kCtrl | ImGuiKey_I } },

    { "undo", "Undo", kEdit, { kCtrl | ImGuiKey_Z } },
    { "redo", "Redo", kEdit, { kCtrl | ImGuiKey_Y, kCtrl | kShift | ImGuiKey_Z } },
    { "duplicate", "Duplicate selection", kEdit, { kCtrl | ImGuiKey_D } },
    { "delete", "Delete selection", kEdit, { ImGuiKey_Delete } },
    { "deselect", "Deselect", kEdit, { ImGuiKey_Escape } },
    { "rename", "Rename object", kEdit, { ImGuiKey_F2 } },
    { "focus", "Focus selection", kEdit, { ImGuiKey_F } },
    { "selectAll", "Select all objects", kEdit, { kCtrl | ImGuiKey_A } },
    { "copy", "Copy objects", kEdit, { kCtrl | ImGuiKey_C } },
    { "cut", "Cut objects", kEdit, { kCtrl | ImGuiKey_X } },
    { "paste", "Paste objects", kEdit, { kCtrl | ImGuiKey_V } },
    { "dropToFloor", "Drop selection to the surface below", kEdit, { ImGuiKey_End } },
    { "rotateClockwise", "Rotate selection 90 degrees clockwise", kEdit, { ImGuiKey_R } },
    { "rotateCounterClockwise", "Rotate selection 90 degrees counter-clockwise", kEdit, { kShift | ImGuiKey_R } },
    { "hide", "Hide or show selection", kEdit, { ImGuiKey_H } },
    { "unhideAll", "Show all objects", kEdit, { kAlt | ImGuiKey_H } },

    { "selectTool", "Select tool", kTools, { ImGuiKey_Q } },
    { "moveTool", "Move tool", kTools, { ImGuiKey_1 } },
    { "rotateTool", "Rotate tool", kTools, { ImGuiKey_2 } },
    { "scaleTool", "Scale tool", kTools, { ImGuiKey_3 } },
    { "toggleSnap", "Toggle grid snapping", kTools, {} },
    { "gridSmaller", "Smaller grid", kTools, { ImGuiKey_LeftBracket } },
    { "gridLarger", "Larger grid", kTools, { ImGuiKey_RightBracket } },
    { "eyedropper", "Pick material under the mouse", kTools, { ImGuiKey_I } },

    { "objectMode", "Object mode", kLevel, { kCtrl | ImGuiKey_1 } },
    { "faceMode", "Face mode", kLevel, { kCtrl | ImGuiKey_2 } },
    { "edgeMode", "Edge mode", kLevel, { kCtrl | ImGuiKey_3 } },
    { "vertexMode", "Vertex mode", kLevel, { kCtrl | ImGuiKey_4 } },
    { "partMode", "Part mode (parts of a united shape)", kLevel, { kCtrl | ImGuiKey_5 } },
    { "editShape", "Edit the selected shape / finish editing", kLevel, { ImGuiKey_Tab } },
    { "drawShape", "Draw shape tool", kLevel, { ImGuiKey_B } },
    { "uniteShapes", "Unite selected shapes into one", kLevel, { kCtrl | ImGuiKey_G } },
    { "separateShape", "Separate shape into its parts", kLevel, { kCtrl | kShift | ImGuiKey_G } },
    { "saveAsPrefab", "Save selection as prefab", kLevel, {} },
    { "groupObjects", "Group selected objects", kEdit, { kAlt | ImGuiKey_G } },
    { "ungroupObjects", "Ungroup selected objects", kEdit, { kAlt | kShift | ImGuiKey_G } },
    { "copyFace", "Copy face material and UVs", kLevel, { kCtrl | kShift | ImGuiKey_C } },
    { "pasteFace", "Paste face material and UVs", kLevel, { kCtrl | kShift | ImGuiKey_V } },
    { "facePushOut", "Push faces out one grid cell", kLevel, { kCtrl | ImGuiKey_UpArrow }, true },
    { "facePullIn", "Pull faces in one grid cell", kLevel, { kCtrl | ImGuiKey_DownArrow }, true },
    { "faceExtrude", "Extrude faces one grid cell", kLevel, { kCtrl | ImGuiKey_E } },
    { "faceExtrudeIn", "Extrude faces inwards one grid cell (cuts into the shape)", kLevel, { kCtrl | kShift | ImGuiKey_E } },
    { "selectSurface", "Select the whole flat surface of the face", kLevel, { kCtrl | kShift | ImGuiKey_A } },
    { "uvLeft", "Nudge UVs left", kLevel, { ImGuiKey_LeftArrow }, true },
    { "uvRight", "Nudge UVs right", kLevel, { ImGuiKey_RightArrow }, true },
    { "uvUp", "Nudge UVs up", kLevel, { ImGuiKey_UpArrow }, true },
    { "uvDown", "Nudge UVs down", kLevel, { ImGuiKey_DownArrow }, true },
    { "uvRotateLeft", "Rotate UVs left", kLevel, { kCtrl | ImGuiKey_LeftArrow }, true },
    { "uvRotateRight", "Rotate UVs right", kLevel, { kCtrl | ImGuiKey_RightArrow }, true },
    { "uvScaleUp", "Scale UVs up", kLevel, { kAlt | ImGuiKey_UpArrow }, true },
    { "uvScaleDown", "Scale UVs down", kLevel, { kAlt | ImGuiKey_DownArrow }, true },

    { "faceDrawApply", "Finish drawing", kFaceDrawing, { ImGuiKey_Enter, ImGuiKey_KeypadEnter } },
    { "faceDrawRemovePoint", "Remove last point", kFaceDrawing, { ImGuiKey_Backspace }, true },
    { "faceDrawCancel", "Cancel drawing", kFaceDrawing, { ImGuiKey_Escape } },

    { "toggleGrid", "Toggle grid", kView, {} },
    { "toggleFlyMode", "Toggle fly mode (hide UI)", kView, { kShift | ImGuiKey_GraveAccent } },
    { "play", "Play scene", kView, { ImGuiKey_F5 } },
    { "showControls", "Show controls", kView, { ImGuiKey_F1 } },
    { "showKeymap", "Keyboard shortcuts window", kView, {} },

    { "cameraForward", "Move forward", kCamera, { ImGuiKey_W } },
    { "cameraBack", "Move back", kCamera, { ImGuiKey_S } },
    { "cameraLeft", "Move left", kCamera, { ImGuiKey_A } },
    { "cameraRight", "Move right", kCamera, { ImGuiKey_D } },
    { "cameraUp", "Move up", kCamera, { ImGuiKey_E } },
    { "cameraDown", "Move down", kCamera, { ImGuiKey_C } },
    { "cameraFast", "Move faster (hold)", kCamera, { ImGuiKey_LeftShift, ImGuiKey_RightShift } },
};
static_assert(std::size(kActions) == static_cast<size_t>(EditorKeymap::kActionCount),
    "kActions must list every EditorAction in order");

struct KeyAlias {
    ImGuiKey key;
    const char* name;
};

// Shorter or more familiar names than ImGui's ("LeftBracket", "GraveAccent"...).
const KeyAlias kKeyAliases[] = {
    { ImGuiKey_LeftArrow, "Left" }, { ImGuiKey_RightArrow, "Right" }, { ImGuiKey_UpArrow, "Up" },
    { ImGuiKey_DownArrow, "Down" }, { ImGuiKey_Escape, "Esc" }, { ImGuiKey_Delete, "Del" },
    { ImGuiKey_Apostrophe, "'" }, { ImGuiKey_Comma, "," }, { ImGuiKey_Minus, "-" }, { ImGuiKey_Period, "." },
    { ImGuiKey_Slash, "/" }, { ImGuiKey_Semicolon, ";" }, { ImGuiKey_Equal, "=" }, { ImGuiKey_LeftBracket, "[" },
    { ImGuiKey_Backslash, "\\" }, { ImGuiKey_RightBracket, "]" }, { ImGuiKey_GraveAccent, "`" },
    { ImGuiKey_LeftCtrl, "Left Ctrl" }, { ImGuiKey_LeftShift, "Left Shift" }, { ImGuiKey_LeftAlt, "Left Alt" },
    { ImGuiKey_LeftSuper, "Left Super" }, { ImGuiKey_RightCtrl, "Right Ctrl" },
    { ImGuiKey_RightShift, "Right Shift" }, { ImGuiKey_RightAlt, "Right Alt" },
    { ImGuiKey_RightSuper, "Right Super" },
};

ImGuiKey chordKey(ImGuiKeyChord chord)
{
    return static_cast<ImGuiKey>(chord & ~ImGuiMod_Mask_);
}

ImGuiKeyChord chordMods(ImGuiKeyChord chord)
{
    return chord & ImGuiMod_Mask_;
}

const char* keyName(ImGuiKey key)
{
    for (const KeyAlias& alias : kKeyAliases)
        if (alias.key == key)
            return alias.name;
    return ImGui::GetKeyName(key);
}

ImGuiKeyChord sdlMods(SDL_Keymod mod)
{
    ImGuiKeyChord mods = 0;
    if (mod & SDL_KMOD_CTRL) mods |= ImGuiMod_Ctrl;
    if (mod & SDL_KMOD_SHIFT) mods |= ImGuiMod_Shift;
    if (mod & SDL_KMOD_ALT) mods |= ImGuiMod_Alt;
    if (mod & SDL_KMOD_GUI) mods |= ImGuiMod_Super;
    return mods;
}

bool equalsIgnoreCase(const std::string& a, const char* b)
{
    const size_t length = std::strlen(b);
    if (a.size() != length)
        return false;
    for (size_t i = 0; i < length; ++i)
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    return true;
}

std::string trim(const std::string& text)
{
    const size_t begin = text.find_first_not_of(" \t");
    if (begin == std::string::npos)
        return {};
    return text.substr(begin, text.find_last_not_of(" \t") - begin + 1);
}

} // namespace

bool isModifierKey(ImGuiKey key)
{
    return key >= ImGuiKey_LeftCtrl && key <= ImGuiKey_RightSuper;
}

bool isKeyboardKey(ImGuiKey key)
{
    return key >= ImGuiKey_NamedKey_BEGIN && key < ImGuiKey_GamepadStart;
}

std::string keyChordName(ImGuiKeyChord chord)
{
    const ImGuiKey key = chordKey(chord);
    if (key == ImGuiKey_None)
        return {};
    std::string name;
    if (chord & ImGuiMod_Ctrl) name += "Ctrl+";
    if (chord & ImGuiMod_Shift) name += "Shift+";
    if (chord & ImGuiMod_Alt) name += "Alt+";
    if (chord & ImGuiMod_Super) name += "Super+";
    return name + keyName(key);
}

ImGuiKeyChord parseKeyChord(const std::string& text)
{
    ImGuiKeyChord mods = 0;
    size_t start = 0;
    for (size_t plus = text.find('+'); plus != std::string::npos; plus = text.find('+', start)) {
        const std::string token = trim(text.substr(start, plus - start));
        if (equalsIgnoreCase(token, "Ctrl")) mods |= ImGuiMod_Ctrl;
        else if (equalsIgnoreCase(token, "Shift")) mods |= ImGuiMod_Shift;
        else if (equalsIgnoreCase(token, "Alt")) mods |= ImGuiMod_Alt;
        else if (equalsIgnoreCase(token, "Super")) mods |= ImGuiMod_Super;
        else return ImGuiKey_None;
        start = plus + 1;
    }
    const std::string keyText = trim(text.substr(start));
    for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_GamepadStart; ++k) {
        const ImGuiKey key = static_cast<ImGuiKey>(k);
        if (equalsIgnoreCase(keyText, keyName(key)) || equalsIgnoreCase(keyText, ImGui::GetKeyName(key)))
            return mods | key;
    }
    return ImGuiKey_None;
}

bool pushKeyChordEvents(ImGuiKeyChord chord, bool down, uint32_t windowId)
{
    // The scancode whose unmodified keycode ImGui maps to `key`.
    const auto sdlKey = [](ImGuiKey key, SDL_Scancode& scancode, SDL_Keycode& keycode) {
        for (int sc = 0; sc < SDL_SCANCODE_COUNT; ++sc) {
            const SDL_Keycode code = SDL_GetKeyFromScancode(static_cast<SDL_Scancode>(sc), SDL_KMOD_NONE, false);
            if (ImGui_ImplSDL3_KeyEventToImGuiKey(code, static_cast<SDL_Scancode>(sc)) == key) {
                scancode = static_cast<SDL_Scancode>(sc);
                keycode = code;
                return true;
            }
        }
        return false;
    };
    struct ModKey {
        ImGuiKeyChord flag;
        ImGuiKey key;
        SDL_Keymod mod;
    };
    const ModKey modKeys[] = {
        { ImGuiMod_Ctrl, ImGuiKey_LeftCtrl, SDL_KMOD_LCTRL }, { ImGuiMod_Shift, ImGuiKey_LeftShift, SDL_KMOD_LSHIFT },
        { ImGuiMod_Alt, ImGuiKey_LeftAlt, SDL_KMOD_LALT }, { ImGuiMod_Super, ImGuiKey_LeftSuper, SDL_KMOD_LGUI },
    };

    // Each event reports the modifiers held once it has happened, like real key events.
    struct KeyEvent {
        ImGuiKey key;
        SDL_Keymod mods;
    };
    std::vector<KeyEvent> events;
    SDL_Keymod held = SDL_KMOD_NONE;
    for (const ModKey& modKey : modKeys) {
        if (!(chord & modKey.flag))
            continue;
        held = static_cast<SDL_Keymod>(held | modKey.mod);
        if (down)
            events.push_back({ modKey.key, held });
    }
    if (down) {
        events.push_back({ chordKey(chord), held });
    }
    else {
        events.push_back({ chordKey(chord), held });
        for (auto it = std::rbegin(modKeys); it != std::rend(modKeys); ++it) {
            if (!(chord & it->flag))
                continue;
            held = static_cast<SDL_Keymod>(held & ~it->mod);
            events.push_back({ it->key, held });
        }
    }

    for (const KeyEvent& keyEvent : events) {
        SDL_Event event{};
        if (!sdlKey(keyEvent.key, event.key.scancode, event.key.key))
            return false;
        event.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
        event.key.timestamp = SDL_GetTicksNS();
        event.key.windowID = windowId;
        event.key.mod = keyEvent.mods;
        event.key.down = down;
        event.key.repeat = false;
        if (!SDL_PushEvent(&event))
            return false;
    }
    return true;
}

const EditorActionInfo& EditorKeymap::info(EditorAction action)
{
    return kActions[static_cast<int>(action)];
}

void EditorKeymap::resetAll()
{
    for (int a = 0; a < kActionCount; ++a)
        reset(static_cast<EditorAction>(a));
}

void EditorKeymap::reset(EditorAction action)
{
    const EditorActionInfo& actionInfo = info(action);
    for (int slot = 0; slot < kSlots; ++slot)
        m_bindings[static_cast<int>(action)][slot] = actionInfo.defaults[slot];
}

bool EditorKeymap::isDefault(EditorAction action) const
{
    const EditorActionInfo& actionInfo = info(action);
    for (int slot = 0; slot < kSlots; ++slot)
        if (m_bindings[static_cast<int>(action)][slot] != actionInfo.defaults[slot])
            return false;
    return true;
}

ImGuiKeyChord EditorKeymap::binding(EditorAction action, int slot) const
{
    return m_bindings[static_cast<int>(action)][slot];
}

void EditorKeymap::setBinding(EditorAction action, int slot, ImGuiKeyChord chord)
{
    m_bindings[static_cast<int>(action)][slot] = chord;
}

bool EditorKeymap::pressed(EditorAction action) const
{
    const bool repeat = info(action).repeat;
    for (ImGuiKeyChord chord : m_bindings[static_cast<int>(action)]) {
        const ImGuiKey key = chordKey(chord);
        if (key == ImGuiKey_None || !ImGui::IsKeyPressed(key, repeat))
            continue;
        if (isModifierKey(key) || ImGui::GetIO().KeyMods == chordMods(chord))
            return true;
    }
    return false;
}

bool EditorKeymap::keyDown(ImGuiKey key) const
{
    return key >= ImGuiKey_NamedKey_BEGIN && key < ImGuiKey_NamedKey_END && m_down[key - ImGuiKey_NamedKey_BEGIN];
}

bool EditorKeymap::held(EditorAction action) const
{
    const ImGuiKeyChord mods = sdlMods(SDL_GetModState());
    for (ImGuiKeyChord chord : m_bindings[static_cast<int>(action)]) {
        const ImGuiKey key = chordKey(chord);
        if (key == ImGuiKey_None || !keyDown(key))
            continue;
        if (isModifierKey(key))
            return true;
        const ImGuiKeyChord wanted = chordMods(chord);
        if ((mods & wanted) == wanted && (mods & ~wanted & ~ImGuiMod_Shift) == 0)
            return true;
    }
    return false;
}

bool EditorKeymap::matches(EditorAction action, const SDL_KeyboardEvent& event) const
{
    const ImGuiKey key = ImGui_ImplSDL3_KeyEventToImGuiKey(event.key, event.scancode);
    if (key == ImGuiKey_None)
        return false;
    const ImGuiKeyChord mods = sdlMods(event.mod);
    for (ImGuiKeyChord chord : m_bindings[static_cast<int>(action)])
        if (chordKey(chord) == key && (isModifierKey(key) || chordMods(chord) == mods))
            return true;
    return false;
}

void EditorKeymap::processEvent(const SDL_Event& event)
{
    if (event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP) {
        const ImGuiKey key = ImGui_ImplSDL3_KeyEventToImGuiKey(event.key.key, event.key.scancode);
        if (key >= ImGuiKey_NamedKey_BEGIN && key < ImGuiKey_NamedKey_END)
            m_down[key - ImGuiKey_NamedKey_BEGIN] = event.type == SDL_EVENT_KEY_DOWN;
    }
    else if (event.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
        // Key-up events go to the focused window; without this, keys held while switching away stay down.
        m_down.fill(false);
    }
}

std::string EditorKeymap::shortcutLabel(EditorAction action) const
{
    for (ImGuiKeyChord chord : m_bindings[static_cast<int>(action)])
        if (chordKey(chord) != ImGuiKey_None)
            return keyChordName(chord);
    return {};
}

std::string EditorKeymap::allShortcutsLabel(EditorAction action) const
{
    std::string label;
    for (ImGuiKeyChord chord : m_bindings[static_cast<int>(action)]) {
        if (chordKey(chord) == ImGuiKey_None)
            continue;
        if (!label.empty())
            label += ", ";
        label += keyChordName(chord);
    }
    return label;
}

std::string EditorKeymap::conflicts(EditorAction action, int slot) const
{
    const ImGuiKeyChord chord = binding(action, slot);
    if (chordKey(chord) == ImGuiKey_None || std::strcmp(info(action).category, kFaceDrawing) == 0)
        return {};
    std::string others;
    for (int a = 0; a < kActionCount; ++a) {
        const EditorActionInfo& other = kActions[a];
        if (a == static_cast<int>(action) || std::strcmp(other.category, kFaceDrawing) == 0)
            continue;
        for (ImGuiKeyChord otherChord : m_bindings[a]) {
            if (otherChord != chord)
                continue;
            if (!others.empty())
                others += ", ";
            others += other.label;
            break;
        }
    }
    return others;
}

bool EditorKeymap::load(const std::string& path)
{
    resetAll();
    std::ifstream file(path);
    if (!file)
        return false;
    const nlohmann::json json = nlohmann::json::parse(file, nullptr, false);
    if (json.is_discarded() || !json.is_object()) {
        LOG_ERROR("[KEYMAP] " << path << " is not valid JSON; using the default shortcuts\n");
        return false;
    }
    for (int a = 0; a < kActionCount; ++a) {
        const auto it = json.find(kActions[a].id);
        if (it == json.end() || !it->is_array())
            continue;
        for (int slot = 0; slot < kSlots; ++slot) {
            ImGuiKeyChord chord = ImGuiKey_None;
            if (slot < static_cast<int>(it->size()) && (*it)[slot].is_string()) {
                const std::string text = (*it)[slot].get<std::string>();
                chord = parseKeyChord(text);
                if (chord == ImGuiKey_None && !text.empty())
                    LOG_ERROR("[KEYMAP] Unknown shortcut '" << text << "' for " << kActions[a].id << "\n");
            }
            m_bindings[a][slot] = chord;
        }
    }
    return true;
}

bool EditorKeymap::save(const std::string& path) const
{
    nlohmann::json json = nlohmann::json::object();
    for (int a = 0; a < kActionCount; ++a) {
        if (isDefault(static_cast<EditorAction>(a)))
            continue;
        nlohmann::json chords = nlohmann::json::array();
        for (ImGuiKeyChord chord : m_bindings[a])
            chords.push_back(keyChordName(chord));
        json[kActions[a].id] = std::move(chords);
    }
    std::ofstream file(path);
    if (!file) {
        LOG_ERROR("[KEYMAP] Failed to write " << path << "\n");
        return false;
    }
    file << json.dump(4) << "\n";
    return static_cast<bool>(file);
}
