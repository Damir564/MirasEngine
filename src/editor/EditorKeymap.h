#pragma once
#include <array>
#include <cstdint>
#include <string>
#include "imgui.h"

union SDL_Event;
struct SDL_KeyboardEvent;

// Every rebindable editor shortcut, in the order the Keyboard Shortcuts window lists them.
enum class EditorAction {
    NewScene, OpenScene, SaveScene, SaveSceneAs, ImportModel,
    Undo, Redo, Duplicate, Delete, Deselect, Rename, Focus,
    SelectAll, Copy, Cut, Paste, DropToFloor, RotateClockwise, RotateCounterClockwise, Hide, UnhideAll,
    SelectTool, MoveTool, RotateTool, ScaleTool, ToggleSnap, GridSmaller, GridLarger, Eyedropper,
    ObjectMode, FaceMode, EdgeMode, VertexMode, PartMode, DrawShape, UniteShapes, SeparateShape, SaveAsPrefab,
    GroupObjects, UngroupObjects,
    CopyFaceAttributes, PasteFaceAttributes, FacePushOut, FacePullIn, FaceExtrude, FaceExtrudeIn, SelectSurface,
    UvLeft, UvRight, UvUp, UvDown, UvRotateLeft, UvRotateRight, UvScaleUp, UvScaleDown,
    FaceDrawApply, FaceDrawRemovePoint, FaceDrawCancel,
    ToggleGrid, ToggleFlyMode, Play, ShowControls, ShowKeymap,
    CameraForward, CameraBack, CameraLeft, CameraRight, CameraUp, CameraDown, CameraFast,
    Count,
};

struct EditorActionInfo {
    const char* id;       // key in the keymap file
    const char* label;
    const char* category;
    ImGuiKeyChord defaults[2];
    bool repeat;          // fires again with key repeat while held
};

// The editor's key bindings: up to two chords (ImGuiKey | ImGuiMod_*) per action. A binding may also be a
// lone modifier key (e.g. ImGuiKey_LeftShift), for held actions like the fast camera.
class EditorKeymap {
public:
    static constexpr int kSlots = 2;
    static constexpr int kActionCount = static_cast<int>(EditorAction::Count);
    static const EditorActionInfo& info(EditorAction action);

    EditorKeymap() { resetAll(); }

    void resetAll();
    void reset(EditorAction action);
    bool isDefault(EditorAction action) const;
    ImGuiKeyChord binding(EditorAction action, int slot) const;
    void setBinding(EditorAction action, int slot, ImGuiKeyChord chord);

    // Pressed this ImGui frame with exactly the bound modifiers (only while the UI is visible).
    bool pressed(EditorAction action) const;
    // Held down, from SDL events so it also works in fly mode. Shift may be held in addition (it speeds up
    // the camera); Ctrl and Alt must match, so Ctrl+S does not also move the camera back.
    bool held(EditorAction action) const;
    // The SDL key-down event is one of the action's chords; for shortcuts that also work in fly mode.
    bool matches(EditorAction action, const SDL_KeyboardEvent& key) const;
    // Tracks which keys are down for held(); pass every SDL event.
    void processEvent(const SDL_Event& event);

    // First bound chord for menus and tooltips, e.g. "Ctrl+S"; empty when unbound.
    std::string shortcutLabel(EditorAction action) const;
    // Every bound chord, e.g. "Ctrl+Y, Ctrl+Shift+Z".
    std::string allShortcutsLabel(EditorAction action) const;
    // Labels of other actions bound to the same chord, comma-separated; empty when there are none.
    // Drawing keys only apply while drawing, so they never conflict.
    std::string conflicts(EditorAction action, int slot) const;

    // Missing or unknown entries keep their defaults. Only bindings that differ from the defaults are saved.
    bool load(const std::string& path);
    bool save(const std::string& path) const;

private:
    bool keyDown(ImGuiKey key) const;

    std::array<std::array<ImGuiKeyChord, kSlots>, kActionCount> m_bindings{};
    std::array<bool, ImGuiKey_NamedKey_COUNT> m_down{};
};

inline constexpr const char* kEditorKeymapPath = "editor_keys.json";

bool isModifierKey(ImGuiKey key);
// Keyboard keys only (no mouse or gamepad), the range shortcuts can use.
bool isKeyboardKey(ImGuiKey key);
// "Ctrl+Shift+S", "[", "Esc"...; empty for ImGuiKey_None.
std::string keyChordName(ImGuiKeyChord chord);
// Accepts keyChordName() output and ImGui key names, case-insensitively; ImGuiKey_None when invalid.
ImGuiKeyChord parseKeyChord(const std::string& text);
// Queues SDL key events for the chord as if it were typed into the window: modifier keys go down first and
// up last. False when a key has no SDL equivalent. For the press_keys MCP command.
bool pushKeyChordEvents(ImGuiKeyChord chord, bool down, uint32_t windowId);
