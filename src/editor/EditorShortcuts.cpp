#include "Editor.h"
#include <SDL3/SDL.h>
#include <cstring>
#include "EditorStyle.h"

namespace {
constexpr const char* kCapturePopup = "Press a shortcut";
constexpr float kBindingWidth = 150.0f;
}

void Editor::drawKeymapWindow()
{
    ImGui::SetNextWindowSize(ImVec2(px(600.0f), px(540.0f)), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Keyboard Shortcuts", &m_showKeymap)) {
        // Collapsed: the capture popup lives in this window, so recording can't go on.
        m_keyCapture = {};
        ImGui::End();
        return;
    }

    const float resetWidth = ImGui::CalcTextSize("Reset all").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    m_keymapFilter.Draw("##filter", ImGui::GetContentRegionAvail().x - resetWidth - ImGui::GetStyle().ItemSpacing.x);
    if (!m_keymapFilter.IsActive() && !ImGui::IsItemActive()) {
        // Placeholder text over the empty filter box.
        const ImVec2 min = ImGui::GetItemRectMin();
        ImGui::GetWindowDrawList()->AddText(ImVec2(min.x + ImGui::GetStyle().FramePadding.x,
            min.y + ImGui::GetStyle().FramePadding.y), ImGui::GetColorU32(ImGuiCol_TextDisabled), "Search actions or keys");
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset all")) {
        m_keymap.resetAll();
        m_keymap.save(kEditorKeymapPath);
        setStatus("Keyboard shortcuts reset to the defaults");
    }
    ImGui::TextDisabled("Click a shortcut to change it, right-click to clear it. Saved to %s.", kEditorKeymapPath);

    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("##keymap", 4, flags)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Shortcut", ImGuiTableColumnFlags_WidthFixed, px(kBindingWidth));
        ImGui::TableSetupColumn("Alternate", ImGuiTableColumnFlags_WidthFixed, px(kBindingWidth));
        ImGui::TableSetupColumn("##reset", ImGuiTableColumnFlags_WidthFixed,
            ImGui::CalcTextSize("Reset").x + ImGui::GetStyle().FramePadding.x * 2.0f);
        ImGui::TableHeadersRow();

        const char* category = nullptr;
        for (int a = 0; a < EditorKeymap::kActionCount; ++a) {
            const EditorAction action = static_cast<EditorAction>(a);
            const EditorActionInfo& info = EditorKeymap::info(action);
            const std::string keys = m_keymap.allShortcutsLabel(action);
            if (!m_keymapFilter.PassFilter(info.label) && !m_keymapFilter.PassFilter(info.category) &&
                !m_keymapFilter.PassFilter(keys.c_str()))
                continue;
            if (!category || std::strcmp(category, info.category) != 0) {
                category = info.category;
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(EditorStyle::kHighlight, "%s", category);
            }
            drawKeymapRow(action);
        }
        ImGui::EndTable();
    }

    updateKeyCapture();
    ImGui::End();
}

void Editor::drawKeymapRow(EditorAction action)
{
    const int index = static_cast<int>(action);
    const EditorActionInfo& info = EditorKeymap::info(action);
    ImGui::PushID(index);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(info.label);

    for (int slot = 0; slot < EditorKeymap::kSlots; ++slot) {
        ImGui::TableNextColumn();
        ImGui::PushID(slot);
        const ImGuiKeyChord chord = m_keymap.binding(action, slot);
        const std::string name = keyChordName(chord);
        const std::string conflict = m_keymap.conflicts(action, slot);
        const bool recording = m_keyCapture.action == index && m_keyCapture.slot == slot;
        if (!conflict.empty())
            ImGui::PushStyleColor(ImGuiCol_Text, EditorStyle::kError);
        if (ImGui::Button(recording ? "..." : name.empty() ? "-" : name.c_str(), ImVec2(-FLT_MIN, 0.0f))) {
            m_keyCapture = {};
            m_keyCapture.action = index;
            m_keyCapture.slot = slot;
            m_keyCapture.openPopup = true;
        }
        if (!conflict.empty())
            ImGui::PopStyleColor();
        if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && chord != ImGuiKey_None) {
            m_keymap.setBinding(action, slot, ImGuiKey_None);
            m_keymap.save(kEditorKeymapPath);
        }
        if (!conflict.empty())
            ImGui::SetItemTooltip("Also used by: %s", conflict.c_str());
        else
            ImGui::SetItemTooltip("Click to change, right-click to clear");
        ImGui::PopID();
    }

    ImGui::TableNextColumn();
    if (!m_keymap.isDefault(action)) {
        if (ImGui::Button("Reset")) {
            m_keymap.reset(action);
            m_keymap.save(kEditorKeymapPath);
        }
        std::string defaults;
        for (ImGuiKeyChord chord : info.defaults) {
            if (chord == ImGuiKey_None)
                continue;
            defaults += (defaults.empty() ? "" : ", ") + keyChordName(chord);
        }
        ImGui::SetItemTooltip("Back to the default: %s", defaults.empty() ? "none" : defaults.c_str());
    }
    ImGui::PopID();
}

void Editor::pumpSimulatedKeys()
{
    SimulatedKeys& keys = m_simulatedKeys;
    if (keys.down != ImGuiKey_None) {
        if (--keys.framesLeft > 0)
            return;
        pushKeyChordEvents(keys.down, false, SDL_GetWindowID(m_window));
        keys.down = ImGuiKey_None;
        return;
    }
    if (keys.next >= keys.queue.size()) {
        keys.queue.clear();
        keys.next = 0;
        return;
    }
    keys.down = keys.queue[keys.next++];
    keys.framesLeft = keys.holdFrames;
    pushKeyChordEvents(keys.down, true, SDL_GetWindowID(m_window));
}

bool Editor::uiOwnsMouse() const
{
    return ImGui::GetIO().WantCaptureMouse && !m_mouseSimulated;
}

void Editor::pumpSimulatedMouse()
{
    if (m_nextMouseStep >= m_mouseSteps.size()) {
        m_mouseSteps.clear();
        m_nextMouseStep = 0;
        return;
    }
    const MouseStep step = m_mouseSteps[m_nextMouseStep++];
    const float x = m_sceneView.x + step.position.x * m_sceneView.width;
    const float y = m_sceneView.y + step.position.y * m_sceneView.height;
    SDL_Event event{};
    if (step.action == MouseStep::Action::Move) {
        event.type = SDL_EVENT_MOUSE_MOTION;
        event.motion.windowID = SDL_GetWindowID(m_window);
        event.motion.x = x;
        event.motion.y = y;
    }
    else {
        event.type = step.action == MouseStep::Action::Down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
        event.button.windowID = SDL_GetWindowID(m_window);
        event.button.button = step.button;
        event.button.down = step.action == MouseStep::Action::Down;
        event.button.clicks = step.clicks;
        event.button.x = x;
        event.button.y = y;
    }
    m_mouseSimulated = true;
    m_simulatedMods = step.mods;
    onEvent(event);
    m_mouseSimulated = false;
}

void Editor::updateKeyCapture()
{
    if (!capturingKey())
        return;
    if (m_keyCapture.openPopup) {
        ImGui::OpenPopup(kCapturePopup);
        m_keyCapture.openPopup = false;
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    // No nav inputs: Enter, Space and the arrows are keys to record, not buttons to press.
    if (!ImGui::BeginPopupModal(kCapturePopup, nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs)) {
        m_keyCapture = {};
        return;
    }

    const EditorAction action = static_cast<EditorAction>(m_keyCapture.action);
    const EditorActionInfo& info = EditorKeymap::info(action);
    ImGui::Text("New %s shortcut for", m_keyCapture.slot == 0 ? "main" : "alternate");
    ImGui::TextColored(EditorStyle::kHighlight, "%s", info.label);
    ImGui::Spacing();
    const ImGuiKeyChord mods = ImGui::GetIO().KeyMods;
    std::string heldMods;
    if (mods & ImGuiMod_Ctrl) heldMods += "Ctrl+";
    if (mods & ImGuiMod_Shift) heldMods += "Shift+";
    if (mods & ImGuiMod_Alt) heldMods += "Alt+";
    if (mods & ImGuiMod_Super) heldMods += "Super+";
    ImGui::Text("Press the keys now...  %s", heldMods.c_str());
    ImGui::TextDisabled("Esc cancels. A modifier key pressed and released on its own\n(e.g. Shift) is bound by itself.");

    bool done = false;
    bool cancelled = false;
    ImGuiKeyChord result = ImGuiKey_None;
    for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END && !done; ++k) {
        const ImGuiKey key = static_cast<ImGuiKey>(k);
        if (!isKeyboardKey(key))
            continue;
        if (isModifierKey(key)) {
            if (ImGui::IsKeyPressed(key, false))
                m_keyCapture.lonelyModifier = key;
            else if (ImGui::IsKeyReleased(key) && m_keyCapture.lonelyModifier == key) {
                result = key;
                done = true;
            }
            continue;
        }
        if (!ImGui::IsKeyPressed(key, false))
            continue;
        // Plain Esc cancels; with a modifier it can still be bound.
        cancelled = key == ImGuiKey_Escape && mods == ImGuiMod_None;
        result = mods | key;
        done = true;
    }

    ImGui::Spacing();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
        done = true;
        cancelled = true;
    }

    if (done) {
        if (!cancelled) {
            m_keymap.setBinding(action, m_keyCapture.slot, result);
            m_keymap.save(kEditorKeymapPath);
            const std::string conflict = m_keymap.conflicts(action, m_keyCapture.slot);
            std::string message = std::string(info.label) + ": " + keyChordName(result);
            if (!conflict.empty())
                message += " (also used by " + conflict + ")";
            setStatus(message, !conflict.empty());
        }
        m_keyCapture = {};
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}
