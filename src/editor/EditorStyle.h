#pragma once
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include "imgui.h"

namespace EditorStyle {

// The UI is rendered into an sRGB swapchain, which treats vertex colors as linear and brightens
// them. Colors below are authored in sRGB and converted so they appear as designed.
inline ImVec4 lin(const ImVec4& c) {
    auto f = [](float v) { return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f); };
    return ImVec4(f(c.x), f(c.y), f(c.z), c.w);
}

// Colors shared by the editor widgets. The first five follow the theme and accent: apply() sets them.
inline ImVec4 kAccent = lin(ImVec4(0.26f, 0.52f, 0.92f, 1.00f));
inline ImVec4 kAccentHover = lin(ImVec4(0.35f, 0.60f, 0.97f, 1.00f));
inline ImVec4 kAccentActive = lin(ImVec4(0.20f, 0.44f, 0.82f, 1.00f));
inline ImVec4 kTextDim = lin(ImVec4(0.58f, 0.60f, 0.64f, 1.00f));
inline ImVec4 kHighlight = lin(ImVec4(0.95f, 0.78f, 0.30f, 1.00f));
inline const ImVec4 kAxisX = lin(ImVec4(0.78f, 0.22f, 0.20f, 1.00f));
inline const ImVec4 kAxisY = lin(ImVec4(0.30f, 0.64f, 0.22f, 1.00f));
inline const ImVec4 kAxisZ = lin(ImVec4(0.20f, 0.40f, 0.80f, 1.00f));
inline const ImVec4 kError = lin(ImVec4(0.95f, 0.40f, 0.35f, 1.00f));
inline const ImVec4 kDanger = lin(ImVec4(0.70f, 0.25f, 0.22f, 1.00f));

enum class Theme { Dark, Midnight, Light };
inline constexpr const char* kThemeNames[] = { "Dark", "Midnight", "Light" };

struct StyleOptions {
    Theme theme = Theme::Dark;
    ImVec4 accent{ 0.26f, 0.52f, 0.92f, 1.00f }; // sRGB
    float rounding = 3.0f;  // frames; windows and popups get one pixel more
    bool compact = false;   // tighter padding and spacing
    float scale = 1.0f;     // fonts and sizes
};

inline void loadFonts(ImGuiIO& io) {
    // Segoe UI is the Windows UI font; fall back to Arial, then ImGui's built-in font.
    const char* candidates[] = { "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf" };
    for (const char* path : candidates) {
        if (std::filesystem::exists(path)) {
            io.Fonts->AddFontFromFileTTF(path, 17.0f, nullptr, io.Fonts->GetGlyphRangesCyrillic());
            return;
        }
    }
    io.Fonts->AddFontDefault();
}

// A theme's shades, in sRGB.
struct Palette {
    ImVec4 bg0;         // deepest: docking background, title and menu bars
    ImVec4 bg1;         // panels
    ImVec4 bg2;         // frames
    ImVec4 bg3;         // hovered frames
    ImVec4 frameActive;
    ImVec4 popup;
    ImVec4 border;
    ImVec4 tableLight;
    ImVec4 text;
    ImVec4 textDim;
    ImVec4 highlight;   // emphasized text (kHighlight)
    ImVec4 rowAlt;      // blended in linear space, so keep it subtle
    float modalDim;
};

inline Palette palette(Theme theme) {
    switch (theme) {
    case Theme::Midnight:
        return { { 0.060f, 0.062f, 0.070f, 1 }, { 0.085f, 0.088f, 0.098f, 1 }, { 0.130f, 0.134f, 0.148f, 1 },
            { 0.180f, 0.185f, 0.200f, 1 }, { 0.220f, 0.225f, 0.245f, 1 }, { 0.075f, 0.078f, 0.088f, 1 },
            { 0.030f, 0.031f, 0.036f, 1 }, { 0.060f, 0.062f, 0.070f, 1 }, { 0.92f, 0.93f, 0.95f, 1 },
            { 0.52f, 0.54f, 0.58f, 1 }, { 0.95f, 0.78f, 0.30f, 1 }, { 1, 1, 1, 0.010f }, 0.60f };
    case Theme::Light:
        return { { 0.780f, 0.790f, 0.810f, 1 }, { 0.900f, 0.905f, 0.915f, 1 }, { 0.820f, 0.825f, 0.840f, 1 },
            { 0.760f, 0.768f, 0.785f, 1 }, { 0.700f, 0.708f, 0.725f, 1 }, { 0.950f, 0.952f, 0.958f, 1 },
            { 0.650f, 0.655f, 0.670f, 1 }, { 0.800f, 0.805f, 0.815f, 1 }, { 0.10f, 0.11f, 0.13f, 1 },
            { 0.40f, 0.42f, 0.46f, 1 }, { 0.72f, 0.42f, 0.00f, 1 }, { 0, 0, 0, 0.035f }, 0.30f };
    case Theme::Dark:
    default:
        return { { 0.110f, 0.114f, 0.125f, 1 }, { 0.145f, 0.149f, 0.161f, 1 }, { 0.196f, 0.200f, 0.216f, 1 },
            { 0.251f, 0.255f, 0.275f, 1 }, { 0.290f, 0.294f, 0.318f, 1 }, { 0.125f, 0.129f, 0.141f, 1 },
            { 0.070f, 0.072f, 0.080f, 1 }, { 0.10f, 0.10f, 0.11f, 1 }, { 0.88f, 0.89f, 0.91f, 1 },
            { 0.58f, 0.60f, 0.64f, 1 }, { 0.95f, 0.78f, 0.30f, 1 }, { 1, 1, 1, 0.008f }, 0.55f };
    }
}

inline ImVec4 mixColor(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

inline ImVec4 withAlpha(ImVec4 color, float alpha) {
    color.w = alpha;
    return color;
}

// Rebuilds the whole style; call between frames (before ImGui::NewFrame()).
inline void apply(const StyleOptions& options = {}) {
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle();
    if (options.theme == Theme::Light)
        ImGui::StyleColorsLight(&style);
    else
        ImGui::StyleColorsDark(&style);

    if (options.compact) {
        style.WindowPadding = ImVec2(6, 6);
        style.FramePadding = ImVec2(5, 2);
        style.CellPadding = ImVec2(4, 2);
        style.ItemSpacing = ImVec2(6, 3);
        style.ItemInnerSpacing = ImVec2(4, 3);
    }
    else {
        style.WindowPadding = ImVec2(8, 8);
        style.FramePadding = ImVec2(6, 4);
        style.CellPadding = ImVec2(6, 3);
        style.ItemSpacing = ImVec2(8, 5);
        style.ItemInnerSpacing = ImVec2(5, 4);
    }
    style.IndentSpacing = 16.0f;
    style.ScrollbarSize = 12.0f;
    style.GrabMinSize = 10.0f;

    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.TabBorderSize = 0.0f;

    const float rounding = std::clamp(options.rounding, 0.0f, 12.0f);
    style.WindowRounding = rounding > 0.0f ? rounding + 1.0f : 0.0f;
    style.ChildRounding = rounding;
    style.FrameRounding = rounding;
    style.PopupRounding = style.WindowRounding;
    style.ScrollbarRounding = rounding * 2.0f;
    style.GrabRounding = rounding;
    style.TabRounding = rounding;

    style.WindowTitleAlign = ImVec2(0.0f, 0.5f);
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.SeparatorTextBorderSize = 1.0f;

    const Palette p = palette(options.theme);
    const ImVec4 accent = withAlpha(options.accent, 1.0f);
    const ImVec4 accentHover = mixColor(accent, ImVec4(1, 1, 1, 1), 0.12f);
    const ImVec4 accentActive = ImVec4(accent.x * 0.85f, accent.y * 0.85f, accent.z * 0.85f, 1.0f);

    ImVec4* c = style.Colors;
    c[ImGuiCol_Text] = p.text;
    c[ImGuiCol_TextDisabled] = p.textDim;
    c[ImGuiCol_WindowBg] = p.bg1;
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = p.popup;
    c[ImGuiCol_Border] = p.border;
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = p.bg2;
    c[ImGuiCol_FrameBgHovered] = p.bg3;
    c[ImGuiCol_FrameBgActive] = p.frameActive;
    c[ImGuiCol_TitleBg] = p.bg0;
    c[ImGuiCol_TitleBgActive] = p.bg0;
    c[ImGuiCol_TitleBgCollapsed] = p.bg0;
    c[ImGuiCol_MenuBarBg] = p.bg0;
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = p.bg3;
    c[ImGuiCol_ScrollbarGrabHovered] = mixColor(p.bg3, p.text, 0.15f);
    c[ImGuiCol_ScrollbarGrabActive] = mixColor(p.bg3, p.text, 0.25f);
    c[ImGuiCol_CheckMark] = accentHover;
    c[ImGuiCol_SliderGrab] = accent;
    c[ImGuiCol_SliderGrabActive] = accentHover;
    c[ImGuiCol_Button] = p.bg2;
    c[ImGuiCol_ButtonHovered] = p.bg3;
    c[ImGuiCol_ButtonActive] = accentActive;
    c[ImGuiCol_Header] = withAlpha(accent, 0.35f);
    c[ImGuiCol_HeaderHovered] = withAlpha(accent, 0.22f);
    c[ImGuiCol_HeaderActive] = withAlpha(accent, 0.50f);
    c[ImGuiCol_Separator] = p.border;
    c[ImGuiCol_SeparatorHovered] = accent;
    c[ImGuiCol_SeparatorActive] = accentHover;
    c[ImGuiCol_ResizeGrip] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripHovered] = withAlpha(accent, 0.50f);
    c[ImGuiCol_ResizeGripActive] = accent;
    c[ImGuiCol_Tab] = p.bg0;
    c[ImGuiCol_TabHovered] = p.bg3;
    c[ImGuiCol_TabSelected] = p.bg1;
    c[ImGuiCol_TabSelectedOverline] = accent;
    c[ImGuiCol_TabDimmed] = p.bg0;
    c[ImGuiCol_TabDimmedSelected] = p.bg1;
    c[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_DockingPreview] = withAlpha(accent, 0.60f);
    c[ImGuiCol_DockingEmptyBg] = p.bg0;
    c[ImGuiCol_PlotLines] = accentHover;
    c[ImGuiCol_PlotHistogram] = accent;
    c[ImGuiCol_TableHeaderBg] = p.bg0;
    c[ImGuiCol_TableBorderStrong] = p.border;
    c[ImGuiCol_TableBorderLight] = p.tableLight;
    c[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = p.rowAlt;
    c[ImGuiCol_TextSelectedBg] = withAlpha(accent, 0.40f);
    c[ImGuiCol_NavCursor] = accent;
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, p.modalDim);

    for (int i = 0; i < ImGuiCol_COUNT; ++i)
        c[i] = lin(c[i]);
    kAccent = lin(accent);
    kAccentHover = lin(accentHover);
    kAccentActive = lin(accentActive);
    kTextDim = lin(p.textDim);
    kHighlight = lin(p.highlight);

    const float scale = std::clamp(options.scale, 0.5f, 3.0f);
    style.ScaleAllSizes(scale);
    style.FontScaleMain = scale;
}

// Label on the left, widget filling the rest of the row (Unity/Unreal inspector style).
inline void propertyLabel(const char* label, float labelWidth = 110.0f) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(labelWidth);
    ImGui::SetNextItemWidth(-FLT_MIN);
}

// Three colored X/Y/Z drag fields; clicking an axis button resets that component.
inline bool vec3Control(const char* label, float* values, float resetValue, float speed,
    float minValue = 0.0f, float maxValue = 0.0f, float labelWidth = 110.0f) {
    bool changed = false;
    ImGui::PushID(label);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(labelWidth);

    const float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
    const float lineHeight = ImGui::GetFrameHeight();
    const ImVec2 buttonSize(lineHeight, lineHeight);
    const float fieldWidth = (ImGui::GetContentRegionAvail().x - 3 * buttonSize.x - 2 * spacing * 2) / 3.0f;

    const char* axes[3] = { "X", "Y", "Z" };
    const ImVec4 colors[3] = { kAxisX, kAxisY, kAxisZ };
    for (int i = 0; i < 3; ++i) {
        if (i > 0) ImGui::SameLine(0, spacing * 2);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        ImGui::PushStyleColor(ImGuiCol_Button, colors[i]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(colors[i].x + 0.1f, colors[i].y + 0.1f, colors[i].z + 0.1f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors[i]);
        if (ImGui::Button(axes[i], buttonSize)) {
            values[i] = resetValue;
            changed = true;
        }
        ImGui::PopStyleColor(3);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Reset %s to %.2f", axes[i], resetValue);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(std::max(fieldWidth, 20.0f));
        ImGui::PushID(i);
        changed |= ImGui::DragFloat("##v", &values[i], speed, minValue, maxValue, "%.2f");
        ImGui::PopID();
        ImGui::PopStyleVar();
    }
    ImGui::PopID();
    return changed;
}

// Two-column read-only key/value row inside an active table.
inline void keyValueRow(const char* key, const char* value) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextColored(kTextDim, "%s", key);
    ImGui::TableNextColumn();
    ImGui::TextWrapped("%s", (value && *value) ? value : "-");
}

} // namespace EditorStyle
