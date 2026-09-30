#include "SettingsUi.h"
#include "imgui.h"

namespace {

// Combo over a fixed list of int values; values above maxValue are hidden.
bool comboInt(const char* label, int& value, const int* values, const char* const* names, int count, int maxValue)
{
    const char* preview = "Custom";
    for (int i = 0; i < count; ++i)
        if (values[i] == value) preview = names[i];

    bool changed = false;
    if (ImGui::BeginCombo(label, preview)) {
        for (int i = 0; i < count; ++i) {
            if (values[i] > maxValue) continue;
            const bool selected = values[i] == value;
            if (ImGui::Selectable(names[i], selected) && !selected) {
                value = values[i];
                changed = true;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}

} // namespace

bool drawGraphicsSettings(GraphicsSettings& settings, const RenderCapabilities& caps)
{
    bool changed = false;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.55f);

    ImGui::SeparatorText("Display");
    changed |= ImGui::Checkbox("VSync", &settings.vsync);
    {
        static const int values[] = { 0, 30, 60, 120, 144, 165, 240 };
        static const char* const names[] = { "Unlimited", "30", "60", "120", "144", "165", "240" };
        changed |= comboInt("Max FPS", settings.maxFps, values, names, IM_ARRAYSIZE(values), 1 << 30);
    }
    {
        static const int values[] = { 1, 2, 4, 8 };
        static const char* const names[] = { "Off", "MSAA 2x", "MSAA 4x", "MSAA 8x" };
        changed |= comboInt("Anti-aliasing", settings.msaaSamples, values, names, IM_ARRAYSIZE(values), caps.maxMsaaSamples);
    }

    ImGui::SeparatorText("Shadows");
    changed |= ImGui::Checkbox("Shadows", &settings.shadows);
    ImGui::BeginDisabled(!settings.shadows);
    {
        static const int values[] = { 1024, 2048, 4096 };
        static const char* const names[] = { "Low (1024)", "Medium (2048)", "High (4096)" };
        changed |= comboInt("Shadow quality", settings.shadowMapSize, values, names, IM_ARRAYSIZE(values), 1 << 30);
    }
    changed |= ImGui::SliderFloat("Shadow distance", &settings.shadowDistance, 10.0f, 1000.0f, "%.0f m",
        ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
    ImGui::EndDisabled();

    ImGui::SeparatorText("World");
    changed |= ImGui::SliderFloat("View distance", &settings.viewDistance, 100.0f, 20000.0f, "%.0f m",
        ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
    changed |= ImGui::Checkbox("Fog", &settings.fog);
    changed |= ImGui::Checkbox("Sun", &settings.sun);
    ImGui::SetItemTooltip("Sun disk and direct sunlight. Turning it off also disables shadows.");

    ImGui::PopItemWidth();
    ImGui::Spacing();
    if (ImGui::Button("Reset to defaults") && !(settings == GraphicsSettings{})) {
        settings = GraphicsSettings{};
        if (settings.msaaSamples > caps.maxMsaaSamples) settings.msaaSamples = caps.maxMsaaSamples;
        changed = true;
    }
    return changed;
}
