#include "Editor.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <tuple>
#include "imgui_internal.h"
#include "EditorStyle.h"

namespace {

constexpr const char* kIniTypeName = "MirasEditor";
// Same order as Editor::panelFlags().
constexpr const char* kPanelNames[] = { "Hierarchy", "Inspector", "Camera Animation", "Statistics", "Level", "Materials" };
constexpr int kPanelCount = static_cast<int>(std::size(kPanelNames));

// What the imgui.ini handler reads into and writes from. ImGui writes the file once more when its context is
// destroyed, after the editor, so the last known visibility is kept here rather than only in the editor.
struct PanelIni {
    std::array<bool*, kPanelCount> flags{}; // the editor's, while it exists
    uint32_t bits = (1u << kPanelCount) - 1;
};
PanelIni g_panelIni;

void* panelIniOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name)
{
    return std::strcmp(name, "Panels") == 0 ? &g_panelIni : nullptr;
}

void panelIniReadLine(ImGuiContext*, ImGuiSettingsHandler*, void*, const char* line)
{
    const char* equals = std::strchr(line, '=');
    if (!equals)
        return;
    const std::string key(line, equals);
    const bool shown = std::atoi(equals + 1) != 0;
    for (int i = 0; i < kPanelCount; ++i) {
        if (key != kPanelNames[i])
            continue;
        g_panelIni.bits = shown ? g_panelIni.bits | (1u << i) : g_panelIni.bits & ~(1u << i);
        if (g_panelIni.flags[i])
            *g_panelIni.flags[i] = shown;
    }
}

void panelIniWriteAll(ImGuiContext*, ImGuiSettingsHandler* handler, ImGuiTextBuffer* out)
{
    out->appendf("[%s][Panels]\n", handler->TypeName);
    for (int i = 0; i < kPanelCount; ++i) {
        if (g_panelIni.flags[i])
            g_panelIni.bits = *g_panelIni.flags[i] ? g_panelIni.bits | (1u << i) : g_panelIni.bits & ~(1u << i);
        out->appendf("%s=%d\n", kPanelNames[i], (g_panelIni.bits >> i) & 1u);
    }
    out->append("\n");
}

// Letters, digits, spaces, '-' and '_' only, so the name is a safe file name.
std::string layoutFileName(const std::string& name)
{
    std::string clean;
    for (char c : name)
        if (std::isalnum(static_cast<unsigned char>(c)) || c == ' ' || c == '-' || c == '_')
            clean += c;
    const size_t begin = clean.find_first_not_of(' ');
    if (begin == std::string::npos)
        return {};
    return clean.substr(begin, clean.find_last_not_of(' ') - begin + 1);
}

} // namespace

std::array<Editor::PanelFlag, 6> Editor::panelFlags()
{
    static_assert(kPanelCount == 6);
    return { {
        { kPanelNames[0], &m_showHierarchy },
        { kPanelNames[1], &m_showInspector },
        { kPanelNames[2], &m_showAnimationPanel },
        { kPanelNames[3], &m_showStatisticsPanel },
        { kPanelNames[4], &m_showLevelPanel },
        { kPanelNames[5], &m_showMaterialsPanel },
    } };
}

void Editor::registerIniHandler()
{
    const auto panels = panelFlags();
    for (int i = 0; i < kPanelCount; ++i)
        g_panelIni.flags[i] = panels[i].shown;
    if (ImGui::FindSettingsHandler(kIniTypeName))
        return;
    ImGuiSettingsHandler handler;
    handler.TypeName = kIniTypeName;
    handler.TypeHash = ImHashStr(kIniTypeName);
    handler.ReadOpenFn = panelIniOpen;
    handler.ReadLineFn = panelIniReadLine;
    handler.WriteAllFn = panelIniWriteAll;
    ImGui::AddSettingsHandler(&handler);
}

void Editor::detachIniHandler()
{
    for (int i = 0; i < kPanelCount; ++i) {
        if (g_panelIni.flags[i])
            g_panelIni.bits = *g_panelIni.flags[i] ? g_panelIni.bits | (1u << i) : g_panelIni.bits & ~(1u << i);
        g_panelIni.flags[i] = nullptr;
    }
}

void Editor::beforeUiFrame()
{
    // The toolbar edits the camera speed directly.
    m_prefs.cameraSpeed = m_camera.speed;
    m_prefs = sanitizeEditorPrefs(m_prefs);
    m_camera.speed = m_prefs.cameraSpeed;
    m_camera.sensitivity = m_prefs.lookSensitivity;

    const auto styleFields = [](const EditorPrefs& p) {
        return std::tie(p.theme, p.accent, p.uiScale, p.rounding, p.compact);
    };
    if (styleFields(m_prefs) != styleFields(m_appliedStyle)) {
        EditorStyle::apply(m_prefs.styleOptions());
        m_appliedStyle = m_prefs;
    }

    if (!m_pendingLayout.empty()) {
        ImGui::LoadIniSettingsFromMemory(m_pendingLayout.data(), m_pendingLayout.size());
        m_pendingLayout.clear();
    }

    uint32_t bits = 0;
    const auto panels = panelFlags();
    for (int i = 0; i < kPanelCount; ++i)
        if (*panels[i].shown)
            bits |= 1u << i;
    if (bits != m_panelBits) {
        m_panelBits = bits;
        ImGui::MarkIniSettingsDirty();
    }

    // Once the edit is done, not on every frame of a slider drag.
    if (!(m_prefs == m_savedPrefs) && !ImGui::IsAnyItemActive()) {
        saveEditorPrefs(m_prefs);
        m_savedPrefs = m_prefs;
    }
}

// ---------------------------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------------------------

void Editor::buildLayout(ImGuiID dockspaceId, LayoutPreset preset)
{
    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspaceId, ImGui::GetMainViewport()->WorkSize);
    ImGuiID center = dockspaceId;
    const auto split = [](ImGuiID& node, ImGuiDir dir, float ratio) {
        return ImGui::DockBuilderSplitNode(node, dir, ratio, nullptr, &node);
    };
    const auto dock = [](const char* window, ImGuiID node) { ImGui::DockBuilderDockWindow(window, node); };

    m_showHierarchy = m_showInspector = m_showLevelPanel = m_showMaterialsPanel = true;
    switch (preset) {
    case LayoutPreset::Default: {
        const ImGuiID left = split(center, ImGuiDir_Left, 0.22f);
        const ImGuiID right = split(center, ImGuiDir_Right, 0.30f);
        const ImGuiID bottom = split(center, ImGuiDir_Down, 0.30f);
        dock("Hierarchy", left);
        dock("Inspector", right);
        dock("Level", right);
        dock("Materials", bottom);
        dock("Camera Animation", bottom);
        dock("Statistics", bottom);
        m_showAnimationPanel = m_showStatisticsPanel = true;
        break;
    }
    case LayoutPreset::LevelDesign: {
        // Shapes and materials side by side with a tall viewport; animation and statistics closed.
        ImGuiID left = split(center, ImGuiDir_Left, 0.20f);
        ImGuiID right = split(center, ImGuiDir_Right, 0.28f);
        const ImGuiID leftBottom = split(left, ImGuiDir_Down, 0.45f);
        const ImGuiID rightBottom = split(right, ImGuiDir_Down, 0.40f);
        dock("Hierarchy", left);
        dock("Materials", leftBottom);
        dock("Level", right);
        dock("Inspector", rightBottom);
        dock("Camera Animation", rightBottom);
        dock("Statistics", rightBottom);
        m_showAnimationPanel = m_showStatisticsPanel = false;
        break;
    }
    case LayoutPreset::Compact: {
        // One column on the right; the viewport gets the rest of the window.
        ImGuiID right = split(center, ImGuiDir_Right, 0.24f);
        const ImGuiID rightBottom = split(right, ImGuiDir_Down, 0.62f);
        dock("Hierarchy", right);
        dock("Inspector", rightBottom);
        dock("Level", rightBottom);
        dock("Materials", rightBottom);
        dock("Camera Animation", rightBottom);
        dock("Statistics", rightBottom);
        m_showAnimationPanel = m_showStatisticsPanel = false;
        break;
    }
    }
    ImGui::DockBuilderFinish(dockspaceId);
}

std::vector<std::string> Editor::savedLayouts() const
{
    namespace fs = std::filesystem;
    std::vector<std::string> names;
    std::error_code ec;
    for (const fs::directory_entry& entry : fs::directory_iterator(kLayoutsRoot, ec)) {
        std::error_code fileEc;
        if (entry.is_regular_file(fileEc) && entry.path().extension() == ".ini")
            names.push_back(entry.path().stem().string());
    }
    std::sort(names.begin(), names.end());
    return names;
}

void Editor::saveLayout(const std::string& name)
{
    const std::string fileName = layoutFileName(name);
    if (fileName.empty()) {
        setStatus("Enter a layout name (letters, digits, spaces, - and _)", true);
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(kLayoutsRoot, ec);
    const std::filesystem::path path = std::filesystem::path(kLayoutsRoot) / (fileName + ".ini");
    std::ofstream file(path, std::ios::binary);
    size_t size = 0;
    const char* data = ImGui::SaveIniSettingsToMemory(&size);
    file.write(data, static_cast<std::streamsize>(size));
    if (!file) {
        setStatus("Failed to write " + path.string(), true);
        return;
    }
    setStatus("Layout saved: " + fileName);
}

void Editor::loadLayout(const std::string& name)
{
    const std::filesystem::path path = std::filesystem::path(kLayoutsRoot) / (name + ".ini");
    std::ifstream file(path, std::ios::binary);
    std::stringstream contents;
    contents << file.rdbuf();
    if (!file || contents.str().empty()) {
        setStatus("Failed to read " + path.string(), true);
        return;
    }
    m_pendingLayout = contents.str();
    setStatus("Layout loaded: " + name);
}

void Editor::deleteLayout(const std::string& name)
{
    std::error_code ec;
    if (std::filesystem::remove(std::filesystem::path(kLayoutsRoot) / (name + ".ini"), ec))
        setStatus("Layout deleted: " + name);
    else
        setStatus("Failed to delete layout " + name, true);
}

void Editor::drawLayoutMenuItems()
{
    if (ImGui::MenuItem("Default")) m_layoutRequest = LayoutPreset::Default;
    if (ImGui::MenuItem("Level Design")) m_layoutRequest = LayoutPreset::LevelDesign;
    if (ImGui::MenuItem("Compact")) m_layoutRequest = LayoutPreset::Compact;
    const std::vector<std::string> saved = savedLayouts();
    if (!saved.empty())
        ImGui::SeparatorText("Saved");
    for (const std::string& name : saved)
        if (ImGui::MenuItem(name.c_str()))
            loadLayout(name);
    ImGui::Separator();
    if (ImGui::MenuItem("Save Layout...")) {
        m_showPreferences = true;
        m_preferencesTab = 2;
    }
}

// ---------------------------------------------------------------------------------------------
// Preferences window
// ---------------------------------------------------------------------------------------------

void Editor::drawPreferencesWindow()
{
    ImGui::SetNextWindowSize(ImVec2(px(480.0f), px(440.0f)), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Preferences", &m_showPreferences)) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginTabBar("##preferencesTabs")) {
        const auto tab = [this](const char* label, int index) {
            return ImGui::BeginTabItem(label, nullptr, m_preferencesTab == index ? ImGuiTabItemFlags_SetSelected : 0);
        };
        if (tab("Interface", 0)) {
            drawInterfacePrefs();
            ImGui::EndTabItem();
        }
        if (tab("Viewport", 1)) {
            drawViewportPrefs();
            ImGui::EndTabItem();
        }
        if (tab("Layouts", 2)) {
            drawLayoutPrefs();
            ImGui::EndTabItem();
        }
        if (tab("Files", 3)) {
            drawFilePrefs();
            ImGui::EndTabItem();
        }
        m_preferencesTab = -1;
        ImGui::EndTabBar();
    }
    ImGui::Separator();
    if (ImGui::Button("Keyboard Shortcuts..."))
        m_showKeymap = true;
    ImGui::SameLine();
    ImGui::TextDisabled("Saved to %s", kEditorPrefsPath);
    ImGui::End();
}

void Editor::drawInterfacePrefs()
{
    ImGui::PushItemWidth(px(220.0f));
    int theme = static_cast<int>(m_prefs.theme);
    if (ImGui::Combo("Theme", &theme, EditorStyle::kThemeNames, IM_ARRAYSIZE(EditorStyle::kThemeNames)))
        m_prefs.theme = static_cast<EditorStyle::Theme>(theme);

    ImGui::ColorEdit3("Accent color", &m_prefs.accent.x, ImGuiColorEditFlags_NoInputs);
    static const glm::vec3 kAccents[] = { { 0.26f, 0.52f, 0.92f }, { 0.95f, 0.55f, 0.15f }, { 0.30f, 0.70f, 0.35f },
        { 0.60f, 0.40f, 0.90f }, { 0.85f, 0.30f, 0.30f }, { 0.15f, 0.65f, 0.70f } };
    for (int i = 0; i < IM_ARRAYSIZE(kAccents); ++i) {
        ImGui::SameLine();
        ImGui::PushID(i);
        // The UI is drawn in linear color, so show the swatch as the accent will look.
        const glm::vec3& color = kAccents[i];
        if (ImGui::ColorButton("##accent", EditorStyle::lin(ImVec4(color.r, color.g, color.b, 1.0f)),
                ImGuiColorEditFlags_NoTooltip))
            m_prefs.accent = color;
        ImGui::PopID();
    }

    static const float kScales[] = { 0.75f, 0.9f, 1.0f, 1.1f, 1.25f, 1.5f, 1.75f, 2.0f };
    char preview[16];
    snprintf(preview, sizeof(preview), "%.0f%%", m_prefs.uiScale * 100.0f);
    // A list rather than a slider: the window rescales as soon as the value changes.
    if (ImGui::BeginCombo("UI scale", preview)) {
        for (float scale : kScales) {
            char label[16];
            snprintf(label, sizeof(label), "%.0f%%", scale * 100.0f);
            if (ImGui::Selectable(label, std::abs(scale - m_prefs.uiScale) < 1e-3f))
                m_prefs.uiScale = scale;
        }
        ImGui::EndCombo();
    }
    ImGui::SliderFloat("Corner rounding", &m_prefs.rounding, 0.0f, 12.0f, "%.0f px", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Checkbox("Compact spacing", &m_prefs.compact);
    ImGui::PopItemWidth();

    ImGui::Spacing();
    if (ImGui::Button("Reset interface")) {
        const EditorPrefs defaults;
        m_prefs.theme = defaults.theme;
        m_prefs.accent = defaults.accent;
        m_prefs.uiScale = defaults.uiScale;
        m_prefs.rounding = defaults.rounding;
        m_prefs.compact = defaults.compact;
    }
}

void Editor::drawViewportPrefs()
{
    constexpr ImGuiSliderFlags kClamp = ImGuiSliderFlags_AlwaysClamp;
    ImGui::PushItemWidth(px(220.0f));
    ImGui::SliderFloat("Field of view", &m_prefs.fieldOfView, 30.0f, 120.0f, "%.0f\xC2\xB0", kClamp);
    ImGui::SetItemTooltip("Vertical field of view of the editor camera.");
    ImGui::SliderFloat("Look sensitivity", &m_prefs.lookSensitivity, 0.01f, 1.0f, "%.2f",
        kClamp | ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Invert look Y", &m_prefs.invertLookY);
    ImGui::SliderFloat("Camera speed", &m_camera.speed, 0.5f, 200.0f, "%.1f m/s", kClamp | ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Keyboard movement speed; also on the toolbar.");
    ImGui::PopItemWidth();
    ImGui::Checkbox("Help overlay", &m_prefs.showViewportHelp);
    ImGui::SetItemTooltip("Tool name and mouse/keyboard hints in the top-left corner of the viewport.");
    ImGui::Checkbox("Orientation gizmo", &m_prefs.showOrientationGizmo);
    ImGui::SetItemTooltip("The axis widget in the top-right corner; click an axis to look along it.");

    ImGui::Spacing();
    if (ImGui::Button("Reset viewport")) {
        const EditorPrefs defaults;
        m_prefs.fieldOfView = defaults.fieldOfView;
        m_prefs.lookSensitivity = defaults.lookSensitivity;
        m_prefs.invertLookY = defaults.invertLookY;
        m_camera.speed = defaults.cameraSpeed;
        m_prefs.showViewportHelp = defaults.showViewportHelp;
        m_prefs.showOrientationGizmo = defaults.showOrientationGizmo;
    }
}

void Editor::drawFilePrefs()
{
    ImGui::PushItemWidth(px(220.0f));
    ImGui::SliderInt("Autosave", &m_prefs.autosaveMinutes, 0, 30,
        m_prefs.autosaveMinutes == 0 ? "Off" : "every %d min", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Unsaved changes are written to %s in the editor's folder; the scene's own file is "
        "left alone.", kAutosavePath);
    ImGui::PopItemWidth();
    ImGui::Spacing();
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Recent scenes: %zu", m_prefs.recentScenes.size());
    ImGui::SameLine();
    ImGui::BeginDisabled(m_prefs.recentScenes.empty());
    if (ImGui::Button("Clear"))
        m_prefs.recentScenes.clear();
    ImGui::EndDisabled();
}

void Editor::drawLayoutPrefs()
{
    ImGui::TextDisabled("Built-in layouts");
    if (ImGui::Button("Default")) m_layoutRequest = LayoutPreset::Default;
    ImGui::SetItemTooltip("Hierarchy left, inspector and level right, materials and animation below.");
    ImGui::SameLine();
    if (ImGui::Button("Level Design")) m_layoutRequest = LayoutPreset::LevelDesign;
    ImGui::SetItemTooltip("Hierarchy and materials left, level and inspector right; tall viewport.");
    ImGui::SameLine();
    if (ImGui::Button("Compact")) m_layoutRequest = LayoutPreset::Compact;
    ImGui::SetItemTooltip("Every panel in one column on the right; the largest viewport.");

    ImGui::SeparatorText("Saved layouts");
    const std::vector<std::string> saved = savedLayouts();
    if (saved.empty())
        ImGui::TextDisabled("None yet.");
    for (const std::string& name : saved) {
        ImGui::PushID(name.c_str());
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(name.c_str());
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - px(120.0f));
        if (ImGui::Button("Load", ImVec2(px(56.0f), 0.0f)))
            loadLayout(name);
        ImGui::SameLine();
        if (ImGui::Button("Delete", ImVec2(px(56.0f), 0.0f)))
            deleteLayout(name);
        ImGui::PopID();
    }

    ImGui::Spacing();
    ImGui::SetNextItemWidth(px(220.0f));
    const bool entered = ImGui::InputTextWithHint("##layoutName", "Layout name", m_layoutName, sizeof(m_layoutName),
        ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if (ImGui::Button("Save current layout") || entered)
        saveLayout(m_layoutName);
    ImGui::TextDisabled("Saves panel positions, docking and which panels are open, to %s/.", kLayoutsRoot);
}
