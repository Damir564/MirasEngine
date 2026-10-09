#include "Editor.h"
#include <SDL3/SDL.h>
#include <utility>
#include "engine/SceneManager.h"

// ---------------------------------------------------------------------------------------------
// Unsaved changes
// ---------------------------------------------------------------------------------------------

uint64_t Editor::sceneStateId() const
{
    return m_undoStack.empty() ? m_baseState : m_undoStack.back().serial;
}

bool Editor::sceneDirty() const
{
    // A change still being made (a drag) has no history entry yet.
    return sceneStateId() != m_savedState || m_sceneChanged || m_levelEditPending ||
        m_scenes.settings() != m_savedSceneSettings;
}

void Editor::markSceneSaved(const std::string& path)
{
    m_savedState = sceneStateId();
    m_savedSceneSettings = m_scenes.settings();
    m_autosavedState = m_savedState;
    // Saved from a script or command batch: its edits become a history entry only afterwards, and the
    // file already holds them.
    m_savedBeforeHistory = m_batch.active || m_sceneChanged || m_levelEditPending;
    addRecentScene(m_prefs, path);
}

void Editor::settleSavedState()
{
    if (m_savedBeforeHistory && !editInProgress() && !m_batch.active) {
        m_savedState = m_autosavedState = sceneStateId();
        m_savedBeforeHistory = false;
    }
}

bool Editor::confirmQuit()
{
    if (!sceneDirty())
        return true;
    // The prompt is part of the UI.
    if (m_flyMode)
        setFlyMode(false);
    requestSceneAction(SceneAction::Quit);
    return false;
}

void Editor::requestSceneAction(SceneAction action, const std::string& path)
{
    if (!sceneDirty()) {
        runSceneAction(action, path);
        return;
    }
    m_pendingSceneAction = action;
    m_pendingScenePath = path;
    m_openUnsavedPopup = true;
}

void Editor::runSceneAction(SceneAction action, const std::string& path)
{
    switch (action) {
    case SceneAction::New:
        newScene();
        break;
    case SceneAction::Open:
        openScene(path);
        break;
    case SceneAction::OpenDialog:
        openSceneDialog();
        break;
    case SceneAction::Quit:
        m_quitRequested = true;
        break;
    case SceneAction::None:
        break;
    }
}

void Editor::finishPendingSceneAction()
{
    if (!std::exchange(m_pendingAfterSaveAs, false))
        return;
    const SceneAction action = std::exchange(m_pendingSceneAction, SceneAction::None);
    if (!sceneDirty())
        runSceneAction(action, m_pendingScenePath);
}

void Editor::drawUnsavedChangesPopup()
{
    constexpr const char* kPopup = "Unsaved Changes";
    if (m_openUnsavedPopup) {
        ImGui::OpenPopup(kPopup);
        m_openUnsavedPopup = false;
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (!ImGui::BeginPopupModal(kPopup, nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings))
        return;

    const std::string path = m_scenes.currentPath();
    const char* before = m_pendingSceneAction == SceneAction::Quit ? "quitting"
        : m_pendingSceneAction == SceneAction::New ? "starting a new scene" : "opening another scene";
    ImGui::Text("Save changes to %s before %s?", path.empty() ? "the untitled scene" : path.c_str(), before);
    ImGui::TextDisabled("Changes you don't save will be lost.");
    ImGui::Spacing();

    const ImVec2 size(px(110.0f), 0.0f);
    SceneAction run = SceneAction::None;
    bool close = false;
    if (ImGui::Button("Save", size) || ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
        close = true;
        if (path.empty()) {
            // Save As first; the action follows once the file is saved (finishPendingSceneAction()).
            m_pendingAfterSaveAs = true;
            saveSceneAsDialog();
        }
        else {
            saveSceneTo(path);
            if (!sceneDirty())
                run = m_pendingSceneAction;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Don't Save", size)) {
        close = true;
        run = m_pendingSceneAction;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", size) || ImGui::IsKeyPressed(ImGuiKey_Escape, false))
        close = true;
    if (close) {
        ImGui::CloseCurrentPopup();
        if (!m_pendingAfterSaveAs)
            m_pendingSceneAction = SceneAction::None;
    }
    ImGui::EndPopup();
    if (run != SceneAction::None)
        runSceneAction(run, m_pendingScenePath);
}

void Editor::drawRecentScenesMenu()
{
    if (!ImGui::BeginMenu("Open Recent", !m_prefs.recentScenes.empty()))
        return;
    std::string chosen;
    for (const std::string& path : m_prefs.recentScenes)
        if (ImGui::MenuItem(path.c_str(), nullptr, path == m_scenes.currentPath()))
            chosen = path;
    ImGui::Separator();
    if (ImGui::MenuItem("Clear Recent Scenes"))
        m_prefs.recentScenes.clear();
    ImGui::EndMenu();
    if (!chosen.empty())
        requestSceneAction(SceneAction::Open, chosen);
}

// ---------------------------------------------------------------------------------------------
// Autosave
// ---------------------------------------------------------------------------------------------

void Editor::updateAutosave()
{
    const uint64_t now = SDL_GetTicks();
    if (m_prefs.autosaveMinutes <= 0) {
        m_lastAutosaveTicks = now;
        return;
    }
    if (now - m_lastAutosaveTicks < static_cast<uint64_t>(m_prefs.autosaveMinutes) * 60'000)
        return;
    // Not mid-drag or while a scene loads; checked again next frame.
    if (m_scenes.isLoading() || editInProgress())
        return;
    m_lastAutosaveTicks = now;
    const uint64_t state = sceneStateId();
    if (!sceneDirty() || state == m_autosavedState)
        return;
    const std::string path = autosavePath();
    if (m_scenes.save(path)) {
        m_autosavedState = state;
        setStatus("Autosaved to " + path);
    }
    else {
        setStatus("Autosave to " + path + " failed", true);
    }
}
