#include "FileDialog.h"

#include <mutex>
#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_properties.h>
#include "engine/Log.h"

struct FileDialogState {
    std::mutex mutex;
    std::string key;
    // SDL reads the filter on the dialog thread, so it lives as long as the dialog.
    std::string filterName;
    std::string patterns;
    SDL_DialogFileFilter filter{};
    bool done = false;
    std::string path;
};

namespace {

// userdata is a heap copy of the state's shared_ptr, so a dialog outliving the editor stays valid.
void SDLCALL onDialogClosed(void* userdata, const char* const* files, int)
{
    auto* holder = static_cast<std::shared_ptr<FileDialogState>*>(userdata);
    FileDialogState& state = **holder;
    {
        std::lock_guard lock(state.mutex);
        if (!files)
            LOG_ERROR("[EDITOR] File dialog failed: " << SDL_GetError() << "\n");
        else if (files[0])
            state.path = files[0];
        state.done = true;
    }
    delete holder;
}

} // namespace

void FileDialog::open(SDL_Window* window, const char* key, const char* title, const char* filterName,
    const char* patterns, const std::string& root, const char* defaultFileName, bool save)
{
    if (isOpen())
        return;
    namespace fs = std::filesystem;
    std::error_code error;
    fs::create_directories(root, error);
    fs::path location = fs::weakly_canonical(fs::absolute(root), error);
    if (save && defaultFileName)
        location /= defaultFileName;

    m_state = std::make_shared<FileDialogState>();
    m_state->key = key;
    m_state->filterName = filterName;
    m_state->patterns = patterns;
    m_state->filter = { m_state->filterName.c_str(), m_state->patterns.c_str() };

    const SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, &m_state->filter);
    SDL_SetNumberProperty(props, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, 1);
    SDL_SetPointerProperty(props, SDL_PROP_FILE_DIALOG_WINDOW_POINTER, window);
    SDL_SetStringProperty(props, SDL_PROP_FILE_DIALOG_LOCATION_STRING, location.make_preferred().string().c_str());
    SDL_SetStringProperty(props, SDL_PROP_FILE_DIALOG_TITLE_STRING, title);
    SDL_ShowFileDialogWithProperties(save ? SDL_FILEDIALOG_SAVEFILE : SDL_FILEDIALOG_OPENFILE, onDialogClosed,
        new std::shared_ptr<FileDialogState>(m_state), props);
    SDL_DestroyProperties(props);
}

bool FileDialog::isOpen() const
{
    if (!m_state)
        return false;
    std::lock_guard lock(m_state->mutex);
    return !m_state->done;
}

bool FileDialog::take(const char* key, std::string& path)
{
    if (!m_state || m_state->key != key)
        return false;
    {
        std::lock_guard lock(m_state->mutex);
        if (!m_state->done)
            return false;
        path = std::move(m_state->path);
    }
    m_state.reset();
    return true;
}
