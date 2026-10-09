#pragma once
#include "engine/GraphicsSettings.h"

// ImGui controls for the global graphics settings (no window of its own). Returns true if anything changed.
bool drawGraphicsSettings(GraphicsSettings& settings, const RenderCapabilities& caps);
// ImGui controls for the open scene's look; `global` greys out what its effects switches turn off.
bool drawSceneSettings(SceneSettings& settings, const GraphicsSettings& global);
