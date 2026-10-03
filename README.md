# MirasEngine

A Vulkan rendering engine with a built-in scene editor, written in C++20 for Windows. It loads glTF/GLB models, lets you build and save scenes in an ImGui editor, and plays them as a small first-person game with Jolt physics.

## Features

**Rendering**
- Vulkan 1.2 baseline with dynamic rendering, synchronization2 and shader objects
- GPU-driven indirect draws with per-draw data in storage buffers
- Physically based atmosphere: sky LUT, sun disk, haze and height fog
- Cascaded shadow maps (up to 4 cascades) with soft PCSS penumbrae and screen-space contact shadows
- Depth prepass, half-resolution ambient occlusion, HDR scene pass, bloom
- Tone mapping (ACES, AgX, Khronos PBR Neutral, Reinhard) plus exposure, contrast, saturation and vignette
- MSAA (up to 8x) and FXAA
- Automatic backend fallback: native driver -> emulated features (bundled Khronos layers) -> software (bundled lavapipe)

**Editor**
- Dockable panels: Hierarchy, Inspector, Camera Animation, Statistics
- Select / move / rotate / scale gizmos, rename, duplicate, delete
- glTF/GLB import with a model cache for faster reloads
- Scene save/load (`.scn`)
- Camera path animation
- Play-in-editor (F5 to stop)

**Game mode**
- Main menu, pause menu and in-game graphics settings
- First-person walking controller on a Jolt Physics capsule, collision against the scene geometry

## Requirements

- Windows 10/11 (x64)
- Visual Studio 2022 with the C++ desktop workload (MSVC, CMake, Ninja)
- [Vulkan SDK](https://vulkan.lunarg.com/) (`glslangValidator` must be on `PATH`)
- A GPU with Vulkan 1.2 support (or the software fallback)

All other dependencies are downloaded by CMake `FetchContent`: Vulkan-Hpp, volk, VMA, vk-bootstrap, SDL3, glm, stb, fastgltf, Dear ImGui (docking), nlohmann/json and Jolt Physics.

## Building

Open the folder in Visual Studio and pick the `x64-debug` or `x64-release` preset, or from a Developer PowerShell:

```powershell
cmake --preset x64-debug
cmake --build out/build/x64-debug
```

The executable is written to `out/build/<preset>/src/engine.exe`. Shaders are compiled to SPIR-V and copied next to it; re-run the configure step after adding a new shader.

### Release package

```powershell
cmake --build --preset x64-release-package
```

This produces `out/build/x64-release/MirasEngine-win64.zip`. Options:

| Option | Description |
| --- | --- |
| `MIRAS_PACKAGE_SCENES` | Include scene files in the package |
| `MIRAS_PACKAGE_MODEL_CACHES` | Include prebuilt model caches |
| `MIRAS_BUNDLE_VULKAN_RUNTIME` | Bundle the emulation layers and software Vulkan driver |

## Running

```
engine.exe [options]
```

| Option | Description |
| --- | --- |
| `--editor` | Start in the editor (default) |
| `--game` | Start in game mode |
| `--scene <file>` | Scene to open |
| `--settings <file>` | Graphics settings file (default `settings.json`) |
| `--validation` | Enable Vulkan validation layers |
| `--vulkan auto\|emulated\|software` | First Vulkan backend to try |
| `--exit-after-frames <N>` | Quit after N frames |
| `--help` | Show usage |

Debug builds open a console with log output; release builds have no console.

## Controls

**Editor**

| Key | Action |
| --- | --- |
| Ctrl+O / Ctrl+S | Open / save scene |
| Ctrl+I | Import model |
| F2 | Rename selected |
| Ctrl+D | Duplicate selected |
| Delete | Delete selected |
| F1 | Controls help |

**Game**

| Key | Action |
| --- | --- |
| W A S D | Move |
| Shift | Sprint |
| Space | Jump |
| Mouse | Look |
| Esc | Pause / resume |
| F5 | Return to the editor (play-in-editor) |

## License

MirasEngine is released under the [MIT License](LICENSE). Third-party dependencies keep their own licenses.

## Project layout

```
src/
  app/      Application, launch options, mode switching, shared settings UI
  engine/   Vulkan context, swapchain, renderer and effects, shadows, atmosphere,
            model loading/caching, scenes, camera, physics
  editor/   Editor mode: menus, toolbar, panels, gizmos
  game/     Game mode: menus, player controller
  shaders/  GLSL 460 shaders (compiled to SPIR-V at build time)
external/   FetchContent dependency declarations
cmake/      Packaging scripts
```

Graphics settings (quality, shadows, AO, bloom, tone mapping, sun position, fog, ...) are stored in `settings.json` in the working directory and can be changed from the Graphics settings window in both modes.
