# MirasEngine
Game Engine with Vulkan API

## Packaging

Build a portable ZIP (engine, compiled shaders, launchers and the MSVC runtime DLLs; all other
dependencies are linked statically):

```
cmake --preset x64-release
cmake --build out/build/x64-release
cmake --build out/build/x64-release --target package
```

The result is `out/build/x64-release/MirasEngine-0.1.0-Windows-x64.zip`. Unzip it anywhere and run
`Editor.bat` or `Game.bat` (drop a `.scn` on it to play that level). The target PC needs a GPU driver
with Vulkan support.
