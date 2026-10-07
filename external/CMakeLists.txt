project(external LANGUAGES CXX)

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
  VulkanHpp
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Hpp.git
  GIT_TAG v1.4.336
  GIT_SHALLOW TRUE
)

FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG 1.4.304
  GIT_SHALLOW TRUE
)

set(VOLK_HEADERS_ONLY OFF CACHE BOOL "" FORCE)
set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  VMA
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG v3.3.0
)

set(VMA_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(VMA_BUILD_SAMPLE OFF CACHE BOOL "" FORCE)
set(VMA_BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  vk_bootstrap
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
  GIT_TAG v1.4.336
  GIT_SHALLOW TRUE
)

set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG release-3.4.0
  GIT_SHALLOW TRUE
)

# Windowing, events, Vulkan surfaces and audio are used; drop the other subsystems and renderers.
foreach(SDL_UNUSED_OPTION
    SDL_GPU SDL_RENDER SDL_CAMERA SDL_JOYSTICK SDL_HAPTIC SDL_HIDAPI SDL_POWER SDL_SENSOR
    SDL_DIALOG SDL_TRAY SDL_OPENGL SDL_OPENGLES SDL_TEST_LIBRARY)
  set(${SDL_UNUSED_OPTION} OFF CACHE BOOL "" FORCE)
endforeach()
set(SDL_AUDIO ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.3
  GIT_SHALLOW TRUE
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)


FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    fastgltf
    GIT_REPOSITORY https://github.com/spnda/fastgltf.git
    GIT_TAG v0.9.0
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.5-docking
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
  JoltPhysics
  GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
  GIT_TAG v5.6.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR Build
)

# Jolt defaults to the static CRT (/MT), which would clash with the /MD used by everything else.
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
set(TARGET_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(TARGET_HELLO_WORLD OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES OFF CACHE BOOL "" FORCE)
set(TARGET_VIEWER OFF CACHE BOOL "" FORCE)
# Prebuilt Vulkan runtime shipped next to engine.exe (see cmake/Packaging.cmake): the loader, for PCs
# without a Vulkan driver, and Mesa's lavapipe, a Vulkan 1.4 driver that renders on the CPU. Only
# downloaded and extracted; SOURCE_SUBDIR none keeps FetchContent from looking for a CMake project.
if (MIRAS_BUNDLE_VULKAN_RUNTIME)
    FetchContent_Declare(vulkan_runtime
        URL https://sdk.lunarg.com/sdk/download/1.4.335.0/windows/VulkanRT-X64-1.4.335.0-Components.zip
        URL_HASH SHA256=4f856da356ffb1c16fbfc792cae70f6e6873680e6b62bc132b9d7131cbc97fe2
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR none)
    # Not 26.2.x: its lavapipe corrupts the heap in vkDestroySwapchainKHR on Windows (even vkcube crashes).
    FetchContent_Declare(mesa_dist
        URL https://github.com/pal1000/mesa-dist-win/releases/download/26.1.8/mesa3d-26.1.8-release-msvc.7z
        URL_HASH SHA256=4c6d32e653e0ff9ad07796e40c0bcfabf2764d849e3ce4f3b1590112c87e42f9
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR none)
    # The binary release has no license file; this is the MIT license that covers lavapipe.
    FetchContent_Declare(mesa_license
        URL https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-26.1.8/licenses/MIT
        URL_HASH SHA256=323c587d0ccf10e376f8bf9a7f31fb4ca6078105194b42e0b1e0ee2bc9bde71f
        DOWNLOAD_NO_EXTRACT TRUE
        SOURCE_SUBDIR none)
    FetchContent_MakeAvailable(vulkan_runtime mesa_dist mesa_license)
endif()

set(FASTGLTF_COMPILE_AS_CPP20 ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_compile_definitions(IMGUI_IMPL_VULKAN_NO_PROTOTYPES)


FetchContent_MakeAvailable(
  VulkanHpp
  volk
  VMA
  vk_bootstrap
  SDL3
  glm
  stb
  fastgltf
  imgui
  nlohmann_json
  JoltPhysics
)

# SDL forces /GL- on SDL_mslibc.c, which conflicts with LTCG (D9025/C4652), so SDL is left out of it.
set_target_properties(SDL3-static PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR} 
    ${imgui_SOURCE_DIR}/backends
)

add_library(stb_image STATIC
    stb_image.cpp
)

target_include_directories(stb_image
    PUBLIC ${stb_SOURCE_DIR}
)

target_link_libraries(imgui PUBLIC
    SDL3::SDL3
    Vulkan::Headers 
)
