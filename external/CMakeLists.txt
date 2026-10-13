project(external LANGUAGES CXX)

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Sources are downloaded once into out/deps/<name> and shared by every preset, generator and Visual
# Studio version, so a new or wiped build dir doesn't clone again (only the <name>-build dirs are per
# build). <name>.stamp holds a hash of the declaration; changing the tag/URL downloads again.
set(MIRAS_DEPS_DIR "${CMAKE_SOURCE_DIR}/out/deps" CACHE PATH "Shared download cache for FetchContent sources")

macro(FetchContent_Declare name)
    string(TOLOWER "${name}" _miras_lower)
    string(TOUPPER "${name}" _miras_upper)
    string(SHA1 _miras_key "${ARGN}")
    set(_miras_src "${MIRAS_DEPS_DIR}/${_miras_lower}")
    set(_miras_stamp "${MIRAS_DEPS_DIR}/${_miras_lower}.stamp")
    # FetchContent keeps an empty FETCHCONTENT_SOURCE_DIR_<NAME> cache entry; the normal variable overrides it.
    set(FETCHCONTENT_SOURCE_DIR_${_miras_upper} "")
    if (EXISTS "${_miras_stamp}" AND IS_DIRECTORY "${_miras_src}")
        file(READ "${_miras_stamp}" _miras_have)
        if (_miras_have STREQUAL _miras_key)
            set(FETCHCONTENT_SOURCE_DIR_${_miras_upper} "${_miras_src}")
        endif()
    endif()
    if (NOT FETCHCONTENT_SOURCE_DIR_${_miras_upper})
        set_property(GLOBAL APPEND PROPERTY MIRAS_DEPS_TO_STAMP "${_miras_stamp}|${_miras_key}")
    endif()
    _FetchContent_Declare(${name} ${ARGN} SOURCE_DIR "${_miras_src}")
endmacro()

# Called after FetchContent_MakeAvailable: marks the fresh downloads as complete.
function(miras_stamp_deps)
    get_property(entries GLOBAL PROPERTY MIRAS_DEPS_TO_STAMP)
    foreach(entry IN LISTS entries)
        string(REPLACE "|" ";" parts "${entry}")
        list(GET parts 0 stamp)
        list(GET parts 1 key)
        file(WRITE "${stamp}" "${key}")
    endforeach()
    set_property(GLOBAL PROPERTY MIRAS_DEPS_TO_STAMP "")
endfunction()

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

# Windowing, events, Vulkan surfaces, audio and the editor's file dialogs are used; drop the other
# subsystems and renderers.
foreach(SDL_UNUSED_OPTION
    SDL_GPU SDL_RENDER SDL_CAMERA SDL_JOYSTICK SDL_HAPTIC SDL_HIDAPI SDL_POWER SDL_SENSOR
    SDL_TRAY SDL_OPENGL SDL_OPENGLES SDL_TEST_LIBRARY)
  set(${SDL_UNUSED_OPTION} OFF CACHE BOOL "" FORCE)
endforeach()
set(SDL_AUDIO ON CACHE BOOL "" FORCE)
set(SDL_DIALOG ON CACHE BOOL "" FORCE)

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
    miras_stamp_deps()
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
miras_stamp_deps()

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
