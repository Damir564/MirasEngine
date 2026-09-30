cmake_minimum_required (VERSION 3.21)

project ("MirasEngine" VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
      "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>"
      CACHE STRING "" FORCE)
endif()

if(MSVC)
  add_compile_options(
    $<$<CONFIG:Debug>:/Zi>
    $<$<CONFIG:RelWithDebInfo>:/Zi>
    $<$<CONFIG:Release>:/Zi>          # keep PDBs in Release too if you want to debug it
  )
  add_link_options(
    $<$<CONFIG:Debug>:/DEBUG>
    $<$<CONFIG:RelWithDebInfo>:/DEBUG>
  )
endif()

find_package(Vulkan REQUIRED)

# Smaller release binaries: whole-program optimization (dependencies included) and per-variable
# COMDATs so the linker can drop unreferenced data as well as unreferenced functions.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
if (MSVC)
  add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/Gw>")
endif()

add_subdirectory(external)

add_subdirectory(src)

file(COPY ${CMAKE_SOURCE_DIR}/src/shaders
     DESTINATION ${CMAKE_BINARY_DIR}/src)

# ==========================================
# Shader compilation CMake snippet
# ==========================================

# 1️⃣ Source and output directories
set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/src/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/src/shaders)

# Ensure output directory exists
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

# 2️⃣ Glob the shader stages; *.glsl files are shared includes and only act as dependencies.
file(GLOB SHADERS "${SHADER_SRC_DIR}/*.vert" "${SHADER_SRC_DIR}/*.frag" "${SHADER_SRC_DIR}/*.comp")
file(GLOB SHADER_INCLUDES "${SHADER_SRC_DIR}/*.glsl")

# List to hold SPIR-V output files
set(SPIRV_FILES "")

# 3️⃣ Loop through each shader and create a custom command
foreach(SHADER_FILE ${SHADERS})
    # Get filename without path
    get_filename_component(FILE_NAME_WE ${SHADER_FILE} NAME_WE)
    # Get the extension (vert, frag, etc.)
    get_filename_component(FILE_EXT ${SHADER_FILE} EXT)
    string(REPLACE "." "" FILE_EXT_CLEAN ${FILE_EXT}) # remove dot
    
    # Output file path
    set(OUTPUT_FILE ${SHADER_OUT_DIR}/${FILE_NAME_WE}.${FILE_EXT_CLEAN}.spv)
    
    # Add to SPIR-V list
    list(APPEND SPIRV_FILES ${OUTPUT_FILE})
    
    # Add custom command to compile shader
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${CMAKE_COMMAND} -E echo "Compiling ${SHADER_FILE} -> ${OUTPUT_FILE}"
        COMMAND glslangValidator -V ${SHADER_FILE} -I${SHADER_SRC_DIR} -o ${OUTPUT_FILE}
        DEPENDS ${SHADER_FILE} ${SHADER_INCLUDES}
        COMMENT "Compiling GLSL shader ${SHADER_FILE} to SPIR-V"
        VERBATIM
    )
endforeach()

# 4️⃣ Add a custom target to build all shaders
add_custom_target(Shaders ALL DEPENDS ${SPIRV_FILES})

# 5️⃣ Example: Link shaders to your executable target
add_dependencies(engine Shaders)



# ==========================================
# Packaging: cmake --build <dir> --target package  ->  MirasEngine-<version>-<system>-x64.zip
# ==========================================
# Every dependency is linked statically, so the package is engine.exe, its SPIR-V shaders, the
# launchers and (MSVC) the C/C++ runtime DLLs. The Vulkan loader ships with the GPU driver.
# Everything goes into the "MirasEngine" component so install rules of FetchContent deps
# (SDL, fastgltf, json, ...) stay out of the package.
set(PACKAGE_COMPONENT MirasEngine)

install(TARGETS engine RUNTIME DESTINATION . COMPONENT ${PACKAGE_COMPONENT})
install(FILES ${SPIRV_FILES} DESTINATION shaders COMPONENT ${PACKAGE_COMPONENT})
if (WIN32)
  install(FILES packaging/Editor.bat packaging/Game.bat DESTINATION . COMPONENT ${PACKAGE_COMPONENT})
endif()

set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION .)
set(CMAKE_INSTALL_SYSTEM_RUNTIME_COMPONENT ${PACKAGE_COMPONENT})
set(CMAKE_INSTALL_UCRT_LIBRARIES TRUE)
include(InstallRequiredSystemLibraries)

set(CPACK_PACKAGE_NAME "MirasEngine")
set(CPACK_PACKAGE_VENDOR "MirasEngine")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Game Engine with Vulkan API")
set(CPACK_GENERATOR ZIP)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
set(CPACK_PACKAGE_FILE_NAME "MirasEngine-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-x64")
set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};MirasEngine;${PACKAGE_COMPONENT};/")
include(CPack)
