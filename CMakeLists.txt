cmake_minimum_required (VERSION 3.21)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("MirasEngine")

find_package(Vulkan REQUIRED)

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

# 2️⃣ Glob all GLSL shader files (vert, frag, etc.)
file(GLOB SHADERS "${SHADER_SRC_DIR}/*.*")

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
        COMMAND glslangValidator -V ${SHADER_FILE} -o ${OUTPUT_FILE}
        DEPENDS ${SHADER_FILE}
        COMMENT "Compiling GLSL shader ${SHADER_FILE} to SPIR-V"
        VERBATIM
    )
endforeach()

# 4️⃣ Add a custom target to build all shaders
add_custom_target(Shaders ALL DEPENDS ${SPIRV_FILES})

# 5️⃣ Example: Link shaders to your executable target
add_dependencies(engine Shaders)


