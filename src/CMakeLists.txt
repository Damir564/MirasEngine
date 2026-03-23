add_executable (engine 
  vma.cpp
  main.cpp
  vk_globals.cpp 
  external/portable-file-dialogs.h
  external/ImGuiFileDialog.h
  external/ImGuiFileDialogConfig.h
  external/ImGuiFileDialog.cpp
  engine/Buffers.h 
  engine/Vertex.h 
  engine/Camera.h 
  engine/ModelInstance.h 
  engine/Shadow.h 
  engine/ModelTypes.h 
  engine/ModelManager.h 
  engine/ModelManager.cpp 
  engine/Shadow.cpp 
  engine/Gizmo.h 
  engine/SceneSerializer.h 
  engine/CameraAnimation.h 
  engine/CameraAnimation.cpp 
  engine/FileDialog.h 
  engine/IfcConverter.h 
  engine/IfcLayerInfo.h 
  engine/IfcNodeParser.h) 

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET engine PROPERTY CXX_STANDARD 20)
endif()

target_sources(engine PRIVATE vma.cpp)

target_compile_definitions(engine PRIVATE
    VK_NO_PROTOTYPES    
    VULKAN_HPP_NO_EXCEPTIONS
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
)

target_link_libraries(engine PRIVATE
    Vulkan::Headers
    volk
    VulkanMemoryAllocator
    vk-bootstrap::vk-bootstrap
    SDL3::SDL3
    glm::glm-header-only
    assimp
    stb_image
    fastgltf::fastgltf
    imgui
)


