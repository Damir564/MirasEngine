add_executable (engine 
  vma.cpp
  main.cpp
   vk_globals.cpp)

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
)


