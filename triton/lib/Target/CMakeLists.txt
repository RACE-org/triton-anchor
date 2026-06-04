add_subdirectory(LLVMIR)
if(TRITON_ENABLE_SPIRV AND NOT NO_TTGIR)
  add_subdirectory(SPIRV)
endif()

if(NOT NO_TTGIR AND TARGET SPIRV-Tools-shared)
  set_target_properties(SPIRV-Tools-shared PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
