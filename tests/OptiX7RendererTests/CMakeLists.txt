if (NOT TARGET OptiX7Renderer)
  message(STATUS "OptiX7Renderer not found. Skipping OptiX7Renderer tests.")
  return()
endif()

set(PROJECT_NAME "OptiX7RendererTests")

set(SRCS 
  main.cpp
)

set(CUDA_SRCS
  CUDA/DeviceArrayTest.h
)

add_executable(${PROJECT_NAME} ${SRCS} ${CUDA_SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE .)

target_link_libraries(${PROJECT_NAME}
  gtest
  Bifrost
  OptiX7Renderer
)

source_group("" FILES ${SRCS})
source_group("CUDA" FILES ${CUDA_SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Tests"
)

if (WIN32)
  add_definitions(/wd4723) # Disable potential divide by zero warning.
  add_definitions(/wd4756) # Disable overflow in constant arithmetic warning.

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Od") # The default is /O2, but that has less precise floating point arithmetic.
endif()
