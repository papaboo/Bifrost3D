set(PROJECT_NAME "AlbedoComputation")

# Add current path to CMake's module path, so we can use our own FindOptiX.cmake.
set(CMAKE_MODULE_PATH
  "${COGWHEEL_COGS_DIR}/OptiXRenderer"
  ${CMAKE_MODULE_PATH}
)

# TODO Is it possible to inherit these from OptiXRenderer?
find_package(CUDA 7.5)
find_package(OptiX)

set(SRCS 
  main.cpp
)

add_executable(${PROJECT_NAME} ${SRCS})

include_directories(. ${CUDA_INCLUDE_DIRS} ${OPTIX_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
  Gogwheel
  OptiXRenderer
  StbImageWriter
  ${CUDA_LIBRARIES}
  ${OPTIX_LIBRARIES}
)

source_group("" FILES ${SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps/Dev"
)

if (WIN32)
  add_definitions(-DNOMINMAX) # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
endif()