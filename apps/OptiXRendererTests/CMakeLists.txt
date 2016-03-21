set(PROJECT_NAME "OptiXRendererTests")

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
  Utils.h
)

set(BSDFS_SRCS
  BSDFs/LambertTest.h
)

set(LIGHTSOURCES_SRCS
  LightSources/SphereLightTest.h
)

add_executable(${PROJECT_NAME} ${SRCS} ${BSDFS_SRCS} ${LIGHTSOURCES_SRCS})

include_directories(. ${CUDA_INCLUDE_DIRS} ${OPTIX_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
  gtest
  Gogwheel
  OptiXRenderer
  ${CUDA_LIBRARIES}
  ${OPTIX_LIBRARIES}
)

source_group("" FILES ${SRCS})
source_group("BSDFs" FILES ${BSDFS_SRCS})
source_group("LightSources" FILES ${LIGHTSOURCES_SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps"
)

if (WIN32)
  add_definitions(-DNOMINMAX) # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
endif()