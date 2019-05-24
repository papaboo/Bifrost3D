set(PROJECT_NAME "ProgressiveMultijitteredSamples")

find_package(OpenGL REQUIRED)

set(SRCS
  main.cpp
  ProgressiveJittered.h
  ProgressiveMultijittered.h
  ProgressiveMultijitteredBlueNoise.h
  Tests.h
)

add_executable(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE . ${OpenGL_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
  Bifrost
  StbImageWriter
  ${OPENGL_LIBRARIES}
)

source_group("" FILES ${SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps/Dev"
)