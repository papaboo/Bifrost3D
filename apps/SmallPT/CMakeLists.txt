SET(PROJECT_NAME "SmallPT")

SET(SRCS 
  main.cpp
  smallpt.h
)

add_executable(${PROJECT_NAME} ${SRCS})

include_directories(.)

source_group("" FILES ${SRCS})

# Find OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
	# Add OpenMP to host C++ compiler flags
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
  Gogwheel
  glfw
  ${OPENGL_LIBRARIES}
)