cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME "Bifrost3D")
project(${PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BIFROST_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(BIFROST_EXTENSIONS_DIR "${BIFROST_ROOT_DIR}/extensions")
set(BIFROST_LIBS_DIR "${BIFROST_ROOT_DIR}/libs")
set(BIFROST_APPS_DIR "${BIFROST_ROOT_DIR}/apps")
set(BIFROST_DEV_APPS_DIR "${BIFROST_ROOT_DIR}/apps/dev")
set(BIFROST_TESTS_DIR "${BIFROST_ROOT_DIR}/tests")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(BIFROST_DATA_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Data)
file(MAKE_DIRECTORY ${BIFROST_DATA_OUTPUT_DIRECTORY})

# Global cmake configuration
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Build configurations" FORCE)
include("cmake/SetupVS.cmake")

# Functions
# Copy file to build folder.
function (install_file FILE)
  get_filename_component(NAME ${FILE} NAME)
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    set(PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILE} ${PATH}/${NAME})
  endforeach()
endfunction()

# Copy files passed via ARGN to build folder.
macro(install_files)
  foreach(FILE ${ARGN})
    install_file(${FILE})
  endforeach()
endmacro()

set (INCLUDED_EXTENSIONS "")
macro(add_extension EXT_NAME)
  list(FIND INCLUDED_EXTENSIONS ${EXT_NAME} EXT_ALREADY_INCLUDED)
  if (${EXT_ALREADY_INCLUDED} EQUAL -1)
    message(STATUS "Adding extension: ${EXT_NAME}")
    set(INCLUDED_EXTENSIONS ${INCLUDED_EXTENSIONS} ${EXT_NAME})

    set(FULL_DIR "${BIFROST_EXTENSIONS_DIR}/${EXT_NAME}")
    subdirs(${FULL_DIR})
  else()
    message(STATUS "Extension already linked: ${EXT_NAME} ${EXT_ALREADY_INCLUDED}")
  endif()
endmacro()

#include Bifrost, our core datamodel, first.
message(STATUS "Creating Bifrost")
subdirs("${BIFROST_ROOT_DIR}/core/Bifrost")

  add_extension("DX11Renderer")
  add_extension("OptiXRenderer")
add_extension("DX11OptiXAdapter") # Depends on OptiXRenderer and DX11Renderer
add_extension("AntTweakBar")
add_extension("GLFWDriver")
add_extension("ImageOperations")
add_extension("ImGui") # Depends on DX11Renderer ... for now.
add_extension("ObjLoader")
add_extension("StbImageLoader")
add_extension("StbImageWriter")
add_extension("glTFLoader") # Depends on StbImageLoader and StbImageWriter
add_extension("TinyExr")
add_extension("Win32Driver")
add_extension("gtest")

macro(add_applications DIR_TO_APPS APP_TYPE)
  file(GLOB APP_DIRECTORIES RELATIVE ${DIR_TO_APPS} "${DIR_TO_APPS}/*")
  foreach(APP_DIR ${APP_DIRECTORIES})
    set(FULL_DIR "${DIR_TO_APPS}/${APP_DIR}")
    set(CMAKE_FILE ${FULL_DIR}/CMakeLists.txt)
    if (EXISTS ${CMAKE_FILE})
      message(STATUS "Adding ${APP_TYPE}: ${APP_DIR}")
      subdirs(${FULL_DIR})
    endif()
  endforeach()
endmacro()

# Setup applications
add_applications(${BIFROST_APPS_DIR} "Application")
add_applications(${BIFROST_DEV_APPS_DIR} "Developer App")
add_applications(${BIFROST_TESTS_DIR} "Test")
