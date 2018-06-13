cmake_minimum_required(VERSION 3.6)

set(PROJECT_NAME "Cogwheel")
project(${PROJECT_NAME})

set(COGWHEEL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(COGWHEEL_COGS_DIR "${COGWHEEL_ROOT_DIR}/cogs")
set(COGWHEEL_LIBS_DIR "${COGWHEEL_ROOT_DIR}/libs")
set(COGWHEEL_APPS_DIR "${COGWHEEL_ROOT_DIR}/apps")
set(COGWHEEL_DEV_APPS_DIR "${COGWHEEL_ROOT_DIR}/apps/dev")
set(COGWHEEL_TESTS_DIR "${COGWHEEL_ROOT_DIR}/tests")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(COGWHEEL_DATA_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Data)
file(MAKE_DIRECTORY ${COGWHEEL_DATA_OUTPUT_DIRECTORY})

# Global cmake configuration
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

set (INCLUDED_COGS "")
macro(include_cog COG_NAME)
  list(FIND INCLUDED_COGS ${COG_NAME} COG_ALREADY_INCLUDED)
  if (${COG_ALREADY_INCLUDED} EQUAL -1)
    message(STATUS "Linking Cog: ${COG_NAME}")
    set(INCLUDED_COGS ${INCLUDED_COGS} ${COG_NAME})

    set(FULL_DIR "${COGWHEEL_COGS_DIR}/${COG_NAME}")
    subdirs(${FULL_DIR})
  else()
    message(STATUS "Cog already linked: ${COG_NAME} ${COG_ALREADY_INCLUDED}")
  endif()
endmacro()

include_cog("Cogwheel")
  include_cog("DX11Renderer")
  include_cog("OptiXRenderer")
include_cog("DX11OptiXAdapter") # Depends on OptiXRenderer and DX11Renderer
include_cog("AntTweakBar")
include_cog("GLFWDriver")
include_cog("ImageOperations")
include_cog("Imgui") # Depends on DX11Renderer ... for now.
include_cog("ObjLoader")
include_cog("StbImageLoader")
include_cog("StbImageWriter")
include_cog("TinyExr")
include_cog("Win32Driver")
include_cog("gtest")

macro(include_applications DIR_TO_APPS APP_TYPE)
  file(GLOB APP_DIRECTORIES RELATIVE ${DIR_TO_APPS} "${DIR_TO_APPS}/*")
  foreach(APP_DIR ${APP_DIRECTORIES})
    set(FULL_DIR "${DIR_TO_APPS}/${APP_DIR}")
    set(CMAKE_FILE ${FULL_DIR}/CMakeLists.txt)
    if (EXISTS ${CMAKE_FILE})
      message(STATUS "Linking ${APP_TYPE}: ${APP_DIR}")
      subdirs(${FULL_DIR})
    endif()
  endforeach()
endmacro()

# Setup applications
include_applications(${COGWHEEL_APPS_DIR} "Application")
include_applications(${COGWHEEL_DEV_APPS_DIR} "Developer App")
include_applications(${COGWHEEL_TESTS_DIR} "Test")
