cmake_minimum_required(VERSION 3.3)

set(PROJECT_NAME "Cogwheel")
project(${PROJECT_NAME})

set(COGWHEEL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(COGWHEEL_COGS_DIR "${COGWHEEL_ROOT_DIR}/cogs")
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

# Setup cogs
file(GLOB COG_SUB_DIRECTORIES RELATIVE ${COGWHEEL_COGS_DIR} "${COGWHEEL_COGS_DIR}/*")
foreach(COG_DIR ${COG_SUB_DIRECTORIES})
  message(STATUS "Linking Cog: ${COG_DIR}")
  set(FULL_DIR "${COGWHEEL_COGS_DIR}/${COG_DIR}")
  set(SETUP_FILE ${FULL_DIR}/Setup.cmake)
  if (EXISTS ${SETUP_FILE})
    include(${SETUP_FILE})
  else()
    subdirs(${FULL_DIR})
  endif()
endforeach()

macro(setup_applications DIR_TO_APPS APP_TYPE)
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
setup_applications(${COGWHEEL_APPS_DIR} "Application")
setup_applications(${COGWHEEL_DEV_APPS_DIR} "Developer App")
setup_applications(${COGWHEEL_TESTS_DIR} "Test")
