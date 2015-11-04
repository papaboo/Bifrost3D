cmake_minimum_required(VERSION 3.3)

set(PROJECT_NAME "Cogwheel")
project(${PROJECT_NAME})

set(COGWHEEL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(COGWHEEL_COGS_DIR "${COGWHEEL_ROOT_DIR}/cogs")
set(COGWHEEL_APPS_DIR "${COGWHEEL_ROOT_DIR}/apps")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Global cmake configuration
include("cmake/SetupVS.cmake")

# Functions
# Copy files to build folder
function (install_files FILES)
  foreach(FILE ${FILES})
    get_filename_component(NAME ${FILE} NAME)
    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
      set(PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/")
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILE} ${PATH}/${NAME})
    endforeach()
  endforeach()
endfunction()

# Setup cogs
file(GLOB COG_SUB_DIRECTORIES RELATIVE ${COGWHEEL_COGS_DIR} "${COGWHEEL_COGS_DIR}/*")
foreach(COG_DIR ${COG_SUB_DIRECTORIES})
  message(STATUS "Linking Cog: ${COG_DIR}")
  set(FULL_DIR "${COGWHEEL_COGS_DIR}/${COG_DIR}")
  set(SETUP_FILE ${FULL_DIR}/Setup.cmake)
  if (EXISTS ${SETUP_FILE})
    include(${SETUP_FILE})
  else()
    include_directories(${FULL_DIR})
    subdirs(${FULL_DIR})
  endif()
endforeach()

# Setup applications
file(GLOB APP_SUB_DIRECTORIES RELATIVE ${COGWHEEL_APPS_DIR} "${COGWHEEL_APPS_DIR}/*")
foreach(APP_DIR ${APP_SUB_DIRECTORIES})
  message(STATUS "Linking Application: ${APP_DIR}")
  set(FULL_DIR "${COGWHEEL_APPS_DIR}/${APP_DIR}")
  subdirs(${FULL_DIR})
endforeach()