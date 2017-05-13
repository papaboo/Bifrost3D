set(GLFW_COG_DIR "${COGWHEEL_COGS_DIR}/GLFWDriver")
set(GLFW_ZIP_URL "https://github.com/glfw/glfw/releases/download/3.2.1/glfw-3.2.1.zip")
set(GLFW_ZIP_DEST "${COGWHEEL_LIBS_DIR}/glfw-3.2.1.zip")
set(GLFW_SOURCE_CMAKE "${COGWHEEL_LIBS_DIR}/glfw-3.2.1.zip")
set(GLFW_LIBS_DIR "${COGWHEEL_LIBS_DIR}/glfw-3.2.1")

# TODO Split into function that downloads, unzips and deletes.
if (NOT EXISTS ${GLFW_LIBS_DIR}/CMakeLists.txt)
  message(STATUS "Downloading '${GLFW_ZIP_URL}'
    to '${GLFW_ZIP_DEST}'
    if the download fails, then manually downloading the zip will work as well."
  )

  file(DOWNLOAD
    "${GLFW_ZIP_URL}"
    "${GLFW_ZIP_DEST}"
    SHOW_PROGRESS
    EXPECTED_MD5;824c99eea073bdd6d2fec76b538f79af
    # no TIMEOUT
    STATUS status
    LOG log
  )

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "error: downloading '${GLFW_ZIP_URL}'
      status_code: ${status_code}
      status_string: ${status_string}
      log: ${log}"
    )

  else()

    # Unzip the source
    message(STATUS "Unzipping glfw-3.2.1.zip")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${GLFW_ZIP_DEST}" WORKING_DIRECTORY "${COGWHEEL_LIBS_DIR}" )

    # Delete the zip file
    message(STATUS "Deleting glfw-3.2.1.zip")
    file(REMOVE "${GLFW_ZIP_DEST}")
  endif()
endif()

if (EXISTS ${GLFW_LIBS_DIR}/CMakeLists.txt)
  # Disable default GLFW options
  option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
  option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
  option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
  option(GLFW_INSTALL "Generate installation target" OFF)

  # Add GLFW lib
  add_subdirectory(${GLFW_LIBS_DIR} ${CMAKE_BINARY_DIR}/glfw)

  # Move GLFW source into Libs folder
  set_target_properties(glfw PROPERTIES
                        FOLDER "Libs")

  # Add the GLFW driver project
  add_library(GLFWDriver GLFWDriver.h GLFWDriver.cpp)

  target_include_directories(GLFWDriver
    PUBLIC .
    PRIVATE "${GLFW_LIBS_DIR}/include"
  )

  target_link_libraries(GLFWDriver
    PUBLIC Cogwheel
    PRIVATE glfw
  )

  source_group("GLFWDriver" FILES GLFWDriver.h GLFWDriver.cpp)

  set_target_properties(GLFWDriver PROPERTIES 
    LINKER_LANGUAGE CXX
    FOLDER "Cogs"
  )

endif()