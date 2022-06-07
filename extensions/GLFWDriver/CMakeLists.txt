set(GLFW_ZIP_URL "https://github.com/glfw/glfw/releases/download/3.3.7/glfw-3.3.7.zip")
set(GLFW_LIBS_DIR "${BIFROST_LIBS_DIR}/glfw-3.3.7")
set(GLFW_ZIP_DEST "${GLFW_LIBS_DIR}.zip")

# TODO Split into function that downloads, unzips and deletes.
if (NOT EXISTS ${GLFW_LIBS_DIR}/CMakeLists.txt)
  if (NOT EXISTS ${GLFW_ZIP_DEST})
    message(STATUS "Downloading '${GLFW_ZIP_URL}'
      to '${GLFW_ZIP_DEST}'
      if the download fails, then manually download ${GLFW_ZIP_URL} and place it in ${BIFROST_LIBS_DIR}.")

    file(DOWNLOAD
      "${GLFW_ZIP_URL}"
      "${GLFW_ZIP_DEST}"
      SHOW_PROGRESS
      EXPECTED_MD5;6985b0ecfba89ff4a3c7bc7e22f85a9d
      # no TIMEOUT
      STATUS status
      LOG log)

    list(GET status 0 status_code)
    list(GET status 1 status_string)

    if(NOT status_code EQUAL 0)
      message(FATAL_ERROR "error: downloading '${GLFW_ZIP_URL}'
        status_code: ${status_code}
        status_string: ${status_string}
        log: ${log}")
      file(REMOVE "${GLFW_ZIP_DEST}") # remove the file to ensure that partially downloaded files won't corrupt a reconfiguration run.
    endif()
  endif()

  if (EXISTS ${GLFW_ZIP_DEST})
    # Unzip the source
    message(STATUS "Unzipping glfw-3.3.7.zip to ${GLFW_LIBS_DIR}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${GLFW_ZIP_DEST}" WORKING_DIRECTORY "${BIFROST_LIBS_DIR}" )

    # Delete the zip file
    message(STATUS "Deleting glfw-3.3.7.zip")
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

  # Move GLFW source into Libs/GLFW folder
  set_target_properties(glfw PROPERTIES FOLDER "Libs/GLFW")
  set_target_properties(update_mappings PROPERTIES FOLDER "Libs/GLFW")

  # Add the GLFW driver project
  add_library(GLFWDriver GLFWDriver.h GLFWDriver.cpp)

  target_include_directories(GLFWDriver
    PUBLIC .
    PRIVATE "${GLFW_LIBS_DIR}/include")

  target_link_libraries(GLFWDriver
    PUBLIC Bifrost
    PRIVATE glfw)

  source_group("GLFWDriver" FILES GLFWDriver.h GLFWDriver.cpp)

  set_target_properties(GLFWDriver PROPERTIES 
    LINKER_LANGUAGE CXX
    FOLDER "Extensions")
else()
  message(FATAL_ERROR "error: GLFW source not found at '${GLFW_LIBS_DIR}'")
endif()