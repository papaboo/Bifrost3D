set(GLFW_COG_DIR "${COGWHEEL_COGS_DIR}/GLFWDriver")
set(GLFW_ZIP_URL "https://github.com/glfw/glfw/releases/download/3.1.2/glfw-3.1.2.zip")
set(GLFW_ZIP_DEST "${GLFW_COG_DIR}/lib/glfw-3.1.2.zip")
set(GLFW_SOURCE_CMAKE "${GLFW_COG_DIR}/lib/glfw-3.1.2.zip")
set(GLFW_LIBS_DIR "${GLFW_COG_DIR}/lib/glfw-3.1.2")

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
    EXPECTED_MD5;8023327bfe979b3fe735e449e2f54842
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
    message(STATUS "Unzipping glfw-3.1.2.zip")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${GLFW_ZIP_DEST}" WORKING_DIRECTORY "${GLFW_COG_DIR}/lib" )
    
    # Delete the zip file
    message(STATUS "Deleting glfw-3.1.2.zip")
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
  add_subdirectory(${GLFW_LIBS_DIR})
  include_directories("${GLFW_LIBS_DIR}/include")
  
  # Move GLFW source into Libs folder
  set_target_properties(glfw PROPERTIES
                        FOLDER "Libs")

  # Add the GLFW driver project
  subdirs(${GLFW_COG_DIR}/src)
  include_directories(${GLFW_COG_DIR}/src)

endif()