# Find AntTweakBar library and include path.
#
# Once done this will define
#   ANT_TWEAK_BAR_FOUND
#   ANT_TWEAK_BAR_INCLUDE_DIRS
#   ANT_TWEAK_BAR_LIBRARIES

if (WIN32)

  set(ANT_TWEAK_BAR_ROOT "${BIFROST_LIBS_DIR}/AntTweakBar")
  
  # Download if library doesn't exist.
  if (NOT EXISTS ${ANT_TWEAK_BAR_ROOT}/include/AntTweakBar.h)
  
    set(ANT_TWEAK_BAR_ZIP_URL "https://github.com/papaboo/AntTweakBarLib/archive/master.zip")
    set(ANT_TWEAK_BAR_LIBS_DIR "${BIFROST_LIBS_DIR}/AntTweakBar")
    set(ANT_TWEAK_BAR_ZIP_DEST "${ANT_TWEAK_BAR_LIBS_DIR}.zip")
  
    message(STATUS "Downloading '${ANT_TWEAK_BAR_ZIP_URL}'
            to '${ANT_TWEAK_BAR_ZIP_DEST}'
            if the download fails, then manually downloading the zip will work as well.")

    file(DOWNLOAD "${ANT_TWEAK_BAR_ZIP_URL}" "${ANT_TWEAK_BAR_ZIP_DEST}"
         SHOW_PROGRESS
         # EXPECTED_MD5;824c99eea073bdd6d2fec76b538f79af
         # no TIMEOUT
         STATUS status
         LOG log)

    list(GET status 0 status_code)
    list(GET status 1 status_string)

    if(NOT status_code EQUAL 0)
      message(FATAL_ERROR "error: downloading '${ANT_TWEAK_BAR_ZIP_URL}'
              status_code: ${status_code}
              status_string: ${status_string}
              log: ${log}")

    else()

      # Unzip the source
      message(STATUS "Unzipping AntTweakBar.zip")
      execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${ANT_TWEAK_BAR_ZIP_DEST}" WORKING_DIRECTORY "${BIFROST_LIBS_DIR}")
      file(RENAME "${ANT_TWEAK_BAR_LIBS_DIR}Lib-master" ${ANT_TWEAK_BAR_LIBS_DIR})

      # Delete the zip file
      message(STATUS "Deleting AntTweakBar.zip")
      file(REMOVE "${ANT_TWEAK_BAR_ZIP_DEST}")
    endif()
  endif()

  find_path(ANT_TWEAK_BAR_INCLUDE_DIRS AntTweakBar.h
            PATHS ${ANT_TWEAK_BAR_ROOT}/include
            DOC "The directory where AntTweakBar.h resides")

  # Determine if we should use 32 or 64 bit lib
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ANT_LIB_NAME AntTweakBar64)
  else()
    set(ANT_LIB_NAME AntTweakBar)
  endif()

  find_library(ANT_TWEAK_BAR_LIBRARIES ${ANT_LIB_NAME}
               PATHS ${ANT_TWEAK_BAR_ROOT}/lib
               DOC "The AntTweakBar library")
			 
else() # OS X or UNIX
  find_path(ANT_TWEAK_BAR_INCLUDE_DIRS AntTweakBar.h
            PATHS
            /usr/local/include
            /usr/X11/include
            /usr/include)

  find_library(ANT_TWEAK_BAR_LIBRARIES AntTweakBar
               PATHS /usr/local /usr/X11 /usr
               PATH_SUFFIXES lib64 lib dylib)
endif()

# Handle the QUIETLY and REQUIRED arguments and set ANT_TWEAK_BAR_FOUND to TRUE if
# all listed variables are set.
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(AntTweakBar DEFAULT_MSG ANT_TWEAK_BAR_LIBRARIES ANT_TWEAK_BAR_INCLUDE_DIRS)

if (WIN32 AND ANT_TWEAK_BAR_FOUND)
  # Copy ant dll to build folder
  string(REPLACE ".lib" ".dll" ANT_TWEAK_BAR_DLL ${ANT_TWEAK_BAR_LIBRARIES})
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    set(BUILD_FOLDER "${CMAKE_BINARY_DIR}/bin/${CONFIG}")
    get_filename_component(FILENAME ${ANT_TWEAK_BAR_DLL} NAME)
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ANT_TWEAK_BAR_DLL} ${BUILD_FOLDER}/${FILENAME})
  endforeach()
endif()