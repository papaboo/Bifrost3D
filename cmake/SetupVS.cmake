if(MSVC)
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    # Add OpenMP to Viual Studio compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()

  function (create_project_VS_user_file PROJECT)
    set(USER_FILENAME ${CMAKE_BINARY_DIR}/${PROJECT}/${PROJECT}.vcxproj.user)
    set(TOOLS_VERSION \"12.0\")
    if (MSVC14)
      set(TOOLS_VERSION \"14.0\")
    endif()
    set(USER_CONFIG "<?xml version=\"1.0\" encoding=\"utf-8\"?>
<Project ToolsVersion=${TOOLS_VERSION} xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">
  <PropertyGroup>
    <LocalDebuggerWorkingDirectory>$(OutputPath)</LocalDebuggerWorkingDirectory>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
  </PropertyGroup>
</Project>")
    file(WRITE ${USER_FILENAME} ${USER_CONFIG})
  endfunction()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")

  # Setup warnings

  add_definitions(-WX) # Warnings as errors
  # add_definitions(-Wall)

  add_definitions(-D_SCL_SECURE_NO_WARNINGS) # VS should not warn about 'unsafe' calls to std::_Copy_impl

  add_definitions(-wd4514) # Unreferenced inline function has been removed.

  add_definitions(-wd4625) # copy constructor was implicitly defined as deleted
  add_definitions(-wd4626) # assignment operator was implicitly defined as deleted

  add_definitions(-wd4710) # Function not inlined
  add_definitions(-wd4711) # Function selected for automatic inline expansion.

  if (MSVC14)
      add_definitions(-wd5026) # move constructor was implicitly defined as deleted
      add_definitions(-wd5027) # move assignment operator was implicitly defined as deleted
  endif()
endif()