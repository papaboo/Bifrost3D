if (MSVC)
  # Add OpenMP to Visual Studio compiler flags
  find_package(OpenMP REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

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

  # Multithreaded compilation for faster compilation
  add_compile_options(/MP)

  # Make some /O2 options explicit to make it obvious in the Visual Studio GUI what is enabled
  add_compile_options("$<$<CONFIG:RELEASE>:/Oi>")
  add_compile_options("$<$<CONFIG:RELEASE>:/Ot>")

  # Whole program optimization
  add_compile_options("$<$<CONFIG:RELEASE>:/GL>")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Tells the linker to use link time code generation /LTCG.

  # Enable AVX2 intrinsics
  add_compile_options(/arch:AVX2)

  # Do not check for security issues
  add_compile_options(/GS-)

  # Disable run-time type information. Projects that absolutely needs this can enable it again, but the default must be to never create code that requires a dynamic cast.
  add_compile_options(/GR-)

  # Setup warnings

  # Remove warnings from system level includes
  add_compile_options(/experimental:external)
  add_compile_options(/external:W0)
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "/external:I ")

  add_definitions(-WX) # Warnings as errors
  # add_definitions(-Wall)

  add_definitions(-D_SCL_SECURE_NO_WARNINGS) # VS should not warn about 'unsafe' calls to std::_Copy_impl

  add_definitions(-wd4514) # Unreferenced inline function has been removed.

  add_definitions(-wd4625) # copy constructor was implicitly defined as deleted
  add_definitions(-wd4626) # assignment operator was implicitly defined as deleted

  add_definitions(-wd4710) # Function not inlined
  add_definitions(-wd4711) # Function selected for automatic inline expansion
  add_definitions(-wd4723) # Potential divide by zero

  add_definitions(-wd5026) # move constructor was implicitly defined as deleted
  add_definitions(-wd5027) # move assignment operator was implicitly defined as deleted

  add_definitions(-wd6993) # Do not warn that OpenMP is analyzed single threaded

  add_definitions(-wd26451) # Arithmetic overflow warning. Pops up all places where ints are used as sizes to allocated arrays.
  add_definitions(-wd26812) # Prefer enum class over enum.
endif()