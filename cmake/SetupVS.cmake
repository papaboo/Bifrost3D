if(MSVC)
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    # Add OpenMP to Viual Studio compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()

  add_definitions(-D_SCL_SECURE_NO_WARNINGS) # VS should not warn about 'unsafe' calls to std::_Copy_impl
endif()