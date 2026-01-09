if (TARGET OptiXRenderer)

  set(PROJECT_NAME "SubsurfaceScatteringTestBed")

  set(SRCS main.cpp )

  add_executable(${PROJECT_NAME} ${SRCS})

  target_link_libraries(${PROJECT_NAME}
    Bifrost OptiXRenderer StbImageWriter
  )

  source_group("" FILES ${SRCS})

  set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "Apps/Dev"
  )

  if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
      NOMINMAX # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
    )
  endif()

  target_compile_options(${PROJECT_NAME} PRIVATE ${CXX_FAST_MATH})
else()
  message(STATUS "OptiXRenderer not found. Skipping subsurface scattering testbed.")
endif()