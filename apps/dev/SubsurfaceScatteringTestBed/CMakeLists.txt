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

  target_compile_options(${PROJECT_NAME} PRIVATE ${CXX_FAST_MATH})
else()
  message(STATUS "OptiXRenderer not found. Skipping subsurface scattering testbed.")
endif()