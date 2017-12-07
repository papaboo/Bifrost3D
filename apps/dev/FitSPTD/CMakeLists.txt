if (TARGET OptiXRenderer)

  set(PROJECT_NAME "FitSPTD")

  set(SRCS Brdf.h main.cpp NelderMead.h)

  add_executable(${PROJECT_NAME} ${SRCS})

  target_include_directories(${PROJECT_NAME} PRIVATE .)

  target_link_libraries(${PROJECT_NAME}
    Cogwheel
    OptiXRenderer
    StbImageWriter
  )

  source_group("" FILES ${SRCS})

  set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "Apps/Dev"
  )

  if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
      NOMINMAX # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
      COGWHEEL_SHADING_DIR="${COGWHEEL_COGS_DIR}/Cogwheel/Cogwheel/Assets/Shading/"
    )
  endif()
else()
  message(STATUS "OptiXRenderer not found. Skipping albedo computation development project.")
endif()