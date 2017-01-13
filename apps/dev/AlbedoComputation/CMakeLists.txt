if (TARGET OptiXRenderer)

  set(PROJECT_NAME "AlbedoComputation")

  set(SRCS 
    main.cpp
  )

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
    add_definitions(-DNOMINMAX) # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
  endif()
else()
  message(STATUS "OptiXRenderer not found. Skipping albedo computation development project.")
endif()