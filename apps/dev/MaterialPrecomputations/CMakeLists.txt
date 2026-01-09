if (TARGET OptiXRenderer)

  set(PROJECT_NAME "MaterialPrecomputations")

  set(SRCS FitRhoApproximation.h GGXAlphaFromMaxPDF.h main.cpp PmjbRNG.h PrecomputeDielectricBSDFRho.h PrecomputeRoughBRDFRho.h)

  add_executable(${PROJECT_NAME} ${SRCS})

  target_include_directories(${PROJECT_NAME} PRIVATE .)

  target_link_libraries(${PROJECT_NAME}
    Bifrost
    OptiXRenderer
    StbImageWriter
  )

  source_group("" FILES ${SRCS})

  set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "Apps/Dev"
  )

  target_compile_definitions(${PROJECT_NAME} PRIVATE 
    BIFROST_SHADING_DIR="${BIFROST_ROOT_DIR}/core/Bifrost/Bifrost/Assets/Shading/"
  )
else()
  message(STATUS "OptiXRenderer not found. Skipping material precomputation development project.")
endif()