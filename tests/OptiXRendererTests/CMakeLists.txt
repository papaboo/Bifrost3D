if (TARGET OptiXRenderer)

  set(PROJECT_NAME "OptiXRendererTests")

  set(SRCS 
    BSDFTestUtils.h
    main.cpp
    MiscTest.h
    Utils.h
  )

  set(ASSETS_SRCS
    Assets/FlagsTest.h
  )

  set(BSDFS_SRCS
    BSDFs/BurleyTest.h
    BSDFs/GGXTest.h
    BSDFs/LambertTest.h
    BSDFs/OrenNayarTest.h
  )

  set(LIGHTSOURCES_SRCS
    LightSources/SphereLightTest.h
    LightSources/SpotLightTest.h
  )

  set(SHADING_MODELS_SRCS
    ShadingModels/DefaultShadingTest.h
    ShadingModels/TransmissiveShadingTest.h
    ShadingModels/ShadingModelTestUtils.h
    ShadingModels/UtilsTest.h
  )

  add_executable(${PROJECT_NAME} ${SRCS} ${ASSETS_SRCS} ${BSDFS_SRCS} ${LIGHTSOURCES_SRCS} ${SHADING_MODELS_SRCS})

  target_include_directories(${PROJECT_NAME} PRIVATE .)

  target_link_libraries(${PROJECT_NAME}
    gtest
    Bifrost
    OptiXRenderer
  )

  source_group("" FILES ${SRCS})
  source_group("Assets" FILES ${ASSETS_SRCS})
  source_group("BSDFs" FILES ${BSDFS_SRCS})
  source_group("LightSources" FILES ${LIGHTSOURCES_SRCS})
  source_group("ShadingModels" FILES ${SHADING_MODELS_SRCS})

  set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "Tests"
  )

  if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
      NOMINMAX # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
    )

    add_definitions(/wd4723) # Disable potential divide by zero warning.
    add_definitions(/wd4756) # Disable overflow in constant arithmetic warning.

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Od") # The default is /O2, but that has less precise floating point arithmetic.
  endif()
else()
  message(STATUS "OptiXRenderer not found. Skipping OptiXRenderer tests.")
endif()