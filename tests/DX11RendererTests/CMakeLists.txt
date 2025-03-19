if (TARGET DX11Renderer)

  set(PROJECT_NAME "DX11RendererTests")

  set(SRCS 
    BloomTest.h
    ExposureHistogramTest.h
    LogAverageLuminanceTest.h
    main.cpp
    PrefixSumTest.h
    Utils.h
  )

  set(MANAGERS_SRCS
    Managers/MaterialManagerTest.h
    Managers/MeshManagerTest.h
    Managers/MeshModelManagerTest.h
    Managers/TextureManagerTest.h
  )

  add_executable(${PROJECT_NAME} ${SRCS} ${MANAGERS_SRCS})

  target_include_directories(${PROJECT_NAME} PRIVATE .)

  target_link_libraries(${PROJECT_NAME}
    gtest
    Bifrost
    DX11Renderer
  )

  source_group("" FILES ${SRCS})
  source_group("Managers" FILES ${MANAGERS_SRCS})

  set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "Tests"
  )

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Od") # The default is /O2, but that has less precise floating point arithmetic.
else()
  message(STATUS "DX11Renderer not found. Skipping DX11Renderer tests.")
endif()