if (TARGET DX11Renderer)

  set(PROJECT_NAME "DX11RendererTests")

  set(SRCS 
    main.cpp
    PrefixSumTest.h
    Utils.h
  )

  add_executable(${PROJECT_NAME} ${SRCS})

  target_include_directories(${PROJECT_NAME} PRIVATE .)

  target_link_libraries(${PROJECT_NAME}
    gtest
    Cogwheel
    DX11Renderer
  )

  source_group("" FILES ${SRCS})

  set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "Tests"
  )

  # target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Od") # The default is /O2, but that has less precise floating point arithmetic.
  target_compile_definitions(${PROJECT_NAME} PRIVATE 
    # NOMINMAX # Disable this silly flag that just breaks so much code.
    DX11_SHADER_ROOT=L"${COGWHEEL_COGS_DIR}/DX11Renderer/DX11Renderer/Shaders/"
  )
else()
  message(STATUS "DX11Renderer not found. Skipping DX11Renderer tests.")
endif()