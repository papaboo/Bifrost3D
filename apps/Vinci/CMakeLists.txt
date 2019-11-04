if (TARGET OptiXRenderer)

  set(PROJECT_NAME "Vinci")

  set(SRCS main.cpp SceneGenerator.h)

  add_executable(${PROJECT_NAME} ${SRCS})

  target_include_directories(${PROJECT_NAME} PRIVATE .)

  target_link_libraries(${PROJECT_NAME}
    Bifrost
    DX11OptiXAdaptor
    DX11Renderer
    glTFLoader
    OptiXRenderer
    ObjLoader
    StbImageLoader
    StbImageWriter
    TinyExr
    Win32Driver
  )

  source_group("" FILES ${SRCS})

  set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "Apps"
  )

  if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
      NOMINMAX # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
    )
  endif()
else()
  message(STATUS "OptiXRenderer not found. Skipping Vinci application.")
endif()