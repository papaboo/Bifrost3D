if (TARGET DX11Renderer AND TARGET OptiXRenderer)

  set(LIBRARY_NAME "DX11OptiXAdaptor")

  set(SRCS 
    DX11OptiXAdaptor/Adaptor.h 
    DX11OptiXAdaptor/Adaptor.cpp
  )

  add_library(${LIBRARY_NAME} ${SRCS})

  target_include_directories(${LIBRARY_NAME} PUBLIC .)

  target_link_libraries(${LIBRARY_NAME} PRIVATE
    DX11Renderer
    OptiXRenderer
  )

  source_group("" FILES ${SRCS})

  set_target_properties(${LIBRARY_NAME} PROPERTIES 
    LINKER_LANGUAGE CXX
    FOLDER "Extensions"
  )
else()
  if (NOT TARGET DX11Renderer)
    message(STATUS "DX11Renderer not found. Skipping DirectX OptiX adaptor.")
  endif()
  if (NOT TARGET OptiXRenderer)
    message(STATUS "OptiXRenderer not found. Skipping DirectX OptiX adaptor.")
  endif()
endif()