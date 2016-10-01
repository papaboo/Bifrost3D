set(LIBRARY_NAME "DX12Renderer")

# Add current path to CMake's module path, so we can use our own FindWindowsSDK.cmake.
set(CMAKE_MODULE_PATH
  "${COGWHEEL_COGS_DIR}/DX12Renderer"
  ${CMAKE_MODULE_PATH}
)

find_package(DirectX12)

if (DIRECTX_12_FOUND AND WIN32)
  include_directories(. ${DIRECTX_12_INCLUDE_DIRS})

  set(ROOT_SRC
    DX12Renderer/Renderer.h
    DX12Renderer/Renderer.cpp
  )

  # Add the library.
  add_library(${LIBRARY_NAME}
    ${ROOT_SRC}
  )

  source_group("" FILES ${ROOT_SRC})

  target_link_libraries(${LIBRARY_NAME}
    Gogwheel
    ${DIRECTX_12_LIBRARIES}
  )

  set_target_properties(${LIBRARY_NAME} PROPERTIES 
    LINKER_LANGUAGE CXX
    FOLDER "Cogs"
  )
else()
  if (NOT DIRECTX_12_FOUND)
    message(WARNING "DirectX not found.")
  endif()
  if (NOT WIN32)
    message(WARNING "Only supported on windows.")
  endif()
  message(WARNING "Disabling DX12Renderer.")
endif()