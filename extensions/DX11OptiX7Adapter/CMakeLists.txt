set(LIBRARY_NAME "DX11OptiX7Adaptor")

# Check prerequisite libraries are present

if (NOT TARGET OptiX7Renderer)
  message(WARNING "${LIBRARY_NAME} skipped: OptiX7Renderer not found.")
  return()
endif()

if (NOT TARGET DX11Renderer)
  message(WARNING "${LIBRARY_NAME} skipped: DX11Renderer not found.")
  return()
endif()

find_package(CUDAToolkit 11.1)

if (NOT CUDAToolkit_FOUND)
  message(WARNING "${LIBRARY_NAME} skipped: CUDA not installed.")
  return()
endif()

# Setup library

set(SRCS 
  DX11OptiXAdaptor/Adaptor.h 
  DX11OptiXAdaptor/Adaptor.cpp
)

add_library(${LIBRARY_NAME} ${SRCS})

target_include_directories(${LIBRARY_NAME} PUBLIC . PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(${LIBRARY_NAME} PRIVATE
  Bifrost
  DX11Renderer
  OptiX7Renderer
  CUDA::cuda_driver # CUDA driver API
  CUDA::cudart # CUDA runtime API
)

source_group("" FILES ${SRCS})

set_target_properties(${LIBRARY_NAME} PROPERTIES 
 LINKER_LANGUAGE CXX
 FOLDER "Extensions"
)
