set(IMGUI_SRCS
  ImGui/ImGuiAdaptor.h
  ImGui/ImGuiAdaptor.cpp
)

set(IMGUI_SRC_SRCS
  ImGui/Src/imconfig.h
  ImGui/Src/imgui.h
  ImGui/Src/imgui.cpp
  ImGui/Src/imgui_demo.cpp
  ImGui/Src/imgui_draw.cpp
  ImGui/Src/imgui_internal.h
)

set(IMGUI_RENDERERS
  ImGui/Renderers/DX11Renderer.h
  ImGui/Renderers/DX11Renderer.cpp
)

add_library(ImGui ${IMGUI_SRCS} ${IMGUI_SRC_SRCS} ${IMGUI_RENDERERS})

target_include_directories(ImGui PUBLIC .)

target_link_libraries(ImGui PUBLIC Bifrost PRIVATE DX11Renderer)

source_group("ImGui" FILES ${IMGUI_SRCS})
source_group("ImGui\\Src" FILES ${IMGUI_SRC_SRCS})
source_group("ImGui\\Renderer" FILES ${IMGUI_RENDERERS})

set_target_properties(ImGui PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Extensions"
)