// SimpleViewer rendering GUI.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_GUI_RENDERING_GUI_H_
#define _SIMPLEVIEWER_GUI_RENDERING_GUI_H_

#include <Bifrost/Scene/Camera.h>

#include <ImGui/ImGuiAdaptor.h>

// ------------------------------------------------------------------------------------------------
// Forward declarations
// ------------------------------------------------------------------------------------------------
namespace DX11Renderer { 
class Compositor;
class Renderer;
}
namespace OptiXRenderer {
class Renderer;
}

namespace GUI {

class RenderingGUI : public ImGui::IImGuiFrame {
public:
    RenderingGUI(DX11Renderer::Compositor* compositor, DX11Renderer::Renderer* dx_renderer, OptiXRenderer::Renderer* optix_renderer = nullptr);
    ~RenderingGUI();

    void layout_frame();

private:
    DX11Renderer::Compositor* m_compositor;
    DX11Renderer::Renderer* m_dx_renderer;
    OptiXRenderer::Renderer* m_optix_renderer;

    struct {
        static const unsigned int max_path_length = 1024u;
        char path[max_path_length];
        unsigned int iterations = 0;
        Bifrost::Scene::Cameras::RequestedContent screenshot_content;
    } m_screenshot;

    // Pimpl the state to avoid exposing dependencies.
    struct State;
    State* m_state;
};

} // NS GUI

#endif // _SIMPLEVIEWER_GUI_RENDERING_GUI_H_