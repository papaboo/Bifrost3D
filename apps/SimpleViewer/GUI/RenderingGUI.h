// SimpleViewer rendering GUI.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_GUI_RENDERING_GUI_H_
#define _SIMPLEVIEWER_GUI_RENDERING_GUI_H_

#include <ImGui/ImGuiAdaptor.h>

// ------------------------------------------------------------------------------------------------
// Forward declarations
// ------------------------------------------------------------------------------------------------
namespace DX11Renderer { 
class Compositor;
class Renderer;
}

namespace GUI {

class RenderingGUI : public ImGui::IImGuiFrame {
public:
    RenderingGUI(DX11Renderer::Compositor* compositor, DX11Renderer::Renderer* renderer)
        : m_compositor(compositor), m_renderer(renderer) {}

    void layout_frame();

private:
    DX11Renderer::Compositor* m_compositor;
    DX11Renderer::Renderer* m_renderer;
};

} // NS GUI

#endif // _SIMPLEVIEWER_GUI_RENDERING_GUI_H_