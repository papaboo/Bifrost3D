// Cogwheel ImGui DX11 Renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_IMGUI_RENDERERS_DX11_RENDERER_H_
#define _COGWHEEL_IMGUI_RENDERERS_DX11_RENDERER_H_

#include <DX11Renderer/Compositor.h>

// ------------------------------------------------------------------------------------------------
// Forward declarations.
// ------------------------------------------------------------------------------------------------
/*
struct ID3D11Device1;
struct ID3D11DeviceContext1;
namespace DX11Renderer {
template <typename T> class OwnedResourcePtr;
using ODevice1 = DX11Renderer::OwnedResourcePtr<ID3D11Device1>;
using ODeviceContext1 = DX11Renderer::OwnedResourcePtr<ID3D11DeviceContext1>;
}
*/

namespace ImGui {
namespace Renderers {

// ------------------------------------------------------------------------------------------------
// Dear IMGUI DX11 renderer.
// Heavily based on the DirextX 11 Dear IMGUI sample.
// ------------------------------------------------------------------------------------------------
class DX11Renderer : public ::DX11Renderer::IGuiRenderer {
public:
    DX11Renderer(::DX11Renderer::ODevice1& device);
    void render(::DX11Renderer::ODeviceContext1& context);

private:
    // Pimpl the state to avoid exposing DirectX dependencies.
    struct Implementation;
    Implementation* m_impl;
};

} // NS Renderers
} // NS ImGui

#endif // _COGWHEEL_IMGUI_RENDERERS_DX11_RENDERER_H_