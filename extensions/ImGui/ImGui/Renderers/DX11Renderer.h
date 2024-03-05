// Bifrost ImGui DX11 Renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_IMGUI_RENDERERS_DX11_RENDERER_H_
#define _BIFROST_IMGUI_RENDERERS_DX11_RENDERER_H_

#include <DX11Renderer/Compositor.h>

namespace ImGui::Renderers {

// ------------------------------------------------------------------------------------------------
// Dear IMGUI DX11 renderer.
// Heavily based on the DirextX 11 Dear IMGUI sample.
// ------------------------------------------------------------------------------------------------
class DX11Renderer : public ::DX11Renderer::IGuiRenderer {
public:
    DX11Renderer(::DX11Renderer::ODevice1& device, const std::filesystem::path& data_directory);
    static IGuiRenderer* initialize(::DX11Renderer::ODevice1& device, const std::filesystem::path& data_directory) { return new DX11Renderer(device, data_directory); }
    void render(::DX11Renderer::ODeviceContext1& context);

private:
    // Pimpl the state to avoid exposing DirectX dependencies.
    struct Implementation;
    Implementation* m_impl;
};

} // NS ImGui::Renderers

#endif // _BIFROST_IMGUI_RENDERERS_DX11_RENDERER_H_
