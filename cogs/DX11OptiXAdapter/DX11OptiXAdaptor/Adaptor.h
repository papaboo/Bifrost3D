// Adaptor for the OptiX renderer that allows it to render to a DX11 buffer.
//-------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _DX11_OPTIX_ADAPTOR_H_
#define _DX11_OPTIX_ADAPTOR_H_

#include <Dx11Renderer/Compositor.h>

//-------------------------------------------------------------------------------------------------
// Forward declarations.
//-------------------------------------------------------------------------------------------------
namespace OptiXRenderer {
class Renderer;
}

namespace DX11OptiXAdaptor {

//-------------------------------------------------------------------------------------------------
// Adaptor for the OptiX renderer that allows it to render to a DX11 buffer.
// Future work:
// * We should probably set the OMSetDepthStencilState and RSSetState before rendering.
// * Try storing everything in a command list and execute that instead.
//-------------------------------------------------------------------------------------------------
class Adaptor final : public DX11Renderer::IRenderer {
public:
    static DX11Renderer::IRenderer* initialize(ID3D11Device1& device, int width_hint, int height_hint, const std::wstring& data_folder_path);
    ~Adaptor();

    Cogwheel::Core::Renderers::UID get_ID() const { return m_renderer_ID; }
    OptiXRenderer::Renderer* get_renderer();
    OptiXRenderer::Renderer* get_renderer() const;

    void handle_updates();

    void render(Cogwheel::Scene::Cameras::UID camera_ID, int width, int height);

private:

    Adaptor(ID3D11Device1& device, int width_hint, int height_hint, const std::wstring& data_folder_path);

    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Adaptor(Adaptor& other) = delete;
    Adaptor& operator=(const Adaptor& rhs) = delete;

    Cogwheel::Core::Renderers::UID m_renderer_ID;

    // Pimpl the state to avoid exposing DirectX and OptiX dependencies.
    class Implementation;
    Implementation* m_impl;
};

} // NS DX11OptiXAdaptor

#endif // _DX11_OPTIX_ADAPTOR_H_