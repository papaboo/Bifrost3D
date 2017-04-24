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

namespace DX11OptiXAdaptor {

//-------------------------------------------------------------------------------------------------
// Adaptor for the OptiX renderer that allows it to render to a DX11 buffer.
//-------------------------------------------------------------------------------------------------
class DX11OptiXAdaptor final : public DX11Renderer::IRenderer {
public:
    static DX11Renderer::IRenderer* initialize(ID3D11Device1* device);
    ~DX11OptiXAdaptor();

    void handle_updates();

    void render(Cogwheel::Scene::Cameras::UID camera_ID);

private:

    DX11OptiXAdaptor(ID3D11Device1* device);

    // Delete copy constructors to avoid having multiple versions of the same renderer.
    DX11OptiXAdaptor(DX11OptiXAdaptor& other) = delete;
    DX11OptiXAdaptor& operator=(const DX11OptiXAdaptor& rhs) = delete;

    // Pimpl the state to avoid exposing DirectX and OptiX dependencies.
    class Implementation;
    Implementation* m_impl;
};

} // NS DX11OptiXAdaptor

#endif // _DX11_OPTIX_ADAPTOR_H_