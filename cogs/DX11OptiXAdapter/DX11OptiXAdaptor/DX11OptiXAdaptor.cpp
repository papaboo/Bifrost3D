// Adaptor for the OptiX renderer that allows it to render to a DX11 buffer.
//-------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#include <DX11OptiXAdaptor/DX11OptiXAdaptor.h>
#include <DX11Renderer/Renderer.h>
#include <OptiXRenderer/Renderer.h>

namespace DX11OptiXAdaptor {

class DX11OptiXAdaptor::Implementation {
    OptiXRenderer::Renderer* m_optix_renderer;

public:

    DX11OptiXAdaptor::Implementation(ID3D11Device1* device) {

        // Get CUDA device from DX11 context.

        // Create OptiX Renderer on device.

        // Create DX texture and map it. Or a buffer?

    }

    bool is_valid() const {
        return false;
    }

    void handle_updates() {
        // m_optix_renderer->handle_updates();
    }

    void render(Cogwheel::Scene::Cameras::UID camera_ID) {
        // m_optix_renderer->render(camera_ID);
        // Fill mapped buffer with foobar values.
    }
};

DX11Renderer::IRenderer* DX11OptiXAdaptor::initialize(ID3D11Device1* device) {
    DX11OptiXAdaptor* r = new DX11OptiXAdaptor(device);
    if (r->m_impl->is_valid())
        return r;
    else {
        delete r;
        return nullptr;
    }
}

DX11OptiXAdaptor::DX11OptiXAdaptor(ID3D11Device1* device) {
    m_impl = new Implementation(device);
}

DX11OptiXAdaptor::~DX11OptiXAdaptor() {
    delete m_impl;
}

void DX11OptiXAdaptor::handle_updates() {
    m_impl->handle_updates();
}

void DX11OptiXAdaptor::render(Cogwheel::Scene::Cameras::UID camera_ID) {
    m_impl->render(camera_ID);
}

} // NS DX11OptiXAdaptor
