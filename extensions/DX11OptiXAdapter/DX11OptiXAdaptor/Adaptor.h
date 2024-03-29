// Adaptor for the OptiX renderer that allows it to render to a DX11 buffer.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _DX11_OPTIX_ADAPTOR_H_
#define _DX11_OPTIX_ADAPTOR_H_

#include <Dx11Renderer/Compositor.h>

//-------------------------------------------------------------------------------------------------
// Forward declarations.
//-------------------------------------------------------------------------------------------------
namespace OptiXRenderer { class Renderer; }

namespace DX11OptiXAdaptor {

//-------------------------------------------------------------------------------------------------
// Adaptor for the OptiX renderer that allows it to render to a DX11 buffer.
// Future work:
// * Render directly to DX11 texture. Wrap a mapped DX11 texture in a CUDA surface and copy the accumulation buffer to it.
// * We should probably set the OMSetDepthStencilState and RSSetState before rendering.
// * Try storing everything in a command list and execute that instead.
//-------------------------------------------------------------------------------------------------
class Adaptor final : public DX11Renderer::IRenderer {
public:
    Adaptor(DX11Renderer::ODevice1& device, const std::filesystem::path& data_directory);
    static DX11Renderer::IRenderer* initialize(DX11Renderer::ODevice1& device, const std::filesystem::path& data_directory) { return new Adaptor(device, data_directory); }
    ~Adaptor();

    Bifrost::Core::RendererID get_ID() const;
    OptiXRenderer::Renderer* get_renderer();

    void handle_updates();

    DX11Renderer::RenderedFrame render(Bifrost::Scene::CameraID camera_ID, Bifrost::Math::Vector2i frame_size);
    std::vector<Bifrost::Scene::Screenshot> request_auxiliary_buffers(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::Cameras::ScreenshotContent content_requested, Bifrost::Math::Vector2i frame_size);

private:

    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Adaptor(Adaptor& other) = delete;
    Adaptor& operator=(const Adaptor& rhs) = delete;

    // Pimpl the state to avoid exposing DirectX and OptiX dependencies.
    class Implementation;
    Implementation* m_impl;
};

} // NS DX11OptiXAdaptor

#endif // _DX11_OPTIX_ADAPTOR_H_