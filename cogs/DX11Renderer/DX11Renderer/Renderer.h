// DirectX 11 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_H_
#define _DX11RENDERER_RENDERER_H_

#include <Dx11Renderer/Compositor.h>

namespace DX11Renderer {

//----------------------------------------------------------------------------
// DirectX 11 renderer.
// Future work
// * Device should be a reference instead of a pointer.
// * SSAO
// * HDR backbuffer.
// * Frustum culling.
// * Sort models by material ID as well and use the info while rendering.
//   If the material hasn't changed then don't rebind it or the textures.
//----------------------------------------------------------------------------
class Renderer final : public IRenderer {
public:
    static IRenderer* initialize(ID3D11Device1* device, int width_hint, int height_hint);
    ~Renderer();

    Cogwheel::Core::Renderers::UID get_ID() const { return m_renderer_ID; }

    void handle_updates();

    void render(Cogwheel::Scene::Cameras::UID camera_ID, int width, int height);

private:

    Renderer(ID3D11Device1* device, int width_hint, int height_hint);

    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Renderer(Renderer& other) = delete;
    Renderer& operator=(const Renderer& rhs) = delete;

    Cogwheel::Core::Renderers::UID m_renderer_ID;

    // Pimpl the state to avoid exposing DirectX dependencies.
    class Implementation;
    Implementation* m_impl;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_H_