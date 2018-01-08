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
// * SSAO
// ** Bent normals use case https://80.lv/articles/ssrtgi-toughest-challenge-in-real-time-3d/
// * SSR
// ** An Adaptive Acceleration Structure for Screen-space Ray Tracing, http://jankautz.com/publications/AcceleratedSSRT_HPG15.pdf
// ** Screen Space Reflections in Killing Floor 2, https://sakibsaikia.github.io/graphics/2016/12/25/Screen-Space-Reflection-in-Killing-Floor-2.html
// ** Real-Time Global Illumination using Precomputed Light Field Probes, http://graphics.cs.williams.edu/papers/LightFieldI3D17/McGuire2017LightField.pdf
// ** Reprojection reflections, http://bitsquid.blogspot.dk/2017/06/reprojecting-reflections_22.html
// ** http://bitsquid.blogspot.dk/2017/08/notes-on-screen-space-hiz-tracing.html
// * HDR backbuffer.
// * Frustum culling.
// * Sort models by material ID as well and use the info while rendering.
//   If the material hasn't changed then don't rebind it or the textures.
//----------------------------------------------------------------------------
class Renderer final : public IRenderer {
public:
    static IRenderer* initialize(ID3D11Device1& device, int width_hint, int height_hint);
    ~Renderer();

    Cogwheel::Core::Renderers::UID get_ID() const { return m_renderer_ID; }

    void handle_updates();

    void render(Cogwheel::Scene::Cameras::UID camera_ID, int width, int height);

private:

    Renderer(ID3D11Device1& device, int width_hint, int height_hint);

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