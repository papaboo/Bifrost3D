// DirectX 11 compositor.
//-------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_COMPOSITOR_H_
#define _DX11RENDERER_COMPOSITOR_H_

#include <Cogwheel/Scene/Camera.h>

//-------------------------------------------------------------------------------------------------
// Forward declarations.
//-------------------------------------------------------------------------------------------------
namespace Cogwheel {
namespace Core {
class Engine;
class Window;
}
}
struct HWND__;
typedef HWND__* HWND;
struct ID3D11Device1;

namespace DX11Renderer {

//-------------------------------------------------------------------------------------------------
// Renderer interface.
// Future work:
// * Pass masked areas (rects) from overlapping (opaque) cameras in render, to help with masking and culling.
//-------------------------------------------------------------------------------------------------
class IRenderer {
public:
    virtual ~IRenderer() {}
    virtual void handle_updates() = 0;
    virtual void render(Cogwheel::Scene::Cameras::UID camera_ID) = 0; // TODO Pass in pixel coord rect.
};

typedef IRenderer*(*RendererCreator)(ID3D11Device1*);

//-------------------------------------------------------------------------------------------------
// DirectX 11 compositor.
// Composits the rendered images from various cameras attached to the window.
// TODO:
// * The render backends should have a UID data model representation,
//   such that the cameras can safely reference them.
//-------------------------------------------------------------------------------------------------
class Compositor final {
public:
    static Compositor* initialize(HWND& hwnd, const Cogwheel::Core::Window& window, RendererCreator renderer_creator);
    ~Compositor();

    IRenderer* attach_renderer(RendererCreator renderer_creator);
    void set_active_renderer(IRenderer* renderer); // TODO Remove and let the cameras pick their renderer.

    void render();

private:

    Compositor(HWND& hwnd, const Cogwheel::Core::Window& window);

    // Delete copy constructors to avoid having multiple versions of the same Compositor.
    Compositor(Compositor& other) = delete;
    Compositor& operator=(const Compositor& rhs) = delete;

    // Pimpl the state to avoid exposing DirectX dependencies.
    class Implementation;
    Implementation* m_impl;
};

static inline void render_callback(const Cogwheel::Core::Engine& engine, void* compositor) {
    static_cast<Compositor*>(compositor)->render();
}

} // NS DX11Renderer

#endif // _DX11RENDERER_COMPOSITOR_H_