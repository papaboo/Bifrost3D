// DirectX 11 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_H_
#define _DX11RENDERER_RENDERER_H_

//----------------------------------------------------------------------------
// Forward declarations.
//----------------------------------------------------------------------------
namespace Cogwheel {
namespace Core {
class Engine;
class Window;
}
}
struct HWND__;
typedef HWND__* HWND;

namespace DX11Renderer {

//----------------------------------------------------------------------------
// DirectX 11 renderer.
// TODO
// * Upload models.
// * Light sources.
// * Render environment.
// * Materials.
// ** Material properties buffer.
// ** Textures.
// * SSAO + IBL.
// * Create an mvp matrix array, compute model * view_proj on the GPU and index into this array.
//----------------------------------------------------------------------------
class Renderer final {
public:
    static Renderer* initialize(HWND& hwnd, const Cogwheel::Core::Window& window);
    ~Renderer();

    void render(const Cogwheel::Core::Engine& engine);

private:

    Renderer(HWND& hwnd, const Cogwheel::Core::Window& window);
    
    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Renderer(Renderer& other) = delete;
    Renderer& operator=(const Renderer& rhs) = delete;

    // Pimpl the state to avoid exposing DirectX dependencies.
    class Implementation;
    Implementation* m_impl;
};

static inline void render_callback(const Cogwheel::Core::Engine& engine, void* renderer) {
    static_cast<Renderer*>(renderer)->render(engine);
}

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_H_