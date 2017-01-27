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
// Future work
// * DX 11.1
//   * Material manager. Store all materials in one constant buffer and offset it when binding, 
//     so it starts at the active material.
//   * Create a model to world matrix constant buffer.
// * SSAO
// * HDR backbuffer.
// * Frustum culling.
// * Proper IBL. Scale the IBL lookup by the diffuse and glossy rho. 
//   Requires actually approximating the glossy rho first (and maybe Burley rho as well.)
// * Sort models by material ID as well and use the info while rendering.
//   If the material hasn't changed then don't rebind it or the textures.
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