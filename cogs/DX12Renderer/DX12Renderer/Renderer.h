// DirectX 12 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX12RENDERER_RENDERER_H_
#define _DX12RENDERER_RENDERER_H_

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

namespace DX12Renderer {

//----------------------------------------------------------------------------
// DirectX 12 renderer.
// TODO
// * Move everything to a RendererImpl class.
// * Reimplement MiniEngine's CommandListManager and all dependencies.
// * Debug layer define.
// Future work:
// * Post processing.
// ** Gamma correction.
// ** SSAO
// ** Screen space reflections.
// ** Temporal antialising.
// * IBL
// * Shadows
// * Area lights.
// * Signed distance field traced shadows.
//----------------------------------------------------------------------------
class Renderer final {
public:
    static Renderer* initialize(HWND& hwnd, const Cogwheel::Core::Window& window);
    ~Renderer();

    void render();

private:

    Renderer(HWND& hwnd, const Cogwheel::Core::Window& window);
    
    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Renderer(Renderer& other) = delete;
    Renderer& operator=(const Renderer& rhs) = delete;

    void release_state();

    void wait_for_previous_frame();

    void handle_updates();

    // Pimpl the state to avoid exposing DirectX dependencies.
    struct State;
    State* m_state;
};

static inline void render_callback(const Cogwheel::Core::Engine& engine, void* renderer) {
    static_cast<Renderer*>(renderer)->render();
}

} // NS DX12Renderer

#endif // _DX12RENDERER_RENDERER_H_