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
// * Is FindDirectX needed in VS 2015?
// * Draw model(s).
// * Abstract Mesh and Material. Possible? What is a material in DX12? PSO?
// ** Perhaps store the mesh as a set of buffers and their description.
// ** The material is a bunch of material parameters.
// ** The model combines a mesh, with material parameters and a configuration ID that matches the two. The configuration ID is used to connect to multiple PSO's. One for the Z-prepass, one for rendering, one for shadows and so on and so forth.
// * Actual material and integrator.
// * SRGB backbuffer format, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB.
// * Reimplement MiniEngine's CommandListManager and all dependencies.
// * Can I create an interface similar to CUDA's streams?
// ** I would need to reset whatever is associated with the stream each frame. 
// ** The main components would of course be a command list pr stream and fences, but what else needs to be reset pr frame?
// * Debug layer define.
// ** Compile shaders with debug flags and no optimizations. Load HLSL files instead of cso's.
// * Stochastic coverage.
// Future work:
// * Post processing.
// ** SSAO
// ** Screen space reflections.
// ** Temporal antialising.
// * IBL
// * Area lights.
// * Signed distance field traced shadows.
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

} // NS DX12Renderer

#endif // _DX12RENDERER_RENDERER_H_