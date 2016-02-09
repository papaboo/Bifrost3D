// OptiX renderer manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RENDERER_H_
#define _OPTIXRENDERER_RENDERER_H_

//----------------------------------------------------------------------------
// Forward declarations.
//----------------------------------------------------------------------------
namespace Cogwheel {
namespace Core {
class Engine;
}
}

namespace OptiXRenderer {

//----------------------------------------------------------------------------
// OptiX renderer.
// Supports several different rendering modes; Normal visualization and 
// path tracing. 
// Future work
// * Create Initialization method that only returns a valid pointer in case 
//   a Renderer could be created. Should take a GL (ES) Context and 
//   a device ID begin/end iterator.
// * Several backbuffer modes: VBO, data transfer over the CPU, .. more? PBO?
//   See OptiX samples SampleScene.cpp.
// * ImageComposer that composes images pr camera from several renderers, 
//   probably requires an IRenderer interface.
// * Tone mapping and gamma correction as part of the image composer.
// * Define for switching between float and double backbuffer.
//   Float should generally be used to save memory, 
//   but double is useful for ground truth images.
// * Logarithmic upload of the accumulated image.
// * Have path tracer stop per bounce, filter and display the result.
//   Should be good for interactivity and convergence. :)
//----------------------------------------------------------------------------
class Renderer final {
public:
    Renderer();

    inline bool could_initialize() const { return m_device_ids.optix>= 0;  }

    void render();

private:

    struct {
        int optix;
        int cuda;
    } m_device_ids;

    // Pimpl the state to avoid exposing OptiX headers.
    struct State;
    State* m_state;
};

static inline void render_callback(const Cogwheel::Core::Engine& engine, void* renderer) {
    static_cast<Renderer*>(renderer)->render();
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RENDERER_H_