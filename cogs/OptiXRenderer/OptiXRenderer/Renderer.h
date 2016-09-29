// OptiX renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RENDERER_H_
#define _OPTIXRENDERER_RENDERER_H_

#include <Cogwheel\Scene\SceneRoot.h>

//----------------------------------------------------------------------------
// Forward declarations.
//----------------------------------------------------------------------------
namespace Cogwheel {
namespace Core {
class Engine;
class Window;
}
}

namespace OptiXRenderer {

//----------------------------------------------------------------------------
// OptiX renderer.
// Supports several different rendering modes; Normal visualization and 
// path tracing. 
// Future work
// * Cache acceleration structures for meshes not in the scene, but still active.
//   Test if it speeds up the bullet creation of my boxgun!
// * Create Initialization method that only returns a valid pointer in case 
//   a Renderer could be created. Should take a GL (ES) Context and 
//   possibly a device ID begin/end iterator. But how to uniquely enumerate devices?
// * Several backbuffer modes: VBO, data transfer over the CPU, .. more? PBO?
//   See OptiX samples SampleScene.cpp.
// * ImageComposer that composes images pr camera from several renderers, 
//   probably requires an IRenderer interface.
// * Tone mapping and gamma correction as part of the image composer.
// * Logarithmic upload of the accumulated image.
// * Have path tracer stop per bounce, filter and display the result.
//   Should be good for interactivity and convergence. :)
// * Add a genesis event when rendering the very very very first frame. Can also 
//   be used if the renderer 'falls behind' and it's faster to simply reinitialize
//----------------------------------------------------------------------------
class Renderer final {
public:
    Renderer(const Cogwheel::Core::Window& window);

    inline bool could_initialize() const { return m_device_ids.optix >= 0;  }

    float get_scene_epsilon(Cogwheel::Scene::SceneRoots::UID scene_root_ID) const;
    void set_scene_epsilon(Cogwheel::Scene::SceneRoots::UID scene_root_ID, float scene_epsilon);

    void render();

private:

    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Renderer(Renderer& other) = delete;
    Renderer& operator=(const Renderer& rhs) = delete;

    void handle_updates();

    // Pimpl the state to avoid exposing OptiX headers.
    struct State;
    State* m_state;

    struct {
        int optix;
        int cuda;
    } m_device_ids;
};

static inline void render_callback(const Cogwheel::Core::Engine& engine, void* renderer) {
    static_cast<Renderer*>(renderer)->render();
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RENDERER_H_