// OptiX renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RENDERER_H_
#define _OPTIXRENDERER_RENDERER_H_

#include <Cogwheel\Scene\Camera.h>
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

namespace optix {
template<class T> class Handle;
class BufferObj;
typedef Handle<BufferObj> Buffer;
class ContextObj;
typedef Handle<ContextObj> Context;
}

struct ID3D11Resource;

namespace OptiXRenderer {

//----------------------------------------------------------------------------
// OptiX renderer.
// Supports several different rendering modes; Normal visualization and 
// path tracing. 
// Future work
// * Cache acceleration structures for meshes not in the scene, but still active.
//   Test if it speeds up the bullet creation of my boxgun!
// * Tone mapping and gamma correction as part of the image composer.
// * Logarithmic upload of the accumulated image.
// * Have path tracer stop per bounce, filter and display the result.
//   Should be good for interactivity and convergence. :)
// * Add a genesis event when rendering the very very very first frame. Can also 
//   be used if the renderer 'falls behind' and it's faster to simply reinitialize
//----------------------------------------------------------------------------
class Renderer final {
public:
    static Renderer* initialize(int cuda_device_ID, int width_hint, int height_hint);

    inline bool is_valid() const { return m_device_IDs.optix >= 0;  }

    float get_scene_epsilon(Cogwheel::Scene::SceneRoots::UID scene_root_ID) const;
    void set_scene_epsilon(Cogwheel::Scene::SceneRoots::UID scene_root_ID, float scene_epsilon);

    void handle_updates();

    void render(Cogwheel::Scene::Cameras::UID camera_ID, optix::Buffer buffer, int width, int height);

    optix::Context& get_context();

private:

    Renderer(int cuda_device_ID, int width_hint, int height_hint);

    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Renderer(Renderer& other) = delete;
    Renderer& operator=(const Renderer& rhs) = delete;

    // Pimpl the state to avoid exposing OptiX headers.
    struct State;
    State* m_state;

    struct {
        int optix;
        int cuda;
    } m_device_IDs;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RENDERER_H_