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

namespace OptiXRenderer {

enum class Backend {
    None,
    PathTracing,
    AlbedoVisualization,
    NormalVisualization,
};

//----------------------------------------------------------------------------
// OptiX renderer.
// Supports several different rendering modes; Normal visualization and 
// path tracing. 
// Future work
// * Cache acceleration structures for meshes not in the scene, but still active.
//   Test if it speeds up the bullet creation of my boxgun!
// * Have path tracer stop per bounce, filter and display the result.
//   Should be good for interactivity and convergence. :)
// * Clamp fireflies at vertices in a ray path. The current path PDF or 
//   ray differentials should be able to tell us if a connection with
//   a high intensity light source is a firefly.
// * Add a genesis event when rendering the very very very first frame. Can also 
//   be used if the renderer 'falls behind' and it's faster to simply reinitialize
//----------------------------------------------------------------------------
class Renderer final {
public:
    static Renderer* initialize(int cuda_device_ID, int width_hint, int height_hint, const std::string& data_folder_path, Cogwheel::Core::Renderers::UID renderer_ID);

    float get_scene_epsilon(Cogwheel::Scene::SceneRoots::UID scene_root_ID) const;
    void set_scene_epsilon(Cogwheel::Scene::SceneRoots::UID scene_root_ID, float scene_epsilon);

    Backend get_backend(Cogwheel::Scene::Cameras::UID camera_ID) const;
    void set_backend(Cogwheel::Scene::Cameras::UID camera_ID, Backend backend);

    void handle_updates();

    unsigned int render(Cogwheel::Scene::Cameras::UID camera_ID, optix::Buffer buffer, int width, int height);

    optix::Context& get_context();

private:

    Renderer(int cuda_device_ID, int width_hint, int height_hint, const std::string& data_folder_path, Cogwheel::Core::Renderers::UID renderer_ID);

    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Renderer(Renderer& other) = delete;
    Renderer& operator=(const Renderer& rhs) = delete;

    // Pimpl the state to avoid exposing OptiX headers.
    struct Implementation;
    Implementation* m_impl;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RENDERER_H_