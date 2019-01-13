// OptiX renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RENDERER_H_
#define _OPTIXRENDERER_RENDERER_H_

#include <Bifrost\Scene\Camera.h>
#include <Bifrost\Scene\SceneRoot.h>

// ------------------------------------------------------------------------------------------------
// Forward declarations.
// ------------------------------------------------------------------------------------------------
namespace optix {
template<class T> class Handle;
class BufferObj;
typedef Handle<BufferObj> Buffer;
class ContextObj;
typedef Handle<ContextObj> Context;
}

namespace OptiXRenderer {

// ------------------------------------------------------------------------------------------------
// Different rendering backends.
// ------------------------------------------------------------------------------------------------
enum class Backend {
    None,
    PathTracing,
    AlbedoVisualization,
    NormalVisualization,
};

// ------------------------------------------------------------------------------------------------
// OptiX renderer for rendering with global illumination.
// Future work
// * Have path tracer stop per bounce, filter and display the result.
//   Should be good for interactivity and convergence. :)
// * Clamp fireflies at vertices in a ray path. The current path PDF or 
//   ray differentials should be able to tell us if a connection with
//   a high intensity light source is a firefly.
// * Add a genesis event when rendering the very very very first frame. Can also 
//   be used if the renderstate 'falls behind' and it's faster to simply reinitialize.
// ------------------------------------------------------------------------------------------------
class Renderer final {
public:
    static Renderer* initialize(int cuda_device_ID, int width_hint, int height_hint, const std::string& data_folder_path, Bifrost::Core::Renderers::UID renderer_ID);

    float get_scene_epsilon(Bifrost::Scene::SceneRoots::UID scene_root_ID) const;
    void set_scene_epsilon(Bifrost::Scene::SceneRoots::UID scene_root_ID, float scene_epsilon);

    unsigned int get_max_bounce_count(Bifrost::Scene::Cameras::UID camera_ID) const;
    void set_max_bounce_count(Bifrost::Scene::Cameras::UID camera_ID, unsigned int bounce_count);

    unsigned int get_max_accumulation_count(Bifrost::Scene::Cameras::UID camera_ID) const;
    void set_max_accumulation_count(Bifrost::Scene::Cameras::UID camera_ID, unsigned int accumulation_count);

    Backend get_backend(Bifrost::Scene::Cameras::UID camera_ID) const;
    void set_backend(Bifrost::Scene::Cameras::UID camera_ID, Backend backend);

    void handle_updates();

    unsigned int render(Bifrost::Scene::Cameras::UID camera_ID, optix::Buffer buffer, int width, int height);

    optix::Context& get_context();

private:

    Renderer(int cuda_device_ID, int width_hint, int height_hint, const std::string& data_folder_path, Bifrost::Core::Renderers::UID renderer_ID);

    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Renderer(Renderer& other) = delete;
    Renderer& operator=(const Renderer& rhs) = delete;

    // Pimpl the state to avoid exposing OptiX headers.
    struct Implementation;
    Implementation* m_impl;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RENDERER_H_