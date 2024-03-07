// OptiX renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RENDERER_H_
#define _OPTIXRENDERER_RENDERER_H_

#include <OptiXRenderer/PublicTypes.h>

#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Utils/IdDeclarations.h>

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

namespace std::filesystem{ class path; }

namespace OptiXRenderer {

// ------------------------------------------------------------------------------------------------
// OptiX renderer for rendering with global illumination.
// Future work
// * Have path tracer stop per bounce, filter and display the result.
//   Should be good for interactivity and convergence. :)
// * Add a genesis event when rendering the very very very first frame. Can also 
//   be used if the renderstate 'falls behind' and it's faster to simply reinitialize.
// ------------------------------------------------------------------------------------------------
class Renderer final {
public:
    static Renderer* initialize(int cuda_device_ID, const std::filesystem::path& data_directory);
    ~Renderer();

    Bifrost::Core::RendererID get_renderer_ID() const { return m_renderer_ID; }

    float get_scene_epsilon(Bifrost::Scene::SceneRootID scene_root_ID) const;
    void set_scene_epsilon(Bifrost::Scene::SceneRootID scene_root_ID, float scene_epsilon);

    unsigned int get_max_bounce_count(Bifrost::Scene::CameraID camera_ID) const;
    void set_max_bounce_count(Bifrost::Scene::CameraID camera_ID, unsigned int bounce_count);

    unsigned int get_max_accumulation_count(Bifrost::Scene::CameraID camera_ID) const;
    void set_max_accumulation_count(Bifrost::Scene::CameraID camera_ID, unsigned int accumulation_count);

    Backend get_backend(Bifrost::Scene::CameraID camera_ID) const;
    void set_backend(Bifrost::Scene::CameraID camera_ID, Backend backend);

    PathRegularizationSettings get_path_regularization_settings() const;
    void set_path_regularization_settings(PathRegularizationSettings settings);

    AIDenoiserFlags get_AI_denoiser_flags() const;
    void set_AI_denoiser_flags(AIDenoiserFlags flags);

    void handle_updates();

    unsigned int render(Bifrost::Scene::CameraID camera_ID, optix::Buffer buffer, Bifrost::Math::Vector2i frame_size);

    std::vector<Bifrost::Scene::Screenshot> request_auxiliary_buffers(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::Cameras::ScreenshotContent content_requested, Bifrost::Math::Vector2i frame_size);

    optix::Context& get_context();

private:

    Renderer(int cuda_device_ID, const std::filesystem::path& data_directory);

    // Delete copy constructors to avoid having multiple versions of the same renderer.
    Renderer(Renderer& other) = delete;
    Renderer& operator=(const Renderer& rhs) = delete;

    Bifrost::Core::RendererID m_renderer_ID;

    // Pimpl the state to avoid exposing OptiX headers.
    struct Implementation;
    Implementation* m_impl;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RENDERER_H_