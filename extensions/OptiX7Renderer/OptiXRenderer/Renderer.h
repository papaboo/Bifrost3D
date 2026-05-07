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

#include <cuda.h>

// ------------------------------------------------------------------------------------------------
// Forward declarations.
// ------------------------------------------------------------------------------------------------
// namespace optix {
// template<class T> class Handle;
// class BufferObj;
// typedef Handle<BufferObj> Buffer;
// class ContextObj;
// typedef Handle<ContextObj> Context;
// }

struct cudaGraphicsResource;

namespace std::filesystem{ class path; }

namespace OptiXRenderer {

// ------------------------------------------------------------------------------------------------
// OptiX renderer for rendering with global illumination.
// ------------------------------------------------------------------------------------------------
class Renderer final {
public:
    static Renderer* initialize(CUcontext cuda_context, const std::filesystem::path& data_directory);
    ~Renderer();

    Bifrost::Core::RendererID get_renderer_ID() const { return m_renderer_ID; }

    Backend get_backend(Bifrost::Scene::CameraID camera_ID) const;
    void set_backend(Bifrost::Scene::CameraID camera_ID, Backend backend);

    unsigned int get_max_bounce_count(Bifrost::Scene::CameraID camera_ID) const;
    void set_max_bounce_count(Bifrost::Scene::CameraID camera_ID, unsigned int bounce_count);

    unsigned int get_max_accumulation_count(Bifrost::Scene::CameraID camera_ID) const;
    void set_max_accumulation_count(Bifrost::Scene::CameraID camera_ID, unsigned int accumulation_count);

    unsigned int get_next_event_sample_count(Bifrost::Scene::SceneRootID scene_root_ID) const;
    void set_next_event_sample_count(Bifrost::Scene::SceneRootID scene_root_ID, unsigned int sample_count);

    PathRegularizationSettings get_path_regularization_settings() const;
    void set_path_regularization_settings(PathRegularizationSettings settings);

    AIDenoiserFlags get_AI_denoiser_flags() const;
    void set_AI_denoiser_flags(AIDenoiserFlags flags);

    void handle_updates();

    unsigned int render(Bifrost::Scene::CameraID camera_ID, cudaGraphicsResource* backbuffer, Bifrost::Math::Vector2i frame_size);

    std::vector<Bifrost::Scene::Screenshot> request_auxiliary_buffers(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::Cameras::ScreenshotContent content_requested, Bifrost::Math::Vector2i frame_size);

private:

    Renderer(CUcontext cuda_context, const std::filesystem::path& data_directory);

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