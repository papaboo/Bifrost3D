// OptiX renderer backend interface.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_IBACKEND_H_
#define _OPTIXRENDERER_IBACKEND_H_

#include <Bifrost/Math/Vector.h>

#include <OptiXRenderer/PublicTypes.h>

#include <optixu/optixpp_namespace.h>

namespace OptiXRenderer {

// ------------------------------------------------------------------------------------------------
// OptiX renderer backend interface.
// ------------------------------------------------------------------------------------------------
class IBackend {
public:
    virtual ~IBackend() { };
    virtual void render(optix::Context& context, Bifrost::Math::Vector2i frame_size, int accumulation_count) = 0;
};

// ------------------------------------------------------------------------------------------------
// Simple OptiX renderer backend.
// Launches the ray generation program and outputs directly to the output buffer.
// ------------------------------------------------------------------------------------------------
class SimpleBackend : public IBackend {
public:
    SimpleBackend(int entry_point) : m_entry_point(entry_point) { }
    ~SimpleBackend() { }
    void render(optix::Context& context, Bifrost::Math::Vector2i frame_size, int accumulation_count) {
        context->launch(m_entry_point, frame_size.x, frame_size.y);
    }
private:
    int m_entry_point;
};

// ------------------------------------------------------------------------------------------------
// Path tracing filtered using OptiX' AI denoiser.
// ------------------------------------------------------------------------------------------------
class AIDenoisedBackend : public IBackend {
public:
    AIDenoisedBackend(optix::Context& context, AIDenoiserFlags* flags);
    ~AIDenoisedBackend() { }
    void render(optix::Context& context, Bifrost::Math::Vector2i frame_size, int accumulation_count);
private:
    void resize_backbuffers(Bifrost::Math::Vector2i frame_size);

    AIDenoiserFlags* m_flags; // Reference to renderer's denoise flags so we can check when they are changed.

    Bifrost::Math::Vector2i m_frame_size;

    optix::PostprocessingStage m_denoiser;
    optix::CommandList m_presenting_command_list;
    optix::CommandList m_not_presenting_command_list;

    optix::Buffer m_noisy_pixels; // float4 buffer
    optix::Buffer m_filtered_pixels; // float4 buffer
    optix::Buffer m_albedo; // float4 buffer
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_IBACKEND_H_