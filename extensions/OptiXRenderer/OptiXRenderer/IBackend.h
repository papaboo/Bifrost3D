// OptiX renderer backend interface.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_IBACKEND_H_
#define _OPTIXRENDERER_IBACKEND_H_

#include <OptiXRenderer/PublicTypes.h>

#include <optixu/optixpp_namespace.h>

namespace OptiXRenderer {

// ------------------------------------------------------------------------------------------------
// OptiX renderer backend interface.
// ------------------------------------------------------------------------------------------------
class IBackend {
public:
    virtual ~IBackend() { };
    virtual void resize_backbuffers(int width, int height) = 0;
    virtual void render(optix::Context& context, int width, int height) = 0;
};

// ------------------------------------------------------------------------------------------------
// Simple OptiX renderer backend.
// Launches the ray generation program and outputs directly to the output buffer.
// ------------------------------------------------------------------------------------------------
class SimpleBackend : public IBackend {
public:
    SimpleBackend(int entry_point) : m_entry_point(entry_point) { }
    ~SimpleBackend() { }
    void resize_backbuffers(int width, int height) {}
    void render(optix::Context& context, int width, int height) {
        context->launch(m_entry_point, width, height);
    }
private:
    int m_entry_point;
};

// ------------------------------------------------------------------------------------------------
// Path tracing filtered using OptiX' AI denoiser.
// ------------------------------------------------------------------------------------------------
class AIDenoisedBackend : public IBackend {
public:
    AIDenoisedBackend(optix::Context& context, AIDenoiserFlags* flags, int width, int height);
    ~AIDenoisedBackend() { }
    void resize_backbuffers(int width, int height);
    void render(optix::Context& context, int width, int height);
private:
    AIDenoiserFlags* m_flags; // Reference to renderer's denoise flags so we can check when they are changed.
    optix::CommandList m_command_list;
    optix::PostprocessingStage m_denoiser;
    optix::Buffer m_noisy_pixels; // float4 buffer
    optix::Buffer m_filtered_pixels; // float4 buffer
    optix::Buffer m_albedo; // float4 buffer
    optix::Buffer m_normals; // float4 buffer
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_IBACKEND_H_