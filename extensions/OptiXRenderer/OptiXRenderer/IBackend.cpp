// OptiX renderer backend interface.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/IBackend.h>
#include <OptiXRenderer/Types.h>

using namespace optix;

namespace OptiXRenderer {

AIFilteredBackend::AIFilteredBackend(optix::Context& context, int width, int height) {

    m_noisy_pixels = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4);
    m_filtered_pixels = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4);

    m_denoiser = context->createBuiltinPostProcessingStage("DLDenoiser");
    m_denoiser->declareVariable("input_buffer");
    m_denoiser->declareVariable("output_buffer");
    unsigned int hdr_enabled = 1u;
    m_denoiser->declareVariable("hdr")->set1uiv(&hdr_enabled);
    // TODO m_denoiser->declareVariable("input_albedo_buffer");
    // TODO m_denoiser->declareVariable("input_normal_buffer");

    // Initialize dummy command list to store the context. No rendering can be done until resize_backbuffers has been called.
    m_command_list = nullptr;

    resize_backbuffers(width, height);
}

void AIFilteredBackend::resize_backbuffers(int width, int height) {
    auto context = m_denoiser->getContext();
    
    // Resize internal buffers
    m_noisy_pixels->setSize(width, height);
    m_filtered_pixels->setSize(width, height);

    // Recreate command list
    m_command_list = context->createCommandList();
    m_command_list->appendLaunch(EntryPoints::AIDenoiserPathTracing, width, height);
    m_command_list->appendPostprocessingStage(m_denoiser, width, height);
    m_command_list->appendLaunch(EntryPoints::AIDenoiserCopyOutput, width, height);
    m_command_list->finalize();
}

void AIFilteredBackend::render(optix::Context& context, int width, int height) {

    AIDenoiserStateGPU denoiser_state;
    denoiser_state.noisy_pixels_buffer = m_noisy_pixels->getId();
    denoiser_state.denoised_pixels_buffer = m_filtered_pixels->getId();

    context["g_AI_denoiser_state"]->setUserData(sizeof(denoiser_state), &denoiser_state);
    m_denoiser["input_buffer"]->set(m_noisy_pixels);
    m_denoiser["output_buffer"]->set(m_filtered_pixels);
    m_command_list->execute();
}

} // NS OptiXRenderer