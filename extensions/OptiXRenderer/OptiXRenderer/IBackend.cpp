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

AIDenoisedBackend::AIDenoisedBackend(optix::Context& context, AIDenoiserFlags* flags, int width, int height)
    : m_flags(flags) {

    m_noisy_pixels = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4);
    m_filtered_pixels = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4);
    m_albedo = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4);
    m_normals = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4);

    m_denoiser = context->createBuiltinPostProcessingStage("DLDenoiser");
    unsigned int hdr_enabled = 1u;
    m_denoiser->declareVariable("hdr")->set1uiv(&hdr_enabled);
    m_denoiser->declareVariable("input_buffer")->set(m_noisy_pixels);
    m_denoiser->declareVariable("output_buffer")->set(m_filtered_pixels);
    m_denoiser->declareVariable("input_albedo_buffer")->set(m_albedo);
    m_denoiser->declareVariable("input_normal_buffer")->set(m_normals);

    // Initialize dummy command list to store the context. No rendering can be done until resize_backbuffers has been called.
    m_command_list = nullptr;

    resize_backbuffers(width, height);
}

void AIDenoisedBackend::resize_backbuffers(int width, int height) {
    auto context = m_denoiser->getContext();

    // Resize internal buffers
    m_noisy_pixels->setSize(width, height);
    m_filtered_pixels->setSize(width, height);
    m_albedo->setSize(width, height);
    m_normals->setSize(width, height);

    // Recreate command list
    m_command_list = context->createCommandList();
    m_command_list->appendLaunch(EntryPoints::AIDenoiserPathTracing, width, height);
    m_command_list->appendPostprocessingStage(m_denoiser, width, height);
    m_command_list->appendLaunch(EntryPoints::AIDenoiserCopyOutput, width, height);
    m_command_list->finalize();
}

void AIDenoisedBackend::render(optix::Context& context, int width, int height) {

    RTsize albedoSize;
    m_albedo->getSize(albedoSize);
    bool reset_albedo_accumulation = m_flags->is_set(AIDenoiserFlag::Albedo) && albedoSize == 0;
    if (reset_albedo_accumulation)
        m_albedo->setSize(width, height);
    else if (!m_flags->is_set(AIDenoiserFlag::Albedo) && albedoSize != 0)
        m_albedo->setSize(RTsize(0), RTsize(0));

    RTsize normalSize;
    m_normals->getSize(normalSize);
    // The normal buffer is only supported as input if albedo is used as well.
    bool use_normal_buffer = m_flags->all_set(AIDenoiserFlag::Albedo, AIDenoiserFlag::Normals);
    bool reset_normal_accumulation = use_normal_buffer && normalSize == 0;
    if (reset_normal_accumulation)
        m_normals->setSize(width, height);
    else if (!use_normal_buffer && normalSize != 0)
        m_normals->setSize(RTsize(0), RTsize(0));

    AIDenoiserStateGPU denoiser_state = {};
    denoiser_state.flags = m_flags->raw();
    if (!(denoiser_state.flags & int(AIDenoiserFlag::Albedo)))
        denoiser_state.flags &= ~int(AIDenoiserFlag::VisualizeAlbedo);
    if (!use_normal_buffer)
        denoiser_state.flags &= ~int(AIDenoiserFlag::VisualizeNormals);
    if (reset_albedo_accumulation)
        denoiser_state.flags |= AIDenoiserStateGPU::ResetAlbedoAccumulation;
    if (reset_normal_accumulation)
        denoiser_state.flags |= AIDenoiserStateGPU::ResetNormalAccumulation;
    denoiser_state.noisy_pixels_buffer = m_noisy_pixels->getId();
    denoiser_state.denoised_pixels_buffer = m_filtered_pixels->getId();
    denoiser_state.albedo_buffer = m_flags->is_set(AIDenoiserFlag::Albedo) ? m_albedo->getId() : 0;
    denoiser_state.normals_buffer = use_normal_buffer ? m_normals->getId() : 0;
    context["AIDenoiser::g_AI_denoiser_state"]->setUserData(sizeof(denoiser_state), &denoiser_state);

    m_command_list->execute();
}

} // NS OptiXRenderer