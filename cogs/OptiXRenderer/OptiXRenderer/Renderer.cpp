// OptiX renderer manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Renderer.h>

#include <OptiXRenderer/Kernel.h>
#include <OptiXRenderer/Types.h>

#include <Core/Engine.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <cuda.h>

using namespace optix;
using namespace Cogwheel::Core;

namespace OptiXRenderer {

static inline std::string get_ptx_path(std::string shader_filename) {
    return std::string(OPTIXRENDERER_PTX_DIR) + "/OptiXRenderer_generated_" + shader_filename + ".ptx";
}

struct Renderer::State {
    uint2 screensize;
    Context context;
};

Renderer::Renderer()
    : m_device_ids( {-1, -1} )
    , m_state(new State()) {
    
    if (Context::getDeviceCount() == 0)
        return;

    m_state->context = Context::create();
    Context& context = m_state->context;
        
    printf("Devices: %u\n", Context::getDeviceCount());

    m_device_ids.optix = 0;
    context->setDevices(&m_device_ids.optix, &m_device_ids.optix + 1);

    std::vector<int> devices = context->getEnabledDevices();
    context->getDeviceAttribute(devices[m_device_ids.optix], RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(m_device_ids.cuda), &m_device_ids.cuda);

    printf("OptiX ID: %u, CUDA ID: %u\n", m_device_ids.optix, m_device_ids.cuda);

    printf("%s\n", OPTIXRENDERER_PTX_DIR);

    context->setRayTypeCount(int(RayTypes::Count));
    context->setEntryPointCount(int(EntryPoints::Count));
    context->setStackSize(960);

    context["g_frame_number"]->setFloat(2.0f);

    // Screen buffers
    const Window& window = Engine::get_instance()->get_window();
    m_state->screensize = make_uint2(window.get_width(), window.get_height());
    Buffer accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, m_state->screensize.x, m_state->screensize.y);
    context["g_accumulation_buffer"]->set(accumulation_buffer);

    std::string ptxPath = get_ptx_path("PathTracing.cu");
    context->setRayGenerationProgram(int(EntryPoints::PathTracing),
        context->createProgramFromPTXFile(ptxPath, "PathTracing"));

    context->validate();
    context->compile();

    // launch();
}

void Renderer::apply() {
    Context& context = m_state->context;

    context["g_frame_number"]->setFloat(float(Engine::get_instance()->get_time().get_ticks()));

    Buffer accumulation_buffer = context["g_accumulation_buffer"]->getBuffer();

    context->launch(int(EntryPoints::PathTracing), m_state->screensize.x, m_state->screensize.y);

    float4* accumulation_buffer_mapped = (float4*)accumulation_buffer->map();
    printf("[%f, %f, %f, %f]\n", accumulation_buffer_mapped->x, accumulation_buffer_mapped->y, accumulation_buffer_mapped->z, accumulation_buffer_mapped->w);
    accumulation_buffer->unmap();
}

std::string Renderer::get_name() {
    return "OptiXRenderer";
}

} // NS OptiXRenderer