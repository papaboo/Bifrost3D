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

#include <GL/gl.h> // TODO #ifdef to work on OS X as well. But do that when we create a special GL context.

using namespace optix;
using namespace Cogwheel::Core;

namespace OptiXRenderer {

struct Renderer::State {
    uint2 screensize;
    Context context;

    Buffer accumulation_buffer;

    GLuint backbuffer_gl_id; // TODO Should also be used when we later support interop by rendering to a VBO/PBO.
};

static inline std::string get_ptx_path(std::string shader_filename) {
    return std::string(OPTIXRENDERER_PTX_DIR) + "/OptiXRenderer_generated_" + shader_filename + ".ptx";
}

Renderer::Renderer()
    : m_device_ids( {-1, -1} )
    , m_state(new State()) {
    
    if (Context::getDeviceCount() == 0)
        return;

    m_state->context = Context::create();
    Context& context = m_state->context;
        
    m_device_ids.optix = 0;
    context->setDevices(&m_device_ids.optix, &m_device_ids.optix + 1);

    std::vector<int> devices = context->getEnabledDevices();
    context->getDeviceAttribute(devices[m_device_ids.optix], RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(m_device_ids.cuda), &m_device_ids.cuda);

    printf("OptiX ID: %u, CUDA ID: %u\n", m_device_ids.optix, m_device_ids.cuda);

    context->setRayTypeCount(int(RayTypes::Count));
    context->setEntryPointCount(int(EntryPoints::Count));
    context->setStackSize(960);

    context["g_frame_number"]->setFloat(2.0f);

    { // Screen buffers
        const Window& window = Engine::get_instance()->get_window();
        m_state->screensize = make_uint2(window.get_width(), window.get_height());
        m_state->accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, m_state->screensize.x, m_state->screensize.y);
        context["g_accumulation_buffer"]->set(m_state->accumulation_buffer);

        { // Setup back buffer texture used for copying data to OpenGL
            glEnable(GL_TEXTURE_2D);
            glGenTextures(1, &m_state->backbuffer_gl_id);
            glBindTexture(GL_TEXTURE_2D, m_state->backbuffer_gl_id);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        }
    }

    { // Path tracing setup.
        std::string ptx_path = get_ptx_path("PathTracing.cu");
        context->setRayGenerationProgram(int(EntryPoints::PathTracing),
            context->createProgramFromPTXFile(ptx_path, "path_tracing"));
    }

    { // Normal visualization setup.
        std::string ptx_path = get_ptx_path("NormalRendering.cu");
        context->setRayGenerationProgram(int(EntryPoints::NormalVisualization),
            context->createProgramFromPTXFile(ptx_path, "normal_visualization"));
    }

    context->validate();
    context->compile();

    // launch();
}

void Renderer::apply() {
    Context& context = m_state->context;

    context["g_frame_number"]->setFloat(float(Engine::get_instance()->get_time().get_ticks()));

    context->launch(int(EntryPoints::PathTracing), m_state->screensize.x, m_state->screensize.y);

    { // Update the backbuffer.
        glViewport(0, 0, m_state->screensize.x, m_state->screensize.y);

        { // Setup matrices. I really don't need to do this every frame, since they never change.
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(-1, 1, -1.f, 1.f, 1.f, -1.f);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
        }

        float4* mapped_accumulation_buffer = (float4*)m_state->accumulation_buffer->map();
        glBindTexture(GL_TEXTURE_2D, m_state->backbuffer_gl_id);
        const GLint BASE_IMAGE_LEVEL = 0;
        const GLint noBorder = 0;
        glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGBA, m_state->screensize.x, m_state->screensize.y, noBorder, GL_RGBA, GL_FLOAT, mapped_accumulation_buffer);
        m_state->accumulation_buffer->unmap();

        glBegin(GL_QUADS); {

            glTexCoord2f(0.0f, 1.0f);
            glVertex3f(-1.0f, -1.0f, 0.f);

            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(1.0f, -1.0f, 0.f);

            glTexCoord2f(1.0f, 0.0f);
            glVertex3f(1.0f, 1.0f, 0.f);

            glTexCoord2f(0.0f, 0.0f);
            glVertex3f(-1.0f, 1.0f, 0.f);

        } glEnd();
    }
}

std::string Renderer::get_name() {
    return "OptiXRenderer";
}

} // NS OptiXRenderer