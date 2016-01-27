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
#include <Math/Math.h>
#include <Scene/Camera.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <GL/gl.h> // TODO #ifdef to work on OS X as well. But do that when we create a special GL context.

using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;
using namespace optix;

namespace OptiXRenderer {

struct Renderer::State {
    uint2 screensize;
    Context context;

    Buffer accumulation_buffer;

    GLuint backbuffer_gl_id; // TODO Should also be used when we later support interop by rendering to a VBO/PBO.

    optix::GeometryGroup root_node;
};

static inline std::string get_ptx_path(std::string shader_filename) {
    return std::string(OPTIXRENDERER_PTX_DIR) + "/OptiXRenderer_generated_" + shader_filename + ".cu.ptx";
}

static inline void setup_debug_scene(optix::Context& context, optix::GeometryGroup& geoGroup) {
    optix::Geometry mesh = context->createGeometry();

    std::string intersection_ptx_path = get_ptx_path("IntersectSphere");
    mesh->setIntersectionProgram(context->createProgramFromPTXFile(intersection_ptx_path, "intersect"));
    mesh->setBoundingBoxProgram(context->createProgramFromPTXFile(intersection_ptx_path, "bounds"));
    mesh->setPrimitiveCount(1);

    optix::float4 sphere = optix::make_float4(0,0,0,1);
    mesh["sphere"]->setFloat(sphere);
    mesh->validate();

    optix::Material material = context->createMaterial();
    {
        std::string monte_carlo_ptx_path = get_ptx_path("MonteCarlo");
        material->setClosestHitProgram(int(RayTypes::MonteCarlo), context->createProgramFromPTXFile(monte_carlo_ptx_path, "closest_hit"));
        
        std::string normal_vis_ptx_path = get_ptx_path("NormalRendering");
        material->setClosestHitProgram(int(RayTypes::NormalVisualization), context->createProgramFromPTXFile(normal_vis_ptx_path, "closest_hit"));
        material->validate();
    }

    optix::GeometryInstance model = context->createGeometryInstance(mesh, &material, &material + 1);
    model->validate();

    geoGroup->addChild(model);
    geoGroup->validate();
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
    int2 compute_capability;
    context->getDeviceAttribute(m_device_ids.optix, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(compute_capability), &compute_capability);
    printf("OptiXRenderer using device %u: '%s' with compute capability %u.%u.\n", m_device_ids.optix, context->getDeviceName(m_device_ids.optix).c_str(), compute_capability.x, compute_capability.y);

    context->getDeviceAttribute(m_device_ids.optix, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(m_device_ids.cuda), &m_device_ids.cuda);

    context->setRayTypeCount(int(RayTypes::Count));
    context->setEntryPointCount(int(EntryPoints::Count));
    context->setStackSize(960);

    context["g_frame_number"]->setFloat(2.0f);

    { // Setup root node
        optix::Acceleration acceleration = context->createAcceleration("Bvh", "Bvh");
        acceleration->setProperty("refit", "1");

        m_state->root_node = context->createGeometryGroup();
        m_state->root_node->setAcceleration(acceleration);

        context["g_scene_root"]->set(m_state->root_node);

        setup_debug_scene(context, m_state->root_node);
    }

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
        std::string rgp_ptx_path = get_ptx_path("PathTracing");
        context->setRayGenerationProgram(int(EntryPoints::PathTracing), context->createProgramFromPTXFile(rgp_ptx_path, "path_tracing"));

        std::string monte_carlo_miss_ptx_path = get_ptx_path("MonteCarlo");
        context->setMissProgram(int(RayTypes::MonteCarlo), context->createProgramFromPTXFile(monte_carlo_miss_ptx_path, "miss"));
    }

    { // Normal visualization setup.
        std::string ptx_path = get_ptx_path("NormalRendering");
        context->setRayGenerationProgram(int(EntryPoints::NormalVisualization), context->createProgramFromPTXFile(ptx_path, "ray_generation"));
        context->setMissProgram(int(RayTypes::NormalVisualization), context->createProgramFromPTXFile(ptx_path, "miss"));
    }

    context->validate();
    context->compile();
}

void Renderer::apply() {
    Context& context = m_state->context;

    const Window& window = Engine::get_instance()->get_window();
    const uint2 current_screensize = make_uint2(window.get_width(), window.get_height());
    if (current_screensize != m_state->screensize) {
        // Screen buffers should be resized.
        m_state->accumulation_buffer->setSize(window.get_width(), window.get_height());
        m_state->screensize = make_uint2(window.get_width(), window.get_height());
    }

    context["g_frame_number"]->setFloat(float(Engine::get_instance()->get_time().get_ticks()));

    { // Upload camera parameters.
        Vector3f cam_pos = Vector3f::zero();
        Matrix4x4f inverse_view_projection_matrix;
        if (Cameras::begin() == Cameras::end()) {
            // Some default camera.
            Matrix4x4f view_projection_matrix; // Unused!
            CameraUtils::compute_perspective_projection(0.1f, 1000.0f, degrees_to_radians(60.0f), m_state->screensize.x / float(m_state->screensize.y),
                                                        view_projection_matrix, inverse_view_projection_matrix);
        } else {
            Cameras::UID camera_ID = *Cameras::begin();

            Matrix4x4f inverse_view_matrix = to_matrix4x4(Cameras::get_inverse_view_transform(camera_ID));
            Matrix4x4f inverse_projection_matrix = Cameras::get_inverse_projection_matrix(camera_ID);
            inverse_view_projection_matrix = inverse_view_matrix * inverse_projection_matrix;

            SceneNode camera_node = Cameras::get_parent_ID(camera_ID);
            cam_pos = camera_node.get_global_transform().translation;
        }

        context["g_inverted_view_projection_matrix"]->setMatrix4x4fv(false, inverse_view_projection_matrix.begin());
        float4 camera_position = make_float4(cam_pos.x, cam_pos.y, cam_pos.z, 0.0f);
        context["g_camera_position"]->setFloat(camera_position);
    }

    context->launch(int(EntryPoints::NormalVisualization), m_state->screensize.x, m_state->screensize.y);

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