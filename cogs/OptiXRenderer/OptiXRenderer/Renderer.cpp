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

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Math/Math.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/SceneNode.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <GL/gl.h>

using namespace Cogwheel;
using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;
using namespace optix;

namespace OptiXRenderer {

struct Renderer::State {
    uint2 screensize;
    Context context;

    // Per camera members.
    Buffer accumulation_buffer;
    unsigned int accumulations;

    GLuint backbuffer_gl_id;

    optix::Group root_node;
};

static inline std::string get_ptx_path(std::string shader_filename) {
    return std::string(OPTIXRENDERER_PTX_DIR) + "/OptiXRenderer_generated_" + shader_filename + ".cu.ptx";
}

static inline optix::GeometryGroup setup_debug_scene(optix::Context& context) {
    optix::Acceleration acceleration = context->createAcceleration("Bvh", "Bvh");
    acceleration->setProperty("refit", "1");
    acceleration->validate();

    optix::GeometryGroup geoGroup = context->createGeometryGroup();
    geoGroup->setAcceleration(acceleration);

    optix::Material material = context->createMaterial();
    {
        std::string monte_carlo_ptx_path = get_ptx_path("MonteCarlo");
        material->setClosestHitProgram(int(RayTypes::MonteCarlo), context->createProgramFromPTXFile(monte_carlo_ptx_path, "closest_hit"));

        std::string normal_vis_ptx_path = get_ptx_path("NormalRendering");
        material->setClosestHitProgram(int(RayTypes::NormalVisualization), context->createProgramFromPTXFile(normal_vis_ptx_path, "closest_hit"));
        material->validate();
    }

    std::string intersection_ptx_path = get_ptx_path("IntersectSphere");
    optix::Program sphere_intersection_program = context->createProgramFromPTXFile(intersection_ptx_path, "intersect");
    optix::Program sphere_bounds_program = context->createProgramFromPTXFile(intersection_ptx_path, "bounds");

    { // Center
        optix::Geometry mesh = context->createGeometry();
        mesh->setIntersectionProgram(sphere_intersection_program);
        mesh->setBoundingBoxProgram(sphere_bounds_program);
        mesh->setPrimitiveCount(1);

        mesh["sphere"]->setFloat(make_float4(0, 0, 0, 0.5f));

        optix::GeometryInstance model = context->createGeometryInstance(mesh, &material, &material + 1);
        model["g_color"]->setFloat(make_float3(1, 1, 1));
        geoGroup->addChild(model);
    }

    { // Up
        optix::Geometry mesh = context->createGeometry();
        mesh->setIntersectionProgram(sphere_intersection_program);
        mesh->setBoundingBoxProgram(sphere_bounds_program);
        mesh->setPrimitiveCount(1);

        mesh["sphere"]->setFloat(make_float4(0, 2, 0, 0.5f));

        optix::GeometryInstance model = context->createGeometryInstance(mesh, &material, &material + 1);
        model["g_color"]->setFloat(make_float3(0, 1, 0));
        geoGroup->addChild(model);
    }

    { // Down
        optix::Geometry mesh = context->createGeometry();
        mesh->setIntersectionProgram(sphere_intersection_program);
        mesh->setBoundingBoxProgram(sphere_bounds_program);
        mesh->setPrimitiveCount(1);

        mesh["sphere"]->setFloat(make_float4(0, -2, 0, 0.5f));

        optix::GeometryInstance model = context->createGeometryInstance(mesh, &material, &material + 1);
        model["g_color"]->setFloat(make_float3(1, 0, 1));
        geoGroup->addChild(model);
    }

    { // Right
        optix::Geometry mesh = context->createGeometry();
        mesh->setIntersectionProgram(sphere_intersection_program);
        mesh->setBoundingBoxProgram(sphere_bounds_program);
        mesh->setPrimitiveCount(1);

        mesh["sphere"]->setFloat(make_float4(2, 0, 0, 0.5f));

        optix::GeometryInstance model = context->createGeometryInstance(mesh, &material, &material + 1);
        model["g_color"]->setFloat(make_float3(1, 0, 0));
        geoGroup->addChild(model);
    }

    { // Left
        optix::Geometry mesh = context->createGeometry();
        mesh->setIntersectionProgram(sphere_intersection_program);
        mesh->setBoundingBoxProgram(sphere_bounds_program);
        mesh->setPrimitiveCount(1);

        mesh["sphere"]->setFloat(make_float4(-2, 0, 0, 0.5f));

        optix::GeometryInstance model = context->createGeometryInstance(mesh, &material, &material + 1);
        model["g_color"]->setFloat(make_float3(0, 1, 1));
        geoGroup->addChild(model);
    }

    { // Forward
        optix::Geometry mesh = context->createGeometry();
        mesh->setIntersectionProgram(sphere_intersection_program);
        mesh->setBoundingBoxProgram(sphere_bounds_program);
        mesh->setPrimitiveCount(1);

        mesh["sphere"]->setFloat(make_float4(0, 0, 2, 0.5f));

        optix::GeometryInstance model = context->createGeometryInstance(mesh, &material, &material + 1);
        model["g_color"]->setFloat(make_float3(0, 0, 1));
        geoGroup->addChild(model);
    }

    { // Backwards
        optix::Geometry mesh = context->createGeometry();
        mesh->setIntersectionProgram(sphere_intersection_program);
        mesh->setBoundingBoxProgram(sphere_bounds_program);
        mesh->setPrimitiveCount(1);

        mesh["sphere"]->setFloat(make_float4(0, 0, -2, 0.5f));

        optix::GeometryInstance model = context->createGeometryInstance(mesh, &material, &material + 1);
        model["g_color"]->setFloat(make_float3(1, 1, 0));
        geoGroup->addChild(model);
    }

    geoGroup->validate();

    return geoGroup;
}

//----------------------------------------------------------------------------
// Model loading.
//----------------------------------------------------------------------------

static inline size_t size_of(RTformat format) {
    switch (format) {
    case RT_FORMAT_FLOAT:
        return sizeof(float);
    case RT_FORMAT_FLOAT2:
        return sizeof(float2);
    case RT_FORMAT_FLOAT3:
        return sizeof(float3);
    case RT_FORMAT_FLOAT4:
        return sizeof(float4);
    case RT_FORMAT_INT:
        return sizeof(int);
    case RT_FORMAT_INT2:
        return sizeof(int2);
    case RT_FORMAT_INT3:
        return sizeof(int3);
    case RT_FORMAT_INT4:
        return sizeof(int4);
    case RT_FORMAT_UNSIGNED_INT:
        return sizeof(unsigned int);
    case RT_FORMAT_UNSIGNED_INT2:
        return sizeof(uint2);
    case RT_FORMAT_UNSIGNED_INT3:
        return sizeof(uint3);
    case RT_FORMAT_UNSIGNED_INT4:
        return sizeof(uint4);
    default:
        printf("ERROR: OptiXRenderer::Renderer::size_of does not support format: %u\n", (unsigned int)format);
        return 0;
    }
}

static inline optix::Buffer create_buffer(optix::Context& context, unsigned int buffer_type, RTformat format, RTsize element_count, void* data) {
    optix::Buffer buffer = context->createBuffer(buffer_type, format, element_count);
    memcpy(buffer->map(), data, element_count * size_of(format));
    buffer->unmap();
    return buffer;
}

static inline optix::Transform load_model(optix::Context& context, MeshModel model) {
    // TODO Check if we gain any loading performance by caching intersection and closest hit programs.

    optix::Geometry optixMesh = context->createGeometry();
    {
        // TODO Handle nulled index buffers. Can I check if a buffer is null on the GPU?
        Mesh& mesh = Meshes::get_mesh(model.m_mesh_ID);

        std::string intersection_ptx_path = get_ptx_path("IntersectTriangle");
        optixMesh->setIntersectionProgram(context->createProgramFromPTXFile(intersection_ptx_path, "intersect"));
        optixMesh->setBoundingBoxProgram(context->createProgramFromPTXFile(intersection_ptx_path, "bounds"));

        optix::Buffer index_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, mesh.m_indices_count, mesh.m_indices);
        optixMesh["index_buffer"]->setBuffer(index_buffer);
        optixMesh->setPrimitiveCount(mesh.m_indices_count);

        // Vertex attributes
        optix::Buffer position_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh.m_vertex_count, mesh.m_positions);
        optixMesh["position_buffer"]->setBuffer(position_buffer);
        optix::Buffer normal_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh.m_vertex_count, mesh.m_positions);
        optixMesh["normal_buffer"]->setBuffer(normal_buffer);
        optix::Buffer texcoord_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh.m_vertex_count, mesh.m_positions);
        optixMesh["texcoord_buffer"]->setBuffer(texcoord_buffer);
        optixMesh->validate(); // TODO debug validate macro.
    }

    optix::Material optixMaterial = context->createMaterial();
    {
        std::string monte_carlo_ptx_path = get_ptx_path("MonteCarlo");
        optixMaterial->setClosestHitProgram(int(RayTypes::MonteCarlo), context->createProgramFromPTXFile(monte_carlo_ptx_path, "closest_hit"));

        std::string normal_vis_ptx_path = get_ptx_path("NormalRendering");
        optixMaterial->setClosestHitProgram(int(RayTypes::NormalVisualization), context->createProgramFromPTXFile(normal_vis_ptx_path, "closest_hit"));
        optixMaterial->validate();
    }

    optix::GeometryInstance optixModel = context->createGeometryInstance(optixMesh, &optixMaterial, &optixMaterial + 1);
    optixModel["g_color"]->setFloat(make_float3(0.5f, 0.5f, 0.5f));
    optixModel->validate();

    optix::Acceleration acceleration = context->createAcceleration("Bvh", "Bvh");
    acceleration->setProperty("index_buffer_name", "index_buffer");
    acceleration->setProperty("vertex_buffer_name", "position_buffer");
    acceleration->markDirty();
    acceleration->validate();

    optix::GeometryGroup geometryGroup = context->createGeometryGroup(&optixModel, &optixModel + 1);
    geometryGroup->setAcceleration(acceleration);
    geometryGroup->validate();

    optix::Transform optixTransform = context->createTransform();
    {
        Math::Transform transform = SceneNodes::get_global_transform(model.m_scene_node_ID);
        Math::Transform inverse_transform = invert(transform);
        optixTransform->setMatrix(false, to_matrix4x4(transform).begin(), to_matrix4x4(inverse_transform).begin());
        optixTransform->setChild(geometryGroup);
        optixTransform->validate();
    }

    return optixTransform;
}

//----------------------------------------------------------------------------
// Renderer implementation.
//----------------------------------------------------------------------------

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

    m_state->accumulations = 0u;

    context["g_frame_number"]->setFloat(2.0f);

    { // Setup root node
        optix::Acceleration root_acceleration = context->createAcceleration("Bvh", "Bvh");
        root_acceleration->setProperty("refit", "1");

        m_state->root_node = context->createGroup();
        m_state->root_node->setAcceleration(root_acceleration);
        m_state->root_node->validate();

        context["g_scene_root"]->set(m_state->root_node);

        // optix::Transform transform = context->createTransform();
        // transform->setChild(setup_debug_scene(context));
        // transform->validate();
        // m_state->root_node->addChild(transform);

        for (MeshModels::ConstUIDIterator model_itr = MeshModels::begin();
            model_itr != MeshModels::end(); ++model_itr) {
            MeshModel model = MeshModels::get_model(*model_itr);
            m_state->root_node->addChild(load_model(context, model));
        }
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

    // context->setPrintEnabled(true);
    // context->setPrintLaunchIndex(0, 0);
    // context->setExceptionEnabled(RT_EXCEPTION_ALL, true);

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

    context["g_accumulations"]->setFloat(float(m_state->accumulations));

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

    context->launch(int(EntryPoints::PathTracing), m_state->screensize.x, m_state->screensize.y);

    m_state->accumulations += 1u;

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
        const GLint NO_BORDER = 0;
        glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGBA, m_state->screensize.x, m_state->screensize.y, NO_BORDER, GL_RGBA, GL_FLOAT, mapped_accumulation_buffer);
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

} // NS OptiXRenderer