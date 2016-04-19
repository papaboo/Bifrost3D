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
#include <Cogwheel/Core/Array.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Math/Math.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/LightSource.h>
#include <Cogwheel/Scene/SceneNode.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <GL/gl.h>

#include <assert.h>
#include <vector>

using namespace Cogwheel;
using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;
using namespace optix;

// Validate macro. Will validate the optix object in debug mode.
#ifdef _DEBUG
#define OPTIX_VALIDATE(o) o->validate()
#else
#define OPTIX_VALIDATE(o)
#endif

namespace OptiXRenderer {

struct Renderer::State {
    uint2 screensize;
    optix::Context context;

    // Per camera members.
    optix::Buffer accumulation_buffer;
    optix::Buffer output_buffer;
    unsigned int accumulations;
    Math::Transform camera_transform;

    GLuint backbuffer_gl_id;

    optix::Group root_node;

    std::vector<optix::Transform> transforms = std::vector<optix::Transform>(0);
    std::vector<optix::Geometry> meshes = std::vector<optix::Geometry>(0);

    optix::Material default_material;
    optix::Buffer material_parameters;
    unsigned int material_parameter_count;

    optix::Program triangle_intersection_program;
    optix::Program triangle_bounds_program;

    struct {
        Core::Array<unsigned int> ID_to_index;
        Core::Array<LightSources::UID> index_to_ID;
        optix::Buffer sources;
        unsigned int count;

        optix::GeometryGroup area_lights;
        optix::Geometry area_lights_geometry;
    } lights;
};

static inline std::string get_ptx_path(std::string shader_filename) {
    return std::string(OPTIXRENDERER_PTX_DIR) + "/OptiXRenderer_generated_" + shader_filename + ".cu.ptx";
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

static inline optix::Geometry load_mesh(optix::Context& context, Meshes::UID mesh_ID, 
                                        optix::Program intersection_program, optix::Program bounds_program) {
    optix::Geometry optixMesh = context->createGeometry();
    
    const Mesh& mesh = Meshes::get_mesh(mesh_ID);

    optixMesh->setIntersectionProgram(intersection_program);
    optixMesh->setBoundingBoxProgram(bounds_program);

    optix::Buffer index_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, mesh.indices_count, mesh.indices);
    optixMesh["index_buffer"]->setBuffer(index_buffer);
    optixMesh->setPrimitiveCount(mesh.indices_count);

    // Vertex attributes
    optix::Buffer position_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh.vertex_count, mesh.positions);
    optixMesh["position_buffer"]->setBuffer(position_buffer);
    optix::Buffer normal_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh.vertex_count, mesh.normals);
    optixMesh["normal_buffer"]->setBuffer(normal_buffer);
    optix::Buffer texcoord_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, mesh.vertex_count, mesh.texcoords);
    optixMesh["texcoord_buffer"]->setBuffer(texcoord_buffer);
    OPTIX_VALIDATE(optixMesh);

    return optixMesh;
}

static inline optix::Transform load_model(optix::Context& context, MeshModel model, optix::Geometry* meshes, optix::Material optix_material) {
    optix::Geometry optix_mesh = meshes[model.mesh_ID];

    assert(optix_mesh);

    optix::GeometryInstance optix_model = context->createGeometryInstance(optix_mesh, &optix_material, &optix_material + 1);
    optix_model["material_index"]->setInt(model.material_ID.get_ID());
    unsigned char mesh_flags = Meshes::get_normals(model.mesh_ID) != nullptr ? MeshFlags::Normals : MeshFlags::None;
    mesh_flags |= Meshes::get_texcoords(model.mesh_ID) != nullptr ? MeshFlags::Texcoords : MeshFlags::None;
    optix_model["mesh_flags"]->setInt(mesh_flags);
    OPTIX_VALIDATE(optix_model);

    optix::Acceleration acceleration = context->createAcceleration("Bvh", "Bvh");
    acceleration->setProperty("index_buffer_name", "index_buffer");
    acceleration->setProperty("vertex_buffer_name", "position_buffer");
    OPTIX_VALIDATE(acceleration);

    optix::GeometryGroup geometry_group = context->createGeometryGroup(&optix_model, &optix_model + 1);
    geometry_group->setAcceleration(acceleration);
    OPTIX_VALIDATE(geometry_group);

    optix::Transform optix_transform = context->createTransform();
    {
        Math::Transform transform = SceneNodes::get_global_transform(model.scene_node_ID);
        Math::Transform inverse_transform = invert(transform);
        optix_transform->setMatrix(false, to_matrix4x4(transform).begin(), to_matrix4x4(inverse_transform).begin());
        optix_transform->setChild(geometry_group);
        OPTIX_VALIDATE(optix_transform);
    }

    return optix_transform;
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
    context->setStackSize(1280);

    m_state->accumulations = 0u;

    context["g_frame_number"]->setFloat(0.0f);

    { // Setup root node
        optix::Acceleration root_acceleration = context->createAcceleration("Bvh", "Bvh");
        root_acceleration->setProperty("refit", "1");

        m_state->root_node = context->createGroup();
        m_state->root_node->setAcceleration(root_acceleration);
        OPTIX_VALIDATE(m_state->root_node);

        context["g_scene_root"]->set(m_state->root_node);
        context["g_scene_epsilon"]->setFloat(0.0001f); // TODO, base on scene size. Can I query the scene bounds from OptiX?
        context["g_max_bounce_count"]->setInt(4);
    }

    { // Light sources
        m_state->lights.sources = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 0);
        m_state->lights.sources->setElementSize(sizeof(SphereLight));
        m_state->lights.count = 0;
        context["g_lights"]->set(m_state->lights.sources);
        context["g_light_count"]->setInt(m_state->lights.count);
        
        // Analytical area light geometry.
        m_state->lights.area_lights_geometry = context->createGeometry();
        std::string light_intersection_ptx_path = get_ptx_path("LightSources");
        m_state->lights.area_lights_geometry->setIntersectionProgram(context->createProgramFromPTXFile(light_intersection_ptx_path, "intersect"));
        m_state->lights.area_lights_geometry->setBoundingBoxProgram(context->createProgramFromPTXFile(light_intersection_ptx_path, "bounds"));
        m_state->lights.area_lights_geometry->setPrimitiveCount(0u);
        OPTIX_VALIDATE(m_state->lights.area_lights_geometry);

        // Analytical area light material.
        optix::Material material = context->createMaterial();
        std::string monte_carlo_ptx_path = get_ptx_path("MonteCarlo");
        material->setClosestHitProgram(int(RayTypes::MonteCarlo), context->createProgramFromPTXFile(monte_carlo_ptx_path, "light_closest_hit"));
        std::string normal_vis_ptx_path = get_ptx_path("NormalRendering");
        material->setClosestHitProgram(int(RayTypes::NormalVisualization), context->createProgramFromPTXFile(normal_vis_ptx_path, "closest_hit"));
        OPTIX_VALIDATE(material);

        optix::Acceleration acceleration = context->createAcceleration("Bvh", "Bvh");
        OPTIX_VALIDATE(acceleration);

        optix::GeometryInstance area_lights = context->createGeometryInstance(m_state->lights.area_lights_geometry, &material, &material + 1);
        OPTIX_VALIDATE(area_lights);

        m_state->lights.area_lights = context->createGeometryGroup(&area_lights, &area_lights + 1);
        m_state->lights.area_lights->setAcceleration(acceleration);
        OPTIX_VALIDATE(m_state->lights.area_lights);

        m_state->root_node->addChild(m_state->lights.area_lights);
    }

    { // Setup default material.
        m_state->default_material = context->createMaterial();

        std::string monte_carlo_ptx_path = get_ptx_path("MonteCarlo");
        m_state->default_material->setClosestHitProgram(int(RayTypes::MonteCarlo), context->createProgramFromPTXFile(monte_carlo_ptx_path, "closest_hit"));
        m_state->default_material->setAnyHitProgram(int(RayTypes::Shadow), context->createProgramFromPTXFile(monte_carlo_ptx_path, "shadow_any_hit"));

        std::string normal_vis_ptx_path = get_ptx_path("NormalRendering");
        m_state->default_material->setClosestHitProgram(int(RayTypes::NormalVisualization), context->createProgramFromPTXFile(normal_vis_ptx_path, "closest_hit"));

        OPTIX_VALIDATE(m_state->default_material);

        std::string trangle_intersection_ptx_path = get_ptx_path("IntersectTriangle");
        m_state->triangle_intersection_program = context->createProgramFromPTXFile(trangle_intersection_ptx_path, "intersect");
        m_state->triangle_bounds_program = context->createProgramFromPTXFile(trangle_intersection_ptx_path, "bounds");

        m_state->material_parameter_count = 0;
        m_state->material_parameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_state->material_parameter_count);
        m_state->material_parameters->setElementSize(sizeof(OptiXRenderer::Material));
        context["g_materials"]->set(m_state->material_parameters);
    }

    { // Screen buffers
        const Window& window = Engine::get_instance()->get_window();
        m_state->screensize = make_uint2(window.get_width(), window.get_height());
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
        m_state->accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, m_state->screensize.x, m_state->screensize.y);
        m_state->accumulation_buffer->setElementSize(sizeof(double) * 4);
#else
        m_state->accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, m_state->screensize.x, m_state->screensize.y);
#endif
        context["g_accumulation_buffer"]->set(m_state->accumulation_buffer);

        m_state->output_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, m_state->screensize.x, m_state->screensize.y);
        context["g_output_buffer"]->set(m_state->output_buffer);

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

        m_state->camera_transform = Math::Transform::identity();
    }

    { // Path tracing setup.
        std::string rgp_ptx_path = get_ptx_path("PathTracing");
        context->setRayGenerationProgram(int(EntryPoints::PathTracing), context->createProgramFromPTXFile(rgp_ptx_path, "path_tracing"));
        context->setExceptionProgram(int(EntryPoints::PathTracing), context->createProgramFromPTXFile(rgp_ptx_path, "exceptions"));
        context->setMissProgram(int(RayTypes::MonteCarlo), context->createProgramFromPTXFile(rgp_ptx_path, "miss"));
    }

    { // Normal visualization setup.
        std::string ptx_path = get_ptx_path("NormalRendering");
        context->setRayGenerationProgram(int(EntryPoints::NormalVisualization), context->createProgramFromPTXFile(ptx_path, "ray_generation"));
        context->setMissProgram(int(RayTypes::NormalVisualization), context->createProgramFromPTXFile(ptx_path, "miss"));
    }

#ifdef _DEBUG
    context->setPrintEnabled(true);
    context->setPrintLaunchIndex(0, 0);
    context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
#endif

    OPTIX_VALIDATE(context);
    context->compile();
}

void Renderer::render() {

    if (Cameras::begin() == Cameras::end())
        return;

    handle_updates();

    Context& context = m_state->context;

    const Window& window = Engine::get_instance()->get_window();
    const uint2 current_screensize = make_uint2(window.get_width(), window.get_height());
    if (current_screensize != m_state->screensize) {
        // Screen buffers should be resized.
        m_state->accumulation_buffer->setSize(window.get_width(), window.get_height());
        m_state->output_buffer->setSize(window.get_width(), window.get_height());
        m_state->screensize = make_uint2(window.get_width(), window.get_height());
        m_state->accumulations = 0u;
#ifdef _DEBUG
        // context->setPrintLaunchIndex(window.get_width() / 2, window.get_height() / 2);
#endif
    }

    { // Upload camera parameters.
        Cameras::UID camera_ID = *Cameras::begin();

        Matrix4x4f inverse_view_matrix = to_matrix4x4(Cameras::get_inverse_view_transform(camera_ID));
        Matrix4x4f inverse_projection_matrix = Cameras::get_inverse_projection_matrix(camera_ID);
        Matrix4x4f inverse_view_projection_matrix = inverse_view_matrix * inverse_projection_matrix;

        SceneNode camera_node = Cameras::get_node_ID(camera_ID);
        Vector3f cam_pos = camera_node.get_global_transform().translation;

        context["g_inverted_view_projection_matrix"]->setMatrix4x4fv(false, inverse_view_projection_matrix.begin());
        float4 camera_position = make_float4(cam_pos.x, cam_pos.y, cam_pos.z, 0.0f);
        context["g_camera_position"]->setFloat(camera_position);

        // Reset accumulations if camera transform has changed.
        SceneNodes::UID camera_node_ID = Cameras::get_node_ID(camera_ID);
        if (m_state->camera_transform != SceneNodes::get_global_transform(camera_node_ID)) {
            m_state->camera_transform = SceneNodes::get_global_transform(camera_node_ID);
            m_state->accumulations = 0u;
        }
    }

    context["g_accumulations"]->setInt(m_state->accumulations);

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

        float4* mapped_output_buffer = (float4*)m_state->output_buffer->map();
        glBindTexture(GL_TEXTURE_2D, m_state->backbuffer_gl_id);
        const GLint BASE_IMAGE_LEVEL = 0;
        const GLint NO_BORDER = 0;
        glTexImage2D(GL_TEXTURE_2D, BASE_IMAGE_LEVEL, GL_RGBA, m_state->screensize.x, m_state->screensize.y, NO_BORDER, GL_RGBA, GL_FLOAT, mapped_output_buffer);
        m_state->output_buffer->unmap();

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

void Renderer::handle_updates() {
    { // Mesh updates.
        for (Meshes::UID mesh_ID : Meshes::get_destroyed_meshes()) {
            m_state->meshes[mesh_ID]->destroy();
            m_state->meshes[mesh_ID] = NULL;
        }

        for (Meshes::UID mesh_ID : Meshes::get_created_meshes()) {
            if (m_state->meshes.size() <= mesh_ID)
                m_state->meshes.resize(Meshes::capacity());
            m_state->meshes[mesh_ID] = load_mesh(m_state->context, mesh_ID, m_state->triangle_intersection_program, m_state->triangle_bounds_program);
        }
    }

    { // Material updates.
        static auto upload_material = [](Materials::UID material_ID, OptiXRenderer::Material* device_materials) {
            OptiXRenderer::Material& device_material = device_materials[material_ID];
            Assets::Material host_material = material_ID;
            device_material.base_tint.x = host_material.get_base_tint().r;
            device_material.base_tint.y = host_material.get_base_tint().g;
            device_material.base_tint.z = host_material.get_base_tint().b;
            device_material.base_roughness = host_material.get_base_roughness();
            device_material.specularity = host_material.get_specularity() * 0.08f; // See Physically-Based Shading at Disney bottom of page 8 for why we remap. TODO Consider moving this into Cogwheel or maybe even remove completely in favor of just letting GUI handle this.
            device_material.metallic = host_material.get_metallic();
        };

        if (!Materials::get_changed_materials().is_empty()) {
            if (m_state->material_parameter_count < Materials::capacity()) {
                // Buffer size changed. Re-upload all parameters.
                m_state->material_parameter_count = Materials::capacity();
                m_state->material_parameters->setSize(m_state->material_parameter_count);

                OptiXRenderer::Material* device_materials = (OptiXRenderer::Material*)m_state->material_parameters->map();
                upload_material(Materials::UID::invalid_UID(), device_materials); // Upload invalid material params as well.
                for (Materials::UID material_ID : Materials::get_iterable())
                    upload_material(material_ID, device_materials);
                m_state->material_parameters->unmap();
            } else {
                // Update new and changed materials. Just ignore destroyed ones.
                OptiXRenderer::Material* device_materials = (OptiXRenderer::Material*)m_state->material_parameters->map();
                for (Materials::UID material_ID : Materials::get_changed_materials())
                    if (!Materials::has_changes(material_ID, Materials::Changes::Destroyed))
                        upload_material(material_ID, device_materials);
                m_state->material_parameters->unmap();
            }
        }
    }

    { // Light updates.
        if (!LightSources::get_created_lights().is_empty() || !LightSources::get_destroyed_lights().is_empty()) {
            if (m_state->lights.ID_to_index.size() < LightSources::capacity()) {
                m_state->lights.ID_to_index.resize(LightSources::capacity());
                m_state->lights.index_to_ID.resize(LightSources::capacity());
            }

            // Deferred area light geometry update helper. Keeps track of the highest delta light index updated.
            int highest_area_light_index_updated = -1;

            LightSources::light_created_iterator created_lights_begin = LightSources::get_created_lights().begin();
            LightSources::light_created_iterator created_lights_end = LightSources::get_created_lights().end();
            LightSources::light_destroyed_iterator destroyed_lights_begin = LightSources::get_destroyed_lights().begin();
            LightSources::light_destroyed_iterator destroyed_lights_end = LightSources::get_destroyed_lights().end();

            unsigned int lights_created_count = unsigned int(created_lights_end - created_lights_begin);
            unsigned int lights_destroyed_count = unsigned int(destroyed_lights_end - destroyed_lights_begin);
            unsigned int old_light_count = m_state->lights.count;
            unsigned int rolling_light_count = old_light_count;
            m_state->lights.count += lights_created_count - lights_destroyed_count;

            // Light creation helper method.
            static auto light_creation = [](LightSources::UID light_ID, unsigned int light_index, SphereLight* device_lights,
                                            int& highest_area_light_index_updated) {
                SphereLight& light = device_lights[light_index];

                SceneNodes::UID node_ID = LightSources::get_node_ID(light_ID);
                Vector3f position = SceneNodes::get_global_transform(node_ID).translation;
                memcpy(&light.position, &position, sizeof(light.position));

                RGB power = LightSources::get_power(light_ID);
                memcpy(&light.power, &power, sizeof(light.power));

                light.radius = LightSources::get_radius(light_ID);

                if (!LightSources::is_delta_light(light_ID))
                    highest_area_light_index_updated = max(highest_area_light_index_updated, light_index);
            };

            if (old_light_count < m_state->lights.count) {
                // Resize to add room for new light sources.
                m_state->lights.sources->setSize(m_state->lights.count);

                // Resizing removes old data, so see this as an opportunity to linearize the light data.
                SphereLight* device_lights = (SphereLight*)m_state->lights.sources->map();
                unsigned int light_index = 0;
                for (LightSources::UID light_ID : LightSources::get_iterable()) {
                    m_state->lights.ID_to_index[light_ID] = light_index;
                    m_state->lights.index_to_ID[light_index] = light_ID;

                    light_creation(light_ID, light_index, device_lights, highest_area_light_index_updated);
                    ++light_index;
                }
                m_state->lights.sources->unmap();

            } else {

                SphereLight* device_lights = (SphereLight*)m_state->lights.sources->map();

                for (LightSources::UID light_ID : Iterable<LightSources::light_destroyed_iterator>(destroyed_lights_begin, destroyed_lights_end)) {
                    unsigned int light_index = m_state->lights.ID_to_index[light_ID];

                    if (!LightSources::is_delta_light(light_ID))
                        highest_area_light_index_updated = max(highest_area_light_index_updated, light_index);

                    if (created_lights_begin != created_lights_end) {
                        // Replace deleted light by new light source.
                        LightSources::UID new_light_ID = *created_lights_begin++;
                        light_creation(new_light_ID, light_index, device_lights, highest_area_light_index_updated);
                        m_state->lights.ID_to_index[new_light_ID] = light_index;
                        m_state->lights.index_to_ID[light_index] = new_light_ID;
                    } else {
                        // Replace deleted light by light from the end of the array.
                        --rolling_light_count;
                        if (light_index != rolling_light_count) {
                            memcpy(device_lights + light_index, device_lights + rolling_light_count, sizeof(SphereLight));

                            // Rewire light ID and index maps.
                            m_state->lights.index_to_ID[light_index] = m_state->lights.index_to_ID[rolling_light_count];
                            m_state->lights.ID_to_index[m_state->lights.index_to_ID[light_index]] = light_index;

                            highest_area_light_index_updated = max(highest_area_light_index_updated, rolling_light_count);
                        }
                    }
                }

                for (LightSources::UID light_ID : Iterable<LightSources::light_destroyed_iterator>(created_lights_begin, created_lights_end)) {
                    unsigned int light_index = rolling_light_count++;
                    m_state->lights.ID_to_index[light_ID] = light_index;
                    m_state->lights.index_to_ID[light_index] = light_ID;

                    light_creation(light_ID, light_index, device_lights, highest_area_light_index_updated);
                }

                m_state->lights.sources->unmap();
            }

            m_state->context["g_light_count"]->setInt(m_state->lights.count);
            m_state->accumulations = 0u;

            // Update area light geometry if needed.
            if (highest_area_light_index_updated >= 0) {
                // Some area light was updated.
                m_state->lights.area_lights->getAcceleration()->markDirty();
                m_state->root_node->getAcceleration()->markDirty();

                // Increase primitive count if new area lights have been added.
                int primitive_count = m_state->lights.area_lights_geometry->getPrimitiveCount();
                if (primitive_count < (highest_area_light_index_updated + 1)) {
                    m_state->lights.area_lights_geometry->setPrimitiveCount(highest_area_light_index_updated + 1);
                    primitive_count = highest_area_light_index_updated + 1;
                }

                // And reduce primitive count if lights have been removed.
                if (int(m_state->lights.count) < primitive_count)
                    m_state->lights.area_lights_geometry->setPrimitiveCount(m_state->lights.count);
            }
        }
    }

    { // Transform updates.
        // We're only interested in changes in the transforms that are connected to renderables, such as meshes.
        bool important_transform_changed = false; 
        for (SceneNodes::UID node_ID : SceneNodes::get_changed_transforms()) {
            if (node_ID < m_state->transforms.size()) {
                optix::Transform optixTransform = m_state->transforms[node_ID];
                if (optixTransform) {
                    Math::Transform transform = SceneNodes::get_global_transform(node_ID);
                    Math::Transform inverse_transform = invert(transform);
                    optixTransform->setMatrix(false, to_matrix4x4(transform).begin(), to_matrix4x4(inverse_transform).begin());
                    important_transform_changed = true;
                }
            }
        }

        if (important_transform_changed) {
            m_state->root_node->getAcceleration()->markDirty();
            m_state->accumulations = 0u;
        }
    }

    { // Model updates.
        // TODO Properly handle reused model ID's. Is it faster to reuse the rt components then it is to destroy and recreate them? Perhaps even keep a list of 'ready to use' components?

        bool models_changed = false;
        for (MeshModels::UID model_ID : MeshModels::get_changed_models()) {
            MeshModel model = MeshModels::get_model(model_ID);

            if (MeshModels::get_changes(model_ID) == MeshModels::Changes::Destroyed) {
                optix::Transform optixTransform = m_state->transforms[model.scene_node_ID];
                m_state->root_node->removeChild(optixTransform);
                optixTransform->destroy();
                m_state->transforms[model.scene_node_ID] = NULL;
                // TODO check if I need to destroy the subgraph. I think reference counting might take care of that.

                models_changed = true;
            }

            if (MeshModels::get_changes(model_ID) == MeshModels::Changes::Created) {
                MeshModel model = MeshModels::get_model(model_ID);
                optix::Transform transform = load_model(m_state->context, model, m_state->meshes.data(), m_state->default_material);
                m_state->root_node->addChild(transform);

                if (m_state->transforms.size() <= model.scene_node_ID)
                    m_state->transforms.resize(SceneNodes::capacity());
                m_state->transforms[model.scene_node_ID] = transform;

                models_changed = true;
            }
        }

        if (models_changed) {
            m_state->root_node->getAcceleration()->markDirty();
            m_state->accumulations = 0u;
        }
    }
}

} // NS OptiXRenderer