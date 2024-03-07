// OptiX renderer manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Renderer.h>

#include <OptiXRenderer/EnvironmentMap.h>
#include <OptiXRenderer/IBackend.h>
#include <OptiXRenderer/PresampledEnvironmentMap.h>
#include <OptiXRenderer/Types.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Shading/Fittings.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Assets/Texture.h>
#include <Bifrost/Core/Array.h>
#include <Bifrost/Core/Engine.h>
#include <Bifrost/Math/OctahedralNormal.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <StbImageWriter/StbImageWriter.h>

#include <assert.h>
#include <memory>
#include <set>
#include <vector>

using namespace Bifrost;
using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;
using namespace optix;

// #define ENABLE_OPTIX_DEBUG

namespace OptiXRenderer {

inline float3 to_float3(const RGB v) { return { v.r, v.g, v.b }; }
inline float3 to_float3(const Vector3f v) { return { v.x, v.y, v.z }; }

struct half4 { __half x, y, z, w; };

static inline size_t size_of(RTformat format) {
    switch (format) {
    case RT_FORMAT_FLOAT: return sizeof(float);
    case RT_FORMAT_FLOAT2: return sizeof(float2);
    case RT_FORMAT_FLOAT3: return sizeof(float3);
    case RT_FORMAT_FLOAT4: return sizeof(float4);
    case RT_FORMAT_INT: return sizeof(int);
    case RT_FORMAT_INT2: return sizeof(int2);
    case RT_FORMAT_INT3: return sizeof(int3);
    case RT_FORMAT_INT4: return sizeof(int4);
    case RT_FORMAT_UNSIGNED_INT: return sizeof(unsigned int);
    case RT_FORMAT_UNSIGNED_INT2: return sizeof(uint2);
    case RT_FORMAT_UNSIGNED_INT3: return sizeof(uint3);
    case RT_FORMAT_UNSIGNED_INT4: return sizeof(uint4);
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

//-------------------------------------------------------------------------------------------------
// Model loading.
//-------------------------------------------------------------------------------------------------

static inline optix::GeometryTriangles load_mesh(optix::Context& context, MeshID mesh_ID, optix::Program attribute_program) {

    Mesh mesh = mesh_ID;

    optix::Buffer index_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, mesh.get_primitive_count(), mesh.get_primitives());

    // Position and normal buffer
    unsigned int vertex_count = mesh.get_vertex_count();
    optix::Buffer geometry_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, vertex_count);
    geometry_buffer->setElementSize(sizeof(VertexGeometry));
    VertexGeometry* mapped_geometry = (VertexGeometry*)geometry_buffer->map();
    for (unsigned int i = 0; i < vertex_count; ++i) {
        Vector3f position = mesh.get_positions()[i];
        mapped_geometry[i].position = optix::make_float3(position.x, position.y, position.z);
        if (mesh.get_normals() != nullptr) {
            Vector3f normal = mesh.get_normals()[i];
            Math::OctahedralNormal encoded_normal = Math::OctahedralNormal::encode_precise(normal.x, normal.y, normal.z);
            mapped_geometry[i].normal = { optix::make_short2(encoded_normal.encoding.x, encoded_normal.encoding.y) };
        }
    }
    geometry_buffer->unmap();

    optix::Buffer texcoord_buffer = mesh.get_texcoords() != nullptr ?
        create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, vertex_count, mesh.get_texcoords()) :
        context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0); // TODO Use shared default buffer or bind default to context.

    { // Setup triangle geometry representation.
        optix::GeometryTriangles triangle_mesh = context->createGeometryTriangles();

        triangle_mesh->setPrimitiveCount(mesh.get_primitive_count());
        triangle_mesh->setTriangleIndices(index_buffer, RT_FORMAT_UNSIGNED_INT3);
        RTsize vertex_buffer_byte_offset = 0;
        RTsize vertex_byte_stride = sizeof(VertexGeometry);
        triangle_mesh->setVertices(vertex_count, geometry_buffer, vertex_buffer_byte_offset, vertex_byte_stride, RT_FORMAT_FLOAT3);
        triangle_mesh->setBuildFlags(RT_GEOMETRY_BUILD_FLAG_RELEASE_BUFFERS);

        triangle_mesh->setAttributeProgram(attribute_program);
        triangle_mesh["index_buffer"]->setBuffer(index_buffer);
        triangle_mesh["geometry_buffer"]->setBuffer(geometry_buffer);
        triangle_mesh["texcoord_buffer"]->setBuffer(texcoord_buffer);

        OPTIX_VALIDATE(triangle_mesh);
        return triangle_mesh;
    }
}

static inline optix::Transform load_model(optix::Context& context, MeshModel model, optix::GeometryTriangles* meshes, optix::Material optix_material) {
    Mesh mesh = model.get_mesh();
    optix::GeometryTriangles optix_mesh = meshes[mesh.get_ID()];

    assert(optix_mesh);

    optix::GeometryInstance optix_model = context->createGeometryInstance(optix_mesh, optix_material);
    optix_model["material_index"]->setInt(model.get_material().get_ID());
    unsigned char mesh_flags = mesh.get_normals() != nullptr ? MeshFlags::Normals : MeshFlags::None;
    mesh_flags |= mesh.get_texcoords() != nullptr ? MeshFlags::Texcoords : MeshFlags::None;
    optix_model["mesh_flags"]->setInt(mesh_flags);
    OPTIX_VALIDATE(optix_model);

    optix::Acceleration acceleration = context->createAcceleration("Trbvh", "Bvh");
    acceleration->setProperty("index_buffer_name", "index_buffer");
    acceleration->setProperty("vertex_buffer_name", "geometry_buffer");
    acceleration->setProperty("vertex_buffer_stride", "16");
    OPTIX_VALIDATE(acceleration);

    optix::GeometryGroup geometry_group = context->createGeometryGroup(&optix_model, &optix_model + 1);
    geometry_group->setAcceleration(acceleration);
    OPTIX_VALIDATE(geometry_group);

    optix::Transform optix_transform = context->createTransform();
    {
        Math::Transform transform = model.get_scene_node().get_global_transform();
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

struct Renderer::Implementation {

    struct {
        int optix;
        int cuda;
    } device_IDs;

    optix::Context context;

    RendererID owning_renderer_ID;

    // Per camera members.
    struct CameraState {
        uint2 frame_size;
        optix::Buffer accumulation_buffer;
        unsigned int accumulations;
        unsigned int max_accumulation_count;
        unsigned int max_bounce_count;
        Matrix4x4f inverse_view_projection_matrix;
        Backend backend;
        std::unique_ptr<IBackend> backend_impl;

        inline void clear() {
            frame_size = { 0u, 0u };
            accumulation_buffer = nullptr;
            accumulations = 0u;
            max_accumulation_count = UINT_MAX;
            max_bounce_count = 4;
            inverse_view_projection_matrix = Matrix4x4f::identity();
            backend = Backend::None;
            backend_impl = nullptr;
        }
    };
    std::vector<CameraState> per_camera_state;
    inline bool conditional_per_camera_state_resize(int camera_ID) {
        if (per_camera_state.size() <= camera_ID) {
            size_t old_size = per_camera_state.size();
            per_camera_state.resize(Cameras::capacity());
            for (size_t i = old_size; i < per_camera_state.size(); ++i)
                per_camera_state[i].clear();
            return true;
        } else
            return false;
    }

    AIDenoiserFlags AI_denoiser_flags = AIDenoiserFlag::Default;

    // Per scene state.
    struct {
        optix::Group root;
#if PRESAMPLE_ENVIRONMENT_MAP
        PresampledEnvironmentMap environment;
#else
        EnvironmentMap environment;
#endif
        PathRegularizationSettings path_regularization;
        SceneStateGPU GPU_state;
    } scene;

    std::vector<std::set<MeshModelID>> node_to_mesh_models = std::vector<std::set<MeshModelID>>(0);
    std::vector<optix::Transform> mesh_models = std::vector<optix::Transform>(0);
    std::vector<optix::GeometryTriangles> meshes = std::vector<optix::GeometryTriangles>(0);

    std::vector<optix::Buffer> images = std::vector<optix::Buffer>(0);
    std::vector<optix::TextureSampler> textures = std::vector<optix::TextureSampler>(0);

    optix::Material default_material;
    optix::Buffer material_parameters;
    unsigned int active_material_count;

    optix::Program triangle_attribute_program;

    struct {
        Core::Array<unsigned int> ID_to_index;
        Core::Array<LightSourceID> index_to_ID;
        optix::Buffer sources;
        unsigned int count;

        optix::GeometryGroup area_lights;
        optix::Geometry area_lights_geometry;
    } lights;

    Implementation(int cuda_device_ID, const std::filesystem::path& data_directory, RendererID renderer_ID) {

#pragma warning(disable : 4302 4311)

        device_IDs = { -1, -1 };
        owning_renderer_ID = renderer_ID;

        if (Context::getDeviceCount() == 0)
            return;

        int enable_RTX = true;
        if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(enable_RTX), &enable_RTX) != RT_SUCCESS)
            enable_RTX = false;

        context = Context::create();

        // TODO Use cuda_device_ID to select device.
        device_IDs.optix = 0;
        context->setDevices(&device_IDs.optix, &device_IDs.optix + 1);
        int2 compute_capability;
        context->getDeviceAttribute(device_IDs.optix, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(compute_capability), &compute_capability);
        int optix_major = OPTIX_VERSION / 10000, optix_minor = (OPTIX_VERSION % 10000) / 100, optix_micro = OPTIX_VERSION % 100;
        printf("OptiX %u.%u.%u renderer using device %u: '%s' with compute capability %u.%u and RTX %s.\n", 
            optix_major, optix_minor, optix_micro, device_IDs.optix, context->getDeviceName(device_IDs.optix).c_str(), 
            compute_capability.x, compute_capability.y, enable_RTX ? "enabled" : "disabled");

        context->getDeviceAttribute(device_IDs.optix, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(device_IDs.cuda), &device_IDs.cuda);

        context->setRayTypeCount(RayTypes::Count);
        context->setEntryPointCount(EntryPoints::Count);
        if (enable_RTX) {
            context->setMaxTraceDepth(0); // Path tracing is iterative instead of recursive, so the trace depth is always 1 ray deep.
            context->setMaxCallableProgramDepth(0);
        } else
            context->setStackSize(1400);

        // Per camera state
        per_camera_state.resize(1);
        per_camera_state[0].clear(); // Clear sentinel camera state.

        auto shader_prefix = data_directory / "OptiXRenderer" / "ptx" / "OptiXRenderer_generated_";
        auto get_ptx_path = [](const std::filesystem::path& shader_prefix, const std::string& shader_filename) -> std::string {
            return shader_prefix.generic_string() + shader_filename + ".cu.ptx";
        };
        std::string rgp_ptx_path = get_ptx_path(shader_prefix, "SimpleRGPs");

        { // Path tracing setup.
            context->setRayGenerationProgram(EntryPoints::PathTracing, context->createProgramFromPTXFile(rgp_ptx_path, "path_tracing_RPG"));
            context->setMissProgram(RayTypes::MonteCarlo, context->createProgramFromPTXFile(rgp_ptx_path, "miss"));
#ifdef ENABLE_OPTIX_DEBUG
            context->setExceptionProgram(EntryPoints::PathTracing, context->createProgramFromPTXFile(rgp_ptx_path, "exceptions"));
#endif

#if PMJ_RNG
            { // PMJ random samples.
                optix::Buffer pmj_sample_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, PMJSamplerState::MAX_SAMPLE_COUNT);
                Vector2f* pmj_samples = (Vector2f*)pmj_sample_buffer->map();
                Bifrost::Math::RNG::fill_progressive_multijittered_bluenoise_samples(pmj_samples, pmj_samples + PMJSamplerState::MAX_SAMPLE_COUNT);
                pmj_sample_buffer->unmap();
                context["g_pmj_samples"]->set(pmj_sample_buffer);
            }
#endif
        }

        { // AI Denoiser
            context->setRayGenerationProgram(EntryPoints::AIDenoiserPathTracing, context->createProgramFromPTXFile(rgp_ptx_path, "AIDenoiser::path_tracing_RPG"));
            context->setRayGenerationProgram(EntryPoints::AIDenoiserCopyOutput, context->createProgramFromPTXFile(rgp_ptx_path, "AIDenoiser::copy_to_output"));
        }

        { // Auxiliary image visualization setup.
            context->setRayGenerationProgram(EntryPoints::Depth, context->createProgramFromPTXFile(rgp_ptx_path, "depth_RPG"));
            context->setRayGenerationProgram(EntryPoints::Albedo, context->createProgramFromPTXFile(rgp_ptx_path, "albedo_RPG"));
            context->setRayGenerationProgram(EntryPoints::Tint, context->createProgramFromPTXFile(rgp_ptx_path, "tint_RPG"));
            context->setRayGenerationProgram(EntryPoints::Roughness, context->createProgramFromPTXFile(rgp_ptx_path, "roughness_RPG"));
            context->setRayGenerationProgram(EntryPoints::ShadingNormal, context->createProgramFromPTXFile(rgp_ptx_path, "shading_normal_RPG"));
        }

        { // Setup default material.
            default_material = context->createMaterial();

            std::string monte_carlo_ptx_path = get_ptx_path(shader_prefix, "MonteCarlo");
            default_material->setClosestHitProgram(RayTypes::MonteCarlo, context->createProgramFromPTXFile(monte_carlo_ptx_path, "closest_hit"));
            default_material->setAnyHitProgram(RayTypes::Shadow, context->createProgramFromPTXFile(monte_carlo_ptx_path, "shadow_any_hit"));

            OPTIX_VALIDATE(default_material);

            std::string trangle_attributes_ptx_path = get_ptx_path(shader_prefix, "TriangleAttributes");
            triangle_attribute_program = context->createProgramFromPTXFile(trangle_attributes_ptx_path, "interpolate_attributes");

            active_material_count = 0;
            material_parameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, active_material_count);
            material_parameters->setElementSize(sizeof(OptiXRenderer::Material));
            context["g_materials"]->set(material_parameters);

            { // Upload texture for estimating roughness from cos(theta) and max_PDF
                using namespace Bifrost::Assets::Shading;

                // Create buffer.
                int width = Estimate_GGX_bounded_VNDF_alpha::wo_dot_normal_sample_count;
                int height = Estimate_GGX_bounded_VNDF_alpha::max_PDF_sample_count;
                Buffer alpha_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_SHORT, width, height);

                unsigned short* rho_data = static_cast<unsigned short*>(alpha_buffer->map());
                for (int i = 0; i < width * height; ++i)
                    rho_data[i] = unsigned short(Estimate_GGX_bounded_VNDF_alpha::alphas[i] * 65535 + 0.5f);
                alpha_buffer->unmap();
                OPTIX_VALIDATE(alpha_buffer);

                // ... and wrap it in a texture sampler.
                TextureSampler& rho_texture = context->createTextureSampler();
                rho_texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
                rho_texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
                rho_texture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
                rho_texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
                rho_texture->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
                rho_texture->setMaxAnisotropy(1.0f);
                rho_texture->setMipLevelCount(1u);
                rho_texture->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
                rho_texture->setArraySize(1u);
                rho_texture->setBuffer(0u, 0u, alpha_buffer);
                OPTIX_VALIDATE(rho_texture);

                context["estimate_GGX_alpha_texture"]->setTextureSampler(rho_texture);
            }

            { // Upload directional-hemispherical reflectance texture.
                using namespace Bifrost::Assets::Shading;

                // Create buffer.
                unsigned int width = Rho::GGX_with_fresnel_angle_sample_count;
                unsigned int height = Rho::GGX_with_fresnel_roughness_sample_count;
                Buffer rho_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_SHORT2, width, height);

                unsigned short* rho_data = static_cast<unsigned short*>(rho_buffer->map());
                for (unsigned int i = 0; i < width * height; ++i) {
                    rho_data[2 * i] = unsigned short(Rho::GGX_with_fresnel[i] * 65535 + 0.5f); // No specularity
                    rho_data[2 * i + 1] = unsigned short(Rho::GGX[i] * 65535 + 0.5f); // Full specularity
                }
                rho_buffer->unmap();
                OPTIX_VALIDATE(rho_buffer);

                // ... and wrap it in a texture sampler.
                TextureSampler& rho_texture = context->createTextureSampler();
                rho_texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
                rho_texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
                rho_texture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
                rho_texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
                rho_texture->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
                rho_texture->setMaxAnisotropy(1.0f);
                rho_texture->setMipLevelCount(1u);
                rho_texture->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
                rho_texture->setArraySize(1u);
                rho_texture->setBuffer(0u, 0u, rho_buffer);
                OPTIX_VALIDATE(rho_texture);

                context["ggx_with_fresnel_rho_texture"]->setTextureSampler(rho_texture);
            }
        }

        { // Setup scene
            optix::Acceleration root_acceleration = context->createAcceleration("Trbvh", "Bvh");
            root_acceleration->setProperty("refit", "1");

            scene.root = context->createGroup();
            scene.root->setAcceleration(root_acceleration);
            context["g_scene_root"]->set(scene.root);
            OPTIX_VALIDATE(scene.root);

            scene.GPU_state.ray_epsilon = 0.0001f;

            scene.path_regularization.scale = 1.0f;
            scene.path_regularization.decay = 1.0f / 6.0f;

            { // Light sources
                lights.sources = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
                lights.sources->setElementSize(sizeof(Light));
                lights.count = 0;
                scene.GPU_state.light_buffer = lights.sources->getId();
                scene.GPU_state.light_count = lights.count;

                // Analytical area light geometry.
                lights.area_lights_geometry = context->createGeometry();
                std::string light_intersection_ptx_path = get_ptx_path(shader_prefix, "LightSources");
                lights.area_lights_geometry->setIntersectionProgram(context->createProgramFromPTXFile(light_intersection_ptx_path, "intersect"));
                lights.area_lights_geometry->setBoundingBoxProgram(context->createProgramFromPTXFile(light_intersection_ptx_path, "bounds"));
                lights.area_lights_geometry->setPrimitiveCount(0u);
                OPTIX_VALIDATE(lights.area_lights_geometry);

                // Analytical area light material.
                optix::Material material = context->createMaterial();
                std::string monte_carlo_ptx_path = get_ptx_path(shader_prefix, "MonteCarlo");
                material->setClosestHitProgram(RayTypes::MonteCarlo, context->createProgramFromPTXFile(monte_carlo_ptx_path, "light_closest_hit"));
                OPTIX_VALIDATE(material);

                optix::Acceleration acceleration = context->createAcceleration("Trbvh", "Bvh");
                OPTIX_VALIDATE(acceleration);

                optix::GeometryInstance area_lights = context->createGeometryInstance(lights.area_lights_geometry, &material, &material + 1);
                OPTIX_VALIDATE(area_lights);

                lights.area_lights = context->createGeometryGroup(&area_lights, &area_lights + 1);
                lights.area_lights->setAcceleration(acceleration);
                OPTIX_VALIDATE(lights.area_lights);

                scene.root->addChild(lights.area_lights);

                // Empty environment
#if PRESAMPLE_ENVIRONMENT_MAP
                scene.environment = PresampledEnvironmentMap();
#else
                scene.environment = EnvironmentMap();
#endif
                scene.GPU_state.environment_light = {};
            }
        }

        { // Setup dummy texture.
            { // Create red/white pattern image.
                images.resize(1);
                unsigned int width = 16, height = 16;
                images[0] = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);

                uchar4* pixel_data = static_cast<uchar4*>(images[0]->map());
                for (unsigned int y = 0; y < height; ++y)
                    for (unsigned int x = 0; x < width; ++x) {
                        uchar4* pixel = pixel_data + x + y * width;
                        if ((x & 1) == (y & 1))
                            *pixel = make_uchar4(255, 0, 0, 255);
                        else
                            *pixel = make_uchar4(255, 255, 255, 255);
                    }
                images[0]->unmap();
                OPTIX_VALIDATE(images[0]);
            }

            { // ... and wrap it in a texture sampler.
                textures.resize(1);
                TextureSampler& texture = textures[0] = context->createTextureSampler();
                texture->setWrapMode(0, RT_WRAP_REPEAT);
                texture->setWrapMode(1, RT_WRAP_REPEAT);
                texture->setWrapMode(2, RT_WRAP_REPEAT);
                texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
                texture->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
                texture->setMaxAnisotropy(1.0f);
                texture->setMipLevelCount(1u);
                texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
                texture->setArraySize(1u);
                texture->setBuffer(0u, 0u, images[0]);
                OPTIX_VALIDATE(textures[0]);
            }
        }

#ifdef ENABLE_OPTIX_DEBUG
        context->setPrintEnabled(true);
        context->setPrintLaunchIndex(0, 0);
        context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
#else
        context->setPrintEnabled(false);
        context->setExceptionEnabled(RT_EXCEPTION_ALL, false);
#endif

        OPTIX_VALIDATE(context);
    }

    inline bool is_valid() const { return device_IDs.optix >= 0; }

    void handle_updates() {
        bool should_reset_accumulations = false;

        { // Camera updates.
            for (CameraID cam_ID : Cameras::get_changed_cameras()) {
                auto camera_changes = Cameras::get_changes(cam_ID);
                if (camera_changes & Cameras::Change::Destroyed) {
                    if (cam_ID < per_camera_state.size())
                        per_camera_state[cam_ID].clear();

                } else {

                    bool camera_initialized = per_camera_state.size() > cam_ID && per_camera_state[cam_ID].accumulation_buffer.get() != nullptr;
                    bool uses_optix_renderer = owning_renderer_ID == Cameras::get_renderer_ID(cam_ID);
                    bool create_optix_renderer = uses_optix_renderer && camera_changes.is_set(Cameras::Change::Created);
                    bool switch_to_optix_renderer = uses_optix_renderer && camera_changes.is_set(Cameras::Change::Renderer);

                    if (!camera_initialized && (create_optix_renderer || switch_to_optix_renderer)) {
                        conditional_per_camera_state_resize(cam_ID);
                        auto& camera_state = per_camera_state[cam_ID];

                        unsigned int width = 1, height = 1;
                        camera_state.accumulations = 0u;
                        camera_state.frame_size = { width, height };
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
                        camera_state.accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, width, height);
                        camera_state.accumulation_buffer->setElementSize(sizeof(double) * 4);
#else
                        camera_state.accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
#endif
                        camera_state.inverse_view_projection_matrix = {};

                        // Preserve backend if set from outside before handle_updates is called. Yuck!
                        if (camera_state.backend == Backend::None) {
                            camera_state.backend = Backend::PathTracing;
                            camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::PathTracing));
                        }
                    }
                }
            }
        }

        { // Mesh updates.
            for (MeshID mesh_ID : Meshes::get_changed_meshes()) {
                // Destroy a destroyed mesh or a previous one where a new one has been created.
                if (Meshes::get_changes(mesh_ID).any_set(Meshes::Change::Created, Meshes::Change::Destroyed)) {
                    if (mesh_ID < meshes.size() && meshes[mesh_ID]) {

                        meshes[mesh_ID]["index_buffer"]->getBuffer()->destroy();
                        meshes[mesh_ID]["geometry_buffer"]->getBuffer()->destroy();
                        meshes[mesh_ID]["texcoord_buffer"]->getBuffer()->destroy();

                        meshes[mesh_ID]->destroy();
                        meshes[mesh_ID] = nullptr;
                    }
                }

                if (Meshes::get_changes(mesh_ID) & Meshes::Change::Created) {
                    if (meshes.size() <= mesh_ID)
                        meshes.resize(Meshes::capacity());
                    meshes[mesh_ID] = load_mesh(context, mesh_ID, triangle_attribute_program);
                }
            }
        }

        { // Image updates.
            if (!Images::get_changed_images().is_empty()) {
                if (images.size() < Images::capacity())
                    images.resize(Images::capacity());

                for (ImageID image_ID : Images::get_changed_images()) {
                    Image image = image_ID;
                    if (Images::get_changes(image_ID) & Images::Change::Destroyed) {
                        if (images[image_ID]) {
                            images[image_ID]->destroy();
                            images[image_ID] = nullptr;
                        }
                    } else if (Images::get_changes(image_ID) & Images::Change::Created) {
                        RTformat pixel_format = RT_FORMAT_UNKNOWN;
                        switch (image.get_pixel_format()) {
                        case PixelFormat::Alpha8:
                            pixel_format = RT_FORMAT_UNSIGNED_BYTE; break;
                        case PixelFormat::RGB24: // OptiX does not support using ubyte3 buffers as input to samplers.
                        case PixelFormat::RGBA32:
                            pixel_format = RT_FORMAT_UNSIGNED_BYTE4; break;
                        case PixelFormat::RGB_Float:
                            pixel_format = RT_FORMAT_FLOAT3; break;
                        case PixelFormat::RGBA_Float:
                            pixel_format = RT_FORMAT_FLOAT4; break;
                        }

                        // NOTE setting the depth to 1 result in invalid 2D textures for some reason.
                        // Since we know that images attached to materials will be 2D for the foreseeable future, 
                        // we just don't set the depth for now.
                        images[image_ID] = context->createBuffer(RT_BUFFER_INPUT, pixel_format, image.get_width(), image.get_height());

                        unsigned char* optix_pixel_data = (unsigned char*)images[image_ID]->map();
                        if (image.get_pixel_format() == PixelFormat::RGB24) {
                            assert(images[image_ID]->getFormat() == RT_FORMAT_UNSIGNED_BYTE4); // RGB24 images are copied to ubyte4 buffers.
                                                                                               // Copy every pixel individually and set alpha to 255.
                            unsigned char* pixel_data = (unsigned char*)image.get_pixels();
                            unsigned char* pixel_data_end = pixel_data + image.get_pixel_count() * 3;
                            while (pixel_data != pixel_data_end) {
                                *optix_pixel_data++ = *pixel_data++;
                                *optix_pixel_data++ = *pixel_data++;
                                *optix_pixel_data++ = *pixel_data++;
                                *optix_pixel_data++ = 255;
                            }
                        } else
                            std::memcpy(optix_pixel_data, image.get_pixels(), images[image_ID]->getElementSize() * image.get_pixel_count());
                        images[image_ID]->unmap();
                        OPTIX_VALIDATE(images[image_ID]);
                    } else if (Images::get_changes(image_ID) & Images::Change::PixelsUpdated)
                        assert(!"Pixel update not implemented yet.\n");
                }
            }
        }

        { // Texture updates.
            if (!Textures::get_changed_textures().is_empty()) {
                textures.resize(Textures::capacity());

                for (TextureID texture_ID : Textures::get_changed_textures()) {
                    // Destroy a destroyed texture or a previous one where a new one has been created.
                    if (Textures::get_changes(texture_ID).any_set(Textures::Change::Destroyed, Textures::Change::Created)) {
                        if (textures[texture_ID]) {
                            textures[texture_ID]->destroy();
                            textures[texture_ID] = nullptr;
                        }
                    }

                    if (Textures::get_changes(texture_ID) & Textures::Change::Destroyed)
                        continue;

                    static auto convert_wrap_mode = [](WrapMode wrapmode) {
                        switch (wrapmode) {
                        case WrapMode::Clamp: return RT_WRAP_CLAMP_TO_EDGE;
                        case WrapMode::Repeat: return RT_WRAP_REPEAT;
                        }
                        return RT_WRAP_REPEAT;
                    };

                    if (Textures::get_changes(texture_ID) & Textures::Change::Created) {
                        TextureSampler& texture = textures[texture_ID] = context->createTextureSampler();
                        texture->setWrapMode(0, convert_wrap_mode(Textures::get_wrapmode_U(texture_ID)));
                        texture->setWrapMode(1, convert_wrap_mode(Textures::get_wrapmode_V(texture_ID)));
                        texture->setWrapMode(2, convert_wrap_mode(Textures::get_wrapmode_W(texture_ID)));
                        texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
                        ImageID image_ID = Textures::get_image_ID(texture_ID);
                        if (Images::get_gamma(image_ID) == 1.0f)
                            texture->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT); // Image is in linear color space.
                        else
                            texture->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB); // Assume that image is in sRGB color space.
                        texture->setMaxAnisotropy(1.0f);
                        texture->setMipLevelCount(1u);
                        RTfiltermode min_filtermode = Textures::get_minification_filter(texture_ID) == MinificationFilter::None ? RT_FILTER_NEAREST : RT_FILTER_LINEAR;
                        RTfiltermode mag_filtermode = Textures::get_magnification_filter(texture_ID) == MagnificationFilter::Linear ? RT_FILTER_LINEAR : RT_FILTER_NEAREST;
                        texture->setFilteringModes(min_filtermode, mag_filtermode, RT_FILTER_NONE);
                        texture->setArraySize(1u);
                        texture->setBuffer(0u, 0u, images[image_ID]);
                        OPTIX_VALIDATE(texture);
                    }
                }
            }
        }

        { // Material updates.
            static auto upload_material = [](MaterialID material_ID, OptiXRenderer::Material* device_materials, 
                                             optix::TextureSampler* samplers, Buffer* images) {
                OptiXRenderer::Material& device_material = device_materials[material_ID];
                Assets::Material host_material = material_ID;
                device_material.flags = Material::Flags(int(host_material.get_flags())); // The Bifrost and OptiX flags have the same layout.

                device_material.tint = to_float3(host_material.get_tint());
                if (host_material.has_tint_texture()) {
                    // Validate that the image has 4 channels! Otherwise OptiX goes boom boom.
                    TextureID texture_ID = host_material.get_tint_roughness_texture_ID();
                    RTformat pixel_format = images[Textures::get_image_ID(texture_ID)]->getFormat();
                    assert(pixel_format == RT_FORMAT_UNSIGNED_BYTE4 || pixel_format == RT_FORMAT_FLOAT4);
                    device_material.tint_roughness_texture_ID = samplers[texture_ID]->getId();
                } else
                    device_material.tint_roughness_texture_ID = 0;

                device_material.roughness = host_material.get_roughness();
                // Only set a roughness texture if the tint texture is not set and the material has a roughness texture.
                if (device_material.tint_roughness_texture_ID == 0 && host_material.has_roughness_texture()) {
                    // Validate that the image has 1 channel! Otherwise OptiX goes boom boom.
                    TextureID texture_ID = host_material.get_tint_roughness_texture_ID();
                    RTformat pixel_format = images[Textures::get_image_ID(texture_ID)]->getFormat();
                    assert(pixel_format == RT_FORMAT_UNSIGNED_BYTE || pixel_format == RT_FORMAT_FLOAT);
                    device_material.roughness_texture_ID = samplers[texture_ID]->getId();
                }
                else
                    device_material.roughness_texture_ID = 0;

                device_material.specularity = host_material.get_specularity();

                device_material.metallic = host_material.get_metallic();
                if (host_material.get_metallic_texture_ID() != TextureID::invalid_UID()) {
                    // Validate that the image has 1 channel! Otherwise OptiX goes boom boom.
                    TextureID texture_ID = host_material.get_metallic_texture_ID();
                    RTformat pixel_format = images[Textures::get_image_ID(texture_ID)]->getFormat();
                    assert(pixel_format == RT_FORMAT_UNSIGNED_BYTE || pixel_format == RT_FORMAT_FLOAT);
                    device_material.metallic_texture_ID = samplers[texture_ID]->getId();
                }
                else
                    device_material.metallic_texture_ID = 0;

                device_material.coat = host_material.get_coat();
                device_material.coat_roughness = host_material.get_coat_roughness();

                device_material.coverage = host_material.get_coverage();
                if (host_material.get_coverage_texture_ID() != TextureID::invalid_UID()) {
                    // Validate that the image has 1 channel! Otherwise OptiX goes boom boom.
                    TextureID texture_ID = host_material.get_coverage_texture_ID();
                    RTformat pixel_format = images[Textures::get_image_ID(texture_ID)]->getFormat();
                    assert(pixel_format == RT_FORMAT_UNSIGNED_BYTE || pixel_format == RT_FORMAT_FLOAT);
                    device_material.coverage_texture_ID = samplers[texture_ID]->getId();
                } else
                    device_material.coverage_texture_ID = 0;
            };

            if (!Materials::get_changed_materials().is_empty()) {
                if (active_material_count < Materials::capacity()) {
                    // Buffer size changed. Re-upload all parameters.
                    active_material_count = Materials::capacity();
                    material_parameters->setSize(active_material_count);

                    OptiXRenderer::Material* device_materials = (OptiXRenderer::Material*)material_parameters->map();
                    upload_material(MaterialID::invalid_UID(), device_materials, textures.data(), images.data()); // Upload invalid material params as well.
                    for (MaterialID material_ID : Materials::get_iterable())
                        upload_material(material_ID, device_materials, textures.data(), images.data());
                    material_parameters->unmap();
                } else {
                    // Update new and changed materials. Just ignore destroyed ones.
                    OptiXRenderer::Material* device_materials = (OptiXRenderer::Material*)material_parameters->map();
                    for (MaterialID material_ID : Materials::get_changed_materials())
                        if (!Materials::get_changes(material_ID).is_set(Materials::Change::Destroyed)) {
                            upload_material(material_ID, device_materials, textures.data(), images.data());
                            should_reset_accumulations = true;
                        }
                    material_parameters->unmap();
                }
            }
        }

        { // Light updates.
            if (!LightSources::get_changed_lights().is_empty()) {
                // Light creation helper method.
                static auto light_creation = [](LightSourceID light_ID, unsigned int light_index, Light* device_lights,
                    int& highest_area_light_index_updated) {

                    Light& device_light = device_lights[light_index];
                    switch (LightSources::get_type(light_ID)) {
                    case LightSources::Type::Sphere: {
                        Scene::SphereLight host_light = light_ID;

                        device_light.flags = Light::Sphere;

                        device_light.sphere.position = to_float3(host_light.get_node().get_global_transform().translation);
                        device_light.sphere.radius = host_light.get_radius();
                        device_light.sphere.power = to_float3(host_light.get_power());
                        break;
                    }
                    case LightSources::Type::Spot: {
                        Scene::SpotLight host_light = light_ID;
                        auto light_transform = host_light.get_node().get_global_transform();

                        device_light.flags = Light::Spot;

                        device_light.spot.position = to_float3(light_transform.translation);
                        device_light.spot.radius = host_light.get_radius();
                        device_light.spot.power = to_float3(host_light.get_power());
                        device_light.spot.direction = to_float3(light_transform.rotation.forward());
                        device_light.spot.cos_angle = host_light.get_cos_angle();
                        break;
                    }
                    case LightSources::Type::Directional: {
                        Scene::DirectionalLight host_light = light_ID;

                        device_light.flags = Light::Directional;

                        device_light.directional.direction = to_float3(host_light.get_node().get_global_transform().rotation.forward());
                        device_light.directional.radiance = to_float3(host_light.get_radiance());
                        break;
                    }
                    default:
                        printf("OptiXRenderer warning: Unknown light source type %u on light %u\n", LightSources::get_type(light_ID), light_ID.get_index());
                        device_light.flags = Light::Sphere;
                        device_light.sphere.position = { 0, 0, 0 };
                        device_light.sphere.power = { 100000, 0, 100000 };
                        device_light.sphere.radius = 5;
                    }

                    if (!LightSources::is_delta_light(light_ID))
                        highest_area_light_index_updated = std::max<int>(highest_area_light_index_updated, light_index);
                };

                // Deferred area light geometry update helper. Keeps track of the highest delta light index updated.
                int highest_area_light_index_updated = -1;

                if (lights.ID_to_index.size() < LightSources::capacity()) {
                    // Resize the light buffer to hold the new capacity.
                    lights.ID_to_index.resize(LightSources::capacity());
                    lights.index_to_ID.resize(LightSources::capacity());
                    lights.sources->setSize(LightSources::capacity() + 1); // + 1 to allow the environment light to be added at the end.

                    // Resizing removes old data, so this as an opportunity to linearize the light data.
                    Light* device_lights = (Light*)lights.sources->map();
                    unsigned int light_index = 0;
                    for (LightSourceID light_ID : LightSources::get_iterable()) {
                        lights.ID_to_index[light_ID] = light_index;
                        lights.index_to_ID[light_index] = light_ID;

                        light_creation(light_ID, light_index, device_lights, highest_area_light_index_updated);
                        ++light_index;
                    }

                    lights.count = light_index;

                    // Append the environment map, if valid, to the list of light sources.
                    if (scene.environment.next_event_estimation_possible())
                        device_lights[lights.count++] = scene.environment.get_light();

                    lights.sources->unmap();
                } else {
                    // Skip the environment light proxy at the end of the light buffer.
                    if (scene.environment.next_event_estimation_possible())
                        lights.count -= 1;

                    auto destroy_light = [](LightSources::Changes changes) -> bool {
                        return changes.is_set(LightSources::Change::Destroyed) && changes.not_set(LightSources::Change::Created);
                    };

                    Light* device_lights = (Light*)lights.sources->map();

                    // First process destroyed lights to ensure that we don't allocate lights and then afterwards adds holes to the light array.
                    for (LightSourceID light_ID : LightSources::get_changed_lights()) {
                        if (!destroy_light(LightSources::get_changes(light_ID)))
                            continue;

                        unsigned int light_index = lights.ID_to_index[light_ID];
                        // Replace deleted light by light from the end of the array.
                        --lights.count;
                        if (light_index != lights.count) {
                            memcpy(device_lights + light_index, device_lights + lights.count, sizeof(Light));

                            // Rewire light ID and index maps.
                            lights.index_to_ID[light_index] = lights.index_to_ID[lights.count];
                            lights.ID_to_index[lights.index_to_ID[light_index]] = light_index;
                        }

                        if (!LightSources::is_delta_light(light_ID))
                            highest_area_light_index_updated = std::max<int>(highest_area_light_index_updated, light_index);
                    }

                    // Then update or create the rest of the light sources.
                    for (LightSourceID light_ID : LightSources::get_changed_lights()) {
                        auto light_changes = LightSources::get_changes(light_ID);
                        if (destroy_light(light_changes))
                            continue;

                        if (light_changes.is_set(LightSources::Change::Created)) {
                            unsigned int light_index = lights.count++;
                            lights.ID_to_index[light_ID] = light_index;
                            lights.index_to_ID[light_index] = light_ID;

                            light_creation(light_ID, light_index, device_lights, highest_area_light_index_updated);
                        } else if (light_changes.is_set(LightSources::Change::Updated)) {
                            unsigned int light_index = lights.ID_to_index[light_ID];
                            light_creation(light_ID, light_index, device_lights, highest_area_light_index_updated);
                        }
                    }

                    // Append the environment map, if valid, to the list of light sources.
                    if (scene.environment.next_event_estimation_possible())
                        device_lights[lights.count++] = scene.environment.get_light();

                    lights.sources->unmap();
                }

                scene.GPU_state.light_count = lights.count;
                should_reset_accumulations = true;

                // Update area light geometry if needed.
                if (highest_area_light_index_updated >= 0) {
                    // Some area light was updated.
                    lights.area_lights->getAcceleration()->markDirty();
                    scene.root->getAcceleration()->markDirty();

                    // Increase primitive count if new area lights have been added.
                    int primitive_count = lights.area_lights_geometry->getPrimitiveCount();
                    if (primitive_count < (highest_area_light_index_updated + 1)) {
                        lights.area_lights_geometry->setPrimitiveCount(highest_area_light_index_updated + 1);
                        primitive_count = highest_area_light_index_updated + 1;
                    }

                    // And reduce primitive count if lights have been removed.
                    if (int(lights.count) < primitive_count)
                        lights.area_lights_geometry->setPrimitiveCount(lights.count);
                }
            }
        }

        { // Transform updates.
            bool transform_changed = false;
            for (SceneNodeID node_ID : SceneNodes::get_changed_nodes()) {
                // We're only interested in changes in the transforms that are connected to renderables, such as meshes.
                if (node_ID < node_to_mesh_models.size()) {
                    // Clear the set mesh models associated with this scene node on destruction and creation.
                    if (SceneNodes::get_changes(node_ID) == SceneNodes::Change::Destroyed ||
                        SceneNodes::get_changes(node_ID) & SceneNodes::Change::Created)
                        node_to_mesh_models[node_ID].clear();

                    // Update the transformation of all models associated with transformed scene nodes.
                    if (SceneNodes::get_changes(node_ID) & SceneNodes::Change::Transform && !node_to_mesh_models[node_ID].empty()) {
                        Math::Transform transform = SceneNodes::get_global_transform(node_ID);
                        Matrix3x4f transform_matrix = to_matrix3x4(transform);
                        Matrix3x4f inverse_transform_matrix = to_matrix3x4(invert(transform));
                        for (auto mesh_model_index : node_to_mesh_models[node_ID]) {
                            optix::Transform model_transform = mesh_models[mesh_model_index];
                            if (model_transform) {
                                model_transform->setMatrix(false, transform_matrix.begin(), inverse_transform_matrix.begin());
                                transform_changed = true;
                            } else
                                node_to_mesh_models[node_ID].erase(mesh_model_index);
                        }
                    }
                }
            }

            if (transform_changed) {
                scene.root->getAcceleration()->markDirty();
                should_reset_accumulations = true;
            }
        }

        { // Model updates.
            bool models_changed = false;
            for (MeshModel model : MeshModels::get_changed_models()) {
                MeshModelID mesh_model_index = model.get_ID();
                unsigned int transform_index = model.get_scene_node().get_ID();

                auto destroy_mesh_model = [&](unsigned int mesh_model_index) {
                    optix::Transform& optixTransform = mesh_models[mesh_model_index];

                    // Destroy transform and geometry wrappers.
                    optix::GeometryGroup geometry_group = optixTransform->getChild<optix::GeometryGroup>();
                    geometry_group->getAcceleration()->destroy();
                    for (unsigned int i = 0; i < geometry_group->getChildCount(); ++i)
                        geometry_group->getChild(i)->destroy();
                    geometry_group->destroy();

                    scene.root->removeChild(optixTransform);
                    optixTransform->destroy();

                    mesh_models[mesh_model_index] = nullptr;
                };

                if (model.get_changes() & MeshModels::Change::Destroyed) {
                    if (mesh_model_index < mesh_models.size() && mesh_models[mesh_model_index]) {
                        destroy_mesh_model(mesh_model_index);

                        // Remove association to scene node.
                        node_to_mesh_models[transform_index].erase(mesh_model_index);

                        models_changed = true;
                    }
                }
                else if (model.get_changes() & MeshModels::Change::Created) {
                    if (mesh_models.size() <= mesh_model_index)
                        mesh_models.resize(MeshModels::capacity());

                    // Remove old mesh model if present. The ID of the previous scene node cannot be recovered / is not stored,
                    // so the transformation association isn't broken until the transform is next updated or destroyed.
                    if (mesh_models[mesh_model_index])
                        destroy_mesh_model(mesh_model_index);

                    optix::Transform transform = load_model(context, model, meshes.data(), default_material);
                    scene.root->addChild(transform);
                    mesh_models[mesh_model_index] = transform;

                    if (node_to_mesh_models.size() <= transform_index)
                        node_to_mesh_models.resize(SceneNodes::capacity());
                    node_to_mesh_models[transform_index].insert(mesh_model_index);

                    models_changed = true;
                } else if (model.get_changes() & MeshModels::Change::Material) {
                    optix::Transform& optixTransform = mesh_models[mesh_model_index];
                    optix::GeometryGroup geometry_group = optixTransform->getChild<optix::GeometryGroup>();
                    optix::GeometryInstance optix_model = geometry_group->getChild(0);
                    optix_model["material_index"]->setInt(model.get_material().get_ID());
                    should_reset_accumulations = true;
                }
            }

            if (models_changed) {
                scene.root->getAcceleration()->markDirty();
                should_reset_accumulations = true;
            }
        }

        { // Scene root updates
            for (SceneRoot scene_data : SceneRoots::get_changed_scenes()) {
                if (scene_data.get_changes() & SceneRoots::Change::Destroyed)
                {
                    float3 black = {0, 0, 0};
#if PRESAMPLE_ENVIRONMENT_MAP
                    scene.environment = PresampledEnvironmentMap(black);
                    scene.GPU_state.environment_light = scene.environment.get_light().presampled_environment;
#else
                    scene.environment = EnvironmentMap(black);
                    scene.GPU_state.environment_light = scene.environment.get_light().environment;
#endif
                    should_reset_accumulations = true;
                    continue;
                }

                Math::RGB _env_tint = scene_data.get_environment_tint();
                float3 env_tint = make_float3(_env_tint.r, _env_tint.g, _env_tint.b);
                if (scene_data.get_changes().any_set(SceneRoots::Change::EnvironmentTint, SceneRoots::Change::Created)) {
                    scene.environment.set_tint(env_tint);
                    scene.GPU_state.environment_light.set_tint(env_tint);
                    should_reset_accumulations = true;
                }

                if (scene_data.get_changes().any_set(SceneRoots::Change::EnvironmentMap, SceneRoots::Change::Created)) {
                    TextureID environment_map_ID = scene_data.get_environment_map();
                    if (environment_map_ID != scene.environment.get_environment_map_ID()) {
                        Image image = Textures::get_image_ID(environment_map_ID);
                        // Only textures with four channels are supported.
                        if (channel_count(image.get_pixel_format()) == 4) { // TODO Support other formats as well by converting the buffers to float4 and upload.
#if PRESAMPLE_ENVIRONMENT_MAP
                            scene.environment = PresampledEnvironmentMap(context, *scene_data.get_environment_light(), env_tint, textures.data());
                            scene.GPU_state.environment_light = scene.environment.get_light().presampled_environment;
#else
                            scene.environment = EnvironmentMap(context, *scene_data.get_environment_light(), env_tint, textures.data());
                            scene.GPU_state.environment_light = scene.environment.get_light().environment;
#endif
                            if (scene.environment.next_event_estimation_possible()) {
                                // Append environment light to the end of the light source buffer.
                                // NOTE When multi-scene support is added we cannot know if an environment light is available pr scene, 
                                // so we do not know if the environment light is always valid.
                                // This can be solved by making the environment light a proxy that points to the scene environment light, if available.
                                // If not available, then reduce the lightcount by one CPU side before rendering the scene.
                                // That way we should have minimal performance impact GPU side.
#if _DEBUG
                                RTsize light_source_capacity;
                                lights.sources->getSize(light_source_capacity);
                                assert(lights.count + 1 <= light_source_capacity);
#endif
                                Light* device_lights = (Light*)lights.sources->map();
                                device_lights[lights.count++] = scene.environment.get_light();
                                lights.sources->unmap();

                                scene.GPU_state.light_count = lights.count;
                            }
                        } else {
#if PRESAMPLE_ENVIRONMENT_MAP
                            scene.environment = PresampledEnvironmentMap(env_tint);
                            scene.GPU_state.environment_light = scene.environment.get_light().presampled_environment;
#else
                            scene.environment = EnvironmentMap(env_tint);
                            scene.GPU_state.environment_light = scene.environment.get_light().environment;
#endif
                            printf("OptiXRenderer only supports environments with 4 channels. '%s' has %u.\n", image.get_name().c_str(), channel_count(image.get_pixel_format()));
                        }
                        should_reset_accumulations = true;
                    }
                }
            }
        }

        if (should_reset_accumulations)
            for (auto& camera_state : per_camera_state)
                camera_state.accumulations = 0u;
    }

    inline CameraStateGPU prepare_camera_state(Bifrost::Scene::CameraID camera_ID, Vector2i frame_size) {
        int frame_width = frame_size.x;
        int frame_height = frame_size.y;
        auto& camera_state = per_camera_state[camera_ID];
        CameraStateGPU camera_state_GPU = {};

        { // Update camera state
            // Resize screen buffers if necessary.
            uint2 current_frame_size = make_uint2(frame_width, frame_height);
            if (current_frame_size != camera_state.frame_size) {
                camera_state.accumulation_buffer->setSize(frame_width, frame_height);
                camera_state.backend_impl->resize_backbuffers(frame_size);
                camera_state.frame_size = current_frame_size;
                camera_state.accumulations = 0u;
#ifdef ENABLE_OPTIX_DEBUG
                context->setPrintLaunchIndex(width / 2, height / 2);
#endif
            }

            { // Upload camera parameters.
                // Check if the camera transform or projection matrix changed and, if so, upload the new data and reset accumulation.
                Matrix4x4f inverse_view_projection_matrix = Cameras::get_inverse_view_projection_matrix(camera_ID);;

                if (camera_state.inverse_view_projection_matrix != inverse_view_projection_matrix) {
                    camera_state.inverse_view_projection_matrix = inverse_view_projection_matrix;
                    camera_state.accumulations = 0u;
                }

                camera_state_GPU.inverse_view_projection_matrix = optix::Matrix4x4(inverse_view_projection_matrix.begin());

                Matrix3x3f world_to_view_rotation = to_matrix3x3(Cameras::get_view_transform(camera_ID).rotation);
                camera_state_GPU.world_to_view_rotation = optix::Matrix3x3(world_to_view_rotation.begin());
            }
        }

        camera_state_GPU.accumulation_buffer = camera_state.accumulation_buffer->getId();
        camera_state_GPU.accumulations = camera_state.accumulations;
        camera_state_GPU.max_bounce_count = camera_state.max_bounce_count;
        camera_state_GPU.path_regularization_scale = scene.path_regularization.scale * (1.0f + scene.path_regularization.decay * camera_state_GPU.accumulations);
        return camera_state_GPU;
    }

    unsigned int render(Bifrost::Scene::CameraID camera_ID, optix::Buffer buffer, Vector2i frame_size) {
        CameraStateGPU camera_state_GPU = prepare_camera_state(camera_ID, frame_size);
        camera_state_GPU.output_buffer = buffer->getId();

        auto& camera_state = per_camera_state[camera_ID];
        if (camera_state.accumulations >= camera_state.max_accumulation_count)
            return camera_state.accumulations;

        context["g_camera_state"]->setUserData(sizeof(camera_state_GPU), &camera_state_GPU);
        context["g_scene"]->setUserData(sizeof(scene.GPU_state), &scene.GPU_state);

        camera_state.backend_impl->render(context, frame_size, camera_state.accumulations);
        ++camera_state.accumulations;

        return camera_state.accumulations;
    }

    std::vector<Screenshot> request_auxiliary_buffers(CameraID camera_ID, Cameras::ScreenshotContent content_requested, Vector2i frame_size) {
        const Cameras::ScreenshotContent supported_content = { Screenshot::Content::Depth, Screenshot::Content::Albedo, Screenshot::Content::Tint, Screenshot::Content::Roughness };
        if ((content_requested & supported_content) == Screenshot::Content::None)
            return std::vector<Screenshot>();

        int frame_width = frame_size.x;
        int frame_height = frame_size.y;
        int frame_pixel_count = frame_width * frame_height;

        // Rerender for the requested content into a scratch render buffer.
        auto& camera_state = per_camera_state[camera_ID];
        unsigned int accumulation_count = camera_state.accumulations;
        auto output_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_SHORT4, frame_width, frame_height);
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
        auto accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, frame_width, frame_height);
        accumulation_buffer->setElementSize(sizeof(double) * 4);
#else
        auto accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, frame_width, frame_height);
#endif

        auto screenshots = std::vector<Screenshot>();

        // Upload general camera state
        CameraStateGPU camera_state_GPU = prepare_camera_state(camera_ID, frame_size);
        camera_state_GPU.output_buffer = output_buffer->getId();
        camera_state_GPU.accumulation_buffer = accumulation_buffer->getId();
        context["g_scene"]->setUserData(sizeof(scene.GPU_state), &scene.GPU_state);

        auto render_auxiliary_feature = [&](unsigned int entrypoint) {
            for (camera_state_GPU.accumulations = 0; camera_state_GPU.accumulations < accumulation_count; ++camera_state_GPU.accumulations) {
                context["g_camera_state"]->setUserData(sizeof(camera_state_GPU), &camera_state_GPU);
                context->launch(entrypoint, frame_width, frame_height);
            }
        };

        // Render depth
        if (content_requested & Screenshot::Content::Depth) {
            render_auxiliary_feature(EntryPoints::Depth);

            // Readback screenshot data
            float* pixels = new float[frame_pixel_count];
            double4* gpu_pixels = (double4*)accumulation_buffer->map();
            for (int i = 0; i < frame_pixel_count; ++i)
                pixels[i] = float(gpu_pixels[i].x);
            accumulation_buffer->unmap();

            screenshots.emplace_back(frame_width, frame_height, Screenshot::Content::Depth, PixelFormat::Intensity_Float, pixels);
        }

        auto readback_rgba32_screenshot = [&](Screenshot::Content content) {
            // Readback screenshot data
            RGBA32* pixels = new RGBA32[frame_pixel_count];
            half4* gpu_pixels = (half4*)output_buffer->map();
            for (int i = 0; i < frame_pixel_count; ++i) {
                pixels[i].r = unsigned char(gpu_pixels[i].x * 255.0f + 0.5f);
                pixels[i].g = unsigned char(gpu_pixels[i].y * 255.0f + 0.5f);
                pixels[i].b = unsigned char(gpu_pixels[i].z * 255.0f + 0.5f);
                pixels[i].a = 255;
            }
            output_buffer->unmap();

            screenshots.emplace_back(frame_width, frame_height, content, PixelFormat::RGBA32, pixels);
        };

        // Render albedo
        if (content_requested & Screenshot::Content::Albedo) {
            render_auxiliary_feature(EntryPoints::Albedo);
            readback_rgba32_screenshot(Screenshot::Content::Albedo);
        }

        // Render tint
        if (content_requested & Screenshot::Content::Tint) {
            render_auxiliary_feature(EntryPoints::Tint);
            readback_rgba32_screenshot(Screenshot::Content::Tint);
        }

        // Render roughness
        if (content_requested & Screenshot::Content::Roughness) {
            render_auxiliary_feature(EntryPoints::Roughness);

            // Readback screenshot data
            unsigned char* pixels = new unsigned char[frame_pixel_count];
            half4* gpu_pixels = (half4*)output_buffer->map();
            for (int i = 0; i < frame_pixel_count; ++i)
                pixels[i] = unsigned char(gpu_pixels[i].x * 255.0f + 0.5f);
            output_buffer->unmap();

            screenshots.emplace_back(frame_width, frame_height, Screenshot::Content::Roughness, PixelFormat::Intensity8, pixels);
        }

        return screenshots;
    }
};

// ------------------------------------------------------------------------------------------------
// Renderer
// ------------------------------------------------------------------------------------------------

Renderer* Renderer::initialize(int cuda_device_ID, const std::filesystem::path& data_directory) {
    try {
        Renderer* r = new Renderer(cuda_device_ID, data_directory);
        if (r->m_impl->is_valid())
            return r;
        else {
            delete r;
            return nullptr;
        }
    } catch (optix::Exception e) {
        printf("OptiXRenderer failed to initialize:\n%s\n", e.getErrorString().c_str());
        return nullptr;
    }
}

Renderer::Renderer(int cuda_device_ID, const std::filesystem::path& data_directory)
    : m_renderer_ID(Bifrost::Core::Renderers::create("OptiXRenderer")),
      m_impl(new Implementation(cuda_device_ID, data_directory, m_renderer_ID)) {}

Renderer::~Renderer() {
    Bifrost::Core::Renderers::destroy(m_renderer_ID);
    delete m_impl;
}

float Renderer::get_scene_epsilon(Bifrost::Scene::SceneRootID scene_root_ID) const { return m_impl->scene.GPU_state.ray_epsilon; }
void Renderer::set_scene_epsilon(Bifrost::Scene::SceneRootID scene_root_ID, float scene_epsilon) {
    m_impl->scene.GPU_state.ray_epsilon = scene_epsilon;
}

unsigned int Renderer::get_max_bounce_count(CameraID camera_ID) const { 
    m_impl->conditional_per_camera_state_resize(camera_ID);
    return m_impl->per_camera_state[camera_ID].max_bounce_count;
}
void Renderer::set_max_bounce_count(CameraID camera_ID, unsigned int bounce_count) {
    m_impl->conditional_per_camera_state_resize(camera_ID);
    m_impl->per_camera_state[camera_ID].max_bounce_count = bounce_count;
}

unsigned int Renderer::get_max_accumulation_count(CameraID camera_ID) const {
    m_impl->conditional_per_camera_state_resize(camera_ID);
    return m_impl->per_camera_state[camera_ID].max_accumulation_count;
}
void Renderer::set_max_accumulation_count(CameraID camera_ID, unsigned int accumulation_count) {
    m_impl->conditional_per_camera_state_resize(camera_ID); 
    m_impl->per_camera_state[camera_ID].max_accumulation_count = accumulation_count;
}

Backend Renderer::get_backend(CameraID camera_ID) const {
    m_impl->conditional_per_camera_state_resize(camera_ID);
    return m_impl->per_camera_state[camera_ID].backend;
}

void Renderer::set_backend(CameraID camera_ID, Backend backend) {
    if (backend == Backend::None)
        return;

    m_impl->conditional_per_camera_state_resize(camera_ID);

    auto& camera_state = m_impl->per_camera_state[camera_ID];
    camera_state.backend = backend;
    switch (backend) {
    case Backend::PathTracing:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::PathTracing));
        break;
    case Backend::AIDenoisedPathTracing:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new AIDenoisedBackend(m_impl->context, &m_impl->AI_denoiser_flags));
        break;
    case Backend::DepthVisualization:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::Depth));
        break;
    case Backend::AlbedoVisualization:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::Albedo));
        break;
    case Backend::TintVisualization:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::Tint));
        break;
    case Backend::RoughnessVisualization:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::Roughness));
        break;
    case Backend::ShadingNormalVisualization:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::ShadingNormal));
        break;
    default:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::Albedo));
        printf("OptiXRenderer: Backend %u not supported.\n", backend);
    }
    camera_state.accumulations = 0u;
}

PathRegularizationSettings Renderer::get_path_regularization_settings() const { return m_impl->scene.path_regularization; }
void Renderer::set_path_regularization_settings(PathRegularizationSettings settings) { m_impl->scene.path_regularization = settings; }

AIDenoiserFlags Renderer::get_AI_denoiser_flags() const { return m_impl->AI_denoiser_flags; }
void Renderer::set_AI_denoiser_flags(AIDenoiserFlags flags) { m_impl->AI_denoiser_flags = flags; }

void Renderer::handle_updates() { m_impl->handle_updates(); }

unsigned int Renderer::render(CameraID camera_ID, optix::Buffer buffer, Vector2i frame_size) {
    return m_impl->render(camera_ID, buffer, frame_size);
}

std::vector<Screenshot> Renderer::request_auxiliary_buffers(CameraID camera_ID, Cameras::ScreenshotContent content_requested, Vector2i frame_size) {
    return m_impl->request_auxiliary_buffers(camera_ID, content_requested, frame_size);
}

optix::Context& Renderer::get_context() { return m_impl->context; }

} // NS OptiXRenderer