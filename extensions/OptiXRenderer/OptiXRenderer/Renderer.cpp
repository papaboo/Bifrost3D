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
#include <OptiXRenderer/OctahedralNormal.h>
#include <OptiXRenderer/PresampledEnvironmentMap.h>
#include <OptiXRenderer/RhoTexture.h>
#include <OptiXRenderer/Types.h>

#include <Bifrost/Assets/Image.h>
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
#include <vector>

using namespace Bifrost;
using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;
using namespace optix;

// #define ENABLE_OPTIX_DEBUG

namespace OptiXRenderer {

//----------------------------------------------------------------------------
// Model loading.
//----------------------------------------------------------------------------

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

static inline optix::GeometryTriangles load_mesh(optix::Context& context, Meshes::UID mesh_ID, optix::Program attribute_program) {

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

    Renderers::UID owning_renderer_ID;

    // Per camera members.
    struct CameraState {
        uint2 screensize;
        optix::Buffer accumulation_buffer;
        unsigned int accumulations;
        unsigned int max_accumulation_count;
        unsigned int max_bounce_count;
        Matrix4x4f inverse_view_projection_matrix;
        Backend backend;
        std::unique_ptr<IBackend> backend_impl;

        inline void clear() {
            screensize = { 0u, 0u };
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

    // Per scene state.
    struct {
        optix::Group root;
#if PRESAMPLE_ENVIRONMENT_MAP
        PresampledEnvironmentMap environment;
#else
        EnvironmentMap environment;
#endif
        SceneStateGPU GPU_state;
    } scene;

    std::vector<optix::Transform> transforms = std::vector<optix::Transform>(0);
    std::vector<optix::GeometryTriangles> meshes = std::vector<optix::GeometryTriangles>(0);

    std::vector<optix::Buffer> images = std::vector<optix::Buffer>(0);
    std::vector<optix::TextureSampler> textures = std::vector<optix::TextureSampler>(0);

    optix::Material default_material;
    optix::Buffer material_parameters;
    unsigned int active_material_count;

    optix::Program triangle_attribute_program;

    struct {
        Core::Array<unsigned int> ID_to_index;
        Core::Array<LightSources::UID> index_to_ID;
        optix::Buffer sources;
        unsigned int count;

        optix::GeometryGroup area_lights;
        optix::Geometry area_lights_geometry;
    } lights;

    Implementation(int cuda_device_ID, int width_hint, int height_hint, const std::string& data_folder_path, Renderers::UID renderer_ID) {

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
        context->setStackSize(1400);

        // Per camera state
        per_camera_state.resize(1);
        per_camera_state[0].clear(); // Clear sentinel camera state.

        std::string shader_prefix = data_folder_path + "OptiXRenderer\\ptx\\OptiXRenderer_generated_";
        auto get_ptx_path = [](const std::string& shader_prefix, const std::string& shader_filename) -> std::string {
            return shader_prefix + shader_filename + ".cu.ptx";
        };

        { // Path tracing setup.
            std::string rgp_ptx_path = get_ptx_path(shader_prefix, "SimpleRGPs");
            context->setRayGenerationProgram(EntryPoints::PathTracing, context->createProgramFromPTXFile(rgp_ptx_path, "path_tracing_RPG"));
            context->setMissProgram(RayTypes::MonteCarlo, context->createProgramFromPTXFile(rgp_ptx_path, "miss"));
#ifdef ENABLE_OPTIX_DEBUG
            context->setExceptionProgram(EntryPoints::PathTracing, context->createProgramFromPTXFile(rgp_ptx_path, "exceptions"));
#endif

            { // PMJ random samples.
                optix::Buffer pmj_sample_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, PMJSamplerState::MAX_SAMPLE_COUNT);
                Vector2f* pmj_samples = (Vector2f*)pmj_sample_buffer->map();
                Bifrost::Math::RNG::fill_progressive_multijittered_bluenoise_samples(pmj_samples, pmj_samples + PMJSamplerState::MAX_SAMPLE_COUNT);
                pmj_sample_buffer->unmap();
                context["g_pmj_samples"]->set(pmj_sample_buffer);
            }
        }

        { // Albedo visualization setup.
            std::string ptx_path = get_ptx_path(shader_prefix, "SimpleRGPs");
            context->setRayGenerationProgram(EntryPoints::Albedo, context->createProgramFromPTXFile(ptx_path, "albedo_RPG"));
        }

        { // Normal visualization setup.
            std::string ptx_path = get_ptx_path(shader_prefix, "SimpleRGPs");
            context->setRayGenerationProgram(EntryPoints::Normal, context->createProgramFromPTXFile(ptx_path, "normals_RPG"));
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

            // Upload directional-hemispherical reflectance texture.
            context["ggx_with_fresnel_rho_texture"]->setTextureSampler(ggx_with_fresnel_rho_texture(context));
        }

        { // Setup scene
            optix::Acceleration root_acceleration = context->createAcceleration("Trbvh", "Bvh");
            root_acceleration->setProperty("refit", "1");

            scene.root = context->createGroup();
            scene.root->setAcceleration(root_acceleration);
            context["g_scene_root"]->set(scene.root);
            OPTIX_VALIDATE(scene.root);

            scene.GPU_state.ray_epsilon = 0.0001f;

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
        bool should_reset_allocations = false;

        { // Camera updates.
            for (Cameras::UID cam_ID : Cameras::get_changed_cameras()) {
                auto camera_changes = Cameras::get_changes(cam_ID);
                if (camera_changes == Cameras::Change::Destroyed) {
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
                        camera_state.screensize = { width, height };
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
                        camera_state.accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, width, height);
                        camera_state.accumulation_buffer->setElementSize(sizeof(double) * 4);
#else
                        camera_state.accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
#endif
                        camera_state.inverse_view_projection_matrix = {};

                        // Preserve backend if set from otuside before handle_updates is called. Yuck!
                        if (camera_state.backend == Backend::None) {
                            camera_state.backend = Backend::PathTracing;
                            camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::PathTracing));
                        }
                    }
                }
            }
        }

        { // Mesh updates.
            for (Meshes::UID mesh_ID : Meshes::get_changed_meshes()) {
                if (Meshes::get_changes(mesh_ID) == Meshes::Change::Destroyed) {
                    if (mesh_ID < meshes.size() && meshes[mesh_ID]) {
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

                for (Images::UID image_ID : Images::get_changed_images()) {
                    Image image = image_ID;
                    if (Images::get_changes(image_ID) == Images::Change::Destroyed) {
                        if (images[image_ID]) {
                            images[image_ID]->destroy();
                            images[image_ID] = nullptr;
                        }
                    } else if (Images::get_changes(image_ID).is_set(Images::Change::Created)) {
                        RTformat pixel_format = RT_FORMAT_UNKNOWN;
                        switch (image.get_pixel_format()) {
                        case PixelFormat::A8:
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
                    } else if (Images::get_changes(image_ID).is_set(Images::Change::PixelsUpdated))
                        assert(!"Pixel update not implemented yet.\n");
                }
            }
        }

        { // Texture updates.
            if (!Textures::get_changed_textures().is_empty()) {
                textures.resize(Textures::capacity());

                for (Textures::UID texture_ID : Textures::get_changed_textures()) {
                    if (Textures::get_changes(texture_ID) == Textures::Change::Destroyed) {
                        if (texture_ID < textures.size() && textures[texture_ID]) {
                            textures[texture_ID]->destroy();
                            textures[texture_ID] = nullptr;
                        }
                    }

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
                        Images::UID image_ID = Textures::get_image_ID(texture_ID);
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
            static auto upload_material = [](Materials::UID material_ID, OptiXRenderer::Material* device_materials, 
                                             optix::TextureSampler* samplers, Buffer* images) {
                OptiXRenderer::Material& device_material = device_materials[material_ID];
                Assets::Material host_material = material_ID;
                device_material.tint.x = host_material.get_tint().r;
                device_material.tint.y = host_material.get_tint().g;
                device_material.tint.z = host_material.get_tint().b;
                if (host_material.has_tint_texture()) {
                    // Validate that the image has 4 channels! Otherwise OptiX goes boom boom.
                    Textures::UID texture_ID = host_material.get_tint_roughness_texture_ID();
                    RTformat pixel_format = images[Textures::get_image_ID(texture_ID)]->getFormat();
                    assert(pixel_format == RT_FORMAT_UNSIGNED_BYTE4 || pixel_format == RT_FORMAT_FLOAT4);
                    device_material.tint_roughness_texture_ID = samplers[texture_ID]->getId();
                } else
                    device_material.tint_roughness_texture_ID = 0;

                device_material.roughness = host_material.get_roughness();
                // Only set a roughness texture if the tint texture is not set and the material has a roughness texture.
                if (device_material.tint_roughness_texture_ID == 0 && host_material.has_roughness_texture()) {
                    // Validate that the image has 1 channel! Otherwise OptiX goes boom boom.
                    Textures::UID texture_ID = host_material.get_tint_roughness_texture_ID();
                    RTformat pixel_format = images[Textures::get_image_ID(texture_ID)]->getFormat();
                    assert(pixel_format == RT_FORMAT_UNSIGNED_BYTE || pixel_format == RT_FORMAT_FLOAT);
                    device_material.roughness_texture_ID = samplers[texture_ID]->getId();
                }
                else
                    device_material.roughness_texture_ID = 0;

                device_material.specularity = host_material.get_specularity();

                device_material.metallic = host_material.get_metallic();
                if (host_material.get_metallic_texture_ID() != Textures::UID::invalid_UID()) {
                    // Validate that the image has 1 channel! Otherwise OptiX goes boom boom.
                    Textures::UID texture_ID = host_material.get_metallic_texture_ID();
                    RTformat pixel_format = images[Textures::get_image_ID(texture_ID)]->getFormat();
                    assert(pixel_format == RT_FORMAT_UNSIGNED_BYTE || pixel_format == RT_FORMAT_FLOAT);
                    device_material.metallic_texture_ID = samplers[texture_ID]->getId();
                }
                else
                    device_material.metallic_texture_ID = 0;

                device_material.coverage = host_material.get_coverage();
                if (host_material.get_coverage_texture_ID() != Textures::UID::invalid_UID()) {
                    // Validate that the image has 1 channel! Otherwise OptiX goes boom boom.
                    Textures::UID texture_ID = host_material.get_coverage_texture_ID();
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
                    upload_material(Materials::UID::invalid_UID(), device_materials, textures.data(), images.data()); // Upload invalid material params as well.
                    for (Materials::UID material_ID : Materials::get_iterable())
                        upload_material(material_ID, device_materials, textures.data(), images.data());
                    material_parameters->unmap();
                } else {
                    // Update new and changed materials. Just ignore destroyed ones.
                    OptiXRenderer::Material* device_materials = (OptiXRenderer::Material*)material_parameters->map();
                    for (Materials::UID material_ID : Materials::get_changed_materials())
                        if (!Materials::get_changes(material_ID).is_set(Materials::Change::Destroyed)) {
                            upload_material(material_ID, device_materials, textures.data(), images.data());
                            should_reset_allocations = true;
                        }
                    material_parameters->unmap();
                }
            }
        }

        { // Light updates.
            if (!LightSources::get_changed_lights().is_empty()) {
                // Light creation helper method.
                static auto light_creation = [](LightSources::UID light_ID, unsigned int light_index, Light* device_lights,
                    int& highest_area_light_index_updated) {

                    Light& device_light = device_lights[light_index];
                    switch (LightSources::get_type(light_ID)) {
                    case LightSources::Type::Sphere: {
                        Scene::SphereLight host_light = light_ID;

                        device_light.flags = Light::Sphere;

                        Vector3f position = host_light.get_node().get_global_transform().translation;
                        memcpy(&device_light.sphere.position, &position, sizeof(device_light.sphere.position));

                        RGB power = host_light.get_power();
                        memcpy(&device_light.sphere.power, &power, sizeof(device_light.sphere.power));

                        device_light.sphere.radius = host_light.get_radius();

                        if (!host_light.is_delta_light())
                            highest_area_light_index_updated = std::max<int>(highest_area_light_index_updated, light_index);
                        break;
                    }
                    case LightSources::Type::Directional: {
                        Scene::DirectionalLight host_light = light_ID;

                        device_light.flags = Light::Directional;

                        Vector3f direction = host_light.get_node().get_global_transform().rotation.forward();
                        memcpy(&device_light.directional.direction, &direction, sizeof(device_light.directional.direction));

                        RGB radiance = host_light.get_radiance();
                        memcpy(&device_light.directional.radiance, &radiance, sizeof(device_light.directional.radiance));
                        break;
                    }
                    default:
                        printf("OptiXRenderer warning: Unknown light source type %u on light %u\n", LightSources::get_type(light_ID), light_ID.get_index());
                        device_light.flags = Light::Sphere;
                        device_light.sphere.position = { 0, 0, 0 };
                        device_light.sphere.power = { 100000, 0, 100000 };
                        device_light.sphere.radius = 5;
                    }
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
                    for (LightSources::UID light_ID : LightSources::get_iterable()) {
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

                    Light* device_lights = (Light*)lights.sources->map();
                    LightSources::ChangedIterator created_lights_begin = LightSources::get_changed_lights().begin();
                    while (created_lights_begin != LightSources::get_changed_lights().end() &&
                        LightSources::get_changes(*created_lights_begin).not_set(LightSources::Change::Created))
                        ++created_lights_begin;

                    // Process destroyed lights.
                    for (LightSources::UID light_ID : LightSources::get_changed_lights()) {
                        if (LightSources::get_changes(light_ID) != LightSources::Change::Destroyed)
                            continue;

                        unsigned int light_index = lights.ID_to_index[light_ID];

                        if (!LightSources::is_delta_light(light_ID))
                            highest_area_light_index_updated = std::max<int>(highest_area_light_index_updated, light_index);

                        if (created_lights_begin != LightSources::get_changed_lights().end()) {
                            // Replace deleted light by new light source.
                            LightSources::UID new_light_ID = *created_lights_begin;
                            light_creation(new_light_ID, light_index, device_lights, highest_area_light_index_updated);
                            lights.ID_to_index[new_light_ID] = light_index;
                            lights.index_to_ID[light_index] = new_light_ID;

                            // Find next created light.
                            while (created_lights_begin != LightSources::get_changed_lights().end() &&
                                LightSources::get_changes(*created_lights_begin).not_set(LightSources::Change::Created))
                                ++created_lights_begin;
                        } else {
                            // Replace deleted light by light from the end of the array.
                            --lights.count;
                            if (light_index != lights.count) {
                                memcpy(device_lights + light_index, device_lights + lights.count, sizeof(Light));

                                // Rewire light ID and index maps.
                                lights.index_to_ID[light_index] = lights.index_to_ID[lights.count];
                                lights.ID_to_index[lights.index_to_ID[light_index]] = light_index;

                                highest_area_light_index_updated = min(highest_area_light_index_updated, int(lights.count));
                            }
                        }
                    }

                    // If there are still lights that needs to be created, then append them to the list.
                    for (LightSources::UID light_ID : Iterable<LightSources::ChangedIterator>(created_lights_begin, LightSources::get_changed_lights().end())) {
                        if (LightSources::get_changes(light_ID).not_set(LightSources::Change::Created))
                            continue;

                        unsigned int light_index = lights.count++;
                        lights.ID_to_index[light_ID] = light_index;
                        lights.index_to_ID[light_index] = light_ID;

                        light_creation(light_ID, light_index, device_lights, highest_area_light_index_updated);
                    }

                    // Append the environment map, if valid, to the list of light sources.
                    if (scene.environment.next_event_estimation_possible())
                        device_lights[lights.count++] = scene.environment.get_light();

                    lights.sources->unmap();
                }

                scene.GPU_state.light_count = lights.count;
                should_reset_allocations = true;

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

        { // Model updates.
            bool models_changed = false;
            for (MeshModel model : MeshModels::get_changed_models()) {
                unsigned int scene_node_index = model.get_scene_node().get_ID();

                if (model.get_changes() == MeshModels::Change::Destroyed) {
                    if (scene_node_index < transforms.size() && transforms[scene_node_index]) {
                        optix::Transform& optixTransform = transforms[scene_node_index];
                        scene.root->removeChild(optixTransform);
                        optixTransform->destroy();
                        transforms[scene_node_index] = nullptr;

                        models_changed = true;
                    }
                }

                if (model.get_changes() & MeshModels::Change::Created) {
                    optix::Transform transform = load_model(context, model, meshes.data(), default_material);
                    scene.root->addChild(transform);

                    if (transforms.size() <= scene_node_index)
                        transforms.resize(SceneNodes::capacity());
                    transforms[scene_node_index] = transform;

                    models_changed = true;
                } else if (model.get_changes() & MeshModels::Change::Material) {
                    optix::Transform& optixTransform = transforms[scene_node_index];
                    optix::GeometryGroup geometry_group = optixTransform->getChild<optix::GeometryGroup>();
                    optix::GeometryInstance optix_model = geometry_group->getChild(0);
                    optix_model["material_index"]->setInt(model.get_material().get_ID());
                    should_reset_allocations = true;
                }
            }

            if (models_changed) {
                scene.root->getAcceleration()->markDirty();
                should_reset_allocations = true;
            }
        }

        { // Transform updates.
            // We're only interested in changes in the transforms that are connected to renderables, such as meshes.
            bool important_transform_changed = false;
            for (SceneNodes::UID node_ID : SceneNodes::get_changed_nodes()) {
                if (SceneNodes::get_changes(node_ID).not_set(SceneNodes::Change::Transform))
                    continue;

                assert(node_ID < transforms.size());
                optix::Transform optixTransform = transforms[node_ID];
                if (optixTransform) {
                    Math::Transform transform = SceneNodes::get_global_transform(node_ID);
                    Math::Transform inverse_transform = invert(transform);
                    optixTransform->setMatrix(false, to_matrix4x4(transform).begin(), to_matrix4x4(inverse_transform).begin());
                    important_transform_changed = true;
                }
            }

            if (important_transform_changed) {
                scene.root->getAcceleration()->markDirty();
                should_reset_allocations = true;
            }
        }

        { // Scene root updates
            for (SceneRoot scene_data : SceneRoots::get_changed_scenes()) {
                if (scene_data.get_changes().any_set(SceneRoots::Change::EnvironmentTint, SceneRoots::Change::Created)) {
                    Math::RGB env_tint = scene_data.get_environment_tint();
                    scene.GPU_state.environment_tint = make_float3(env_tint.r, env_tint.g, env_tint.b);
                    should_reset_allocations = true;
                }

                if (scene_data.get_changes().any_set(SceneRoots::Change::EnvironmentMap, SceneRoots::Change::Created)) {
                    Textures::UID environment_map_ID = scene_data.get_environment_map();
                    if (environment_map_ID != scene.environment.get_environment_map_ID()) {
                        Image image = Textures::get_image_ID(environment_map_ID);
                        // Only textures with four channels are supported.
                        if (channel_count(image.get_pixel_format()) == 4) { // TODO Support other formats as well by converting the buffers to float4 and upload.
#if PRESAMPLE_ENVIRONMENT_MAP
                            scene.environment = PresampledEnvironmentMap(context, *scene_data.get_environment_light(), textures.data());
                            scene.GPU_state.environment_light = scene.environment.get_light().presampled_environment;
#else
                            scene.environment = EnvironmentMap(context, *scene_data.get_environment_light(), textures.data());
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
                            scene.environment = PresampledEnvironmentMap();
                            scene.GPU_state.environment_light = PresampledEnvironmentLight::none();
#else
                            scene.environment = EnvironmentMap();
                            scene.GPU_state.environment_light = EnvironmentLight::none();
#endif
                            printf("OptiXRenderer only supports environments with 4 channels. '%s' has %u.\n", image.get_name().c_str(), channel_count(image.get_pixel_format()));
                        }
                        should_reset_allocations = true;
                    }
                }
            }
        }

        if (should_reset_allocations)
            for (auto& camera_state : per_camera_state)
                camera_state.accumulations = 0u;
    }

    unsigned int render(Bifrost::Scene::Cameras::UID camera_ID, optix::Buffer buffer, int width, int height) {

        CameraStateGPU camera_state_GPU = {};

        { // Update camera state
            // Resize screen buffers if necessary.
            uint2 current_screensize = make_uint2(width, height);
            auto& camera_state = per_camera_state[camera_ID];
            if (current_screensize != camera_state.screensize) {
                camera_state.accumulation_buffer->setSize(width, height);
                camera_state.backend_impl->resize_backbuffers(width, height);
                camera_state.screensize = make_uint2(width, height);
                camera_state.accumulations = 0u;
#ifdef ENABLE_OPTIX_DEBUG
                context->setPrintLaunchIndex(width / 2, height / 2);
#endif
            }

            { // Upload camera parameters.
              // Check if the camera transforms changed and, if so, upload the new ones and reset accumulation.
                Matrix4x4f inverse_view_projection_matrix = Cameras::get_inverse_view_projection_matrix(camera_ID);
                if (camera_state.inverse_view_projection_matrix != inverse_view_projection_matrix) {
                    camera_state.inverse_view_projection_matrix = inverse_view_projection_matrix;
                    camera_state.accumulations = 0u;
                }

                camera_state_GPU.inverted_view_projection_matrix = optix::Matrix4x4(inverse_view_projection_matrix.begin());
                Vector3f cam_pos = Cameras::get_transform(camera_ID).translation;
                camera_state_GPU.camera_position = make_float4(cam_pos.x, cam_pos.y, cam_pos.z, 0.0f);
            }
        }


        auto& camera_state = per_camera_state[camera_ID];
        if (camera_state.accumulations >= camera_state.max_accumulation_count)
            return camera_state.accumulations;

        camera_state_GPU.output_buffer = buffer->getId();
        camera_state_GPU.accumulation_buffer = camera_state.accumulation_buffer->getId();
        camera_state_GPU.accumulations = camera_state.accumulations;
        camera_state_GPU.max_bounce_count = camera_state.max_bounce_count;
        context["g_camera_state"]->setUserData(sizeof(CameraStateGPU), &camera_state_GPU);

        context["g_scene"]->setUserData(sizeof(SceneStateGPU), &scene.GPU_state);

        camera_state.backend_impl->render(context, width, height);
        ++camera_state.accumulations;

        /*
        if (is_power_of_two(camera_state.accumulations - 1)) {
            Vector4<double>* mapped_output = (Vector4<double>*)camera_state.accumulation_buffer->map();
            Image output = Images::create2D("Output", PixelFormat::RGBA_Float, 1.0, Vector2ui(width, height));
            RGBA* pixels = output.get_pixels<RGBA>();
            for (unsigned int i = 0; i < output.get_pixel_count(); ++i)
                pixels[i] = RGBA(float(mapped_output[i].x), float(mapped_output[i].y), float(mapped_output[i].z), 1.0f);
            camera_state.accumulation_buffer->unmap();
            std::ostringstream filename;
            filename << "C:\\Users\\Asger Hoedt\\Desktop\\cam_" << camera_ID.get_index() << "_image_" << (camera_state.accumulations - 1) << ".png";
            StbImageWriter::write(output, filename.str());
        }
        */

        return camera_state.accumulations;
    }
};

// ------------------------------------------------------------------------------------------------
// Renderer
// ------------------------------------------------------------------------------------------------

Renderer* Renderer::initialize(int cuda_device_ID, int width_hint, int height_hint, const std::string& data_folder_path, Renderers::UID renderer_ID) {
    try {
        Renderer* r = new Renderer(cuda_device_ID, width_hint, height_hint, data_folder_path, renderer_ID);
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

Renderer::Renderer(int cuda_device_ID, int width_hint, int height_hint, const std::string& data_folder_path, Renderers::UID renderer_ID)
    : m_impl(new Implementation(cuda_device_ID, width_hint, height_hint, data_folder_path, renderer_ID)) {}

float Renderer::get_scene_epsilon(Bifrost::Scene::SceneRoots::UID scene_root_ID) const { return m_impl->scene.GPU_state.ray_epsilon; }
void Renderer::set_scene_epsilon(Bifrost::Scene::SceneRoots::UID scene_root_ID, float scene_epsilon) {
    m_impl->scene.GPU_state.ray_epsilon = scene_epsilon;
}

unsigned int Renderer::get_max_bounce_count(Cameras::UID camera_ID) const { 
    m_impl->conditional_per_camera_state_resize(camera_ID);
    return m_impl->per_camera_state[camera_ID].max_bounce_count;
}
void Renderer::set_max_bounce_count(Cameras::UID camera_ID, unsigned int bounce_count) {
    m_impl->conditional_per_camera_state_resize(camera_ID);
    m_impl->per_camera_state[camera_ID].max_bounce_count = bounce_count;
}

unsigned int Renderer::get_max_accumulation_count(Cameras::UID camera_ID) const {
    m_impl->conditional_per_camera_state_resize(camera_ID);
    return m_impl->per_camera_state[camera_ID].max_accumulation_count;
}
void Renderer::set_max_accumulation_count(Cameras::UID camera_ID, unsigned int accumulation_count) {
    m_impl->conditional_per_camera_state_resize(camera_ID); 
    m_impl->per_camera_state[camera_ID].max_accumulation_count = accumulation_count;
}

Backend Renderer::get_backend(Cameras::UID camera_ID) const {
    m_impl->conditional_per_camera_state_resize(camera_ID);
    return m_impl->per_camera_state[camera_ID].backend;
}

void Renderer::set_backend(Cameras::UID camera_ID, Backend backend) {
    if (backend == Backend::None)
        return;

    m_impl->conditional_per_camera_state_resize(camera_ID);

    auto& camera_state = m_impl->per_camera_state[camera_ID];
    camera_state.backend = backend;
    switch (backend) {
    case Backend::PathTracing:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::PathTracing));
        break;
    case Backend::AlbedoVisualization:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::Albedo));
        break;
    case Backend::NormalVisualization:
        camera_state.backend_impl = std::unique_ptr<IBackend>(new SimpleBackend(EntryPoints::Normal));
        break;
    }
    camera_state.accumulations = 0u;
}

void Renderer::handle_updates() {
    m_impl->handle_updates();
}

unsigned int Renderer::render(Cameras::UID camera_ID, optix::Buffer buffer, int width, int height) {
    return m_impl->render(camera_ID, buffer, width, height);
}

optix::Context& Renderer::get_context() {
    return m_impl->context;
}

} // NS OptiXRenderer