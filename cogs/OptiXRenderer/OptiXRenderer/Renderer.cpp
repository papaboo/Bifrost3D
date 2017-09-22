// OptiX renderer manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Renderer.h>

#include <OptiXRenderer/EnvironmentMap.h>
#include <OptiXRenderer/OctahedralNormal.h>
#include <OptiXRenderer/PresampledEnvironmentMap.h>
#include <OptiXRenderer/Kernel.h>
#include <OptiXRenderer/RhoTexture.h>
#include <OptiXRenderer/Types.h>

#include <Cogwheel/Assets/Image.h>
#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Assets/Texture.h>
#include <Cogwheel/Core/Array.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Math/Math.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/LightSource.h>
#include <Cogwheel/Scene/SceneNode.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <StbImageWriter/StbImageWriter.h>

#include <assert.h>
#include <vector>

using namespace Cogwheel;
using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;
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

static inline optix::Geometry load_mesh(optix::Context& context, Meshes::UID mesh_ID,
    optix::Program intersection_program, optix::Program bounds_program) {
    optix::Geometry optix_mesh = context->createGeometry();

    Mesh mesh = mesh_ID;

    optix_mesh->setIntersectionProgram(intersection_program);
    optix_mesh->setBoundingBoxProgram(bounds_program);

    optix::Buffer index_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, mesh.get_primitive_count(), mesh.get_primitives());
    optix_mesh["index_buffer"]->setBuffer(index_buffer);
    optix_mesh->setPrimitiveCount(mesh.get_primitive_count());

    // Vertex attributes
    RTsize vertex_count = mesh.get_vertex_count();
    optix::Buffer geometry_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, vertex_count);
    geometry_buffer->setElementSize(sizeof(VertexGeometry));
    VertexGeometry* mapped_geometry = (VertexGeometry*)geometry_buffer->map();
    for (RTsize i = 0; i < vertex_count; ++i) {
        Vector3f position = mesh.get_positions()[i];
        mapped_geometry[i].position = optix::make_float3(position.x, position.y, position.z);
        if (mesh.get_normals() != nullptr) {
            Vector3f normal = mesh.get_normals()[i];
            Math::OctahedralNormal encoded_normal = Math::OctahedralNormal::encode_precise(normal.x, normal.y, normal.z);
            mapped_geometry[i].normal = { optix::make_short2(encoded_normal.encoding.x, encoded_normal.encoding.y) };
        }
    }
    geometry_buffer->unmap();
    optix_mesh["geometry_buffer"]->setBuffer(geometry_buffer);

    RTsize texcoord_count = mesh.get_texcoords() ? vertex_count : 0;
    optix::Buffer texcoord_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, texcoord_count, mesh.get_texcoords());
    optix_mesh["texcoord_buffer"]->setBuffer(texcoord_buffer);

    OPTIX_VALIDATE(optix_mesh);

    return optix_mesh;
}

static inline optix::Transform load_model(optix::Context& context, MeshModel model, optix::Geometry* meshes, optix::Material optix_material) {
    Mesh mesh = model.get_mesh();
    optix::Geometry optix_mesh = meshes[mesh.get_ID()];

    assert(optix_mesh);

    optix::GeometryInstance optix_model = context->createGeometryInstance(optix_mesh, &optix_material, &optix_material + 1);
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

    // Per camera members.
    uint2 screensize;
    optix::Buffer accumulation_buffer;
    unsigned int accumulations;
    Matrix4x4f camera_inverse_view_projection_matrix;
    Backend backend;

    // Per scene members.
    optix::Group root_node;
    float scene_epsilon;
#if PRESAMPLE_ENVIRONMENT_MAP
    PresampledEnvironmentMap environment;
#else
    EnvironmentMap environment;
#endif

    std::vector<optix::Transform> transforms = std::vector<optix::Transform>(0);
    std::vector<optix::Geometry> meshes = std::vector<optix::Geometry>(0);

    std::vector<optix::Buffer> images = std::vector<optix::Buffer>(0);
    std::vector<optix::TextureSampler> textures = std::vector<optix::TextureSampler>(0);

    optix::Material default_material;
    optix::TextureSampler ggx_with_fresnel_rho;
    optix::Buffer material_parameters;
    unsigned int active_material_count;

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

    Implementation(int cuda_device_ID, int width_hint, int height_hint)
    : backend(Backend::PathTracing) {

        device_IDs = { -1, -1 };
        
        if (Context::getDeviceCount() == 0)
            return;

        context = Context::create();

        // TODO Use cuda_device_ID to select device.
        device_IDs.optix = 0;
        context->setDevices(&device_IDs.optix, &device_IDs.optix + 1);
        int2 compute_capability;
        context->getDeviceAttribute(device_IDs.optix, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(compute_capability), &compute_capability);
        int optix_major = OPTIX_VERSION / 10000, optix_minor = (OPTIX_VERSION % 10000) / 100, optix_micro = OPTIX_VERSION % 100;
        printf("OptiX %u.%u.%u renderer using device %u: '%s' with compute capability %u.%u.\n", 
            optix_major, optix_minor, optix_micro, 
            device_IDs.optix, context->getDeviceName(device_IDs.optix).c_str(), compute_capability.x, compute_capability.y);

        context->getDeviceAttribute(device_IDs.optix, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(device_IDs.cuda), &device_IDs.cuda);

        context->setRayTypeCount(RayTypes::Count);
        context->setEntryPointCount(EntryPoints::Count);
        context->setStackSize(1400);

        accumulations = 0u;

        std::string shader_prefix = Engine::get_instance()->data_path() + "OptiXRenderer\\ptx\\OptiXRenderer_generated_";

        auto get_ptx_path = [](const std::string& shader_prefix, const std::string& shader_filename) -> std::string {
            return shader_prefix + shader_filename + ".cu.ptx";
        };

        { // Path tracing setup.
            std::string rgp_ptx_path = get_ptx_path(shader_prefix, "PathTracing");
            context->setRayGenerationProgram(EntryPoints::PathTracing, context->createProgramFromPTXFile(rgp_ptx_path, "path_tracing"));
            context->setMissProgram(RayTypes::MonteCarlo, context->createProgramFromPTXFile(rgp_ptx_path, "miss"));
#ifdef ENABLE_OPTIX_DEBUG
            context->setExceptionProgram(EntryPoints::PathTracing, context->createProgramFromPTXFile(rgp_ptx_path, "exceptions"));
#endif

            context["g_max_bounce_count"]->setInt(4);
        }

        { // Normal visualization setup.
            std::string ptx_path = get_ptx_path(shader_prefix, "NormalRendering");
            context->setRayGenerationProgram(EntryPoints::NormalVisualization, context->createProgramFromPTXFile(ptx_path, "ray_generation"));
        }

        { // Setup default material.
            default_material = context->createMaterial();

            std::string monte_carlo_ptx_path = get_ptx_path(shader_prefix, "MonteCarlo");
            default_material->setClosestHitProgram(RayTypes::MonteCarlo, context->createProgramFromPTXFile(monte_carlo_ptx_path, "closest_hit"));
            default_material->setAnyHitProgram(RayTypes::Shadow, context->createProgramFromPTXFile(monte_carlo_ptx_path, "shadow_any_hit"));

            OPTIX_VALIDATE(default_material);

            std::string trangle_intersection_ptx_path = get_ptx_path(shader_prefix, "IntersectTriangle");
            triangle_intersection_program = context->createProgramFromPTXFile(trangle_intersection_ptx_path, "intersect");
            triangle_bounds_program = context->createProgramFromPTXFile(trangle_intersection_ptx_path, "bounds");

            active_material_count = 0;
            material_parameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, active_material_count);
            material_parameters->setElementSize(sizeof(OptiXRenderer::Material));
            context["g_materials"]->set(material_parameters);

            // Upload directional-hemispherical reflectance texture.
            ggx_with_fresnel_rho = ggx_with_fresnel_rho_texture(context);
            context["ggx_with_fresnel_rho_texture_ID"]->setInt(ggx_with_fresnel_rho->getId());
        }

        { // Setup scene
            optix::Acceleration root_acceleration = context->createAcceleration("Trbvh", "Bvh");
            root_acceleration->setProperty("refit", "1");

            root_node = context->createGroup();
            root_node->setAcceleration(root_acceleration);
            OPTIX_VALIDATE(root_node);

            context["g_scene_root"]->set(root_node);
            scene_epsilon = 0.0001f;
            context["g_scene_epsilon"]->setFloat(scene_epsilon);
#if PRESAMPLE_ENVIRONMENT_MAP
            PresampledEnvironmentLight environment = {};
            context["g_scene_environment_light"]->setUserData(sizeof(PresampledEnvironmentLight), &environment);
#else
            EnvironmentLight environment = {};
            context["g_scene_environment_light"]->setUserData(sizeof(environment), &environment);
#endif
        }

        { // Light sources
            lights.sources = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
            lights.sources->setElementSize(sizeof(Light));
            lights.count = 0;
            context["g_lights"]->set(lights.sources);
            context["g_light_count"]->setInt(lights.count);

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

            optix::Acceleration acceleration = context->createAcceleration("Bvh", "Bvh");
            OPTIX_VALIDATE(acceleration);

            optix::GeometryInstance area_lights = context->createGeometryInstance(lights.area_lights_geometry, &material, &material + 1);
            OPTIX_VALIDATE(area_lights);

            lights.area_lights = context->createGeometryGroup(&area_lights, &area_lights + 1);
            lights.area_lights->setAcceleration(acceleration);
            OPTIX_VALIDATE(lights.area_lights);

            root_node->addChild(lights.area_lights);
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

        { // Screen buffers
            screensize = make_uint2(width_hint, height_hint);
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
            accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, screensize.x, screensize.y);
            accumulation_buffer->setElementSize(sizeof(double) * 4);
#else
            accumulation_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, screensize.x, screensize.y);
#endif
            context["g_accumulation_buffer"]->set(accumulation_buffer);

            // Temporary output buffer to ensure a valid context.
            context["g_output_buffer"]->set(context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_SHORT4, 1, 1));

            camera_inverse_view_projection_matrix = Math::Matrix4x4f::identity();
        }

#ifdef ENABLE_OPTIX_DEBUG
        context->setPrintEnabled(true);
        context->setPrintLaunchIndex(screensize.x / 2, screensize.y / 2);
        context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
#else
        context->setPrintEnabled(false);
        context->setExceptionEnabled(RT_EXCEPTION_ALL, false);
#endif

        OPTIX_VALIDATE(context);
    }

    inline bool is_valid() const { return device_IDs.optix >= 0; }

    void handle_updates() {
        { // Mesh updates.
            for (Meshes::UID mesh_ID : Meshes::get_changed_meshes()) {
                if (Meshes::get_changes(mesh_ID) == Meshes::Change::Destroyed) {
                    if (mesh_ID < meshes.size() && meshes[mesh_ID]) {
                        meshes[mesh_ID]->destroy();
                        meshes[mesh_ID] = NULL;
                    }
                }

                if (Meshes::get_changes(mesh_ID) & Meshes::Change::Created) {
                    if (meshes.size() <= mesh_ID)
                        meshes.resize(Meshes::capacity());
                    meshes[mesh_ID] = load_mesh(context, mesh_ID, triangle_intersection_program, triangle_bounds_program);
                }
            }
        }

        { // Image updates.
            if (!Images::get_changed_images().is_empty()) {
                if (images.size() < Images::capacity())
                    images.resize(Images::capacity());

                for (Images::UID image_ID : Images::get_changed_images()) {
                    if (Images::get_changes(image_ID) == Images::Change::Destroyed) {
                        if (images[image_ID]) {
                            images[image_ID]->destroy();
                            images[image_ID] = NULL;
                        }
                    } else if (Images::get_changes(image_ID).is_set(Images::Change::Created)) {
                        RTformat pixel_format = RT_FORMAT_UNKNOWN;
                        switch (Images::get_pixel_format(image_ID)) {
                        case PixelFormat::I8:
                            pixel_format = RT_FORMAT_UNSIGNED_BYTE; break;
                        case PixelFormat::RGB24:
                            pixel_format = RT_FORMAT_UNSIGNED_BYTE3; break;
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
                        images[image_ID] = context->createBuffer(RT_BUFFER_INPUT, pixel_format,
                            Images::get_width(image_ID), Images::get_height(image_ID));

                        void* pixel_data = images[image_ID]->map();
                        std::memcpy(pixel_data, Images::get_pixels(image_ID), images[image_ID]->getElementSize() * Images::get_pixel_count(image_ID));
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
                            textures[texture_ID] = NULL;
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
                        Images::UID image_id = Textures::get_image_ID(texture_ID);
                        if (Images::get_gamma(Textures::get_image_ID(texture_ID)) == 1.0f)
                            texture->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT); // Image is in linear color space.
                        else
                            texture->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB); // Assume that image is in sRGB color space.
                        texture->setMaxAnisotropy(1.0f);
                        texture->setMipLevelCount(1u);
                        RTfiltermode min_filtermode = Textures::get_minification_filter(texture_ID) == MinificationFilter::None ? RT_FILTER_NEAREST : RT_FILTER_LINEAR;
                        RTfiltermode mag_filtermode = Textures::get_magnification_filter(texture_ID) == MagnificationFilter::Linear ? RT_FILTER_LINEAR : RT_FILTER_NEAREST;
                        texture->setFilteringModes(min_filtermode, mag_filtermode, RT_FILTER_NONE);
                        texture->setArraySize(1u);
                        texture->setBuffer(0u, 0u, images[image_id]);
                        OPTIX_VALIDATE(texture);
                    }
                }
            }
        }

        { // Material updates.
            static auto upload_material = [](Materials::UID material_ID, OptiXRenderer::Material* device_materials, optix::TextureSampler* samplers) {
                OptiXRenderer::Material& device_material = device_materials[material_ID];
                Assets::Material host_material = material_ID;
                device_material.tint.x = host_material.get_tint().r;
                device_material.tint.y = host_material.get_tint().g;
                device_material.tint.z = host_material.get_tint().b;
                if (host_material.get_tint_texture_ID() != Textures::UID::invalid_UID()) {
                    // Validate that the image has 4 channels! Otherwise OptiX goes boom boom.
                    Textures::UID texture_ID = host_material.get_tint_texture_ID();
                    assert(channel_count(Images::get_pixel_format(Textures::get_image_ID(texture_ID))) == 4);
                    device_material.tint_texture_ID = samplers[texture_ID]->getId();
                } else
                    device_material.tint_texture_ID = 0;
                device_material.roughness = host_material.get_roughness();
                device_material.specularity = host_material.get_specularity();
                device_material.metallic = host_material.get_metallic();
                device_material.coverage = host_material.get_coverage();
                if (host_material.get_coverage_texture_ID() != Textures::UID::invalid_UID()) {
                    // Validate that the image has 1 channel! Otherwise OptiX goes boom boom.
                    Textures::UID texture_ID = host_material.get_coverage_texture_ID();
                    assert(channel_count(Images::get_pixel_format(Textures::get_image_ID(texture_ID))) == 1);
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
                    upload_material(Materials::UID::invalid_UID(), device_materials, textures.data()); // Upload invalid material params as well.
                    for (Materials::UID material_ID : Materials::get_iterable())
                        upload_material(material_ID, device_materials, textures.data());
                    material_parameters->unmap();
                } else {
                    // Update new and changed materials. Just ignore destroyed ones.
                    OptiXRenderer::Material* device_materials = (OptiXRenderer::Material*)material_parameters->map();
                    for (Materials::UID material_ID : Materials::get_changed_materials())
                        if (!Materials::get_changes(material_ID).is_set(Materials::Change::Destroyed))
                            upload_material(material_ID, device_materials, textures.data());
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
                            highest_area_light_index_updated = max(highest_area_light_index_updated, light_index);
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

                    // Append the environment map, if valid, to the list of light sources.
                    if (environment.next_event_estimation_possible())
                        device_lights[lights.count++] = environment.get_light();

                    lights.count = light_index;
                    lights.sources->unmap();
                } else {
                    // Skip the environment light proxy at the end of the light buffer.
                    if (environment.next_event_estimation_possible())
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
                            highest_area_light_index_updated = max(highest_area_light_index_updated, light_index);

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
                    if (environment.next_event_estimation_possible())
                        device_lights[lights.count++] = environment.get_light();

                    lights.sources->unmap();
                }

                context["g_light_count"]->setInt(lights.count);
                accumulations = 0u;

                // Update area light geometry if needed.
                if (highest_area_light_index_updated >= 0) {
                    // Some area light was updated.
                    lights.area_lights->getAcceleration()->markDirty();
                    root_node->getAcceleration()->markDirty();

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
            // TODO Properly handle reused model ID's. Is it faster to reuse the rt components then it is to destroy and recreate them? Perhaps even keep a list of 'ready to use' components?

            bool models_changed = false;
            for (MeshModel model : MeshModels::get_changed_models()) {
                unsigned int scene_node_index = model.get_scene_node().get_ID();

                if (model.get_changes() == MeshModels::Change::Destroyed) {
                    if (scene_node_index < transforms.size() && transforms[scene_node_index]) {
                        optix::Transform& optixTransform = transforms[scene_node_index];
                        root_node->removeChild(optixTransform);
                        optixTransform->destroy();
                        transforms[scene_node_index] = NULL;

                        models_changed = true;
                    }
                }

                if (model.get_changes() & MeshModels::Change::Created) {
                    optix::Transform transform = load_model(context, model, meshes.data(), default_material);
                    root_node->addChild(transform);

                    if (transforms.size() <= scene_node_index)
                        transforms.resize(SceneNodes::capacity());
                    transforms[scene_node_index] = transform;

                    models_changed = true;
                } else if (model.get_changes() & MeshModels::Change::Material) {
                    optix::Transform& optixTransform = transforms[scene_node_index];
                    optix::GeometryGroup geometry_group = optixTransform->getChild<optix::GeometryGroup>();
                    optix::GeometryInstance optix_model = geometry_group->getChild(0);
                    optix_model["material_index"]->setInt(model.get_material().get_ID());
                    accumulations = 0u;
                }
            }

            if (models_changed) {
                root_node->getAcceleration()->markDirty();
                accumulations = 0u;
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
                root_node->getAcceleration()->markDirty();
                accumulations = 0u;
            }
        }

        { // Scene root updates
            for (SceneRoot scene : SceneRoots::get_changed_scenes()) {
                if (scene.get_changes().any_set(SceneRoots::Change::EnvironmentTint, SceneRoots::Change::Created)) {
                    Math::RGB env_tint = scene.get_environment_tint();
                    float3 environment_tint = make_float3(env_tint.r, env_tint.g, env_tint.b);
                    context["g_scene_environment_tint"]->setFloat(environment_tint);
                }

                if (scene.get_changes().any_set(SceneRoots::Change::EnvironmentMap, SceneRoots::Change::Created)) {
                    Textures::UID environment_map_ID = scene.get_environment_map();
                    if (environment_map_ID != environment.get_environment_map_ID()) {
                        Image image = Textures::get_image_ID(environment_map_ID);
                        // Only textures with four channels are supported.
                        if (channel_count(image.get_pixel_format()) == 4) { // TODO Support other formats as well by converting the buffers to float4 and upload.
#if PRESAMPLE_ENVIRONMENT_MAP
                            environment = PresampledEnvironmentMap(context, *scene.get_environment_light(), textures.data());
                            PresampledEnvironmentLight light = environment.get_light().presampled_environment;
#else
                            environment = EnvironmentMap(context, *scene.get_environment_light(), textures.data());
                            EnvironmentLight light = environment.get_light().environment;
#endif
                            context["g_scene_environment_light"]->setUserData(sizeof(light), &light);

                            if (environment.next_event_estimation_possible()) {
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
                                device_lights[lights.count++] = environment.get_light();
                                lights.sources->unmap();

                                context["g_light_count"]->setInt(lights.count);
                            }
                        } else {
                            EnvironmentLight light = EnvironmentLight::none();
                            context["g_scene_environment_light"]->setUserData(sizeof(light), &light);
                            printf("OptiXRenderer only supports environments with 4 channels. '%s' has %u.\n", image.get_name().c_str(), channel_count(image.get_pixel_format()));
                        }
                    }
                }
            }
        }
    }

    void render(Cogwheel::Scene::Cameras::UID camera_ID, optix::Buffer buffer, int width, int height) {
        context["g_output_buffer"]->set(buffer);

        // Resized screen buffers if necessary.
        const uint2 current_screensize = make_uint2(width, height);
        if (current_screensize != screensize) {
            accumulation_buffer->setSize(width, height);
            screensize = make_uint2(width, height);
            accumulations = 0u;
#ifdef ENABLE_OPTIX_DEBUG
            context->setPrintLaunchIndex(width / 2, height / 2);
#endif
        }

        { // Upload camera parameters.
            // Check if the camera transforms changed and, if so, upload the new ones and reset accumulation.
            Matrix4x4f inverse_view_projection_matrix = Cameras::get_inverse_view_projection_matrix(camera_ID);
            if (camera_inverse_view_projection_matrix != inverse_view_projection_matrix) {
                camera_inverse_view_projection_matrix = inverse_view_projection_matrix;

                Vector3f cam_pos = Cameras::get_transform(camera_ID).translation;

                context["g_inverted_view_projection_matrix"]->setMatrix4x4fv(false, inverse_view_projection_matrix.begin());
                float4 camera_position = make_float4(cam_pos.x, cam_pos.y, cam_pos.z, 0.0f);
                context["g_camera_position"]->setFloat(camera_position);

                accumulations = 0u;
            }
        }

        context["g_accumulations"]->setInt(accumulations);
        if (backend == Backend::PathTracing)
            context->launch(EntryPoints::PathTracing, screensize.x, screensize.y);
        else
            context->launch(EntryPoints::NormalVisualization, screensize.x, screensize.y);
        accumulations += 1u;

        /*
        if (is_power_of_two(accumulations - 1)) {
            Vector4<double>* mapped_output = (Vector4<double>*)accumulation_buffer->map();
            Image output = Images::create2D("Output", PixelFormat::RGBA_Float, 1.0, Vector2ui(screensize.x, screensize.y));
            RGBA* pixels = output.get_pixels<RGBA>();
            for (unsigned int i = 0; i < output.get_pixel_count(); ++i)
                pixels[i] = RGBA(float(mapped_output[i].x), float(mapped_output[i].y), float(mapped_output[i].z), 1.0f);
            accumulation_buffer->unmap();
            std::ostringstream filename;
            filename << "C:\\Users\\Asger\\Desktop\\image_" << (accumulations - 1) << ".png";
            StbImageWriter::write(output, filename.str());
        }
        */
    }

};

// ------------------------------------------------------------------------------------------------
// Renderer
// ------------------------------------------------------------------------------------------------

Renderer* Renderer::initialize(int cuda_device_ID, int width_hint, int height_hint) {
    try {
        Renderer* r = new Renderer(cuda_device_ID, width_hint, height_hint);
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

Renderer::Renderer(int cuda_device_ID, int width_hint, int height_hint) 
    : m_impl(new Implementation(cuda_device_ID, width_hint, height_hint)) {}

float Renderer::get_scene_epsilon(Cogwheel::Scene::SceneRoots::UID scene_root_ID) const {
    return m_impl->scene_epsilon;
}

void Renderer::set_scene_epsilon(Cogwheel::Scene::SceneRoots::UID scene_root_ID, float scene_epsilon) {
    m_impl->context["g_scene_epsilon"]->setFloat(m_impl->scene_epsilon);
    m_impl->scene_epsilon = scene_epsilon;
}

Backend Renderer::get_backend() const {
    return m_impl->backend;
}

void Renderer::set_backend(Backend backend) {
    m_impl->backend = backend;
    m_impl->accumulations = 0;
}

void Renderer::handle_updates() {
    m_impl->handle_updates();
}

void Renderer::render(Cogwheel::Scene::Cameras::UID camera_ID, optix::Buffer buffer, int width, int height) {
    m_impl->render(camera_ID, buffer, width, height);
}

optix::Context& Renderer::get_context() {
    return m_impl->context;
}

} // NS OptiXRenderer