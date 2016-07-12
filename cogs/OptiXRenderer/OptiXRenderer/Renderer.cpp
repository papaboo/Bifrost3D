// OptiX renderer manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Renderer.h>

#include <OptiXRenderer/EncodedNormal.h>
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

#include <GL/gl.h>

#include <assert.h>
#include <vector>

#include <StbImageWriter/StbImageWriter.h>

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
// #define ENABLE_OPTIX_DEBUG

namespace OptiXRenderer {

struct Environment {
    TextureND map;
    optix::TextureSampler marginal_CDF;
    optix::TextureSampler conditional_CDF;
    optix::TextureSampler per_pixel_PDF;

    EnvironmentLight to_light_source(optix::TextureSampler* texture_cache) {
        EnvironmentLight light;
        Image image = map.get_image();
        light.width = image.get_width();
        light.height = image.get_height();
        light.environment_map_ID = texture_cache[map.get_ID()]->getId();
        light.marginal_CDF_ID = marginal_CDF->getId();
        light.conditional_CDF_ID = conditional_CDF->getId();
        light.per_pixel_PDF_ID = per_pixel_PDF->getId();
        return light;
    }

    bool is_valid() { return map != Textures::UID::invalid_UID(); }
};

struct Renderer::State {
    uint2 screensize;
    optix::Context context;

    // Per camera members.
    optix::Buffer accumulation_buffer;
    optix::Buffer output_buffer;
    unsigned int accumulations;
    Matrix4x4f camera_inverse_view_projection_matrix;

    GLuint backbuffer_gl_id;

    // Per scene members.
    optix::Group root_node;
    float scene_epsilon;
    Environment environment;

    std::vector<optix::Transform> transforms = std::vector<optix::Transform>(0);
    std::vector<optix::Geometry> meshes = std::vector<optix::Geometry>(0);

    std::vector<optix::Buffer> images = std::vector<optix::Buffer>(0);
    std::vector<optix::TextureSampler> textures = std::vector<optix::TextureSampler>(0);

    optix::Material default_material;
    optix::TextureSampler default_material_rho;
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
    optix::Geometry optix_mesh = context->createGeometry();
    
    const Mesh& mesh = Meshes::get_mesh(mesh_ID);

    optix_mesh->setIntersectionProgram(intersection_program);
    optix_mesh->setBoundingBoxProgram(bounds_program);

    optix::Buffer index_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, mesh.indices_count, mesh.indices);
    optix_mesh["index_buffer"]->setBuffer(index_buffer);
    optix_mesh->setPrimitiveCount(mesh.indices_count);

    // Vertex attributes
    optix::Buffer position_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh.vertex_count, mesh.positions);
    optix_mesh["position_buffer"]->setBuffer(position_buffer);

    RTsize normal_count = mesh.normals ? mesh.vertex_count : 0;
    optix::Buffer normal_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, normal_count);
    normal_buffer->setElementSize(sizeof(EncodedNormal));
    EncodedNormal* mapped_normals = (EncodedNormal*)normal_buffer->map();
    for (RTsize i = 0; i < normal_count; ++i) {
        Vector3f normal = mesh.normals[i];
        mapped_normals[i] = EncodedNormal(normal.x, normal.y, normal.z);
    }
    normal_buffer->unmap();
    optix_mesh["normal_buffer"]->setBuffer(normal_buffer);

    RTsize texcoord_count = mesh.texcoords ? mesh.vertex_count : 0;
    optix::Buffer texcoord_buffer = create_buffer(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, texcoord_count, mesh.texcoords);
    optix_mesh["texcoord_buffer"]->setBuffer(texcoord_buffer);

    OPTIX_VALIDATE(optix_mesh);

    return optix_mesh;
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
// Environment map CDF calculation. Returns true if the CDF is successfully 
// computed, otherwise false.
// The CDF is ill-defined if fx the input image is completely black 
// or contains negative values.
//----------------------------------------------------------------------------
bool compute_environment_CDFs(Image environment, optix::Context& context, 
                              optix::Buffer& marginal_CDF, optix::Buffer& conditional_CDF, optix::Buffer& per_pixel_PDF) {

    unsigned int width = environment.get_width();
    unsigned int height = environment.get_height();

    // Perform computations in double precision to maintain some precision.
    double* marginal_CDFd = new double[height + 1];
    double* conditional_CDFd = new double[(width + 1) * height];
    per_pixel_PDF = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width, height);
    float* per_pixel_PDF_data = static_cast<float*>(per_pixel_PDF->map());

    // TODO Perform CDF reduction and normalization directly on the GPU.
    // TODO Verify that conditional CDF sampling will be in the same row in the array, i.e that the cache behaviour is sensible.

    // Compute conditional CDF.
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < int(height); ++y) {
        // PBRT p. 728. Account for the non-uniform surface area of the pixels, e.g. the higher density near the poles.
        float sin_theta = sinf(PIf * float(y + 0.5f) / float(height)); 
        
        double* conditional_CDF_row = conditional_CDFd + y * (width + 1);
        float* per_pixel_PDF_row = per_pixel_PDF_data + y * width;
        conditional_CDF_row[0] = 0.0;
        for (unsigned int x = 0; x < width; ++x) {
            RGB pixel = environment.get_pixel(Vector2ui(x, y)).rgb();
            // Pixel importance is scaled by sin_theta to avoid oversampling at the poles. See PBRT v2 page 727.
            // TODO Blur a bit to account for linear interpolation.
            float pixel_importance = (pixel.r + pixel.g + pixel.b) * sin_theta; // TODO Use luminance instead? Perhaps define a global importance(RGB / float3) function and use it here and for BRDF sampling.
            conditional_CDF_row[x + 1] = conditional_CDF_row[x] + pixel_importance;
            // Precompute the PDF of the subtended solid angle of each pixel. The PDF must be scaled by 1 / sin_theta before use. See PBRT v2 page 728.
            per_pixel_PDF_row[x] = pixel_importance;
        }
    }

    // Compute marginal CDF.
    marginal_CDFd[0] = 0.0;
    for (unsigned int y = 0; y < height; ++y)
        marginal_CDFd[y + 1] = marginal_CDFd[y] + conditional_CDFd[(y + 1) * (width + 1) - 1];

    // Integral of the environment map.
    float environment_integral = float(marginal_CDFd[height] * (2.0f * PIf * PIf) / (width * height));

    if (environment_integral < 0.00001f)
        return false;

    // Normalize marginal CDF.
    for (unsigned int y = 1; y < height; ++y)
        marginal_CDFd[y] /= marginal_CDFd[height];
    marginal_CDFd[height] = 1.0;

    // Normalize conditional CDF.
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < int(height); ++y) {
        double* conditional_CDF_row = conditional_CDFd + y * (width + 1);
        for (unsigned int x = 1; x < width; ++x)
            conditional_CDF_row[x] /= conditional_CDF_row[width];
        conditional_CDF_row[width] = 1.0f;
    }

    { // Upload data to OptiX buffers.
        // Marginal CDF.
        marginal_CDF = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, height + 1);
        float* marginal_CDF_data = static_cast<float*>(marginal_CDF->map());
        for (unsigned int y = 0; y < height + 1; ++y)
            marginal_CDF_data[y] = float(marginal_CDFd[y]);
        marginal_CDF->unmap();

        // Conditional CDF.
        conditional_CDF = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, width + 1, height);
        float* conditional_CDF_data = static_cast<float*>(conditional_CDF->map());
        for (unsigned int i = 0; i < (width + 1) * height; ++i)
            conditional_CDF_data[i] = float(conditional_CDFd[i]);
        conditional_CDF->unmap();
    
        // Precalculate the PDF and store it in an array for fast lookup.
        // TODO Test if it is just as fast to just compute it on the fly from the CDF tables.
        for (unsigned int i = 0; i < width * height; ++i)
            per_pixel_PDF_data[i] /= environment_integral;
        per_pixel_PDF->unmap();
    }

    delete[] marginal_CDFd;
    delete[] conditional_CDFd;

    return true;
}

//----------------------------------------------------------------------------
// Creates an environment representation from an environment map.
// This includes constructing the environment map CDFs and per pixel PDF.
// In case the CDF's cannot be constructed the environment returned will 
// contain invalid values, e.g. invalud UID and nullptrs.
//----------------------------------------------------------------------------
Environment create_environment(TextureND environment_map, optix::Context& context) {

    optix::Buffer marginal_CDF, conditional_CDF, per_pixel_PDF;
    bool success = compute_environment_CDFs(environment_map.get_image(), context, marginal_CDF, conditional_CDF, per_pixel_PDF);
    if (!success) {
        Environment env = { Textures::UID::invalid_UID(), nullptr, nullptr, nullptr };
        return env;
    }

    Environment environment;
    environment.map = environment_map;

    { // Marginal CDF sampler.
        TextureSampler& texture = environment.marginal_CDF = context->createTextureSampler();
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        texture->setMaxAnisotropy(0.0f);
        texture->setMipLevelCount(1u);
        texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        texture->setArraySize(1u);
        texture->setBuffer(0u, 0u, marginal_CDF);
        OPTIX_VALIDATE(texture);
    }

    { // Conditional CDF sampler.
        TextureSampler& texture = environment.conditional_CDF = context->createTextureSampler();
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        texture->setMaxAnisotropy(0.0f);
        texture->setMipLevelCount(1u);
        texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        texture->setArraySize(1u);
        texture->setBuffer(0u, 0u, conditional_CDF);
        OPTIX_VALIDATE(texture);
    }

    { // Per pixel PDF sampler.
        TextureSampler& texture = environment.per_pixel_PDF = context->createTextureSampler();
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE); // Data is already in floating point format, so no need to normalize it.
        texture->setMaxAnisotropy(0.0f);
        texture->setMipLevelCount(1u);
        texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
        texture->setArraySize(1u);
        texture->setBuffer(0u, 0u, conditional_CDF);
        OPTIX_VALIDATE(texture);
    }

    return environment;
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

    { // Setup scene
        optix::Acceleration root_acceleration = context->createAcceleration("Bvh", "Bvh");
        root_acceleration->setProperty("refit", "1");

        m_state->root_node = context->createGroup();
        m_state->root_node->setAcceleration(root_acceleration);
        OPTIX_VALIDATE(m_state->root_node);

        context["g_scene_root"]->set(m_state->root_node);
        m_state->scene_epsilon = 0.0001f;
        context["g_scene_epsilon"]->setFloat(m_state->scene_epsilon);
        EnvironmentLight environment = {};
        context["g_scene_environment_light"]->setUserData(sizeof(environment), &environment);
    }

    { // Light sources
        m_state->lights.sources = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
        m_state->lights.sources->setElementSize(sizeof(Light));
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

    { // Setup dummy texture.
        { // Create red/white pattern image.
            m_state->images.resize(1);
            unsigned int width = 16, height = 16;
            m_state->images[0] = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);

            uchar4* pixel_data = static_cast<uchar4*>(m_state->images[0]->map());
            for (unsigned int y = 0; y < height; ++y)
                for (unsigned int x = 0; x < width; ++x) {
                    uchar4* pixel = pixel_data + x + y * width;
                    if ((x & 1) == (y & 1))
                        *pixel = make_uchar4(255, 0, 0, 255);
                    else
                        *pixel = make_uchar4(255, 255, 255, 255);
                }
            m_state->images[0]->unmap();
            OPTIX_VALIDATE(m_state->images[0]);
        }

        { // ... and wrap it in a texture sampler.
            m_state->textures.resize(1);
            TextureSampler& texture = m_state->textures[0] = context->createTextureSampler();
            texture->setWrapMode(0, RT_WRAP_REPEAT);
            texture->setWrapMode(1, RT_WRAP_REPEAT);
            texture->setWrapMode(2, RT_WRAP_REPEAT);
            texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
            texture->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
            texture->setMaxAnisotropy(1.0f);
            texture->setMipLevelCount(1u);
            texture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
            texture->setArraySize(1u);
            texture->setBuffer(0u, 0u, m_state->images[0]);
            OPTIX_VALIDATE(m_state->textures[0]);
        }
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

        m_state->active_material_count = 0;
        m_state->material_parameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_state->active_material_count);
        m_state->material_parameters->setElementSize(sizeof(OptiXRenderer::Material));
        context["g_materials"]->set(m_state->material_parameters);

        // Upload directional-hemispherical reflectance texture. TODO Approximate rho using a function instead.
        m_state->default_material_rho = default_shading_rho_texture(context);
        context["default_shading_rho_texture_ID"]->setUint(m_state->default_material_rho->getId());
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
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // TODO Needed here or when rendering? Also, since the elements are larger than char, half4, should it stil be 1?
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        }

        m_state->camera_inverse_view_projection_matrix = Math::Matrix4x4f::identity();
    }

    { // Path tracing setup.
        std::string rgp_ptx_path = get_ptx_path("PathTracing");
        context->setRayGenerationProgram(int(EntryPoints::PathTracing), context->createProgramFromPTXFile(rgp_ptx_path, "path_tracing"));
        context->setMissProgram(int(RayTypes::MonteCarlo), context->createProgramFromPTXFile(rgp_ptx_path, "miss"));
#ifdef ENABLE_OPTIX_DEBUG
        context->setExceptionProgram(int(EntryPoints::PathTracing), context->createProgramFromPTXFile(rgp_ptx_path, "exceptions"));
#endif

        context["g_max_bounce_count"]->setInt(4);
    }

    { // Normal visualization setup.
        std::string ptx_path = get_ptx_path("NormalRendering");
        context->setRayGenerationProgram(int(EntryPoints::NormalVisualization), context->createProgramFromPTXFile(ptx_path, "ray_generation"));
        context->setMissProgram(int(RayTypes::NormalVisualization), context->createProgramFromPTXFile(ptx_path, "miss"));
    }

#ifdef ENABLE_OPTIX_DEBUG
    context->setPrintEnabled(true);
    context->setPrintLaunchIndex(m_state->screensize.x / 2, m_state->screensize.y / 2);
    context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
#endif

    OPTIX_VALIDATE(context);
    context->compile();
}

float Renderer::get_scene_epsilon() const {
    return m_state->scene_epsilon;
}

void Renderer::set_scene_epsilon(float scene_epsilon) {
    m_state->context["g_scene_epsilon"]->setFloat(m_state->scene_epsilon);
    m_state->scene_epsilon = scene_epsilon;
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
#ifdef ENABLE_OPTIX_PRINT
        context->setPrintLaunchIndex(window.get_width() / 2, window.get_height() / 2);
#endif
    }

    { // Upload camera parameters.
        Cameras::UID camera_ID = *Cameras::begin();

        Matrix4x4f inverse_view_matrix = to_matrix4x4(Cameras::get_inverse_view_transform(camera_ID));
        Matrix4x4f inverse_projection_matrix = Cameras::get_inverse_projection_matrix(camera_ID);
        Matrix4x4f inverse_view_projection_matrix = inverse_view_matrix * inverse_projection_matrix;

        // Check if the camera transforms changed and if so, then upload the new ones and reset accumulation.
        if (m_state->camera_inverse_view_projection_matrix != inverse_view_projection_matrix) {
            m_state->camera_inverse_view_projection_matrix = inverse_view_projection_matrix;

            SceneNode camera_node = Cameras::get_node_ID(camera_ID);
            Vector3f cam_pos = camera_node.get_global_transform().translation;

            context["g_inverted_view_projection_matrix"]->setMatrix4x4fv(false, inverse_view_projection_matrix.begin());
            float4 camera_position = make_float4(cam_pos.x, cam_pos.y, cam_pos.z, 0.0f);
            context["g_camera_position"]->setFloat(camera_position);

            m_state->accumulations = 0u;
        }
    }

    if (m_state->accumulations == 0u) {
        Cameras::UID camera_ID = *Cameras::begin();
        SceneRoot scene = Cameras::get_scene_ID(camera_ID);
        Math::RGB bg_color = scene.get_background_color();
        float3 background_color = make_float3(bg_color.r, bg_color.g, bg_color.b);
        context["g_scene_background_color"]->setFloat(background_color);

        // Setup the environment map. TODO Handle this via scene change flags or scene initialization instead.
        Textures::UID environment_map_ID = scene.get_environment_map();
        if (environment_map_ID != Textures::UID::invalid_UID() && !m_state->environment.is_valid()) {
            // Only textures with four channels are supported.
            Image image = Textures::get_image_ID(environment_map_ID);
            if (channel_count(image.get_pixel_format()) == 4) { // TODO Support other formats as well by converting the buffers to float4 and upload.
                m_state->environment = create_environment(environment_map_ID, context);
                if (m_state->environment.is_valid()) {
                    EnvironmentLight light = m_state->environment.to_light_source(m_state->textures.data());
                    context["g_scene_environment_light"]->setUserData(sizeof(light), &light);
                    
                    // Append environment light to the end of the light source buffer.
                    // NOTE When multi scene support is added we cannot know if an environment light is available pr scene, 
                    // so we do not know if the environment light is always valid.
                    // This can be solved by making the environment light a proxy that points to the scene environment light, if available.
                    // If not available, then reduce the lightcount by one CPU side before rendering the scene.
                    // That way we should have minimal performance impact on the GPU code.
#if _DEBUG
                    RTsize light_source_capacity;
                    m_state->lights.sources->getSize(light_source_capacity);
                    assert(m_state->lights.count + 1 <= light_source_capacity);
#endif
                    Light* device_lights = (Light*)m_state->lights.sources->map();
                    Light& device_light = device_lights[m_state->lights.count++];
                    device_light.type = LightTypes::Environment;
                    device_light.environment = m_state->environment.to_light_source(m_state->textures.data());
                    m_state->lights.sources->unmap();

                    context["g_light_count"]->setInt(m_state->lights.count);
                } else {
                    // The environment could not be created, most likely because the environment is practically black, 
                    // which causes the CDF calculation to fail.
                    float3 black = make_float3(0.0f);
                    context["g_scene_background_color"]->setFloat(black);
                }
            } else
                printf("The OptiXRenderer only supports environments with 4 channels. '%s' has %u.\n", image.get_name().c_str(), channel_count(image.get_pixel_format()));
        }
    }

    context["g_accumulations"]->setInt(m_state->accumulations);

    context->launch(int(EntryPoints::PathTracing), m_state->screensize.x, m_state->screensize.y);

    if (false && is_power_of_two(m_state->accumulations)) {
        void* mapped_output_buffer = m_state->output_buffer->map();
        Image output = Images::create("Output", PixelFormat::RGBA_Float, 1.0, Vector2ui(m_state->screensize.x, m_state->screensize.y));
        memcpy(output.get_pixels(), mapped_output_buffer, sizeof(float) * 4 * m_state->screensize.x * m_state->screensize.y);
        m_state->output_buffer->unmap();
        std::ostringstream filename;
        filename << "C:\\Users\\Asger\\Desktop\\env_result\\output_" << m_state->accumulations << ".png";
        StbImageWriter::write(filename.str(), output);
    }

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

        // TODO Render as a single triangle. Also in SmallPT
        glBegin(GL_QUADS); {

            glTexCoord2f(0.0f, 1.0f);
            glVertex3f(-1.0f, 1.0f, 0.f);

            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(1.0f, 1.0f, 0.f);

            glTexCoord2f(1.0f, 0.0f);
            glVertex3f(1.0f, -1.0f, 0.f);

            glTexCoord2f(0.0f, 0.0f);
            glVertex3f(-1.0f, -1.0f, 0.f);

        } glEnd();
    }
}

void Renderer::handle_updates() {
    optix::Context& context = m_state->context;

    { // Mesh updates.
        for (Meshes::UID mesh_ID : Meshes::get_changed_meshes()) {
            if (Meshes::get_changes(mesh_ID) == Meshes::Changes::Destroyed) {
                if (m_state->meshes[mesh_ID]) {
                    m_state->meshes[mesh_ID]->destroy();
                    m_state->meshes[mesh_ID] = NULL;
                }
            }

            if (Meshes::get_changes(mesh_ID) == Meshes::Changes::Created) {
                if (m_state->meshes.size() <= mesh_ID)
                    m_state->meshes.resize(Meshes::capacity());
                m_state->meshes[mesh_ID] = load_mesh(context, mesh_ID, m_state->triangle_intersection_program, m_state->triangle_bounds_program);
            }
        }
    }

    { // Image updates.
        if (!Images::get_changed_images().is_empty()) {
            m_state->images.resize(Images::capacity());

            for (Images::UID image_ID : Images::get_changed_images()) {
                if (Images::has_changes(image_ID, Images::Changes::Destroyed)) {
                    if (m_state->images[image_ID]) {
                        m_state->images[image_ID]->destroy();
                        m_state->images[image_ID] = NULL;
                    }
                } else if (Images::has_changes(image_ID, Images::Changes::Created)) {
                    RTformat pixel_format = RT_FORMAT_UNKNOWN;
                    switch (Images::get_pixel_format(image_ID)) {
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
                    m_state->images[image_ID] = context->createBuffer(RT_BUFFER_INPUT, pixel_format,
                        Images::get_width(image_ID), Images::get_height(image_ID));

                    uchar4* pixel_data = static_cast<uchar4*>(m_state->images[image_ID]->map());
                    std::memcpy(pixel_data, Images::get_pixels(image_ID), m_state->images[image_ID]->getElementSize() * Images::get_pixel_count(image_ID));
                    m_state->images[image_ID]->unmap();
                    OPTIX_VALIDATE(m_state->images[image_ID]);
                } else if (Images::has_changes(image_ID, Images::Changes::PixelsUpdated)) {
                    // TODO Update buffer.
                    assert(!"Pixel update not implemented yet.\n");
                }
            }
        }
    }

    { // Texture updates.
        if (!Textures::get_changed_textures().is_empty()) {
            m_state->textures.resize(Textures::capacity());

            for (Textures::UID texture_ID : Textures::get_changed_textures()) {
                if (Textures::get_changes(texture_ID) == Textures::Changes::Destroyed) {
                    if (m_state->textures[texture_ID]) {
                        m_state->textures[texture_ID]->destroy();
                        m_state->textures[texture_ID] = NULL;
                    }
                }

                static auto convert_wrap_mode = [](WrapMode wrapmode) {
                    switch (wrapmode) {
                    case WrapMode::Clamp: return RT_WRAP_CLAMP_TO_EDGE;
                    case WrapMode::Repeat: return RT_WRAP_REPEAT;
                    }
                    return RT_WRAP_REPEAT;
                };

                if (Textures::get_changes(texture_ID) == Textures::Changes::Created) {
                    TextureSampler& texture = m_state->textures[texture_ID] = context->createTextureSampler();
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
                    texture->setBuffer(0u, 0u, m_state->images[image_id]);
                    OPTIX_VALIDATE(texture);
                }
            }
        }
    }

    { // Material updates.
        static auto upload_material = [](Materials::UID material_ID, OptiXRenderer::Material* device_materials, optix::TextureSampler* samplers) {
            OptiXRenderer::Material& device_material = device_materials[material_ID];
            Assets::Material host_material = material_ID;
            device_material.base_tint.x = host_material.get_base_tint().r;
            device_material.base_tint.y = host_material.get_base_tint().g;
            device_material.base_tint.z = host_material.get_base_tint().b;
            if (host_material.get_base_tint_texture_ID() != Textures::UID::invalid_UID()) {
                // Validate that the image has 4 channels! Otherwise OptiX goes boom boom.
                Textures::UID texture_ID = host_material.get_base_tint_texture_ID();
                assert(channel_count(Images::get_pixel_format(Textures::get_image_ID(texture_ID))) == 4);
                device_material.base_tint_texture_ID = samplers[texture_ID]->getId();
            } else
                device_material.base_tint_texture_ID = 0u;
            device_material.base_roughness = host_material.get_base_roughness();
            device_material.specularity = host_material.get_specularity() * 0.08f; // See Physically-Based Shading at Disney bottom of page 8 for why we remap. TODO Consider moving this into Cogwheel or maybe even remove completely in favor of just letting GUI handle this.
            device_material.metallic = host_material.get_metallic();
        };

        if (!Materials::get_changed_materials().is_empty()) {
            if (m_state->active_material_count < Materials::capacity()) {
                // Buffer size changed. Re-upload all parameters.
                m_state->active_material_count = Materials::capacity();
                m_state->material_parameters->setSize(m_state->active_material_count);

                OptiXRenderer::Material* device_materials = (OptiXRenderer::Material*)m_state->material_parameters->map();
                upload_material(Materials::UID::invalid_UID(), device_materials, m_state->textures.data()); // Upload invalid material params as well.
                for (Materials::UID material_ID : Materials::get_iterable())
                    upload_material(material_ID, device_materials, m_state->textures.data());
                m_state->material_parameters->unmap();
            } else {
                // Update new and changed materials. Just ignore destroyed ones.
                OptiXRenderer::Material* device_materials = (OptiXRenderer::Material*)m_state->material_parameters->map();
                for (Materials::UID material_ID : Materials::get_changed_materials())
                    if (!Materials::has_changes(material_ID, Materials::Changes::Destroyed))
                        upload_material(material_ID, device_materials, m_state->textures.data());
                m_state->material_parameters->unmap();
            }
        }
    }

    { // Light updates.
        if (!LightSources::get_changed_lights().is_empty()) {
            // Light creation helper method.
            static auto light_creation = [](LightSources::UID light_ID, unsigned int light_index, Light* device_lights,
                int& highest_area_light_index_updated) {

                switch (LightSources::get_type(light_ID)) {
                case LightSources::Type::Sphere: {
                    Scene::SphereLight host_light = light_ID;
                    Light& device_light = device_lights[light_index];

                    device_light.type = LightTypes::Sphere;

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
                    Light& device_light = device_lights[light_index];

                    device_light.type = LightTypes::Directional;

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

            if (m_state->lights.ID_to_index.size() < LightSources::capacity()) {
                // Resize the light buffer to hold the new capacity.
                m_state->lights.ID_to_index.resize(LightSources::capacity());
                m_state->lights.index_to_ID.resize(LightSources::capacity());
                m_state->lights.sources->setSize(LightSources::capacity() + 1); // + 1 to allow the environment light to be added at the end.

                // Resizing removes old data, so this as an opportunity to linearize the light data.
                Light* device_lights = (Light*)m_state->lights.sources->map();
                unsigned int light_index = 0;
                for (LightSources::UID light_ID : LightSources::get_iterable()) {
                    m_state->lights.ID_to_index[light_ID] = light_index;
                    m_state->lights.index_to_ID[light_index] = light_ID;

                    light_creation(light_ID, light_index, device_lights, highest_area_light_index_updated);
                    ++light_index;
                }

                // Append the environment map, if valid, to the list of light sources.
                if (m_state->environment.is_valid()) {
                    Light& light = device_lights[light_index++];
                    light.type = LightTypes::Sphere;
                    light.environment = m_state->environment.to_light_source(m_state->textures.data());
                }

                m_state->lights.count = light_index;
                m_state->lights.sources->unmap();
            } else {
                // Skip the environment light proxy at the end of the light buffer.
                if (m_state->environment.is_valid())
                    m_state->lights.count -= 1;

                Light* device_lights = (Light*)m_state->lights.sources->map();
                LightSources::ChangedIterator created_lights_begin = LightSources::get_changed_lights().begin();
                while (created_lights_begin != LightSources::get_changed_lights().end() &&
                    LightSources::get_changes(*created_lights_begin) != LightSources::Changes::Created)
                    ++created_lights_begin;

                // Process destroyed lights.
                for (LightSources::UID light_ID : LightSources::get_changed_lights()) {
                    if (LightSources::get_changes(light_ID) != LightSources::Changes::Destroyed)
                        continue;

                    unsigned int light_index = m_state->lights.ID_to_index[light_ID];

                    if (!LightSources::is_delta_light(light_ID))
                        highest_area_light_index_updated = max(highest_area_light_index_updated, light_index);

                    if (created_lights_begin != LightSources::get_changed_lights().end()) {
                        // Replace deleted light by new light source.
                        LightSources::UID new_light_ID = *created_lights_begin;
                        light_creation(new_light_ID, light_index, device_lights, highest_area_light_index_updated);
                        m_state->lights.ID_to_index[new_light_ID] = light_index;
                        m_state->lights.index_to_ID[light_index] = new_light_ID;

                        // Find next created light.
                        while (created_lights_begin != LightSources::get_changed_lights().end() &&
                            LightSources::get_changes(*created_lights_begin) != LightSources::Changes::Created)
                            ++created_lights_begin;
                    } else {
                        // Replace deleted light by light from the end of the array.
                        --m_state->lights.count;
                        if (light_index != m_state->lights.count) {
                            memcpy(device_lights + light_index, device_lights + m_state->lights.count, sizeof(Light));

                            // Rewire light ID and index maps.
                            m_state->lights.index_to_ID[light_index] = m_state->lights.index_to_ID[m_state->lights.count];
                            m_state->lights.ID_to_index[m_state->lights.index_to_ID[light_index]] = light_index;

                            highest_area_light_index_updated = max(highest_area_light_index_updated, m_state->lights.count);
                        }
                    }
                }

                for (LightSources::UID light_ID : Iterable<LightSources::ChangedIterator>(created_lights_begin, LightSources::get_changed_lights().end())) {
                    if (LightSources::get_changes(light_ID) != LightSources::Changes::Created)
                        continue;
                    
                    unsigned int light_index = m_state->lights.count++;
                    m_state->lights.ID_to_index[light_ID] = light_index;
                    m_state->lights.index_to_ID[light_index] = light_ID;

                    light_creation(light_ID, light_index, device_lights, highest_area_light_index_updated);
                }

                // Append the environment map, if valid, to the list of light sources.
                if (m_state->environment.is_valid()) {
                    Light& light = device_lights[m_state->lights.count++];
                    light.type = LightTypes::Sphere;
                    light.environment = m_state->environment.to_light_source(m_state->textures.data());
                }

                m_state->lights.sources->unmap();
            }
            
            context["g_light_count"]->setInt(m_state->lights.count);
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
        for (SceneNodes::UID node_ID : SceneNodes::get_changed_nodes()) {
            if (!SceneNodes::has_changes(node_ID, SceneNodes::Changes::Transform) )
                continue;

            if (node_ID < m_state->transforms.size()) { // TODO(avh) Assert instead. This should always be true. Possibly move the transform update down past the model updates though as they are the ones that creates the transforms.
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
                if (m_state->transforms[model.scene_node_ID]) {
                    optix::Transform optixTransform = m_state->transforms[model.scene_node_ID];
                    m_state->root_node->removeChild(optixTransform);
                    optixTransform->destroy();
                    m_state->transforms[model.scene_node_ID] = NULL;

                    models_changed = true;
                }
            }

            if (MeshModels::get_changes(model_ID) == MeshModels::Changes::Created) {
                MeshModel model = MeshModels::get_model(model_ID);
                optix::Transform transform = load_model(context, model, m_state->meshes.data(), m_state->default_material);
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