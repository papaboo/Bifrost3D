// OptiX renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/Defines.h>
#include <OptiXRenderer/CUDAUtils.h>
#include <OptiXRenderer/Managers/AccelerationStructureManager.h>
#include <OptiXRenderer/OptiXHostUtils.h>
#include <OptiXRenderer/PtxLoader.h>
#include <OptiXRenderer/Renderer.h>
#include <OptiXRenderer/Types.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace OptiXRenderer {

// Record for the shader binding table
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

const int MAX_NEE_RNG_SAMPLE_OFFSETS = 256;

struct Renderer::Implementation {

    typedef SbtRecord<RayGenData> RayGenSbtRecord;
    typedef SbtRecord<MissShaderData> MissSbtRecord;
    typedef SbtRecord<HitsShaderData> HitsSbtRecord;

    RendererID m_renderer_ID;

    CUcontext m_cuda_context;
    OptixDeviceContext m_context = nullptr;
    OptixModule m_module = nullptr;

    Managers::AccelerationStructureManager m_acceleration_structures;

    // Per renderer pipeline state.
    OptixPipeline m_pipeline = nullptr;
    DevicePtr<PipelineParams> m_pipeline_params;

    RayGenSbtRecord m_raygen_cpu_record;
    DevicePtr<RayGenSbtRecord> m_raygen_stb_record;
    DevicePtr<MissSbtRecord> m_miss_stb_record;
    DevicePtr<HitsSbtRecord> m_hits_stb_record;
    OptixShaderBindingTable m_shader_binding_table = {};

    AIDenoiserFlags AI_denoiser_flags = AIDenoiserFlag::Default;

    // Per camera members.
    struct CameraState {
        uint2 frame_size;
        DeviceArray<AccumulationElementType> accumulation_buffer = DeviceArray<AccumulationElementType>();
        unsigned int accumulations;
        unsigned int max_accumulation_count;
        unsigned int max_bounce_count;
        Matrix4x4f inverse_projection_matrix;
        Matrix4x4f inverse_view_projection_matrix;
        Backend backend;

        inline void clear() {
            frame_size = { 0u, 0u };
            accumulation_buffer.resize(0);
            accumulations = 0u;
            max_accumulation_count = UINT_MAX;
            max_bounce_count = 4;
            inverse_projection_matrix = Matrix4x4f::identity();
            inverse_view_projection_matrix = Matrix4x4f::identity();
            backend = Backend::None;
        }
    };

    std::vector<CameraState> per_camera_state;
    inline CameraState& safe_camera_state_access(int camera_ID) {
        if (per_camera_state.size() <= camera_ID) {
            size_t old_size = per_camera_state.size();
            per_camera_state.resize(Cameras::capacity());
            for (size_t i = old_size; i < per_camera_state.size(); ++i)
                per_camera_state[i].clear();
        }
        return per_camera_state[camera_ID];
    }

    // Per scene state.
    struct {
        PathRegularizationSettings path_regularization;
        unsigned int next_event_sample_count;
    } scene;

    static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
        std::string msg = std::string("[") + std::to_string(level) + "][" + tag + "]: " + message;
        printf("OptiXRenderer log: %s\n", msg.c_str());
    }

    Implementation(CUcontext cuda_context, const std::filesystem::path& data_directory, RendererID renderer_ID)
        : m_cuda_context(cuda_context), m_renderer_ID(RendererID::invalid_UID()) {

        int cuda_device_id;
        CUDA_CHECK(cudaGetDevice(&cuda_device_id));
        cudaDeviceProp device_properties;
        CUDA_CHECK(cudaGetDeviceProperties(&device_properties, cuda_device_id));

        int optix_major = OPTIX_VERSION / 10000, optix_minor = (OPTIX_VERSION % 10000) / 100;
        printf("OptiX %u.%u renderer using device %u: '%s' with compute capability %u.%u.\n",
            optix_major, optix_minor, cuda_device_id, device_properties.name, device_properties.major, device_properties.minor);

        // For error reporting from OptiX creation functions
        const size_t max_error_log_size = 2048;
        size_t error_log_size = max_error_log_size;
        char error_log[2048];

        auto check_error_with_log = [&](OptixResult error, const std::string& file, int line) {
            if (error != OPTIX_SUCCESS) {
                std::string message = "[file:" + file + " line:" + std::to_string(line) + "] CUDA error: " + optixGetErrorName(error) + "\n";
                if (error_log_size < max_error_log_size)
                    message += "log: " + std::string(error_log, error_log_size);
                printf("%s.\n", message.c_str());
                throw std::exception(message.c_str(), error);
            }
            error_log_size = max_error_log_size; // Reset error log size to be ready for next call.
        };
#define OPTIX_LOG_CHECK(error) check_error_with_log(error, __FILE__,__LINE__)

        { // Setup context
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
#ifdef _DEBUG
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;
#endif // _DEBUG
            OPTIX_CHECK(optixDeviceContextCreate(m_cuda_context, &options, &m_context));
        }

        m_acceleration_structures = Managers::AccelerationStructureManager(m_context);

        OptixPipelineCompileOptions pipeline_compile_options = {};
        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
                                                       | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 2;
        pipeline_compile_options.numAttributeValues = 2;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "g_params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

        { // Setup module
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

            std::string ptx_source = PtxLoader::load_ptx("Shading/RGSolidColor.cu");

            OPTIX_LOG_CHECK(optixModuleCreate(m_context,&module_compile_options, &pipeline_compile_options,
                ptx_source.c_str(), ptx_source.size(), error_log, &error_log_size, &m_module));
        }

        OptixProgramGroup raygen_program = nullptr;
        OptixProgramGroup miss_program = nullptr;
        OptixProgramGroup hit_programs = nullptr;
        { // Create programs
            OptixProgramGroupOptions program_group_options = {};

            OptixProgramGroupDesc raygen_program_desc = {};
            raygen_program_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_program_desc.raygen.module = m_module;
            raygen_program_desc.raygen.entryFunctionName = "__raygen__radiance";
            OPTIX_LOG_CHECK(optixProgramGroupCreate(m_context, &raygen_program_desc, 1, &program_group_options,
                error_log, &error_log_size, &raygen_program));

            OptixProgramGroupDesc miss_program_desc = {};
            miss_program_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_program_desc.miss.module = m_module;
            miss_program_desc.miss.entryFunctionName = "__miss__radiance";
            OPTIX_LOG_CHECK(optixProgramGroupCreate(m_context, &miss_program_desc, 1, &program_group_options,
                error_log, &error_log_size, &miss_program));

            OptixProgramGroupDesc hit_programs_desc = {};
            hit_programs_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_programs_desc.hitgroup.moduleCH = m_module;
            hit_programs_desc.hitgroup.entryFunctionNameCH= "__closesthit__radiance";
            OPTIX_LOG_CHECK(optixProgramGroupCreate(m_context, &hit_programs_desc, 1, &program_group_options,
                error_log, &error_log_size, &hit_programs));
        }

        { // Link pipeline
            unsigned int max_traversable_graph_depth = 2; // TLAS + BLAS
            const uint32_t max_trace_depth = 1; // The closest hit program isn't recursive, but returns next ray generation information back to the raygen program, so we never exceed a depth of one.
            OptixProgramGroup program_entries[] = { raygen_program, miss_program, hit_programs };
            unsigned int program_count = sizeof(program_entries) / sizeof(program_entries[0]);

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            OPTIX_LOG_CHECK(optixPipelineCreate(m_context, &pipeline_compile_options, &pipeline_link_options,
                program_entries, program_count, error_log, &error_log_size, &m_pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_entries)
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, m_pipeline));

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                0,  // maxCCDepth
                0,  // maxDCDEpth
                &direct_callable_stack_size_from_traversal, &direct_callable_stack_size_from_state, &continuation_stack_size));

            OPTIX_CHECK(optixPipelineSetStackSize(m_pipeline, direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state, continuation_stack_size,
                max_traversable_graph_depth));

            m_pipeline_params = DevicePtr<PipelineParams>::create();
        }

        { // Set up shader binding table
            m_raygen_cpu_record = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_program, &m_raygen_cpu_record));
            m_raygen_stb_record = DevicePtr<RayGenSbtRecord>::create(m_raygen_cpu_record);

            MissSbtRecord miss_cpu_record = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_program, &miss_cpu_record));
            m_miss_stb_record = DevicePtr<MissSbtRecord>::create(miss_cpu_record);

            HitsSbtRecord hits_cpu_record = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(hit_programs, &hits_cpu_record));
            m_hits_stb_record = DevicePtr<HitsSbtRecord>::create(hits_cpu_record);

            m_shader_binding_table.raygenRecord = m_raygen_stb_record.device_ptr();
            m_shader_binding_table.missRecordBase = m_miss_stb_record.device_ptr();
            m_shader_binding_table.missRecordStrideInBytes = sizeof(MissSbtRecord);
            m_shader_binding_table.missRecordCount = 1;
            m_shader_binding_table.hitgroupRecordBase = m_hits_stb_record.device_ptr();
            m_shader_binding_table.hitgroupRecordStrideInBytes = sizeof(HitsSbtRecord);
            m_shader_binding_table.hitgroupRecordCount = 1;
        }

        // Clean up the state.
        // The pipeline will keep the shader programs alive.
        OPTIX_CHECK(optixProgramGroupDestroy(raygen_program));
        OPTIX_CHECK(optixProgramGroupDestroy(miss_program));
        OPTIX_CHECK(optixProgramGroupDestroy(hit_programs));

        { // Setup scene
            scene.next_event_sample_count = 3;

            // Setup path regularization for fast convergence.
            scene.path_regularization.PDF_scale = 0.5f;
            scene.path_regularization.scale_decay = 0.0f;
        }

        // Set renderer ID at the end of the constructor, so we can use it to flag that the constructor ran to the end and the renderer is valid.
        m_renderer_ID = renderer_ID;
    }

    ~Implementation() {
        // Destroy pipeline state
        OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
        OPTIX_CHECK(optixModuleDestroy(m_module));
        OPTIX_CHECK(optixDeviceContextDestroy(m_context));
    }

    bool is_valid() { return m_renderer_ID != RendererID::invalid_UID(); }

    void handle_updates() {
        { // Camera updates.
            for (CameraID cam_ID : Cameras::get_changed_cameras()) {
                auto camera_changes = Cameras::get_changes(cam_ID);
                if (camera_changes.contains(Cameras::Change::Destroyed)) {
                    if (cam_ID < per_camera_state.size())
                        per_camera_state[cam_ID].clear();

                } else {

                    bool camera_initialized = per_camera_state.size() > cam_ID && per_camera_state[cam_ID].accumulation_buffer.size() != 0u;
                    bool uses_optix_renderer = m_renderer_ID == Cameras::get_renderer_ID(cam_ID);
                    bool create_optix_renderer = uses_optix_renderer && camera_changes.is_set(Cameras::Change::Created);
                    bool switch_to_optix_renderer = uses_optix_renderer && camera_changes.is_set(Cameras::Change::Renderer);

                    if (!camera_initialized && (create_optix_renderer || switch_to_optix_renderer)) {
                        auto& camera_state = safe_camera_state_access(cam_ID);

                        // Preserve backend if set from outside before handle_updates is called. Yuck!
                        if (camera_state.backend == Backend::None)
                            camera_state.backend = Backend::PathTracing;
                    }
                }
            }
        }

        m_acceleration_structures.handle_updates();
    }

    inline PipelineParams prepare_pipeline_params(CameraID camera_ID, Vector2i frame_size) {
        int frame_width = frame_size.x;
        int frame_height = frame_size.y;
        auto& camera_state = per_camera_state[camera_ID];
        PipelineParams pipeline_params = {};

        { // Update camera state
            // Resize screen buffers if necessary.
            if (frame_width != camera_state.frame_size.x || frame_height != camera_state.frame_size.y) {
                camera_state.accumulation_buffer.resize(frame_width * frame_height);

                camera_state.frame_size = make_uint2(frame_width, frame_height);
                camera_state.accumulations = 0u;
            }

            { // Upload camera parameters.
                // Check if the camera transform or projection matrix changed and, if so, reset accumulation.
                Matrix4x4f inverse_projection_matrix = Cameras::get_inverse_projection_matrix(camera_ID);
                Matrix4x4f inverse_view_projection_matrix = Cameras::get_inverse_view_projection_matrix(camera_ID);;

                if (camera_state.inverse_view_projection_matrix != inverse_view_projection_matrix)
                    camera_state.accumulations = 0u;

                camera_state.inverse_view_projection_matrix = inverse_view_projection_matrix;

                pipeline_params.inverse_projection_matrix = inverse_projection_matrix;
                pipeline_params.inverse_view_projection_matrix = inverse_view_projection_matrix;

                Matrix3x3f world_to_view_rotation = to_matrix3x3(Cameras::get_inverse_view_transform(camera_ID).rotation);
                pipeline_params.view_to_world_rotation = world_to_view_rotation;
            }
        }

        pipeline_params.frame_width = frame_width;
        pipeline_params.frame_height = frame_height;
        pipeline_params.accumulation_buffer = camera_state.accumulation_buffer.data();
        pipeline_params.accumulations = camera_state.accumulations;
        pipeline_params.max_bounce_count = camera_state.max_bounce_count;
        pipeline_params.path_regularization_PDF_scale = scene.path_regularization.PDF_scale_at_accumulation(pipeline_params.accumulations);

        { // Scene uploads
            SceneRoot scene_root = Cameras::get_scene_ID(camera_ID);

            pipeline_params.scene.traversable = m_acceleration_structures.get_acceleration_structure();
            pipeline_params.scene.environment_tint = scene_root.get_environment_tint();
        }

        return pipeline_params;
    }

    unsigned int render(CameraID camera_ID, cudaGraphicsResource* backbuffer, Vector2i frame_size) {
        RGBA16f* device_pixels;
        size_t byte_count;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&device_pixels, &byte_count, backbuffer));

        PipelineParams pipeline_params = prepare_pipeline_params(camera_ID, frame_size);
        pipeline_params.output_buffer = device_pixels;
        m_pipeline_params.upload(pipeline_params);

        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        OPTIX_CHECK(optixLaunch(m_pipeline, stream, m_pipeline_params.device_ptr(), sizeof(PipelineParams), &m_shader_binding_table, frame_size.x, frame_size.y, 1));

        return 1u;
    }

    std::vector<Screenshot> request_auxiliary_buffers(CameraID camera_ID, Cameras::ScreenshotContent content_requested, Vector2i frame_size) {
        return std::vector<Screenshot>();
    }
};

// ------------------------------------------------------------------------------------------------
// Renderer
// ------------------------------------------------------------------------------------------------

Renderer* Renderer::initialize(CUcontext cuda_context, const std::filesystem::path& data_directory) {
    try {
        Renderer* r = new Renderer(cuda_context, data_directory);
        if (r->m_impl->is_valid())
            return r;
        else {
            delete r;
            return nullptr;
        }
    }
    catch (std::exception e) {
        printf("OptiXRenderer failed to initialize:\n%s\n", e.what());
        return nullptr;
    }
}

Renderer::Renderer(CUcontext cuda_context, const std::filesystem::path& data_directory)
    : m_renderer_ID(Bifrost::Core::Renderers::create("OptiXRenderer")),
    m_impl(new Implementation(cuda_context, data_directory, m_renderer_ID)) {}

Renderer::~Renderer() {
    Bifrost::Core::Renderers::destroy(m_renderer_ID);
    delete m_impl;
}

unsigned int Renderer::get_next_event_sample_count(Bifrost::Scene::SceneRootID scene_root_ID) const { return m_impl->scene.next_event_sample_count; }
void Renderer::set_next_event_sample_count(Bifrost::Scene::SceneRootID scene_root_ID, unsigned int sample_count) {
    m_impl->scene.next_event_sample_count = min(sample_count, MAX_NEE_RNG_SAMPLE_OFFSETS); // Limit the number of samples to the number of random offsets available.
}

unsigned int Renderer::get_max_bounce_count(CameraID camera_ID) const {
    return m_impl->safe_camera_state_access(camera_ID).max_bounce_count;
}
void Renderer::set_max_bounce_count(CameraID camera_ID, unsigned int bounce_count) {
    m_impl->safe_camera_state_access(camera_ID).max_bounce_count = bounce_count;
}

unsigned int Renderer::get_max_accumulation_count(CameraID camera_ID) const {
    return m_impl->safe_camera_state_access(camera_ID).max_accumulation_count;
}
void Renderer::set_max_accumulation_count(CameraID camera_ID, unsigned int accumulation_count) {
    m_impl->safe_camera_state_access(camera_ID).max_accumulation_count = accumulation_count;
}

Backend Renderer::get_backend(CameraID camera_ID) const {
    return m_impl->safe_camera_state_access(camera_ID).backend;
}

void Renderer::set_backend(CameraID camera_ID, Backend backend) {
    m_impl->safe_camera_state_access(camera_ID).backend = backend;
}

PathRegularizationSettings Renderer::get_path_regularization_settings() const { return m_impl->scene.path_regularization; }
void Renderer::set_path_regularization_settings(PathRegularizationSettings settings) { m_impl->scene.path_regularization = settings; }

AIDenoiserFlags Renderer::get_AI_denoiser_flags() const { return m_impl->AI_denoiser_flags; }
void Renderer::set_AI_denoiser_flags(AIDenoiserFlags flags) { m_impl->AI_denoiser_flags = flags; }

void Renderer::handle_updates() { m_impl->handle_updates(); }

unsigned int Renderer::render(CameraID camera_ID, cudaGraphicsResource* backbuffer, Vector2i frame_size) {
    return m_impl->render(camera_ID, backbuffer, frame_size);
}

std::vector<Screenshot> Renderer::request_auxiliary_buffers(CameraID camera_ID, Cameras::ScreenshotContent content_requested, Vector2i frame_size) {
    return m_impl->request_auxiliary_buffers(camera_ID, content_requested, frame_size);
}

} // NS OptiXRenderer