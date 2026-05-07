// OptiX renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <OptiXRenderer/Defines.h>
#include <OptiXRenderer/CUDAUtils.h>
#include <OptiXRenderer/PtxLoader.h>
#include <OptiXRenderer/Renderer.h>
#include <OptiXRenderer/Types.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace OptiXRenderer {

void optix_check_error(OptixResult error, const std::string& file, int line) {
    if (error != OPTIX_SUCCESS) {
        std::string message = "[file:" + file + " line:" + std::to_string(line) + "] CUDA error: " + optixGetErrorName(error);
        printf("%s.\n", message.c_str());
        throw std::exception(message.c_str(), error);
    }
}
#define OPTIX_CHECK_ERROR(error) optix_check_error(error, __FILE__,__LINE__)

const int MAX_NEE_RNG_SAMPLE_OFFSETS = 256;

struct Renderer::Implementation {

    typedef SbtRecord<RayGenData> RayGenSbtRecord;
    typedef SbtRecord<MissShaderData> MissSbtRecord;

    RendererID m_renderer_ID;

    CUcontext m_cuda_context;
    OptixDeviceContext m_context = nullptr;
    OptixModule m_module = nullptr;

    // Per renderer pipeline state.
    OptixPipeline m_pipeline = nullptr;
    DevicePtr<CameraState> m_pipeline_params;

    RayGenSbtRecord m_raygen_cpu_record;
    DevicePtr<RayGenSbtRecord> m_raygen_stb_record;
    DevicePtr<MissSbtRecord> m_miss_stb_record;
    OptixShaderBindingTable m_shader_binding_table = {};

    AIDenoiserFlags AI_denoiser_flags = AIDenoiserFlag::Default;

    // Per scene state.
    struct {
// #if PRESAMPLE_ENVIRONMENT_MAP
//         PresampledEnvironmentMap environment;
// #else
//         EnvironmentMap environment;
// #endif
        PathRegularizationSettings path_regularization;
        unsigned int next_event_sample_count;
    } scene;

    static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
        std::string msg = std::string("[") + std::to_string(level) + "][" + tag + "]: " + message;
        printf("OptiXRenderer log: %s\n", msg.c_str());
    }

    Implementation(CUcontext cuda_context, const std::filesystem::path& data_directory, RendererID renderer_ID)
        : m_cuda_context(cuda_context), m_renderer_ID(RendererID::invalid_UID()) {

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
#define OPTIX_CHECK(error) check_error_with_log(error, __FILE__,__LINE__)

        { // Setup context
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
#ifdef _DEBUG
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;
#endif // _DEBUG
            OPTIX_CHECK(optixDeviceContextCreate(m_cuda_context, &options, &m_context));
        }

        OptixPipelineCompileOptions pipeline_compile_options = {};
        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues = 2;
        pipeline_compile_options.numAttributeValues = 2;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        { // Setup module
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

            std::string ptx_source = PtxLoader::load_ptx("Shading/RGSolidColor.cu");

            OPTIX_CHECK(optixModuleCreateFromPTX(m_context,&module_compile_options, &pipeline_compile_options,
                ptx_source.c_str(), ptx_source.size(), error_log, &error_log_size, &m_module));
        }

        OptixProgramGroup raygen_program = nullptr;
        OptixProgramGroup miss_program = nullptr;
        { // Create programs
            OptixProgramGroupOptions program_group_options = {};

            OptixProgramGroupDesc raygen_program_desc = {};
            raygen_program_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_program_desc.raygen.module = m_module;
            raygen_program_desc.raygen.entryFunctionName = "__raygen__solid_color";
            OPTIX_CHECK(optixProgramGroupCreate(m_context, &raygen_program_desc, 1, &program_group_options,
                error_log, &error_log_size, &raygen_program));

            // Leave miss group's module and entry function name null
            OptixProgramGroupDesc miss_program_desc = {};
            miss_program_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            OPTIX_CHECK(optixProgramGroupCreate(m_context, &miss_program_desc, 1, &program_group_options,
                error_log, &error_log_size, &miss_program));
        }

        { // Link pipeline
            unsigned int max_traversable_graph_depth = 2;
            const uint32_t max_trace_depth = 1; // The closest hit program isn't recursive, but returns next ray generation information back to the raygen program, so we never exceed a depth of one.
            OptixProgramGroup program_entries[] = { raygen_program };
            unsigned int program_count = sizeof(program_entries) / sizeof(program_entries[0]);

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            OPTIX_CHECK(optixPipelineCreate(m_context, &pipeline_compile_options, &pipeline_link_options,
                program_entries, program_count, error_log, &error_log_size, &m_pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_entries)
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));

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

            m_pipeline_params = DevicePtr<CameraState>::create();
        }

        { // Set up shader binding table
            m_raygen_cpu_record = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_program, &m_raygen_cpu_record));
            m_raygen_stb_record = DevicePtr<RayGenSbtRecord>::create(m_raygen_cpu_record);

            MissSbtRecord miss_cpu_record;
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_program, &miss_cpu_record));
            m_miss_stb_record = DevicePtr<MissSbtRecord>::create(miss_cpu_record);

            m_shader_binding_table.raygenRecord = m_raygen_stb_record.ptr;
            m_shader_binding_table.missRecordBase = m_miss_stb_record.ptr;
            m_shader_binding_table.missRecordStrideInBytes = sizeof(MissSbtRecord);
            m_shader_binding_table.missRecordCount = 1;
        }

        // Clean up the state.
        // The pipeline will keep the shader programs alive.
        OPTIX_CHECK(optixProgramGroupDestroy(raygen_program));
        OPTIX_CHECK(optixProgramGroupDestroy(miss_program));

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
        OPTIX_CHECK_ERROR(optixPipelineDestroy(m_pipeline));
        OPTIX_CHECK_ERROR(optixModuleDestroy(m_module));
        OPTIX_CHECK_ERROR(optixDeviceContextDestroy(m_context));
    }

    bool is_valid() { return m_renderer_ID != RendererID::invalid_UID(); }

    void handle_updates() { }

    unsigned int render(CameraID camera_ID, cudaGraphicsResource* backbuffer, Vector2i frame_size) {
        int width = frame_size.x;
        int height = frame_size.y;

        half4* device_pixels;
        size_t byte_count;
        THROW_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&device_pixels, &byte_count, backbuffer));

        { // Temp upload to ray gen program
            SceneRoot scene_root = Cameras::get_scene_ID(camera_ID);
            RGB env_tint = scene_root.get_environment_tint();

            m_raygen_cpu_record.data.r = env_tint.r;
            m_raygen_cpu_record.data.g = env_tint.g;
            m_raygen_cpu_record.data.b = env_tint.b;
            m_raygen_stb_record.upload(m_raygen_cpu_record);
        }

        CameraState camera_state;
        camera_state.output_buffer = device_pixels;
        camera_state.output_buffer_width = width;
        camera_state.output_buffer_height = height;
        m_pipeline_params.upload(camera_state);

        CUstream stream;
        THROW_CUDA_ERROR(cudaStreamCreate(&stream));

        OPTIX_CHECK_ERROR(optixLaunch(m_pipeline, stream, m_pipeline_params.ptr, sizeof(CameraState), &m_shader_binding_table, width, height, 1));

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
    // TODO m_impl->conditional_per_camera_state_resize(camera_ID);
    // TODO return m_impl->per_camera_state[camera_ID].max_bounce_count;
    return 3u;
}
void Renderer::set_max_bounce_count(CameraID camera_ID, unsigned int bounce_count) {
    // TODO m_impl->conditional_per_camera_state_resize(camera_ID);
    // TODO m_impl->per_camera_state[camera_ID].max_bounce_count = bounce_count;
}

unsigned int Renderer::get_max_accumulation_count(CameraID camera_ID) const {
    // TODO m_impl->conditional_per_camera_state_resize(camera_ID);
    // TODO return m_impl->per_camera_state[camera_ID].max_accumulation_count;
    return 1024u;
}
void Renderer::set_max_accumulation_count(CameraID camera_ID, unsigned int accumulation_count) {
    // TODO m_impl->conditional_per_camera_state_resize(camera_ID);
    // TODO m_impl->per_camera_state[camera_ID].max_accumulation_count = accumulation_count;
}

Backend Renderer::get_backend(CameraID camera_ID) const {
    // TODO m_impl->conditional_per_camera_state_resize(camera_ID);
    // TODO return m_impl->per_camera_state[camera_ID].backend;
    return Backend::PathTracing;
}

void Renderer::set_backend(CameraID camera_ID, Backend backend) {
    // TODO
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