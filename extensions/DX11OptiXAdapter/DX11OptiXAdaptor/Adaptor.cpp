// Adaptor for the OptiX renderer that allows it to render to a DX11 buffer.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#include <DX11OptiXAdaptor/Adaptor.h>
#include <DX11Renderer/Renderer.h>
#include <DX11Renderer/Utils.h>
#include <OptiXRenderer/Defines.h>
#include <OptiXRenderer/Renderer.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include <optixu/optixpp_namespace.h>

#include <DX11Renderer/Types.h>

#include <codecvt>

using namespace DX11Renderer;

// #define DISABLE_INTEROP 1

namespace DX11OptiXAdaptor {

inline void throw_cuda_error(cudaError_t error, const std::string& file, int line) {
    if (error != cudaSuccess) {
        std::string message = "[file:" + file + " line:" + std::to_string(line) + "] CUDA errror: " + std::string(cudaGetErrorString(error));
        printf("%s.\n", message.c_str());
        throw std::exception(message.c_str(), error);
    }
}

#define THROW_CUDA_ERROR(error) throw_cuda_error(error, __FILE__,__LINE__)

class Adaptor::Implementation {
public:
    int m_cuda_device_ID = -1;
    ID3D11Device1& m_device;
    ODeviceContext1 m_render_context;

    ORenderTargetView m_backbuffer_RTV;
    OShaderResourceView m_backbuffer_SRV;

    struct {
        OShaderResourceView dx_SRV;
        optix::Buffer optix_buffer;
        cudaGraphicsResource* cuda_buffer;
        int capacity;
        int width;
        int height;
    } m_render_target;

    OBuffer m_constant_buffer;
    OVertexShader m_vertex_shader;
    OPixelShader m_pixel_shader;

    OptiXRenderer::Renderer* m_optix_renderer;

    Implementation(ID3D11Device1& device, int width_hint, int height_hint, const std::wstring& data_folder_path, Bifrost::Core::Renderers::UID renderer_ID)
        : m_device(device), m_backbuffer_RTV(nullptr), m_backbuffer_SRV(nullptr) {

        device.GetImmediateContext1(&m_render_context);

        { // Get CUDA device from DX11 context.
            ODXGIDevice dxgi_device = nullptr;
            THROW_DX11_ERROR(device.QueryInterface(IID_PPV_ARGS(&dxgi_device)));

            ODXGIAdapter adapter = nullptr;
            THROW_DX11_ERROR(dxgi_device->GetAdapter(&adapter));

            THROW_CUDA_ERROR(cudaD3D11GetDevice(&m_cuda_device_ID, adapter));

            // Create OptiX Renderer on device.
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            std::string data_path = converter.to_bytes(data_folder_path);
            m_optix_renderer = OptiXRenderer::Renderer::initialize(m_cuda_device_ID, width_hint, height_hint, data_path, renderer_ID);
        }

        THROW_DX11_ERROR(create_constant_buffer(m_device, sizeof(float) * 4, &m_constant_buffer));

        m_render_target = {};

        { // Init shaders.
            const char* vertex_src =
                "float4 main_vs(uint vertex_ID : SV_VertexID) : SV_POSITION {\n"
                "   float4 position;\n"
                "   // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }\n"
                "   position.x = vertex_ID == 2 ? 3 : -1;\n"
                "   position.y = vertex_ID == 0 ? -3 : 1;\n"
                "   position.zw = float2(0.5f, 1.0f);\n"
                "   return position;\n"
                "}\n";

            const char* pixel_src =
                "StructuredBuffer<half4> pixels : register(t0);\n"
                "cbuffer constants : register(b0) {\n"
                "    int2 viewport_size;\n"
                "    int2 pixels_size;\n"
                "};\n"
                "float4 main_ps(float4 pixel_pos : SV_POSITION) : SV_TARGET {\n"
                "   return pixels[int(pixel_pos.x) + int(viewport_size.y - pixel_pos.y) * pixels_size.x];\n"
                "}\n";

            static auto compile_shader = [](const char* const shader_src, const char* const target, const char* const entry_point) -> OBlob {
                OBlob shader_bytecode;
                OBlob error_messages = nullptr;
                HRESULT hr = D3DCompile(shader_src, strlen(shader_src), nullptr,
                    nullptr, // macroes
                    D3D_COMPILE_STANDARD_FILE_INCLUDE,
                    entry_point,
                    target,
                    0, // Flags
                    0, // More flags. Unused.
                    &shader_bytecode,
                    &error_messages);
                if (FAILED(hr)) {
                    if (error_messages != nullptr)
                        printf("Shader error: '%s'\n", (char*)error_messages->GetBufferPointer());
                    else
                        printf("Unknown error occured when trying to compile blit shader in DX11OptiXAdaptor.\n");
                    throw std::exception("Shader compilation error.");
                }
                return shader_bytecode;
            };

            OBlob vertex_shader_bytecode = compile_shader(vertex_src, "vs_5_0", "main_vs");
            THROW_DX11_ERROR(m_device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_bytecode), nullptr, &m_vertex_shader));

            OBlob pixel_shader_bytecode = compile_shader(pixel_src, "ps_5_0", "main_ps");
            THROW_DX11_ERROR(m_device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_bytecode), nullptr, &m_pixel_shader));
        }
    }

    ~Implementation() {
        m_render_target.optix_buffer = nullptr;
#ifndef DISABLE_INTEROP
        if (m_render_target.cuda_buffer != nullptr)
            THROW_CUDA_ERROR(cudaGraphicsUnregisterResource(m_render_target.cuda_buffer));
#endif
        delete m_optix_renderer;
    }

    void handle_updates() {
        m_optix_renderer->handle_updates();
    }

    RenderedFrame render(Bifrost::Scene::Cameras::UID camera_ID, int width, int height) {
        if (m_render_target.width < width || m_render_target.height < height) {
            int buffer_width = std::max(m_render_target.width, width);
            int buffer_height = std::max(m_render_target.height, height);

            { // Backbuffer.
                m_backbuffer_RTV.release();
                m_backbuffer_SRV.release();

                create_texture_2D(m_device, DXGI_FORMAT_R16G16B16A16_FLOAT, buffer_width, buffer_height, &m_backbuffer_SRV, nullptr, &m_backbuffer_RTV);
            }

            resize_render_target(buffer_width, buffer_height);
        }

        unsigned int iteration_count = 0;
#ifdef DISABLE_INTEROP
        { // Render and copy to backbuffer.
            iteration_count = m_optix_renderer->render(camera_ID, m_render_target.optix_buffer, width, height);

            void* cpu_buffer = m_render_target.optix_buffer->map();
            OResource dx_buffer = nullptr;
            m_render_target.dx_SRV->GetResource(&dx_buffer);
            m_render_context->UpdateSubresource(dx_buffer, 0, nullptr, cpu_buffer, 0, 0);
            m_render_target.optix_buffer->unmap();
        }
#else
        { // Render to render target.
            THROW_CUDA_ERROR(cudaGraphicsMapResources(1, &m_render_target.cuda_buffer)); // Done before rendering?
            ushort4* pixels;
            size_t byte_count;
            THROW_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&pixels, &byte_count, m_render_target.cuda_buffer));

            OPTIX_VALIDATE(m_optix_renderer->get_context());
            OPTIX_VALIDATE(m_render_target.optix_buffer);
            m_render_target.optix_buffer->setDevicePointer(m_cuda_device_ID, pixels);
            iteration_count = m_optix_renderer->render(camera_ID, m_render_target.optix_buffer, width, height);

            THROW_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_render_target.cuda_buffer));
        }
#endif

        { // Render to back buffer.
            m_render_context->OMSetRenderTargets(1, &m_backbuffer_RTV, nullptr);

            // Create and set the viewport.
            D3D11_VIEWPORT dx_viewport;
            dx_viewport.TopLeftX = dx_viewport.TopLeftY = 0.0f;
            dx_viewport.Width = float(width);
            dx_viewport.Height = float(height);
            dx_viewport.MinDepth = 0.0f;
            dx_viewport.MaxDepth = 1.0f;
            m_render_context->RSSetViewports(1, &dx_viewport);

            m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            m_render_context->OMSetBlendState(nullptr, nullptr, 0xffffffff);

            int constant_data[4] = { width, height, m_render_target.width, m_render_target.height };
            m_render_context->UpdateSubresource(m_constant_buffer, 0, nullptr, &constant_data, 0, 0);

            m_render_context->VSSetShader(m_vertex_shader, 0, 0);
            m_render_context->PSSetShader(m_pixel_shader, 0, 0);

            m_render_context->PSSetConstantBuffers(0, 1, &m_constant_buffer);
            m_render_context->PSSetShaderResources(0, 1, &m_render_target.dx_SRV);

            m_render_context->Draw(3, 0);
        }

        Bifrost::Math::Rect<int> viewport = { 0, 0, width, height };
        return { m_backbuffer_SRV, viewport, iteration_count };
    }

    void resize_render_target(int width, int height) {

        if (m_render_target.capacity < width * height) {
            m_render_target.dx_SRV.release();

            m_render_target.capacity = width * height;

            OBuffer dx_buffer = create_default_buffer(m_device, DXGI_FORMAT_R16G16B16A16_FLOAT, m_render_target.capacity, &m_render_target.dx_SRV);

#ifndef DISABLE_INTEROP
            // Register the buffer with CUDA.
            if (m_render_target.cuda_buffer != nullptr)
                THROW_CUDA_ERROR(cudaGraphicsUnregisterResource(m_render_target.cuda_buffer));
            THROW_CUDA_ERROR(cudaGraphicsD3D11RegisterResource(&m_render_target.cuda_buffer, dx_buffer,
                                                               cudaGraphicsRegisterFlagsNone));
            THROW_CUDA_ERROR(cudaGraphicsResourceSetMapFlags(m_render_target.cuda_buffer, cudaGraphicsMapFlagsWriteDiscard));
#endif
        }

        // Create optix buffer.
        // See https://devtalk.nvidia.com/default/topic/946870/optix/optix-4-and-cuda-interop-new-limitation-with-input-output-buffers/
        // for why the buffer is GPU_LOCAL instead of OUTPUT.
        // A GPU local buffer will obviously not work with multiple GPUs.
        optix::Context optix_context = m_optix_renderer->get_context();
        assert(optix_context->getEnabledDeviceCount() == 1);
#ifndef DISABLE_INTEROP
        m_render_target.optix_buffer = optix_context->createBufferForCUDA(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_HALF4, width, height);
#else
        m_render_target.optix_buffer = optix_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_HALF4, width, height);
#endif

        m_render_target.width = width;
        m_render_target.height = height;
    }
};

IRenderer* Adaptor::initialize(ID3D11Device1& device, int width_hint, int height_hint, const std::wstring& data_folder_path) {
    return new Adaptor(device, width_hint, height_hint, data_folder_path);
}

Adaptor::Adaptor(ID3D11Device1& device, int width_hint, int height_hint, const std::wstring& data_folder_path) {
    m_renderer_ID = Bifrost::Core::Renderers::create("OptiXRenderer");
    m_impl = new Implementation(device, width_hint, height_hint, data_folder_path, m_renderer_ID);
}

Adaptor::~Adaptor() {
    Bifrost::Core::Renderers::destroy(m_renderer_ID);
    delete m_impl;
}

OptiXRenderer::Renderer* Adaptor::get_renderer() {
    return m_impl->m_optix_renderer;
}

OptiXRenderer::Renderer* Adaptor::get_renderer() const {
    return m_impl->m_optix_renderer;
}

void Adaptor::handle_updates() {
    m_impl->handle_updates();
}

RenderedFrame Adaptor::render(Bifrost::Scene::Cameras::UID camera_ID, int width, int height) {
    return m_impl->render(camera_ID, width, height);
}

std::vector<Bifrost::Scene::Screenshot> Adaptor::request_auxiliary_buffers(Bifrost::Scene::Cameras::UID camera_ID, Bifrost::Scene::Cameras::RequestedContent content_requested, int width, int height) {
    return m_impl->m_optix_renderer->request_auxiliary_buffers(camera_ID, content_requested, width, height);
}

} // NS DX11OptiXAdaptor
