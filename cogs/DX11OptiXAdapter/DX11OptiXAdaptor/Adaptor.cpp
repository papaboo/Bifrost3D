// Adaptor for the OptiX renderer that allows it to render to a DX11 buffer.
//-------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
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

#define NOMINMAX
#include <D3D11_1.h>
#include <D3DCompiler.h>
#undef RGB

// #define DISABLE_INTEROP 1

namespace DX11OptiXAdaptor {

inline void throw_on_failure(cudaError_t error, const std::string& file, int line) {
    if (error != cudaSuccess) {
        std::string message = "[file:" + file + " line:" + std::to_string(line) + 
            "] CUDA errror: " + std::string(cudaGetErrorString(error));
        printf("%s.\n", message.c_str());
        throw std::exception(message.c_str(), error);
    }
}

#define THROW_ON_CUDA_FAILURE(error) ::DX11OptiXAdaptor::throw_on_failure(error, __FILE__,__LINE__)

class Adaptor::Implementation {
public:
    int m_cuda_device_ID = -1;
    ID3D11Device1& m_device;
    ID3D11DeviceContext1* m_render_context;

    struct {
        ID3D11ShaderResourceView* dx_SRV;
        optix::Buffer optix_buffer;
        cudaGraphicsResource* cuda_buffer;
        int capacity;
        int width;
        int height;
    } m_render_target;

    ID3D11Buffer* m_constant_buffer;
    ID3D11VertexShader* m_vertex_shader;
    ID3D11PixelShader* m_pixel_shader;

    OptiXRenderer::Renderer* m_optix_renderer;

    Implementation(ID3D11Device1& device, int width_hint, int height_hint)
        : m_device(device) {

        device.GetImmediateContext1(&m_render_context);

        { // Get CUDA device from DX11 context.
            IDXGIDevice* dxgi_device = nullptr;
            HRESULT hr = device.QueryInterface(IID_PPV_ARGS(&dxgi_device));
            THROW_ON_FAILURE(hr);

            IDXGIAdapter* adapter = nullptr;
            hr = dxgi_device->GetAdapter(&adapter);
            THROW_ON_FAILURE(hr); 
            dxgi_device->Release();

            cudaError_t error = cudaD3D11GetDevice(&m_cuda_device_ID, adapter);
            THROW_ON_CUDA_FAILURE(error);
            adapter->Release();

            // Create OptiX Renderer on device.
            m_optix_renderer = OptiXRenderer::Renderer::initialize(m_cuda_device_ID, width_hint, height_hint);
        }

        {
            HRESULT hr = DX11Renderer::create_constant_buffer(m_device, sizeof(float) * 4, &m_constant_buffer);
            THROW_ON_FAILURE(hr);
        }

        {
            m_render_target = {};
            resize_render_target(width_hint, height_hint);
        }

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
                "};\n"
                "float4 main_ps(float4 pixel_pos : SV_POSITION) : SV_TARGET {\n"
                "   return pixels[int(pixel_pos.x) + int(viewport_size.y - pixel_pos.y) * viewport_size.x];\n"
                "}\n";

            static auto compile_shader = [](const char* const shader_src, const char* const target, const char* const entry_point) -> ID3DBlob* {
                ID3DBlob* shader_bytecode;
                ID3DBlob* error_messages = nullptr;
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

            ID3DBlob* vertex_shader_bytecode = compile_shader(vertex_src, "vs_5_0", "main_vs");
            HRESULT hr = m_device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_bytecode), nullptr, &m_vertex_shader);
            THROW_ON_FAILURE(hr);
            vertex_shader_bytecode->Release();

            ID3DBlob* pixel_shader_bytecode = compile_shader(pixel_src, "ps_5_0", "main_ps");
            hr = m_device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_bytecode), nullptr, &m_pixel_shader);
            THROW_ON_FAILURE(hr);
            pixel_shader_bytecode->Release();
        }
    }

    ~Implementation() {
        m_render_target.optix_buffer = nullptr;
#ifndef DISABLE_INTEROP
        THROW_ON_CUDA_FAILURE(cudaGraphicsUnregisterResource(m_render_target.cuda_buffer));
#endif
        DX11Renderer::safe_release(&m_render_target.dx_SRV);

        DX11Renderer::safe_release(&m_constant_buffer);

        delete m_optix_renderer;
    }

    void handle_updates() {
        m_optix_renderer->handle_updates();
    }

    void render(Cogwheel::Scene::Cameras::UID camera_ID, int width, int height) {
        if (m_render_target.width != width || m_render_target.height != height)
            resize_render_target(width, height);

#ifdef DISABLE_INTEROP
        { // Render and copy to backbuffer.
            m_optix_renderer->render(camera_ID, m_render_target.optix_buffer, m_render_target.width, m_render_target.height);

            void* cpu_buffer = m_render_target.optix_buffer->map();
            ID3D11Resource* dx_buffer = nullptr;
            m_render_target.dx_SRV->GetResource(&dx_buffer);
            m_render_context->UpdateSubresource(dx_buffer, 0, nullptr, cpu_buffer, 0, 0);
            m_render_target.optix_buffer->unmap();
        }
#else
        { // Render to render target.
            cudaError_t error = cudaGraphicsMapResources(1, &m_render_target.cuda_buffer); // Done before rendering?
            THROW_ON_CUDA_FAILURE(error);
            ushort4* pixels;
            size_t byte_count;
            error = cudaGraphicsResourceGetMappedPointer((void**)&pixels, &byte_count, m_render_target.cuda_buffer);
            THROW_ON_CUDA_FAILURE(error);

            OPTIX_VALIDATE(m_optix_renderer->get_context());
            OPTIX_VALIDATE(m_render_target.optix_buffer);
            m_render_target.optix_buffer->setDevicePointer(m_cuda_device_ID, pixels);
            m_optix_renderer->render(camera_ID, m_render_target.optix_buffer, m_render_target.width, m_render_target.height);

            THROW_ON_CUDA_FAILURE(cudaGraphicsUnmapResources(1, &m_render_target.cuda_buffer));
        }
#endif

        { // Render to back buffer.
            m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            m_render_context->OMSetBlendState(nullptr, nullptr, 0xffffffff);

            m_render_context->VSSetShader(m_vertex_shader, 0, 0);
            m_render_context->PSSetShader(m_pixel_shader, 0, 0);

            m_render_context->PSSetConstantBuffers(0, 1, &m_constant_buffer);
            m_render_context->PSSetShaderResources(0, 1, &m_render_target.dx_SRV);

            m_render_context->Draw(3, 0);
        }
    }

    void resize_render_target(int width, int height) {

        if (m_render_target.capacity < width * height) {
            DX11Renderer::safe_release(&m_render_target.dx_SRV);

            m_render_target.capacity = width * height;

            D3D11_BUFFER_DESC desc = {};
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.StructureByteStride = sizeof(short) * 4;
            desc.ByteWidth = m_render_target.capacity * sizeof(short) * 4;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            // desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc.MiscFlags = 0;

            ID3D11Buffer* dx_buffer;
            HRESULT hr = m_device.CreateBuffer(&desc, nullptr, &dx_buffer);
            THROW_ON_FAILURE(hr);

            D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
            srv_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            srv_desc.Buffer.NumElements = m_render_target.capacity;

            hr = m_device.CreateShaderResourceView(dx_buffer, &srv_desc, &m_render_target.dx_SRV);
            THROW_ON_FAILURE(hr);

#ifndef DISABLE_INTEROP
            // Register the buffer with CUDA.
            if (m_render_target.cuda_buffer != nullptr)
                THROW_ON_CUDA_FAILURE(cudaGraphicsUnregisterResource(m_render_target.cuda_buffer));
            cudaError_t error = cudaGraphicsD3D11RegisterResource(&m_render_target.cuda_buffer, dx_buffer,
                                                                  cudaGraphicsRegisterFlagsNone);
            THROW_ON_CUDA_FAILURE(error);
            cudaGraphicsResourceSetMapFlags(m_render_target.cuda_buffer, cudaGraphicsMapFlagsWriteDiscard);
            dx_buffer->Release();
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

        // Update the constant buffer to reflect the new dimensions.
        int constant_data[4] = { width, height, 0, 0 };
        m_render_context->UpdateSubresource(m_constant_buffer, 0, nullptr, &constant_data, 0, 0);

        m_render_target.width = width;
        m_render_target.height = height;
    }
};

DX11Renderer::IRenderer* Adaptor::initialize(ID3D11Device1& device, int width_hint, int height_hint) {
    return new Adaptor(device, width_hint, height_hint);
}

Adaptor::Adaptor(ID3D11Device1& device, int width_hint, int height_hint) {
    m_impl = new Implementation(device, width_hint, height_hint);
    m_renderer_ID = Cogwheel::Core::Renderers::create("OptiXRenderer");
}

Adaptor::~Adaptor() {
    Cogwheel::Core::Renderers::destroy(m_renderer_ID);
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

void Adaptor::render(Cogwheel::Scene::Cameras::UID camera_ID, int width, int height) {
    m_impl->render(camera_ID, width, height);
}

} // NS DX11OptiXAdaptor
