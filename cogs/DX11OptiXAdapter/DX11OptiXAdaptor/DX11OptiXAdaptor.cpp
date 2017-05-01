// Adaptor for the OptiX renderer that allows it to render to a DX11 buffer.
//-------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#include <DX11OptiXAdaptor/DX11OptiXAdaptor.h>
#include <DX11Renderer/Renderer.h>
#include <DX11Renderer/Utils.h>
#include <OptiXRenderer/Renderer.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include <optixu/optixpp_namespace.h>

#define NOMINMAX
#include <D3D11_1.h>
#include <D3DCompiler.h>
#undef RGB

namespace DX11OptiXAdaptor {

class DX11OptiXAdaptor::Implementation {
    ID3D11Device1* m_device;
    ID3D11DeviceContext1* m_render_context;

    struct { // Buffer
        ID3D11Buffer* dx_buffer;
        ID3D11ShaderResourceView* dx_SRV;
        optix::Buffer optix_buffer;
        int capacity;
        int width;
        int height;
    } m_render_target;

    ID3D11Buffer* m_constant_buffer;
    ID3D11VertexShader* m_vertex_shader;
    ID3D11PixelShader* m_pixel_shader;

    OptiXRenderer::Renderer* m_optix_renderer;

public:

    Implementation(ID3D11Device1* device, int width_hint, int height_hint)
        : m_device(device) {

        device->GetImmediateContext1(&m_render_context);

        int cuda_device = -1;
        { // Get CUDA device from DX11 context.
            IDXGIDevice* dxgi_device = nullptr;
            HRESULT hr = device->QueryInterface(IID_PPV_ARGS(&dxgi_device));
            THROW_ON_FAILURE(hr);

            IDXGIAdapter* adapter = nullptr;
            hr = dxgi_device->GetAdapter(&adapter);
            THROW_ON_FAILURE(hr); 
            dxgi_device->Release();

            cudaError_t error = cudaD3D11GetDevice(&cuda_device, adapter);
            // TODO Throw on CUDA error
            adapter->Release();

            // Create OptiX Renderer on device.
            m_optix_renderer = OptiXRenderer::Renderer::initialize(cuda_device, width_hint, height_hint);
        }

        {
            HRESULT hr = DX11Renderer::create_constant_buffer(*m_device, sizeof(float) * 4, &m_constant_buffer);
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
                "StructuredBuffer<float4> pixels : register(t0);\n"
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
            HRESULT hr = m_device->CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_bytecode), NULL, &m_vertex_shader);
            THROW_ON_FAILURE(hr);

            ID3DBlob* pixel_shader_bytecode = compile_shader(pixel_src, "ps_5_0", "main_ps");
            hr = m_device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_bytecode), NULL, &m_pixel_shader);
            THROW_ON_FAILURE(hr);
        }
    }

    ~Implementation() {
        DX11Renderer::safe_release(&m_render_target.dx_buffer);
        DX11Renderer::safe_release(&m_render_target.dx_SRV);
        DX11Renderer::safe_release(&m_constant_buffer);

        if (m_render_target.optix_buffer)
            m_render_target.optix_buffer->unregisterD3D11Buffer();

        delete m_optix_renderer;
        m_device = nullptr;
    }

    bool is_valid() const {
        return true;
    }

    void handle_updates() {
        m_optix_renderer->handle_updates();
    }

    void render(Cogwheel::Scene::Cameras::UID camera_ID, int width, int height) {
        if (m_render_target.width != width || m_render_target.height != height)
            resize_render_target(width, height);

        m_optix_renderer->render(camera_ID, m_render_target.optix_buffer, m_render_target.width, m_render_target.height);

        m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m_render_context->OMSetBlendState(0, 0, 0xffffffff);

        m_render_context->VSSetShader(m_vertex_shader, 0, 0);
        m_render_context->PSSetShader(m_pixel_shader, 0, 0);

        m_render_context->PSSetConstantBuffers(0, 1, &m_constant_buffer);
        m_render_context->PSSetShaderResources(0, 1, &m_render_target.dx_SRV);

        m_render_context->Draw(3, 0);
    }

    void resize_render_target(int width, int height) {

        if (m_render_target.capacity < width * height) {
            DX11Renderer::safe_release(&m_render_target.dx_buffer);
            DX11Renderer::safe_release(&m_render_target.dx_SRV);

            m_render_target.capacity = width * height;

            D3D11_BUFFER_DESC desc = {};
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.StructureByteStride = sizeof(float) * 4;
            desc.ByteWidth = m_render_target.capacity * sizeof(float) * 4;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc.MiscFlags = 0;

            HRESULT hr = m_device->CreateBuffer(&desc, nullptr, &m_render_target.dx_buffer);
            THROW_ON_FAILURE(hr);

            D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
            srv_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            srv_desc.Buffer.NumElements = m_render_target.capacity;

            hr = m_device->CreateShaderResourceView(m_render_target.dx_buffer, &srv_desc, &m_render_target.dx_SRV);
            THROW_ON_FAILURE(hr);
        
            // Register the buffer with OptiX
            optix::Context optix_context = m_optix_renderer->get_context();
            if (m_render_target.optix_buffer)
                m_render_target.optix_buffer->unregisterD3D11Buffer();
            m_render_target.optix_buffer = optix_context->createBufferFromD3D11Resource(RT_BUFFER_OUTPUT, m_render_target.dx_buffer);

            assert(m_render_target.optix_buffer->getD3D11Resource() == m_render_target.dx_buffer);
        }

        { // Resize buffer and update DX constant buffer.
            m_render_target.width = width;
            m_render_target.height = height;

            m_render_target.optix_buffer->setSize(width, height);
            m_render_target.optix_buffer->setFormat(RT_FORMAT_FLOAT4);

            // Update the constant buffer to reflect the new dimensions.
            int constant_data[4] = { width, height, 0, 0 };
            m_render_context->UpdateSubresource(m_constant_buffer, 0, NULL, &constant_data, 0, 0);
        }
    }
};

DX11Renderer::IRenderer* DX11OptiXAdaptor::initialize(ID3D11Device1* device, int width_hint, int height_hint) {
    DX11OptiXAdaptor* r = new DX11OptiXAdaptor(device, width_hint, height_hint);
    if (r->m_impl->is_valid())
        return r;
    else {
        delete r;
        return nullptr;
    }
}

DX11OptiXAdaptor::DX11OptiXAdaptor(ID3D11Device1* device, int width_hint, int height_hint) {
    m_impl = new Implementation(device, width_hint, height_hint);
}

DX11OptiXAdaptor::~DX11OptiXAdaptor() {
    delete m_impl;
}

void DX11OptiXAdaptor::handle_updates() {
    m_impl->handle_updates();
}

void DX11OptiXAdaptor::render(Cogwheel::Scene::Cameras::UID camera_ID, int width, int height) {
    m_impl->render(camera_ID, width, height);
}

} // NS DX11OptiXAdaptor
