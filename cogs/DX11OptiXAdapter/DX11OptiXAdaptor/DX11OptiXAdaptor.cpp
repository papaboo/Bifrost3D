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
        int size;
    } m_render_target;

    ID3D11Buffer* m_constant_buffer;
    ID3D11VertexShader* m_vertex_shader;
    ID3D11PixelShader* m_pixel_shader;

    OptiXRenderer::Renderer* m_optix_renderer;

public:

    Implementation(ID3D11Device1* device, int width_hint, int height_hint)
        : m_device(device) {

        device->GetImmediateContext1(&m_render_context);

        int cuda_device = 0;
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
            // m_optix_renderer = OptiXRenderer::Renderer::initialize(cuda_device, width_hint, height_hint);
        }

        {
            HRESULT hr = DX11Renderer::create_constant_buffer(*m_device, sizeof(float) * 4, &m_constant_buffer);
            THROW_ON_FAILURE(hr);
        }

        {
            m_render_target.dx_buffer = nullptr;
            m_render_target.dx_SRV = nullptr;
            resize_render_target(width_hint, height_hint);
        }

        { // Init shaders.
            const char* vertex_src =
                "struct Varyings {\n"
                "   float4 position : SV_POSITION;\n"
                "   float2 texcoord : TEXCOORD;\n"
                "};\n"
                "Varyings main_vs(uint vertex_ID : SV_VertexID) {\n"
                "   Varyings output;\n"
                "   // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }\n"
                "   output.position.x = vertex_ID == 2 ? 3 : -1;\n"
                "   output.position.y = vertex_ID == 0 ? -3 : 1;\n"
                "   output.position.zw = float2(0.5f, 1.0f);\n"
                "   output.texcoord = output.position.xy;\n"
                "   return output;\n"
                "}\n";

            const char* pixel_src =
                "StructuredBuffer<float4> pixels : register(t0);\n"
                "cbuffer constants : register(b0) {\n"
                "    int2 viewport_size;\n"
                "    int2 _padding;\n" // TODO Needed??
                "};\n"
                "struct Varyings {\n"
                "   float4 position : SV_POSITION;\n"
                "   float2 texcoord : TEXCOORD;\n"
                "};\n"
                "float4 main_ps(Varyings input) : SV_TARGET {\n"
                "   float2 tc = input.texcoord  * 0.5 + 0.5;\n"
                "   int2 buffer_index = tc * viewport_size;\n"
                "   return pixels[buffer_index.x + buffer_index.y * viewport_size.x];\n"
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

        // delete m_optix_renderer;
        m_device = nullptr;
    }

    bool is_valid() const {
        return true;
    }

    void handle_updates() {
        // m_optix_renderer->handle_updates();
    }

    void render(Cogwheel::Scene::Cameras::UID camera_ID) {
        // if (m_render_target.size < width * height)
        //     resize_buffer(width, height);

        // m_optix_renderer->render(camera_ID, buffer, widht, height);

        // TODO We should probably set the OMSetDepthStencilState and RSSetState.

        m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m_render_context->OMSetBlendState(0, 0, 0xffffffff);

        m_render_context->VSSetShader(m_vertex_shader, 0, 0);
        m_render_context->PSSetShader(m_pixel_shader, 0, 0);

        m_render_context->PSSetConstantBuffers(0, 1, &m_constant_buffer);
        m_render_context->PSSetShaderResources(0, 1, &m_render_target.dx_SRV);

        m_render_context->Draw(3, 0);
    }

    void resize_render_target(int width, int height) {
        DX11Renderer::safe_release(&m_render_target.dx_buffer);
        DX11Renderer::safe_release(&m_render_target.dx_SRV);
        
        m_render_target.size = width * height;

        D3D11_BUFFER_DESC desc = {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.StructureByteStride = sizeof(float) * 4;
        desc.ByteWidth = m_render_target.size * sizeof(float) * 4;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        float* tmp_data = new float[width * height * 4];
        for (int i = 0; i < width * height; ++i) {
            tmp_data[4 * i + 0] = 1.0F - float(i) / (width * height);
            tmp_data[4 * i + 1] = float(i) / (width * height);
            tmp_data[4 * i + 2] = 0.0f;
            tmp_data[4 * i + 3] = 1.0f;
        }

        D3D11_SUBRESOURCE_DATA tmmp_data = {};
        tmmp_data.pSysMem = tmp_data;

        HRESULT hr = m_device->CreateBuffer(&desc, &tmmp_data, &m_render_target.dx_buffer);
        THROW_ON_FAILURE(hr);

        delete[] tmp_data;

        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
        srv_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srv_desc.Buffer.NumElements = m_render_target.size;

        hr = m_device->CreateShaderResourceView(m_render_target.dx_buffer, &srv_desc, &m_render_target.dx_SRV);
        THROW_ON_FAILURE(hr);

        // TODO Register the buffer with OptiX

        { // Update the constant buffer to reflect the new dimensions.
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

void DX11OptiXAdaptor::render(Cogwheel::Scene::Cameras::UID camera_ID) {
    m_impl->render(camera_ID);
}

} // NS DX11OptiXAdaptor
