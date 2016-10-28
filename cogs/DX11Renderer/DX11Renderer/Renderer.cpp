// DirectX 11 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/Renderer.h>

#define NOMINMAX
#include <D3D11.h>
#include <D3DCompiler.h>
#undef RGB

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Core/Window.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/SceneRoot.h>

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

namespace DX11Renderer {

template<typename ResourcePtr>
void safe_release(ResourcePtr* resource_ptr) {
    if (*resource_ptr) {
        (*resource_ptr)->Release();
        *resource_ptr = nullptr;
    }
}

// TODO Handle cso files and errors related to files not found.
// Specialize blob so I can return it by value?
inline ID3DBlob* compile_shader(std::wstring filename, const char* target) {
    std::wstring qualified_filename = L"../Data/DX12Renderer/Shaders/" + filename;

    ID3DBlob* shader_bytecode;
    ID3DBlob* error_messages = nullptr;
    HRESULT hr = D3DCompileFromFile(qualified_filename.c_str(),
        nullptr, // macroes
        nullptr, // Include dirs. TODO "../Data/DX12Renderer/Shaders/"
        "main",
        target,
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
        0, // More flags. Unused.
        &shader_bytecode,
        &error_messages);
    if (FAILED(hr)) { // File not found not handled? Path not found unhandled as well.
        if (error_messages != nullptr)
            printf("Shader error: '%s'\n", (char*)error_messages->GetBufferPointer());
        return nullptr;
    }

    return shader_bytecode;
}

//----------------------------------------------------------------------------
// DirectX 11 renderer implementation.
//----------------------------------------------------------------------------
class Renderer::Implementation {
private:
    ID3D11Device* m_device;

    IDXGISwapChain* m_swap_chain;

    ID3D11DeviceContext* m_render_context;
    ID3D11RenderTargetView* m_backbuffer_view;

    struct {
        ID3D11Buffer* positions;
        ID3D11InputLayout* vertex_layout;
        ID3D10Blob* vertex_shader_buffer;
        ID3D10Blob* pixel_shader_buffer;
        ID3D11VertexShader* vertex_shader;
        ID3D11PixelShader* pixel_shader;
    } m_triangle;

public:
    bool is_valid() { return m_device != nullptr; }

    Implementation(HWND& hwnd, const Cogwheel::Core::Window& window) {
        { // Create device and swapchain.

            DXGI_MODE_DESC backbuffer_desc;
            backbuffer_desc.Width = window.get_width();
            backbuffer_desc.Height = window.get_height();
            backbuffer_desc.RefreshRate.Numerator = 60;
            backbuffer_desc.RefreshRate.Denominator = 1;
            backbuffer_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // TODO sRGB
            backbuffer_desc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
            backbuffer_desc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;

            DXGI_SWAP_CHAIN_DESC swap_chain_desc = {};
            swap_chain_desc.BufferDesc = backbuffer_desc;
            swap_chain_desc.SampleDesc.Count = 1;
            swap_chain_desc.SampleDesc.Quality = 0;
            swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            swap_chain_desc.BufferCount = 1;
            swap_chain_desc.OutputWindow = hwnd;
            swap_chain_desc.Windowed = TRUE;
            swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

            // TODO Replace by explicit device enumeration and selecting of 'largest' device. Otherwise we risk selecting the integrated GPU.
            D3D_FEATURE_LEVEL feature_level;
            HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                D3D11_SDK_VERSION, &swap_chain_desc, &m_swap_chain, &m_device, &feature_level, &m_render_context);

            if (m_device == nullptr) {
                release_state();
                return;
            }

            IDXGIDevice* gi_device;
            hr = m_device->QueryInterface(IID_PPV_ARGS(&gi_device));
            IDXGIAdapter* adapter;
            gi_device->GetAdapter(&adapter);
            DXGI_ADAPTER_DESC adapter_description;
            adapter->GetDesc(&adapter_description);
            std::string readable_feature_level = feature_level == D3D_FEATURE_LEVEL_11_0 ? "11.0" : "11.1";
            printf("DX11Renderer using device '%S' with feature level %s.\n", adapter_description.Description, readable_feature_level.c_str());
            adapter->Release();
            gi_device->Release();
        }

        // Create backBbuffer.
        ID3D11Texture2D* backbuffer;
        HRESULT hr = m_swap_chain->GetBuffer(0, IID_PPV_ARGS(&backbuffer));

        // Create and set render target.
        hr = m_device->CreateRenderTargetView(backbuffer, nullptr, &m_backbuffer_view);
        backbuffer->Release();
        m_render_context->OMSetRenderTargets(1, &m_backbuffer_view, nullptr);

        setup_triangle();
    }

    void setup_triangle() {
        { // Compile shaders.
            m_triangle.vertex_shader_buffer = compile_shader(L"VertexShader.hlsl", "vs_4_0"); // TODO Do I really need to store the buffer?
            m_triangle.pixel_shader_buffer = compile_shader(L"FragmentShader.hlsl", "ps_4_0");
            
            // Create the shader objects.
            HRESULT hr = m_device->CreateVertexShader(m_triangle.vertex_shader_buffer->GetBufferPointer(), m_triangle.vertex_shader_buffer->GetBufferSize(), NULL, &m_triangle.vertex_shader);
            hr = m_device->CreatePixelShader(m_triangle.pixel_shader_buffer->GetBufferPointer(), m_triangle.pixel_shader_buffer->GetBufferSize(), NULL, &m_triangle.pixel_shader);
        }

        // Set vertex and pixel shaders. TODO Move to actual rendering.
        m_render_context->VSSetShader(m_triangle.vertex_shader, 0, 0);
        m_render_context->PSSetShader(m_triangle.pixel_shader, 0, 0);

        { // Setup position buffer.
            Vector3f positions[] = {
                Vector3f(0.0f, 0.5f, 0.5f),
                Vector3f(0.5f, -0.5f, 0.5f),
                Vector3f(-0.5f, -0.5f, 0.5f),
            };

            D3D11_BUFFER_DESC position_buffer_desc = {};
            position_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
            position_buffer_desc.ByteWidth = sizeof(float) * 3 * 3;
            position_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            position_buffer_desc.CPUAccessFlags = 0;
            position_buffer_desc.MiscFlags = 0;

            D3D11_SUBRESOURCE_DATA positions_buffer_data;
            ZeroMemory(&positions_buffer_data, sizeof(positions_buffer_data)); // TODO needed? = {}
            positions_buffer_data.pSysMem = positions;
            HRESULT hr = m_device->CreateBuffer(&position_buffer_desc, &positions_buffer_data, &m_triangle.positions);

            // Set the vertex buffer
            UINT stride = sizeof(float) * 3;
            UINT offset = 0;
            m_render_context->IASetVertexBuffers(0, 1, &m_triangle.positions, &stride, &offset);
        }

        // Create the input layout
        D3D11_INPUT_ELEMENT_DESC position_layout_desc[] = { // TODO No need for an array.
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };

        m_device->CreateInputLayout(position_layout_desc, 1, m_triangle.vertex_shader_buffer->GetBufferPointer(),
            m_triangle.vertex_shader_buffer->GetBufferSize(), &m_triangle.vertex_layout);

        // Set the Input Layout
        m_render_context->IASetInputLayout(m_triangle.vertex_layout);

        // Set Primitive Topology
        m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    }

    void release_state() {
        if (m_device == nullptr)
            return;

        safe_release(&m_device);
        safe_release(&m_render_context);
        safe_release(&m_swap_chain);
    }

    void render(const Cogwheel::Core::Engine& engine) {
        if (Cameras::begin() == Cameras::end())
            return;

        // ?? wait_for_previous_frame(); // Why isn't this needed?

        if (m_device == nullptr)
            return;

        handle_updates();

        Cameras::UID camera_ID = *Cameras::begin();

        { // Create and set the viewport.
            Cogwheel::Math::Rectf vp = Cameras::get_viewport(camera_ID);
            vp.width *= engine.get_window().get_width();
            vp.height *= engine.get_window().get_height();

            D3D11_VIEWPORT viewport;
            viewport.TopLeftX = 0;
            viewport.TopLeftY = 0;
            viewport.Width = vp.width;
            viewport.Height = vp.height;
            viewport.MinDepth = 0.0f;
            viewport.MaxDepth = 1.0f;
            m_render_context->RSSetViewports(1, &viewport);
        }

        SceneRoot scene = Cameras::get_scene_ID(camera_ID);
        RGBA environment_tint = RGBA(scene.get_environment_tint(), 1.0f);
        m_render_context->ClearRenderTargetView(m_backbuffer_view, environment_tint.begin());

        { // Render triangle.
            m_render_context->Draw(3, 0);
        }
        // Present the backbuffer.
        m_swap_chain->Present(0, 0);
    }

    void handle_updates() {

    }
};

//----------------------------------------------------------------------------
// DirectX 11 renderer.
//----------------------------------------------------------------------------
Renderer* Renderer::initialize(HWND& hwnd, const Cogwheel::Core::Window& window) {
    Renderer* r = new Renderer(hwnd, window);
    if (r->m_impl->is_valid())
        return r;
    else {
        delete r;
        return nullptr;
    }
}

Renderer::Renderer(HWND& hwnd, const Cogwheel::Core::Window& window) {
    m_impl = new Implementation(hwnd, window);
}

Renderer::~Renderer() {
    m_impl->release_state();
}

void Renderer::render(const Cogwheel::Core::Engine& engine) {
    m_impl->render(engine);
}

} // NS DX11Renderer
