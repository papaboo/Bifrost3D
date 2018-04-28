// DirectX 11 compositor.
//-------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/CameraEffects.h>
#include <DX11Renderer/Utils.h>

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Core/Window.h>

using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

namespace DX11Renderer {

using OIDXGISwapChain1 = DX11Renderer::OwnedResourcePtr<IDXGISwapChain1>;

OID3D11Device1 create_performant_device1(unsigned int create_device_flags) {
    // Find the best performing device (apparently the one with the most memory) and initialize that.
    struct WeightedAdapter {
        int index, dedicated_memory;

        inline bool operator<(WeightedAdapter rhs) const {
            return rhs.dedicated_memory < dedicated_memory;
        }
    };

    IDXGIFactory1* dxgi_factory1;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory1));

    IDXGIAdapter1* adapter = nullptr;
    std::vector<WeightedAdapter> sorted_adapters;
    for (int adapter_index = 0; dxgi_factory1->EnumAdapters1(adapter_index, &adapter) != DXGI_ERROR_NOT_FOUND; ++adapter_index) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        // Ignore software rendering adapters.
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;

        WeightedAdapter e = { adapter_index, int(desc.DedicatedVideoMemory >> 20) };
        sorted_adapters.push_back(e);
    }

    std::sort(sorted_adapters.begin(), sorted_adapters.end());

    // Then create the device and render context.
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* render_context = nullptr;
    for (WeightedAdapter a : sorted_adapters) {
        dxgi_factory1->EnumAdapters1(a.index, &adapter);

        D3D_FEATURE_LEVEL feature_level_requested = D3D_FEATURE_LEVEL_11_0;

        D3D_FEATURE_LEVEL feature_level;
        hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, create_device_flags, &feature_level_requested, 1,
            D3D11_SDK_VERSION, &device, &feature_level, &render_context);

        if (SUCCEEDED(hr))
            break;
    }
    dxgi_factory1->Release();

    if (device == nullptr)
        return nullptr;

    OID3D11Device1 device1;
    hr = device->QueryInterface(IID_PPV_ARGS(&device1));
    THROW_ON_FAILURE(hr);
    return device1;
}

OID3D11Device1 create_performant_debug_device1() { return create_performant_device1(D3D11_CREATE_DEVICE_DEBUG); }

//-------------------------------------------------------------------------------------------------
// DirectX 11 compositor implementation.
//-------------------------------------------------------------------------------------------------
class Compositor::Implementation {
private:
    const Window& m_window;
    const std::wstring m_data_folder_path;
    std::vector<IRenderer*> m_renderers;

    OID3D11Device1 m_device;
    OID3D11DeviceContext1 m_render_context;
    OIDXGISwapChain1 m_swap_chain;
    unsigned int m_sync_interval = 1;

    // Backbuffer members.
    Vector2ui m_backbuffer_size;
    OID3D11RenderTargetView m_swap_chain_buffer_view;
    OID3D11RenderTargetView m_backbuffer_RTV;
    OID3D11ShaderResourceView m_backbuffer_SRV;
    OID3D11DepthStencilView m_depth_view;

    double previous_tonemapping_time;
    double counter_hertz;
    CameraEffects m_camera_effects;

public:
    Implementation(HWND& hwnd, const Window& window, const std::wstring& data_folder_path)
        : m_window(window), m_data_folder_path(data_folder_path) {

        m_device = create_performant_device1();
        m_device->GetImmediateContext1(&m_render_context);

        { // Get the device's dxgi factory and create the swap chain.
            IDXGIDevice* dxgi_device = nullptr;
            HRESULT hr = m_device->QueryInterface(IID_PPV_ARGS(&dxgi_device));
            THROW_ON_FAILURE(hr);

            // NOTE Can I use the original adapter for this?
            IDXGIAdapter* adapter = nullptr;
            hr = dxgi_device->GetAdapter(&adapter);
            dxgi_device->Release();
            THROW_ON_FAILURE(hr);

            DXGI_ADAPTER_DESC adapter_description;
            adapter->GetDesc(&adapter_description);
            const char* readable_feature_level = m_device->GetFeatureLevel() == D3D_FEATURE_LEVEL_11_0 ? "11.0" : "11.1";
            printf("DirectX 11 compositor using device '%S' with feature level %s.\n", adapter_description.Description, readable_feature_level);

            IDXGIFactory2* dxgi_factory2 = nullptr;
            hr = adapter->GetParent(IID_PPV_ARGS(&dxgi_factory2));
            adapter->Release();
            THROW_ON_FAILURE(hr);

            // Create swap chain
            DXGI_SWAP_CHAIN_DESC1 swap_chain_desc1 = {};
            swap_chain_desc1.Width = window.get_width();
            swap_chain_desc1.Height = window.get_height();
            swap_chain_desc1.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            swap_chain_desc1.SampleDesc.Count = 1;
            swap_chain_desc1.SampleDesc.Quality = 0;
            swap_chain_desc1.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            swap_chain_desc1.BufferCount = 1;
            swap_chain_desc1.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

            hr = dxgi_factory2->CreateSwapChainForHwnd(m_device, hwnd, &swap_chain_desc1, nullptr, nullptr, &m_swap_chain);
            THROW_ON_FAILURE(hr);

            dxgi_factory2->Release();
        }

        { // Setup backbuffer.
            m_backbuffer_size = Vector2ui::zero();

            // Back- and depth buffer is initialized on demand when the output dimensions are known.
            m_swap_chain_buffer_view = nullptr;
            m_backbuffer_RTV = nullptr;
            m_backbuffer_SRV = nullptr;
            m_depth_view = nullptr;
        }

        { // Setup tonemapper
            { // Setup timer.
                LARGE_INTEGER freq;
                QueryPerformanceFrequency(&freq);
                counter_hertz = 1.0 / freq.QuadPart;
                previous_tonemapping_time = std::numeric_limits<double>::lowest(); // Ensures that the first delta_time is positive infinity, which in turn disables eye adaptation for the first frame.
            }

            std::wstring shader_folder_path = m_data_folder_path + L"DX11Renderer\\Shaders\\";
            m_camera_effects = CameraEffects(m_device, shader_folder_path);
        }
    }

    ~Implementation() {
        for (IRenderer* r : m_renderers)
            delete r;
    }

    bool is_valid() const {
        return m_device.resource != nullptr;
    }

    IRenderer* attach_renderer(RendererCreator renderer_creator) {
        IRenderer* renderer = renderer_creator(*m_device, m_window.get_width(), m_window.get_height(), m_data_folder_path);
        if (renderer == nullptr)
            return nullptr;

        if (m_renderers.size() < Renderers::capacity())
            m_renderers.resize(Renderers::capacity());

        Renderers::UID renderer_ID = renderer->get_ID();
        m_renderers[renderer_ID] = renderer;
        return renderer;
    }

    void render() {
        if (Cameras::begin() == Cameras::end())
            return;

        if (m_device == nullptr)
            return;

        Vector2ui current_backbuffer_size = Vector2ui(m_window.get_width(), m_window.get_height());
        if (m_backbuffer_size != current_backbuffer_size) {
            
            { // Setup swap chain buffer.
                // https://msdn.microsoft.com/en-us/library/windows/desktop/bb205075(v=vs.85).aspx#Handling_Window_Resizing

                m_render_context->OMSetRenderTargets(0, 0, 0);
                if (m_swap_chain_buffer_view) m_swap_chain_buffer_view->Release();

                HRESULT hr = m_swap_chain->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0);
                THROW_ON_FAILURE(hr);

                ID3D11Texture2D* swap_chain_buffer;
                hr = m_swap_chain->GetBuffer(0, IID_PPV_ARGS(&swap_chain_buffer));
                THROW_ON_FAILURE(hr);
                hr = m_device->CreateRenderTargetView(swap_chain_buffer, nullptr, &m_swap_chain_buffer_view);
                THROW_ON_FAILURE(hr);
                swap_chain_buffer->Release();
            }

            { // Setup backbuffer.
                if (m_backbuffer_RTV) m_backbuffer_RTV->Release();
                if (m_backbuffer_SRV) m_backbuffer_SRV->Release();

                D3D11_TEXTURE2D_DESC buffer_desc;
                buffer_desc.Width = current_backbuffer_size.x;
                buffer_desc.Height = current_backbuffer_size.y;
                buffer_desc.MipLevels = 1;
                buffer_desc.ArraySize = 1;
                buffer_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                buffer_desc.SampleDesc.Count = 1;
                buffer_desc.SampleDesc.Quality = 0;
                buffer_desc.Usage = D3D11_USAGE_DEFAULT;
                buffer_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                buffer_desc.CPUAccessFlags = 0;
                buffer_desc.MiscFlags = 0;

                OID3D11Texture2D backbuffer;
                HRESULT hr = m_device->CreateTexture2D(&buffer_desc, nullptr, &backbuffer);
                THROW_ON_FAILURE(hr);
                hr = m_device->CreateRenderTargetView(backbuffer, nullptr, &m_backbuffer_RTV);
                THROW_ON_FAILURE(hr);
                hr = m_device->CreateShaderResourceView(backbuffer, nullptr, &m_backbuffer_SRV);
                THROW_ON_FAILURE(hr);
            }

            { // Setup depth buffer.
                if (m_depth_view) m_depth_view->Release();

                D3D11_TEXTURE2D_DESC depth_desc;
                depth_desc.Width = current_backbuffer_size.x;
                depth_desc.Height = current_backbuffer_size.y;
                depth_desc.MipLevels = 1;
                depth_desc.ArraySize = 1;
                depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
                depth_desc.SampleDesc.Count = 1;
                depth_desc.SampleDesc.Quality = 0;
                depth_desc.Usage = D3D11_USAGE_DEFAULT;
                depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
                depth_desc.CPUAccessFlags = 0;
                depth_desc.MiscFlags = 0;

                OID3D11Texture2D depth_buffer;
                HRESULT hr = m_device->CreateTexture2D(&depth_desc, nullptr, &depth_buffer);
                m_device->CreateDepthStencilView(depth_buffer, nullptr, &m_depth_view);
            }

            m_backbuffer_size = current_backbuffer_size;
        }

        // Tell all renderers to update.
        for (IRenderer* renderer : m_renderers)
            if (renderer)
                renderer->handle_updates();

        m_render_context->OMSetRenderTargets(1, &m_backbuffer_RTV, m_depth_view);
        m_render_context->ClearDepthStencilView(m_depth_view, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

        // Render.
        for (Cameras::UID camera_ID : Cameras::get_iterable()) {

            Rectf viewport = Cameras::get_viewport(camera_ID);
            viewport.x *= m_window.get_width();
            viewport.width *= m_window.get_width();
            viewport.y *= m_window.get_height();
            viewport.height *= m_window.get_height();

            // Create and set the viewport.
            D3D11_VIEWPORT dx_viewport;
            dx_viewport.TopLeftX = viewport.x;
            dx_viewport.TopLeftY = viewport.y;
            dx_viewport.Width = viewport.width;
            dx_viewport.Height = viewport.height;
            dx_viewport.MinDepth = 0.0f;
            dx_viewport.MaxDepth = 1.0f;
            m_render_context->RSSetViewports(1, &dx_viewport);
            
            Renderers::UID renderer = Cameras::get_renderer_ID(camera_ID);
            m_renderers[renderer]->render(camera_ID, int(ceilf(viewport.width)), int(ceilf(viewport.height)));
        }

        // Compute delta time for tone mapping.
        LARGE_INTEGER performance_count;
        QueryPerformanceCounter(&performance_count);
        double current_time = performance_count.QuadPart * counter_hertz;
        float delta_time = float(current_time - previous_tonemapping_time);
        previous_tonemapping_time = current_time;

        // Present the backbuffer.
        Cameras::UID camera_ID = *Cameras::get_iterable().begin();
        auto tonemapping_params = Cameras::get_tonemapping_parameters(camera_ID);
        m_camera_effects.process(m_render_context, tonemapping_params, delta_time, m_backbuffer_SRV, m_swap_chain_buffer_view, m_backbuffer_size.x, m_backbuffer_size.y);
        m_swap_chain->Present(m_sync_interval, 0);
    }

    bool uses_v_sync() const { return m_sync_interval != 0; }
    void set_v_sync(bool use_v_sync) { m_sync_interval = unsigned int(use_v_sync); }
};

//----------------------------------------------------------------------------
// DirectX 11 compositor.
//----------------------------------------------------------------------------
Compositor::Initialization Compositor::initialize(HWND& hwnd, const Cogwheel::Core::Window& window, 
                                                  const std::wstring& data_folder_path, RendererCreator renderer_creator) {
    assert(renderer_creator != nullptr);

    Compositor* c = new Compositor(hwnd, window, data_folder_path);
    if (!c->m_impl->is_valid()) {
        delete c;
        return { nullptr, nullptr };
    }

    IRenderer* renderer = c->attach_renderer(renderer_creator);
    if (renderer == nullptr) {
        delete c;
        return { nullptr, nullptr };
    }

    return { c, renderer };
}

Compositor::Compositor(HWND& hwnd, const Cogwheel::Core::Window& window, const std::wstring& data_folder_path) {
    m_impl = new Implementation(hwnd, window, data_folder_path);
}

Compositor::~Compositor() {
    delete m_impl;
}

IRenderer* Compositor::attach_renderer(RendererCreator renderer_creator) {
    return m_impl->attach_renderer(renderer_creator);
}

void Compositor::render() { m_impl->render(); }
bool Compositor::uses_v_sync() const { return m_impl->uses_v_sync(); }
void Compositor::set_v_sync(bool use_v_sync) { m_impl->set_v_sync(use_v_sync); }

} // NS DX11Renderer
