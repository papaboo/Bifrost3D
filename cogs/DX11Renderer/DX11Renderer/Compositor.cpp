// DirectX 11 compositor.
//-------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/Utils.h>

#define NOMINMAX
#include <D3D11_1.h>
#undef RGB

#include <Cogwheel/Core/Window.h>

using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

namespace DX11Renderer {

//-------------------------------------------------------------------------------------------------
// DirectX 11 compositor implementation.
//-------------------------------------------------------------------------------------------------
class Compositor::Implementation {
private:
    const Window& m_window;
    std::vector<IRenderer*> m_renderers;
    IRenderer* m_active_renderer;

    ID3D11Device1* m_device;
    ID3D11DeviceContext1* m_render_context;
    IDXGISwapChain1* m_swap_chain;

    // Backbuffer members.
    Vector2ui m_backbuffer_size;
    ID3D11RenderTargetView* m_backbuffer_view;
    ID3D11Texture2D* m_depth_buffer;
    ID3D11DepthStencilView* m_depth_view;

public:
    Implementation(HWND& hwnd, const Window& window)
        : m_window(window) {

        // Find the best performing device (apparently the one with the most memory) and initialize that.
        struct WeightedAdapter {
            int index, dedicated_memory;

            inline bool operator<(WeightedAdapter rhs) const {
                return rhs.dedicated_memory < dedicated_memory;
            }
        };

        IDXGIAdapter1* adapter = nullptr;

        IDXGIFactory1* dxgi_factory1;
        HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory1));

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

            UINT create_device_flags = 0;
            D3D_FEATURE_LEVEL feature_level_requested = D3D_FEATURE_LEVEL_11_0;

            D3D_FEATURE_LEVEL feature_level;
            hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, create_device_flags, &feature_level_requested, 1,
                D3D11_SDK_VERSION, &device, &feature_level, &render_context);

            if (SUCCEEDED(hr)) {
                DXGI_ADAPTER_DESC adapter_description;
                adapter->GetDesc(&adapter_description);
                std::string readable_feature_level = feature_level == D3D_FEATURE_LEVEL_11_0 ? "11.0" : "11.1";
                printf("DirectX 11 composer using device '%S' with feature level %s.\n", adapter_description.Description, readable_feature_level.c_str());
                break;
            }
        }
        dxgi_factory1->Release();

        if (device == nullptr)
            return;

        hr = device->QueryInterface(IID_PPV_ARGS(&m_device));
        THROW_ON_FAILURE(hr);

        hr = render_context->QueryInterface(IID_PPV_ARGS(&m_render_context));
        THROW_ON_FAILURE(hr);

        { // Get the device's dxgi factory and create the swap chain.
            IDXGIDevice* dxgi_device = nullptr;
            hr = device->QueryInterface(IID_PPV_ARGS(&dxgi_device));
            THROW_ON_FAILURE(hr);

            // NOTE Can I use the original adapter for this?
            IDXGIAdapter* adapter = nullptr;
            hr = dxgi_device->GetAdapter(&adapter);
            dxgi_device->Release();
            THROW_ON_FAILURE(hr);

            IDXGIFactory1* dxgi_factory = nullptr;
            hr = adapter->GetParent(IID_PPV_ARGS(&dxgi_factory));
            adapter->Release();
            THROW_ON_FAILURE(hr);

            IDXGIFactory2* dxgi_factory2 = nullptr;
            hr = dxgi_factory->QueryInterface(IID_PPV_ARGS(&dxgi_factory2));
            dxgi_factory->Release();
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

            // Get and set render target.
            ID3D11Texture2D* backbuffer;
            HRESULT hr = m_swap_chain->GetBuffer(0, IID_PPV_ARGS(&backbuffer));
            THROW_ON_FAILURE(hr);
            hr = m_device->CreateRenderTargetView(backbuffer, nullptr, &m_backbuffer_view);
            THROW_ON_FAILURE(hr);
            backbuffer->Release();

            // Depth buffer is initialized on demand when the output dimensions are known.
            m_depth_buffer = nullptr;
            m_depth_view = nullptr;
        }
    }

    ~Implementation() {
        safe_release(&m_device);
        safe_release(&m_render_context);
        safe_release(&m_swap_chain);

        safe_release(&m_backbuffer_view);
        safe_release(&m_depth_buffer);
        safe_release(&m_depth_view);

        // TODO Delete all render backends and their scene representations.
        for (IRenderer* r : m_renderers)
            delete r;
    }

    bool is_valid() const {
        return m_device != nullptr;
    }

    Renderers::UID attach_renderer(RendererCreator renderer_creator) {
        IRenderer* renderer = renderer_creator(m_device, m_window.get_width(), m_window.get_height());
        if (renderer == nullptr)
            return Renderers::UID::invalid_UID();

        if (m_renderers.size() < Renderers::capacity())
            m_renderers.resize(Renderers::capacity());

        Renderers::UID renderer_ID = renderer->get_ID();
        m_renderers[renderer_ID] = renderer;
        return renderer_ID;
    }

    void set_active_renderer(Renderers::UID renderer_ID) {
        m_active_renderer = m_renderers[renderer_ID];
    }

    void render() {
        if (Cameras::begin() == Cameras::end())
            return;

        // ?? wait_for_previous_frame();

        if (m_device == nullptr)
            return;

        Vector2ui current_backbuffer_size = Vector2ui(m_window.get_width(), m_window.get_height());
        if (m_backbuffer_size != current_backbuffer_size) {
            
            { // Setup new backbuffer.
                // https://msdn.microsoft.com/en-us/library/windows/desktop/bb205075(v=vs.85).aspx#Handling_Window_Resizing

                m_render_context->OMSetRenderTargets(0, 0, 0);
                if (m_backbuffer_view) m_backbuffer_view->Release();

                HRESULT hr = m_swap_chain->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0);
                THROW_ON_FAILURE(hr);

                ID3D11Texture2D* backbuffer;
                hr = m_swap_chain->GetBuffer(0, IID_PPV_ARGS(&backbuffer));
                THROW_ON_FAILURE(hr);
                hr = m_device->CreateRenderTargetView(backbuffer, nullptr, &m_backbuffer_view);
                THROW_ON_FAILURE(hr);
                backbuffer->Release();
            }

            { // Setup new depth buffer.
                if (m_depth_buffer) m_depth_buffer->Release();
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

                HRESULT hr = m_device->CreateTexture2D(&depth_desc, NULL, &m_depth_buffer);
                m_device->CreateDepthStencilView(m_depth_buffer, NULL, &m_depth_view);
            }

            m_backbuffer_size = current_backbuffer_size;
        }

        m_active_renderer->handle_updates();

        m_render_context->OMSetRenderTargets(1, &m_backbuffer_view, m_depth_view);
        m_render_context->ClearDepthStencilView(m_depth_view, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

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
            
            m_active_renderer->render(camera_ID, int(ceilf(viewport.width)), int(ceilf(viewport.height)));
        }

        // Present the backbuffer.
        m_swap_chain->Present(0, 0);
    }
};

//----------------------------------------------------------------------------
// DirectX 11 compositor.
//----------------------------------------------------------------------------
Compositor::Initialization Compositor::initialize(HWND& hwnd, const Cogwheel::Core::Window& window, RendererCreator renderer_creator) {
    Compositor* c = new Compositor(hwnd, window);
    if (!c->m_impl->is_valid()) {
        delete c;
        return { nullptr, Renderers::UID::invalid_UID() };
    }
    
    Renderers::UID renderer_ID = c->attach_renderer(renderer_creator);
    if (renderer_ID == Renderers::UID::invalid_UID()) {
        delete c;
        return { nullptr, Renderers::UID::invalid_UID() };
    }

    c->set_active_renderer(renderer_ID);

    return { c, renderer_ID };
}

Compositor::Compositor(HWND& hwnd, const Cogwheel::Core::Window& window) {
    m_impl = new Implementation(hwnd, window);
}

Compositor::~Compositor() {
    delete m_impl;
}

Renderers::UID Compositor::attach_renderer(RendererCreator renderer_creator) {
    return m_impl->attach_renderer(renderer_creator);
}

void Compositor::set_active_renderer(Renderers::UID renderer_ID) {
    m_impl->set_active_renderer(renderer_ID);
}

void Compositor::render() {
    m_impl->render();
}

} // NS DX11Renderer
