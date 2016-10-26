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

//----------------------------------------------------------------------------
// DirectX 11 renderer implementation.
//----------------------------------------------------------------------------
class Renderer::Implementation {
private:
    ID3D11Device* m_device;

    IDXGISwapChain* m_swap_chain;

    ID3D11DeviceContext* m_render_context;
    ID3D11RenderTargetView* m_backbuffer_view;

public:
    bool is_valid() { return m_device != nullptr; }

    Implementation(HWND& hwnd, const Cogwheel::Core::Window& window) {
        { // Create device and swapchain.

            DXGI_MODE_DESC backbuffer_description;
            backbuffer_description.Width = window.get_width();
            backbuffer_description.Height = window.get_height();
            backbuffer_description.RefreshRate.Numerator = 60;
            backbuffer_description.RefreshRate.Denominator = 1;
            backbuffer_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // TODO sRGB
            backbuffer_description.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
            backbuffer_description.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;

            DXGI_SWAP_CHAIN_DESC swap_chain_description = {};
            swap_chain_description.BufferDesc = backbuffer_description;
            swap_chain_description.SampleDesc.Count = 1;
            swap_chain_description.SampleDesc.Quality = 0;
            swap_chain_description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            swap_chain_description.BufferCount = 1;
            swap_chain_description.OutputWindow = hwnd;
            swap_chain_description.Windowed = TRUE;
            swap_chain_description.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

            // TODO Replace by explicit device enumeration and selecting of 'largest' device. Otherwise we risk selecting the integrated GPU.
            D3D_FEATURE_LEVEL feature_level;
            HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                D3D11_SDK_VERSION, &swap_chain_description, &m_swap_chain, &m_device, &feature_level, &m_render_context);

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

        // ?? wait_for_previous_frame();

        if (m_device == nullptr)
            return;

        handle_updates();

        RGBA environment_tint = RGBA(1.0f, 0.5f, 0.1f, 1.0f);
        m_render_context->ClearRenderTargetView(m_backbuffer_view, environment_tint.begin());

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
