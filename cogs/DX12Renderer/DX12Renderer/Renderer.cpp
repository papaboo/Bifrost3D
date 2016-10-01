// DirectX 12 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX12Renderer/Renderer.h>

#define NOMINMAX
#include <D3D12.h>
#include <dxgi1_4.h>
#include "d3dx12.h"
#undef RGB

#include <Cogwheel/Core/Window.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/SceneRoot.h>

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

namespace DX12Renderer {

template<typename ResourcePtr>
void safe_release(ResourcePtr* resource_ptr) {
    if (*resource_ptr) {
        (*resource_ptr)->Release();
        *resource_ptr = nullptr;
    }
}

//----------------------------------------------------------------------------
// DirectX 12 renderer state.
//----------------------------------------------------------------------------
struct Renderer::State {
    ID3D12Device* device;

    ID3D12CommandQueue* render_queue;
    IDXGISwapChain3* swap_chain;
    int frame_index; // TODO Do I need to store that here? Can't I just propagate it around while rendering? NOTE Isn's so much a frame_index as a current_backbuffer_ID.

    // TODO Combine all backbuffer 'stuff' into a struct.
    ID3D12DescriptorHeap* backbuffer_descriptors;
    std::vector<ID3D12Resource*> backbuffers;
    std::vector<ID3D12CommandAllocator*> backbuffer_command_allocators;

    std::vector<ID3D12Fence*> fence;
    std::vector<UINT64> fence_value; // this value is incremented each frame. Each fence has its own value.
    HANDLE fence_event; // A handle to an event that occurs when our fence is unlocked/passed by the gpu.

    ID3D12GraphicsCommandList* command_list;

    struct {
        unsigned int CBV_SRV_UAV_descriptor;
        unsigned int sampler_descriptor;
        unsigned int RTV_descriptor;
    } size_of;
};

//----------------------------------------------------------------------------
// DirectX 12 renderer.
//----------------------------------------------------------------------------
Renderer::Renderer(HWND& hwnd, const Cogwheel::Core::Window& window) {
    IDXGIFactory4* dxgi_factory; // TODO Make it a ComPtr? To make cleaning up easy.
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi_factory));
    if (FAILED(hr))
        return;

    ID3D12Device* device = nullptr;
    { // Find the best performing device (apparently the one with the most memory) and initialize that.
        IDXGIAdapter1* adapter = nullptr;

        struct WeightedAdapter {
            int Index, DedicatedMemory;

            inline bool operator<(WeightedAdapter rhs) const {
                return rhs.DedicatedMemory < DedicatedMemory;
            }
        };

        std::vector<WeightedAdapter> sorted_adapters;
        for (int adapter_index = 0; dxgi_factory->EnumAdapters1(adapter_index, &adapter) != DXGI_ERROR_NOT_FOUND; ++adapter_index) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);

            // Ignore software rendering adapters.
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                continue;

            WeightedAdapter e = { adapter_index, int(desc.DedicatedVideoMemory >> 20) };
            sorted_adapters.push_back(e);
        }

        std::sort(sorted_adapters.begin(), sorted_adapters.end());

        for (WeightedAdapter a : sorted_adapters) {
            dxgi_factory->EnumAdapters1(a.Index, &adapter);

            // We need a device that is compatible with direct3d 12 (feature level 11 or higher)
            hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
            if (SUCCEEDED(hr)) {
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);
                printf("DX12Renderer using device %u: '%S' with feature level 11.0.\n", a.Index, desc.Description);
                break;
            }
        }
    }

    if (device == nullptr)
        return;

    m_state = new State();
    m_state->device = device;

    // We have a valid device. Time to initialize the renderer!

    const UINT backbuffer_count = 2;
    m_state->size_of.CBV_SRV_UAV_descriptor = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_state->size_of.sampler_descriptor = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    m_state->size_of.RTV_descriptor = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    { // Create the rendering command queue.
        D3D12_COMMAND_QUEUE_DESC description = {};
        description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        description.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        description.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE; // Can be used to disable TDR.

        hr = device->CreateCommandQueue(&description, IID_PPV_ARGS(&m_state->render_queue));
        if (FAILED(hr)) {
            release_state();
            return;
        }
    }

    { // Swap chain.
        DXGI_SWAP_CHAIN_DESC description = {};
        description.BufferDesc.Width = window.get_width();
        description.BufferDesc.Height = window.get_height();;
        description.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        description.BufferDesc.RefreshRate = { 1, 60 };
        description.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
        description.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
        description.SampleDesc.Count = 1; // No multi sampling. TODO Create enum and add multisample support.
        description.SampleDesc.Quality = 0; // No quality? :)
        description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        description.BufferCount = backbuffer_count;
        description.OutputWindow = hwnd;
        description.Windowed = TRUE;
        description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD; // TODO MiniEngine uses FLIP_SEQUENTIALLY.
        description.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

        IDXGISwapChain* swap_chain_interface;
        hr = dxgi_factory->CreateSwapChain(m_state->render_queue, &description, &swap_chain_interface);
        if (FAILED(hr)) {
            release_state();
            return;
        }
        // Downcast the IDXGISwapChain to a IDXGISwapChain3. NOTE MiniEngine is perfectly happy with the swapchain1. TODO And copy their initialization code as itøs cleaner.
        hr = swap_chain_interface->QueryInterface(__uuidof(IDXGISwapChain3), (void**)&m_state->swap_chain);
        if (FAILED(hr)) {
            release_state();
            return;
        }
        // m_state->swap_chain = static_cast<IDXGISwapChain3*>(swap_chain_interface); // TODO Can I create the SwapChain3 directly? ANd have a look at MiniEngine. They don't need SwapChain3, but SwapChain1.

        m_state->frame_index = m_state->swap_chain->GetCurrentBackBufferIndex();
    }

    { // Create the backbuffer's render target views and their descriptor heap.

        D3D12_DESCRIPTOR_HEAP_DESC description = {};
        description.NumDescriptors = backbuffer_count;
        description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        hr = device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&m_state->backbuffer_descriptors));
        if (FAILED(hr)) {
            release_state();
            return;
        }

        // Create a RTV for each backbuffer.
        D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = m_state->backbuffer_descriptors->GetCPUDescriptorHandleForHeapStart();
        m_state->backbuffers.resize(backbuffer_count);
        for (int i = 0; i < backbuffer_count; ++i) {
            hr = m_state->swap_chain->GetBuffer(i, IID_PPV_ARGS(&m_state->backbuffers[i]));
            if (FAILED(hr)) {
                release_state();
                return;
            }

            device->CreateRenderTargetView(m_state->backbuffers[i], nullptr, rtv_handle);

            rtv_handle.ptr += m_state->size_of.RTV_descriptor;
        }
    }

    { // Create the command allocators pr backbuffer.
        m_state->backbuffer_command_allocators.resize(backbuffer_count);
        for (int i = 0; i < backbuffer_count; ++i) {
            hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_state->backbuffer_command_allocators[i]));
            if (FAILED(hr)) {
                release_state();
                return;
            }
        }
    }

    { // Create the command list.
        const UINT device_0 = 0;
        ID3D12PipelineState* initial_pipeline = nullptr;
        hr = device->CreateCommandList(device_0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_state->backbuffer_command_allocators[0], 
                                       initial_pipeline, IID_PPV_ARGS(&m_state->command_list));
        if (FAILED(hr)) {
            release_state();
            return;
        }
        m_state->command_list->Close(); // Close the command list, as we do not want to start recording yet.
    }

    { // Setup the fences for the backbuffers.

        m_state->fence.resize(backbuffer_count);
        m_state->fence_value.resize(backbuffer_count);

        // Create the fences and set their initial value.
        for (int i = 0; i < backbuffer_count; ++i) {
            const unsigned int initial_value = 0;
            hr = device->CreateFence(initial_value, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_state->fence[i]));
            if (FAILED(hr)) {
                release_state();
                return;
            }
            m_state->fence_value[i] = initial_value;
        }

        // Create handle to a fence event.
        m_state->fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (m_state->fence_event == nullptr) {
            release_state();
            return;
        }
    }
}

Renderer* Renderer::initialize(HWND& hwnd, const Cogwheel::Core::Window& window) {
    Renderer* r = new Renderer(hwnd, window);
    if (r != nullptr && r->m_state != nullptr)
        return r;
    else
        return nullptr;
}

Renderer::~Renderer() {
    release_state();
}

void Renderer::release_state() {
    if (m_state->device == nullptr)
        return;

    // Wait for the gpu to finish all frames
    // TODO Only do this if the GPU has actually started rendering.
    for (int i = 0; i < m_state->backbuffers.size(); ++i) {
        m_state->frame_index = i;
        wait_for_previous_frame();
    }

    // Get swapchain out of full screen before exiting.
    BOOL is_fullscreen_on = false;
    m_state->swap_chain->GetFullscreenState(&is_fullscreen_on, NULL);
    if (is_fullscreen_on)
        m_state->swap_chain->SetFullscreenState(false, NULL);

    safe_release(&m_state->device);
    safe_release(&m_state->swap_chain);
    safe_release(&m_state->render_queue);
    safe_release(&m_state->backbuffer_descriptors);
    safe_release(&m_state->command_list);

    for (int i = 0; i < m_state->backbuffers.size(); ++i) {
        safe_release(&m_state->backbuffers[i]);
        safe_release(&m_state->backbuffer_command_allocators[i]);
        safe_release(&m_state->fence[i]);
    }

    m_state = nullptr;
}

void Renderer::render() {
    if (Cameras::begin() == Cameras::end())
        return;

    wait_for_previous_frame();
    
    if (m_state == nullptr)
        return;

    handle_updates();

    // We can only reset an allocator once the gpu is done with it.
    // Resetting an allocator frees the memory that the command list was stored in.
    ID3D12CommandAllocator* command_allocator = m_state->backbuffer_command_allocators[m_state->frame_index];
    HRESULT hr = command_allocator->Reset();
    if (FAILED(hr))
        return release_state();

    // Reset the command list. Incidentally also sets it to record.
    hr = m_state->command_list->Reset(command_allocator, NULL);
    if (FAILED(hr))
        return release_state();

    { // Record commands.
        ID3D12GraphicsCommandList* command_list = m_state->command_list;

        // Transition the 'frame_index' render target from the present state to the render target state so the command list draws to it starting from here.
        command_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_state->backbuffers[m_state->frame_index], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

        // Here we get the handle to our current render target view so we can set it as the render target in the output merger stage of the pipeline.
        CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handle(m_state->backbuffer_descriptors->GetCPUDescriptorHandleForHeapStart(), m_state->frame_index, m_state->size_of.RTV_descriptor);

        // Set the render target for the output merger stage (the output of the pipeline).
        command_list->OMSetRenderTargets(1, &rtv_handle, FALSE, nullptr);

        // Clear the render target to the background color.
        Cameras::UID camera_ID = *Cameras::begin();
        SceneRoot scene = Cameras::get_scene_ID(camera_ID);
        RGB env_tint = scene.get_environment_tint();
        float environment_tint[] = { env_tint.r, env_tint.g, env_tint.b, 1.0f };
        command_list->ClearRenderTargetView(rtv_handle, environment_tint, 0, nullptr);

        // Transition the frame_index'th render target from the render target state to the present state. If the debug layer is enabled, you will receive a
        // warning if present is called on the render target when it's not in the present state.
        command_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_state->backbuffers[m_state->frame_index], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

        hr = command_list->Close();
        if (FAILED(hr))
            return release_state();
    }

    { // Render, i.e. playback command list.

        // Create an array of command lists and execute.
        ID3D12CommandList* command_list_list[] = { m_state->command_list };
        m_state->render_queue->ExecuteCommandLists(1, command_list_list);

        // This command goes in at the end of our command queue. we will know when our command queue 
        // has finished because the fence value will be set to 'fence_value' from the GPU since the command
        // queue is being executed on the GPU.
        hr = m_state->render_queue->Signal(m_state->fence[m_state->frame_index], m_state->fence_value[m_state->frame_index]);
        if (FAILED(hr))
            return release_state();

        // Present the current backbuffer.
        hr = m_state->swap_chain->Present(0, 0);
        if (FAILED(hr))
            return release_state();
    }
}

void Renderer::wait_for_previous_frame() {
    // Swap the current backbuffer index so we draw on the correct buffer.
    int frame_index = m_state->frame_index = m_state->swap_chain->GetCurrentBackBufferIndex();

    // If the current fence value is still less than 'fence_value', then we know the GPU has not finished executing
    // the command queue since it has not reached the 'commandQueue->Signal(fence, fenceValue)' command.
    if (m_state->fence[frame_index]->GetCompletedValue() < m_state->fence_value[frame_index])
    {
        // We have the fence create an event which is signaled once the fence's current value is 'fence_value'.
        HRESULT hr = m_state->fence[frame_index]->SetEventOnCompletion(m_state->fence_value[frame_index], m_state->fence_event);
        if (FAILED(hr))
            return release_state();

        // We will wait until the fence has triggered the event that it's current value has reached "fenceValue". once it's value
        // has reached 'fence_value', we know the command queue has finished executing.
        WaitForSingleObject(m_state->fence_event, INFINITE);
    }

    // increment fence value for next frame.
    m_state->fence_value[frame_index]++;
}

void Renderer::handle_updates() {

}

} // NS DX12Renderer
