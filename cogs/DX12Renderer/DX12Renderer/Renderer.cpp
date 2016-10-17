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
#include <D3DCompiler.h>
#include <dxgi1_4.h>
#include "d3dx12.h"
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

namespace DX12Renderer {

template<typename ResourcePtr>
void safe_release(ResourcePtr* resource_ptr) {
    if (*resource_ptr) {
        (*resource_ptr)->Release();
        *resource_ptr = nullptr;
    }
}

// TODO Handle cso files and errors related to files not found.
inline D3D12_SHADER_BYTECODE compile_shader(std::wstring filename, const char* target) {
    std::wstring qualified_filename = L"../Data/DX12Renderer/Shaders/" + filename;

    ID3DBlob* shader;
    ID3DBlob* error_messages = nullptr;
    HRESULT hr = D3DCompileFromFile(qualified_filename.c_str(),
        nullptr, // macroes
        nullptr, // Include dirs. TODO "../Data/DX12Renderer/Shaders/"
        "main",
        target,
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
        0, // More flags. Unused.
        &shader,
        &error_messages);
    if (FAILED(hr)) { // File not found not handled? Path not found unhandled as well.
        if (error_messages != nullptr)
            printf("Shader error: '%s'\n", (char*)error_messages->GetBufferPointer());
        return { nullptr, 0 };
    }

    // fill out a shader bytecode structure, which is basically just a pointer
    // to the shader bytecode and the size of the shader bytecode
    D3D12_SHADER_BYTECODE shader_bytecode = {};
    shader_bytecode.BytecodeLength = shader->GetBufferSize();
    shader_bytecode.pShaderBytecode = shader->GetBufferPointer();
    return shader_bytecode;
}

//----------------------------------------------------------------------------
// DirectX 12 renderer implementation.
//----------------------------------------------------------------------------
class Renderer::Implementation {
private:
    ID3D12Device* m_device;

    ID3D12CommandQueue* render_queue;
    IDXGISwapChain3* m_swap_chain;

    ID3D12DescriptorHeap* m_backbuffer_descriptors;
    struct Backbuffer {
        ID3D12Resource* resource; // Can this be typed more explicitly?
        ID3D12CommandAllocator* command_allocator;
        ID3D12Fence* fence;
        UINT64 fence_value;
    };
    std::vector<Backbuffer> m_backbuffers;
    int m_active_backbuffer_index;
    Backbuffer& active_backbuffer() { return m_backbuffers[m_active_backbuffer_index]; }

    HANDLE m_frame_rendered_event; // A handle to an event that occurs when our fence is unlocked/passed by the gpu.

    ID3D12GraphicsCommandList* m_command_list; // Can I query if this is recording? Otherwise wrap and make that available.

    struct {
        unsigned int CBV_SRV_UAV_descriptor;
        unsigned int sampler_descriptor;
        unsigned int RTV_descriptor;
    } size_of;

    struct {
        ID3D12RootSignature* root_signature; // Defines the data that the/a shader will access. Used by multiple shaders / PSO's?
        ID3D12PipelineState* pipeline_state_object;

        // These need to be set every time the command list is reset? Then move them to render!
        D3D12_VIEWPORT viewport; // area that output from rasterizer will be stretched to.
        D3D12_RECT scissor_rect; // the area to draw in. pixels outside that area will not be drawn onto

        ID3D12Resource* vertex_buffer; // a default buffer, perhaps make a typed buffer wrapper? At least wrap it somehow, since a 'resource' is very vague.
        D3D12_VERTEX_BUFFER_VIEW vertex_buffer_view; // NOTE Do we need one of these pr vertex buffer? If so group them or wrap the buffer in a view. Also, are views untyped in DX12?
    } m_triangle;

public:
    bool is_valid() { return m_device != nullptr; }

    Implementation(HWND& hwnd, const Cogwheel::Core::Window& window) {

        IDXGIFactory4* dxgi_factory;
        HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi_factory));
        if (FAILED(hr))
            return;

        { // Find the best performing device (apparently the one with the most memory) and initialize that.
            m_device = nullptr;
            IDXGIAdapter1* adapter = nullptr;

            struct WeightedAdapter {
                int index, dedicated_memory;

                inline bool operator<(WeightedAdapter rhs) const {
                    return rhs.dedicated_memory < dedicated_memory;
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
                dxgi_factory->EnumAdapters1(a.index, &adapter);

                // We need a device that is compatible with direct3d 12 (feature level 11 or higher)
                hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device));
                if (SUCCEEDED(hr)) {
                    DXGI_ADAPTER_DESC1 desc;
                    adapter->GetDesc1(&desc);
                    printf("DX12Renderer using device %u: '%S' with feature level 11.0.\n", a.index, desc.Description);
                    break;
                }
            }
        }

        if (m_device == nullptr)
            return;

        // We have a valid device. Time to initialize the renderer!

        const UINT backbuffer_count = 2;
        m_backbuffers.resize(backbuffer_count);

        size_of.CBV_SRV_UAV_descriptor = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        size_of.sampler_descriptor = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        size_of.RTV_descriptor = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        { // Create the rendering command queue.
            D3D12_COMMAND_QUEUE_DESC description = {};
            description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            description.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
            description.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE; // Can be used to disable TDR.

            hr = m_device->CreateCommandQueue(&description, IID_PPV_ARGS(&render_queue));
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
            hr = dxgi_factory->CreateSwapChain(render_queue, &description, &swap_chain_interface);
            if (FAILED(hr)) {
                release_state();
                return;
            }
            // Downcast the IDXGISwapChain to a IDXGISwapChain3. NOTE MiniEngine is perfectly happy with the swapchain1. TODO And copy their initialization code as itøs cleaner.
            hr = swap_chain_interface->QueryInterface(__uuidof(IDXGISwapChain3), (void**)&m_swap_chain);
            if (FAILED(hr)) {
                release_state();
                return;
            }

            m_active_backbuffer_index = m_swap_chain->GetCurrentBackBufferIndex();
        }

        { // Create the backbuffer's render target views and their descriptor heap.

            D3D12_DESCRIPTOR_HEAP_DESC description = {};
            description.NumDescriptors = backbuffer_count;
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
            hr = m_device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&m_backbuffer_descriptors));
            if (FAILED(hr)) {
                release_state();
                return;
            }

            // Create a RTV for each backbuffer.
            D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = m_backbuffer_descriptors->GetCPUDescriptorHandleForHeapStart();
            for (int i = 0; i < backbuffer_count; ++i) {
                hr = m_swap_chain->GetBuffer(i, IID_PPV_ARGS(&m_backbuffers[i].resource));
                if (FAILED(hr)) {
                    release_state();
                    return;
                }

                m_device->CreateRenderTargetView(m_backbuffers[i].resource, nullptr, rtv_handle);

                rtv_handle.ptr += size_of.RTV_descriptor;
            }
        }

        { // Create the command allocators pr backbuffer.
            for (int i = 0; i < backbuffer_count; ++i) {
                hr = m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_backbuffers[i].command_allocator));
                if (FAILED(hr)) {
                    release_state();
                    return;
                }
            }
        }

        { // Create the command list.
            const UINT device_0 = 0;
            ID3D12PipelineState* initial_pipeline = nullptr;
            hr = m_device->CreateCommandList(device_0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_backbuffers[0].command_allocator, // TODO Why do we only need one command list, but multiple allocators?
                initial_pipeline, IID_PPV_ARGS(&m_command_list));
            if (FAILED(hr)) {
                release_state();
                return;
            }
            // m_command_list->Close(); // TODO Close the command list, as we do not want to start recording yet.
        }

        { // Setup the fences for the backbuffers.

            // Create the fences and set their initial value.
            for (int i = 0; i < backbuffer_count; ++i) {
                const unsigned int initial_value = 0;
                hr = m_device->CreateFence(initial_value, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_backbuffers[i].fence));
                if (FAILED(hr)) {
                    release_state();
                    return;
                }
                m_backbuffers[i].fence_value = initial_value;
            }

            // Create handle to a event that occurs when a frame has been rendered.
            m_frame_rendered_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            if (m_frame_rendered_event == nullptr) {
                release_state();
                return;
            }
        }

        // dxgi_factory->Release();

        if (!initialize_triangle()) {
            release_state();
            return;
        }
    }

    void release_state() {
        if (m_device == nullptr)
            return;

        CloseHandle(m_frame_rendered_event);

        // Wait for the gpu to finish all frames
        // TODO Only do this if the GPU has actually started rendering.
        for (int m_active_backbuffer_index = 0; m_active_backbuffer_index < m_backbuffers.size(); ++m_active_backbuffer_index) {
            wait_for_previous_frame();
        }

        // Get swapchain out of full screen before exiting.
        BOOL is_fullscreen_on = false;
        m_swap_chain->GetFullscreenState(&is_fullscreen_on, NULL);
        if (is_fullscreen_on)
            m_swap_chain->SetFullscreenState(false, NULL);

        safe_release(&m_device);
        safe_release(&m_swap_chain);
        safe_release(&render_queue);
        safe_release(&m_backbuffer_descriptors);
        safe_release(&m_command_list);

        for (int i = 0; i < m_backbuffers.size(); ++i) {
            safe_release(&m_backbuffers[i].resource);
            safe_release(&m_backbuffers[i].command_allocator);
            safe_release(&m_backbuffers[i].fence);
        }
    }

    void render(const Cogwheel::Core::Engine& engine) {
        if (Cameras::begin() == Cameras::end())
            return;

        wait_for_previous_frame();

        if (m_device == nullptr)
            return;

        handle_updates();

        // We can only reset an allocator once the gpu is done with it.
        // Resetting an allocator frees the memory that the command list was stored in.
        ID3D12CommandAllocator* command_allocator = active_backbuffer().command_allocator;
        HRESULT hr = command_allocator->Reset();
        if (FAILED(hr))
            return release_state();

        { // Record commands.

            // Reset the command list. Incidentally also sets it to record.
            hr = m_command_list->Reset(command_allocator, nullptr);
            if (FAILED(hr))
                return release_state();

            ID3D12Resource* backbuffer_resource = active_backbuffer().resource;

            // Transition the 'm_active_backbuffer' render target from the present state to the render target state so the command list draws to it starting from here.
            CD3DX12_RESOURCE_BARRIER render_transition = CD3DX12_RESOURCE_BARRIER::Transition(
                backbuffer_resource, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
            m_command_list->ResourceBarrier(1, &render_transition);

            // Here we get the handle to our current render target view so we can set it as the render target in the output merger stage of the pipeline.
            CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handle(m_backbuffer_descriptors->GetCPUDescriptorHandleForHeapStart(), m_active_backbuffer_index, size_of.RTV_descriptor);

            // Set the render target for the output merger stage (the output of the pipeline).
            m_command_list->OMSetRenderTargets(1, &rtv_handle, FALSE, nullptr);

            // Clear the render target to the background color.
            Cameras::UID camera_ID = *Cameras::begin();
            SceneRoot scene = Cameras::get_scene_ID(camera_ID);
            RGB env_tint = scene.get_environment_tint();
            float environment_tint[] = { env_tint.r, env_tint.g, env_tint.b, 1.0f };
            m_command_list->ClearRenderTargetView(rtv_handle, environment_tint, 0, nullptr);

            { // Setup viewport.
                Cogwheel::Math::Rectf vp = Cameras::get_viewport(camera_ID);
                vp.width *= engine.get_window().get_width();
                vp.height *= engine.get_window().get_height();

                // Fill out a scissor rect.
                m_triangle.scissor_rect.left = LONG(vp.x);
                m_triangle.scissor_rect.top = LONG(vp.y);
                m_triangle.scissor_rect.right = LONG(vp.width);
                m_triangle.scissor_rect.bottom = LONG(vp.height);

                // Fill out the viewport.
                m_triangle.viewport.TopLeftX = vp.x;
                m_triangle.viewport.TopLeftY = vp.y;
                m_triangle.viewport.Width = vp.width;
                m_triangle.viewport.Height = vp.height;
                m_triangle.viewport.MinDepth = 0.0f;
                m_triangle.viewport.MaxDepth = 1.0f;

                m_command_list->RSSetViewports(1, &m_triangle.viewport);
                m_command_list->RSSetScissorRects(1, &m_triangle.scissor_rect);
            }

            { // Draw triangle.
                m_command_list->SetPipelineState(m_triangle.pipeline_state_object);
                m_command_list->SetGraphicsRootSignature(m_triangle.root_signature);
                m_command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                m_command_list->IASetVertexBuffers(0, 1, &m_triangle.vertex_buffer_view);
                m_command_list->DrawInstanced(3, 1, 0, 0);
            }

            // Transition the activerender target from the render target state to the present state.
            CD3DX12_RESOURCE_BARRIER present_transition = CD3DX12_RESOURCE_BARRIER::Transition(
                backbuffer_resource, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
            m_command_list->ResourceBarrier(1, &present_transition);

            hr = m_command_list->Close();
            if (FAILED(hr))
                return release_state();
        }

        { // Render, i.e. playback command list.

            // Create an array of command lists and execute.
            ID3D12CommandList* command_lists[] = { m_command_list };
            render_queue->ExecuteCommandLists(1, command_lists);

            // Signal the fence with the next fence value, so we can check if the fence has been reached.
            hr = render_queue->Signal(active_backbuffer().fence, active_backbuffer().fence_value);
            if (FAILED(hr))
                return release_state();

            // Present the current backbuffer.
            hr = m_swap_chain->Present(0, 0);
            if (FAILED(hr))
                return release_state();
        }
    }

    void wait_for_previous_frame() {
        // Swap the current backbuffer index so we draw on the correct buffer.
        m_active_backbuffer_index = m_swap_chain->GetCurrentBackBufferIndex();

        // If the current fence value is still less than 'fence_value', then we know the GPU has not finished executing
        // the command queue since it has not reached the 'commandQueue->Signal(fence, fenceValue)' command.
        if (active_backbuffer().fence->GetCompletedValue() < active_backbuffer().fence_value)
        {
            // We have the fence create an event which is signaled once the fence's current value is 'fence_value'.
            HRESULT hr = active_backbuffer().fence->SetEventOnCompletion(active_backbuffer().fence_value, m_frame_rendered_event);
            if (FAILED(hr))
                return release_state();

            // We will wait until the fence has triggered the event that it's current value has reached "fenceValue". once it's value
            // has reached 'fence_value', we know the command queue has finished executing.
            WaitForSingleObject(m_frame_rendered_event, INFINITE);
        }

        // increment fence value for next frame.
        active_backbuffer().fence_value++;
    }

    void handle_updates() {

    }

    bool initialize_triangle() {

        { // create root signature

            CD3DX12_ROOT_SIGNATURE_DESC root_signature_desc; // TODO Dont use this wrapper object.
            root_signature_desc.Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

            ID3DBlob* signature;
            HRESULT hr = D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, nullptr);
            if (FAILED(hr))
                return false;

            hr = m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_triangle.root_signature));
            if (FAILED(hr))
                return false;
        }

        { // Vertex input layout. TODO Define the ones that we care about once and store them in the state? Do we need anything other than position, normal and texcoords?

            D3D12_INPUT_ELEMENT_DESC input_elements[] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
            };
            D3D12_INPUT_LAYOUT_DESC input_layout_desc = { input_elements, 1 };

            // Create a pipeline state object.
            // We will need a couple of these PSO's for different kinds of configurations; blending x passes x input_layout(?)
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_description = {};
            pso_description.InputLayout = input_layout_desc;
            pso_description.pRootSignature = m_triangle.root_signature;
            pso_description.VS = compile_shader(L"VertexShader.hlsl", "vs_5_0");
            pso_description.PS = compile_shader(L"FragmentShader.hlsl", "ps_5_0");
            pso_description.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pso_description.NumRenderTargets = 1;
            pso_description.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM; // TODO add _SRGB, but the RTV itself doesn't support that format.
            // NOTE SampleDesc must be the same as used by the swap chain.
            pso_description.SampleDesc.Count = 1; // No multi sampling. TODO Create enum and add multisample support.
            pso_description.SampleDesc.Quality = 0; // No quality? :)
            pso_description.SampleMask = 0xffffffff; // Apparently 0xFFFFFFFF equals no multisampling.
            pso_description.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            pso_description.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            HRESULT hr = m_device->CreateGraphicsPipelineState(&pso_description, IID_PPV_ARGS(&m_triangle.pipeline_state_object));
            if (FAILED(hr))
                return false;
        }

        { // Create vertex buffer. // TODO Positions instead of vertex_buffer
            Vector3f vertex_buffer_data[] = {
                { Vector3f(0.0f, 0.5f, 0.5f) },
                { Vector3f(0.5f, -0.5f, 0.5f) },
                { Vector3f(-0.5f, -0.5f, 0.5f) },
            };
            int vertex_buffer_size = sizeof(vertex_buffer_data);

            // GPU vertex buffer.
            m_device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(vertex_buffer_size),
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr, // Null. Used for render targets and depth/stencil buffers.
                IID_PPV_ARGS(&m_triangle.vertex_buffer));
            m_triangle.vertex_buffer->SetName(L"Triangle vertex buffer");

            // Upload buffer.
            ID3D12Resource* vertex_upload_buffer;
            m_device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(vertex_buffer_size),
                D3D12_RESOURCE_STATE_GENERIC_READ, // Read buffer, i.e. upload buffer.
                nullptr, // Null. Used for render targets and depth/stencil buffers.
                IID_PPV_ARGS(&vertex_upload_buffer));
            vertex_upload_buffer->SetName(L"Vertex upload buffer");

            // Add vertex data to the upload buffer.
            D3D12_SUBRESOURCE_DATA vertex_data = {};
            vertex_data.pData = (void*)vertex_buffer_data;
            vertex_data.RowPitch = vertex_buffer_size;
            vertex_data.SlicePitch = vertex_buffer_size;

            // Copy the data from the upload buffer to the default buffer.
            UpdateSubresources(m_command_list, m_triangle.vertex_buffer, vertex_upload_buffer, 0, 0, 1, &vertex_data);

            // Change the vertex buffer state from a copy destination to a vertex buffer.
            m_command_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_triangle.vertex_buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));

            // Now we execute the command list to upload the initial assets (triangle data)
            m_command_list->Close();
            ID3D12CommandList* command_lists[] = { m_command_list };
            render_queue->ExecuteCommandLists(1, command_lists);

            // Increment the fence value now, otherwise the buffer might not be uploaded by the time we start drawing.
            active_backbuffer().fence_value++;
            HRESULT hr = render_queue->Signal(active_backbuffer().fence, active_backbuffer().fence_value);
            if (FAILED(hr))
                return false;

            // Create a vertex buffer view for the triangle.
            m_triangle.vertex_buffer_view.BufferLocation = m_triangle.vertex_buffer->GetGPUVirtualAddress();
            m_triangle.vertex_buffer_view.StrideInBytes = sizeof(Vector3f); // TODO Make a typed view.
            m_triangle.vertex_buffer_view.SizeInBytes = vertex_buffer_size;
        }

        return true;
    }
};

//----------------------------------------------------------------------------
// DirectX 12 renderer.
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

} // NS DX12Renderer
