// DirectX 11 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/Renderer.h>
#include <DX11Renderer/Types.h>

#define NOMINMAX
#include <D3D11.h>
#include <D3DCompiler.h>
#undef RGB

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Core/Window.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/SceneRoot.h>

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;
using namespace std;

namespace DX11Renderer {

template<typename ResourcePtr>
void safe_release(ResourcePtr* resource_ptr) {
    if (*resource_ptr) {
        (*resource_ptr)->Release();
        *resource_ptr = nullptr;
    }
}

#define UNPACK_BLOB_ARGS(blob) blob->GetBufferPointer(), blob->GetBufferSize()

// TODO Handle cso files and errors related to files not found.
// Specialize blob so I can return it by value?
inline ID3DBlob* compile_shader(std::wstring filename, const char* target) {
    std::wstring qualified_filename = L"../Data/DX11Renderer/Shaders/" + filename;

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

    ID3D11DeviceContext* m_render_context; // Is this the same as the immediate context?

    // Backbuffer members.
    Vector2ui m_backbuffer_size;
    ID3D11RenderTargetView* m_backbuffer_view;
    ID3D11Texture2D* m_depth_buffer;
    ID3D11DepthStencilView* m_depth_view;

    vector<Transform> m_transforms = vector<Transform>(0);
    vector<Dx11Mesh> m_meshes = vector<Dx11Mesh>(0);
    vector<Dx11Model> m_models = vector<Dx11Model>(0);

    struct {
        ID3D11Buffer* positions_buffer;
        ID3D11Buffer* uniforms_buffer;
        ID3D11InputLayout* vertex_layout;
        ID3D10Blob* vertex_shader_buffer;
        ID3D10Blob* pixel_shader_buffer;
        ID3D11VertexShader* vertex_shader;
        ID3D11PixelShader* pixel_shader;
    } m_triangle;

    struct Uniforms {
        Matrix4x4f mvp_matrix;
        Vector4f offset;
        RGBA color;
    };

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

        { // Setup backbuffer.

            m_backbuffer_size = Vector2ui::zero();

            // Get and set render target. // TODO How is resizing of the color buffer handled?
            ID3D11Texture2D* backbuffer;
            HRESULT hr = m_swap_chain->GetBuffer(0, IID_PPV_ARGS(&backbuffer));
            hr = m_device->CreateRenderTargetView(backbuffer, nullptr, &m_backbuffer_view);
            backbuffer->Release();

            // Depth buffer is initialized on demand when the output dimensions are known.
            m_depth_buffer = nullptr;
            m_depth_view = nullptr;
        }

        setup_triangle();
    }

    void setup_triangle() {
        { // Compile shaders.
            m_triangle.vertex_shader_buffer = compile_shader(L"VertexShader.hlsl", "vs_5_0");
            m_triangle.pixel_shader_buffer = compile_shader(L"FragmentShader.hlsl", "ps_5_0");
            
            // Create the shader objects.
            HRESULT hr = m_device->CreateVertexShader(UNPACK_BLOB_ARGS(m_triangle.vertex_shader_buffer), NULL, &m_triangle.vertex_shader);
            hr = m_device->CreatePixelShader(UNPACK_BLOB_ARGS(m_triangle.pixel_shader_buffer), NULL, &m_triangle.pixel_shader);
        }

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

            D3D11_SUBRESOURCE_DATA positions_buffer_data = {};
            positions_buffer_data.pSysMem = positions;
            HRESULT hr = m_device->CreateBuffer(&position_buffer_desc, &positions_buffer_data, &m_triangle.positions_buffer);
        }

        // Create the input layout
        D3D11_INPUT_ELEMENT_DESC position_layout_desc =
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 };

        m_device->CreateInputLayout(&position_layout_desc, 1, UNPACK_BLOB_ARGS(m_triangle.vertex_shader_buffer), &m_triangle.vertex_layout);

        { // Uniforms.
            D3D11_BUFFER_DESC uniforms_desc = {};
            uniforms_desc.Usage = D3D11_USAGE_DEFAULT;
            uniforms_desc.ByteWidth = sizeof(Uniforms);
            uniforms_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            uniforms_desc.CPUAccessFlags = 0;
            uniforms_desc.MiscFlags = 0;

            HRESULT hr = m_device->CreateBuffer(&uniforms_desc, NULL, &m_triangle.uniforms_buffer);
        }
    }

    void release_state() {
        if (m_device == nullptr)
            return;

        safe_release(&m_device);
        safe_release(&m_render_context);
        safe_release(&m_swap_chain);

        safe_release(&m_backbuffer_view);
        safe_release(&m_depth_buffer);
        safe_release(&m_depth_view);

        for (Dx11Mesh mesh : m_meshes) {
            safe_release(&mesh.indices);
            safe_release(&mesh.positions);
        }
    }

    void render(const Cogwheel::Core::Engine& engine) {
        if (Cameras::begin() == Cameras::end())
            return;

        // ?? wait_for_previous_frame(); // Why isn't this needed?

        if (m_device == nullptr)
            return;

        handle_updates();

        Cameras::UID camera_ID = *Cameras::begin();
        const Window& window = engine.get_window();

        Vector2ui current_backbuffer_size = Vector2ui(window.get_width(), window.get_height());
        if (m_backbuffer_size != current_backbuffer_size) {
            if (m_depth_buffer) m_depth_buffer->Release();
            if (m_depth_view) m_depth_view->Release();

            D3D11_TEXTURE2D_DESC depth_desc;
            depth_desc.Width = current_backbuffer_size.x;
            depth_desc.Height = current_backbuffer_size.y;
            depth_desc.MipLevels = 1;
            depth_desc.ArraySize = 1;
            depth_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; // TODO DXGI_FORMAT_D32_FLOAT ?? Do I have to change the viewport min/max as well then?
            depth_desc.SampleDesc.Count = 1;
            depth_desc.SampleDesc.Quality = 0;
            depth_desc.Usage = D3D11_USAGE_DEFAULT;
            depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
            depth_desc.CPUAccessFlags = 0;
            depth_desc.MiscFlags = 0;

            HRESULT hr = m_device->CreateTexture2D(&depth_desc, NULL, &m_depth_buffer);
            m_device->CreateDepthStencilView(m_depth_buffer, NULL, &m_depth_view);

            m_render_context->OMSetRenderTargets(1, &m_backbuffer_view, m_depth_view);

            m_backbuffer_size = current_backbuffer_size;
        }

        { // Create and set the viewport.
            Cogwheel::Math::Rectf vp = Cameras::get_viewport(camera_ID);
            vp.width *= window.get_width();
            vp.height *= window.get_height();

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
        m_render_context->ClearDepthStencilView(m_depth_view, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

        { // Render triangle.
            // Set vertex and pixel shaders.
            m_render_context->VSSetShader(m_triangle.vertex_shader, 0, 0);
            m_render_context->PSSetShader(m_triangle.pixel_shader, 0, 0);

            // Set the vertex buffer
            UINT stride = sizeof(float) * 3;
            UINT offset = 0;
            m_render_context->IASetVertexBuffers(0, 1, &m_triangle.positions_buffer, &stride, &offset);
            
            m_render_context->IASetInputLayout(m_triangle.vertex_layout);
            m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            {
                float t = (float)engine.get_time().get_total_time();
                Uniforms uniforms;
                uniforms.mvp_matrix = Cameras::get_view_projection_matrix(camera_ID);
                uniforms.offset = Vector4f(0.25f * sinf(t), 0.25f * sinf(t * 0.74f + 0.13f), 0.1f * sinf(t), 0);
                uniforms.color = RGBA::green();
                m_render_context->UpdateSubresource(m_triangle.uniforms_buffer, 0, NULL, &uniforms, 0, 0);
                m_render_context->VSSetConstantBuffers(0, 1, &m_triangle.uniforms_buffer);
                m_render_context->PSSetConstantBuffers(0, 1, &m_triangle.uniforms_buffer);
            }

            m_render_context->Draw(3, 0);

            {
                float t = (float)engine.get_time().get_total_time() + 2.0f;
                Uniforms uniforms;
                uniforms.mvp_matrix = Cameras::get_view_projection_matrix(camera_ID);
                uniforms.offset = Vector4f(0.25f * sinf(t), 0.25f * sinf(t * 0.74f + 0.13f), 0.1f * sinf(t), 0);
                uniforms.color = RGBA::blue();
                m_render_context->UpdateSubresource(m_triangle.uniforms_buffer, 0, NULL, &uniforms, 0, 0);
                m_render_context->VSSetConstantBuffers(0, 1, &m_triangle.uniforms_buffer);
                m_render_context->PSSetConstantBuffers(0, 1, &m_triangle.uniforms_buffer);
            }

            m_render_context->Draw(3, 0);
        }

        { // Render models.
            for (Dx11Model model : m_models) {
                if (model.mesh_ID == 0)
                    continue;

                Dx11Mesh mesh = m_meshes[model.mesh_ID];

                // Set vertex and pixel shaders.
                m_render_context->VSSetShader(m_triangle.vertex_shader, 0, 0);
                m_render_context->PSSetShader(m_triangle.pixel_shader, 0, 0);

                m_render_context->IASetInputLayout(m_triangle.vertex_layout);
                m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                { // Set the buffers.
                    if (mesh.index_count != 0)
                        m_render_context->IASetIndexBuffer(mesh.indices, DXGI_FORMAT_R32_UINT, 0);

                    UINT stride = sizeof(float) * 3;
                    UINT offset = 0;
                    m_render_context->IASetVertexBuffers(0, 1, &mesh.positions, &stride, &offset);
                }

                {
                    static RGBA color[] = { RGBA::red(), RGBA::yellow(), RGBA::green(), RGBA::blue() };
                    Uniforms uniforms;
                    uniforms.mvp_matrix = Cameras::get_view_projection_matrix(camera_ID) * to_matrix4x4(m_transforms[model.transform_ID]);
                    uniforms.offset = Vector4f::zero();
                    uniforms.color = color[model.mesh_ID % 4];
                    m_render_context->UpdateSubresource(m_triangle.uniforms_buffer, 0, NULL, &uniforms, 0, 0);
                    m_render_context->VSSetConstantBuffers(0, 1, &m_triangle.uniforms_buffer);
                    m_render_context->PSSetConstantBuffers(0, 1, &m_triangle.uniforms_buffer);
                }

                if (mesh.index_count != 0)
                    m_render_context->DrawIndexed(mesh.index_count, 0, 0);
                else
                    m_render_context->Draw(mesh.vertex_count, 0);
            }
        }

        // Present the backbuffer.
        m_swap_chain->Present(0, 0);
    }

    void handle_updates() {
        // TODO Handle updates in multiple command lists.

        { // Mesh updates.
            for (Meshes::UID mesh_ID : Meshes::get_changed_meshes()) {
                if (Meshes::get_changes(mesh_ID) == Meshes::Changes::Destroyed) {
                    if (mesh_ID < m_meshes.size() && m_meshes[mesh_ID].vertex_count != 0) {
                        m_meshes[mesh_ID].index_count = m_meshes[mesh_ID].vertex_count = 0;
                        safe_release(&m_meshes[mesh_ID].indices);
                        safe_release(&m_meshes[mesh_ID].positions);
                    }
                }

                if (Meshes::get_changes(mesh_ID) & Meshes::Changes::Created) {
                    if (m_meshes.size() <= mesh_ID)
                        m_meshes.resize(Meshes::capacity());

                    Cogwheel::Assets::Mesh mesh = mesh_ID;
                    Dx11Mesh dx_mesh;

                    { // Upload indices.
                        dx_mesh.index_count = mesh.get_index_count() * 3;

                        D3D11_BUFFER_DESC indices_desc = {};
                        indices_desc.Usage = D3D11_USAGE_DEFAULT;
                        indices_desc.ByteWidth = sizeof(unsigned int) * dx_mesh.index_count;
                        indices_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
                        indices_desc.CPUAccessFlags = 0;
                        indices_desc.MiscFlags = 0;

                        D3D11_SUBRESOURCE_DATA indices_data = {};
                        indices_data.pSysMem = mesh.get_indices();
                        HRESULT hr = m_device->CreateBuffer(&indices_desc, &indices_data, &dx_mesh.indices);
                        if (FAILED(hr))
                            printf("Could not upload '%s' index buffer.\n");
                    }

                    { // Upload positions.
                        dx_mesh.vertex_count = mesh.get_vertex_count();

                        D3D11_BUFFER_DESC position_desc = {};
                        position_desc.Usage = D3D11_USAGE_DEFAULT;
                        position_desc.ByteWidth = sizeof(Vector3f) * dx_mesh.vertex_count;
                        position_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
                        position_desc.CPUAccessFlags = 0;
                        position_desc.MiscFlags = 0;

                        D3D11_SUBRESOURCE_DATA positions_data = {};
                        positions_data.pSysMem = mesh.get_positions();
                        HRESULT hr = m_device->CreateBuffer(&position_desc, &positions_data, &dx_mesh.positions);
                        if (FAILED(hr))
                            printf("Could not upload '%s' position buffer.\n");
                    }

                    m_meshes[mesh_ID] = dx_mesh;
                }
            }
        }

        { // Transform updates.
            // TODO We're only interested in changes in the transforms that are connected to renderables, such as meshes.
            bool important_transform_changed = false;
            for (SceneNodes::UID node_ID : SceneNodes::get_changed_nodes()) {
                if (SceneNodes::has_changes(node_ID, SceneNodes::Changes::Created | SceneNodes::Changes::Transform)) {

                    if (m_transforms.size() <= node_ID)
                        m_transforms.resize(SceneNodes::capacity());

                    // NOTE Store the inverse excplicitly?
                    m_transforms[node_ID] = SceneNodes::get_global_transform(node_ID);
                }
            }
        }

        { // Model updates
            for (MeshModel model : MeshModels::get_changed_models()) {
                unsigned int model_index = model.get_ID();

                if (model.get_changes() == MeshModels::Changes::Destroyed) {
                    if (model_index < m_models.size()) {
                        m_models[model_index].material_ID = 0;
                        m_models[model_index].mesh_ID = 0;
                        m_models[model_index].transform_ID = 0;
                    }
                }

                if (model.get_changes() & MeshModels::Changes::Created) {
                    if (m_models.size() <= model_index)
                        m_models.resize(MeshModels::capacity());

                    // This info could actually just be memcopied from the datamodel to the rendermodel.
                    m_models[model_index].material_ID = model.get_material().get_ID();
                    m_models[model_index].mesh_ID = model.get_mesh().get_ID();
                    m_models[model_index].transform_ID = model.get_scene_node().get_ID();
                }
            }
        }
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
