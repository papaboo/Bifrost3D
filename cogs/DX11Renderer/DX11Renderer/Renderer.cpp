// DirectX 11 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/LightManager.h>
#include <DX11Renderer/Renderer.h>
#include <DX11Renderer/Types.h>

#define NOMINMAX
#include <D3D11.h>
#include <D3DCompiler.h>
#undef RGB

#include <Cogwheel/Assets/Material.h>
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
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
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

    vector<Dx11Material> m_materials = vector<Dx11Material>(0);
    vector<Dx11Mesh> m_meshes = vector<Dx11Mesh>(0);
    vector<Dx11Model> m_models = vector<Dx11Model>(0);
    vector<Transform> m_transforms = vector<Transform>(0);

    LightManager m_lights;

    struct {
        ID3D11Buffer* null_buffer;
        ID3D11InputLayout* input_layout;
        ID3D11VertexShader* shader;
    } m_vertex_shading;

    struct {
        ID3D11DepthStencilState* depth_state;
        ID3D11PixelShader* shader;
    } m_opaque;

    // Catch-all uniforms. Split into model/spatial and surface shading.
    struct Uniforms {
        Matrix4x4f mvp_matrix;
        Matrix4x3f to_world_matrix;
        RGBA color;
        Vector4i flags;
    };
    ID3D11Buffer* uniforms_buffer;

public:
    bool is_valid() { return m_device != nullptr; }

    Implementation(HWND& hwnd, const Cogwheel::Core::Window& window) {
        { // Create device and swapchain.

            DXGI_MODE_DESC backbuffer_desc;
            backbuffer_desc.Width = window.get_width();
            backbuffer_desc.Height = window.get_height();
            backbuffer_desc.RefreshRate.Numerator = 60;
            backbuffer_desc.RefreshRate.Denominator = 1;
            backbuffer_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
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

            { // Find the best performing device (apparently the one with the most memory) and initialize that.
                struct WeightedAdapter {
                    int index, dedicated_memory;

                    inline bool operator<(WeightedAdapter rhs) const {
                        return rhs.dedicated_memory < dedicated_memory;
                    }
                };

                m_device = nullptr;
                IDXGIAdapter1* adapter = nullptr;

                IDXGIFactory1* dxgi_factory;
                HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory));

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

                    // TODO Use DX 11.1. See https://blogs.msdn.microsoft.com/chuckw/2014/02/05/anatomy-of-direct3d-11-create-device/
                    D3D_FEATURE_LEVEL feature_level;
                    HRESULT hr = D3D11CreateDeviceAndSwapChain(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
                        D3D11_SDK_VERSION, &swap_chain_desc, &m_swap_chain, &m_device, &feature_level, &m_render_context);
                    if (SUCCEEDED(hr)) {
                        DXGI_ADAPTER_DESC adapter_description;
                        adapter->GetDesc(&adapter_description);
                        std::string readable_feature_level = feature_level == D3D_FEATURE_LEVEL_11_0 ? "11.0" : "11.1";
                        printf("DX11Renderer using device '%S' with feature level %s.\n", adapter_description.Description, readable_feature_level.c_str());
                        break;
                    }
                }
                dxgi_factory->Release();
            }

            if (m_device == nullptr) {
                release_state();
                return;
            }
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

        { // Setup vertex processing.
            ID3D10Blob* vertex_shader_blob = compile_shader(L"VertexShader.hlsl", "vs_5_0");

            // Create the shader objects.
            HRESULT hr = m_device->CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), NULL, &m_vertex_shading.shader);

            // Create the input layout
            D3D11_INPUT_ELEMENT_DESC input_layout_desc[] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 2, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
            };

            hr = m_device->CreateInputLayout(input_layout_desc, 3, UNPACK_BLOB_ARGS(vertex_shader_blob), &m_vertex_shading.input_layout);
            if (FAILED(hr)) {
                release_state();
                return;
            }

            // Create a default emptyish buffer.
            D3D11_BUFFER_DESC empty_desc = {};
            empty_desc.Usage = D3D11_USAGE_DEFAULT;
            empty_desc.ByteWidth = sizeof(Vector4f);
            empty_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

            Vector4f lval = Vector4f::zero();
            D3D11_SUBRESOURCE_DATA empty_data = {};
            empty_data.pSysMem = &lval;
            m_device->CreateBuffer(&empty_desc, &empty_data, &m_vertex_shading.null_buffer);
        }

        { // Setup light sources.
            m_lights = LightManager(*m_device, LightSources::capacity());
        }

        { // Setup opaque rendering.
            D3D11_DEPTH_STENCIL_DESC depth_desc = {};
            depth_desc = CD3D11_DEPTH_STENCIL_DESC();
            m_device->CreateDepthStencilState(&depth_desc, &m_opaque.depth_state);

            ID3D10Blob* pixel_shader_buffer = compile_shader(L"FragmentShader.hlsl", "ps_5_0");
            HRESULT hr = m_device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_buffer), NULL, &m_opaque.shader);
        }

        { // Catch-all uniforms.
            D3D11_BUFFER_DESC uniforms_desc = {};
            uniforms_desc.Usage = D3D11_USAGE_DEFAULT;
            uniforms_desc.ByteWidth = sizeof(Uniforms);
            uniforms_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            uniforms_desc.CPUAccessFlags = 0;
            uniforms_desc.MiscFlags = 0;

            HRESULT hr = m_device->CreateBuffer(&uniforms_desc, NULL, &uniforms_buffer);
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

        m_vertex_shading.input_layout->Release();
        m_vertex_shading.shader->Release();

        m_opaque.depth_state->Release();
        m_opaque.shader->Release();

        uniforms_buffer->Release();

        for (Dx11Mesh mesh : m_meshes) {
            safe_release(&mesh.indices);
            safe_release(mesh.positions_address());
            safe_release(mesh.normals_address());
            safe_release(mesh.texcoords_address());
        }
    }

    void render(const Cogwheel::Core::Engine& engine) {
        if (Cameras::begin() == Cameras::end())
            return;

        // ?? wait_for_previous_frame();

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
            depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
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

        m_render_context->OMSetDepthStencilState(m_opaque.depth_state, 0);

        SceneRoot scene = Cameras::get_scene_ID(camera_ID);
        RGBA environment_tint = RGBA(scene.get_environment_tint(), 1.0f);
        m_render_context->ClearRenderTargetView(m_backbuffer_view, environment_tint.begin());
        m_render_context->ClearDepthStencilView(m_depth_view, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

        { // Render models.
            // Bind light buffer.
            m_render_context->PSSetConstantBuffers(1, 1, m_lights.light_buffer_addr());

            for (Dx11Model model : m_models) {
                if (model.mesh_ID == 0)
                    continue;

                Dx11Mesh mesh = m_meshes[model.mesh_ID];

                // Set vertex and pixel shaders.
                m_render_context->VSSetShader(m_vertex_shading.shader, 0, 0);
                m_render_context->PSSetShader(m_opaque.shader, 0, 0);

                m_render_context->IASetInputLayout(m_vertex_shading.input_layout);
                m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                { // Set the buffers.
                    if (mesh.index_count != 0)
                        m_render_context->IASetIndexBuffer(mesh.indices, DXGI_FORMAT_R32_UINT, 0);

                    m_render_context->IASetVertexBuffers(0, mesh.buffer_count, mesh.buffers, mesh.strides, mesh.offsets);
                }

                {
                    Uniforms uniforms;
                    uniforms.mvp_matrix = Cameras::get_view_projection_matrix(camera_ID) * to_matrix4x4(m_transforms[model.transform_ID]);
                    uniforms.to_world_matrix = to_matrix4x3(m_transforms[model.transform_ID]);
                    uniforms.color = m_materials[model.material_ID].tint;
                    uniforms.flags.x = mesh.texcoords() != m_vertex_shading.null_buffer ? 1 : 0;
                    m_render_context->UpdateSubresource(uniforms_buffer, 0, NULL, &uniforms, 0, 0);
                    m_render_context->VSSetConstantBuffers(0, 1, &uniforms_buffer);
                    m_render_context->PSSetConstantBuffers(0, 1, &uniforms_buffer);
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

    template <typename T>
    HRESULT upload_default_buffer(T* data, int element_count, D3D11_BIND_FLAG flags, ID3D11Buffer** buffer) {
        D3D11_BUFFER_DESC desc = {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = sizeof(T) * element_count;
        desc.BindFlags = flags;

        D3D11_SUBRESOURCE_DATA resource_data = {};
        resource_data.pSysMem = data;
        return m_device->CreateBuffer(&desc, &resource_data, buffer);
    }

    void handle_updates() {
        // TODO Handle updates in multiple command lists.

        m_lights.handle_updates(*m_render_context);

        { // Material updates. // TODO Upload to one buffer, so we don't have to upload it per frame.
            for (Material mat : Materials::get_changed_materials()) {
                unsigned int material_index = mat.get_ID();

                // Just ignore deleted materials. They shouldn't be referenced anyway.
                if (mat.has_changes(Materials::Changes::Created || Materials::Changes::Updated)) {
                    if (m_materials.size() <= material_index)
                        m_materials.resize(MeshModels::capacity());

                    Dx11Material& dx11_material = m_materials[material_index];
                    dx11_material.tint = mat.get_tint();
                    dx11_material.tint_texture_index = mat.get_tint_texture_ID();
                    dx11_material.roughness = mat.get_roughness();
                    dx11_material.specularity = mat.get_specularity() * 0.08f; // See Physically-Based Shading at Disney bottom of page 8 for why we remap.
                    dx11_material.metallic = mat.get_metallic();
                    dx11_material.coverage = mat.get_coverage();
                    dx11_material.coverage_texture_index = mat.get_coverage_texture_ID();
                }
            }
        }

        { // Mesh updates.
            for (Meshes::UID mesh_ID : Meshes::get_changed_meshes()) {
                if (Meshes::get_changes(mesh_ID) == Meshes::Changes::Destroyed) {
                    if (mesh_ID < m_meshes.size() && m_meshes[mesh_ID].vertex_count != 0) {
                        m_meshes[mesh_ID].index_count = m_meshes[mesh_ID].vertex_count = 0;
                        safe_release(&m_meshes[mesh_ID].indices);
                        safe_release(m_meshes[mesh_ID].positions_address());
                        safe_release(m_meshes[mesh_ID].normals_address());
                        safe_release(m_meshes[mesh_ID].texcoords_address());
                    }
                }

                if (Meshes::get_changes(mesh_ID) & Meshes::Changes::Created) {
                    if (m_meshes.size() <= mesh_ID)
                        m_meshes.resize(Meshes::capacity());

                    Cogwheel::Assets::Mesh mesh = mesh_ID;
                    Dx11Mesh dx_mesh = {};

                    { // Setup strides and offsets for the buffers. TODO Make this static??
                        dx_mesh.strides[0] = sizeof(float) * 3;
                        dx_mesh.offsets[0] = 0;
                        dx_mesh.strides[1] = sizeof(float) * 3;
                        dx_mesh.offsets[1] = 0;
                        dx_mesh.strides[2] = sizeof(float) * 2;
                        dx_mesh.offsets[2] = 0;
                    }

                    // Expand the indexed buffers if an index buffer is used, but no normals are given.
                    // In that case we need to compute hard normals per triangle and we can only do that on expanded buffers.
                    // NOTE Alternatively look into storing the hard normals in a buffer and index into it based on the triangle ID?
                    bool expand_indexed_buffers = mesh.get_primitive_count() != 0 && mesh.get_normals() == nullptr;

                    if (!expand_indexed_buffers) { // Upload indices.
                        dx_mesh.index_count = mesh.get_index_count();

                        HRESULT hr = upload_default_buffer(mesh.get_primitives(), dx_mesh.index_count / 3,
                                                           D3D11_BIND_INDEX_BUFFER, &dx_mesh.indices);
                        if (FAILED(hr))
                            printf("Could not upload '%s' index buffer.\n", mesh.get_name().c_str());
                    }

                    dx_mesh.vertex_count = mesh.get_vertex_count();
                    Vector3f* positions = mesh.get_positions();

                    if (expand_indexed_buffers) {
                        // Expand the positions.
                        dx_mesh.vertex_count = mesh.get_index_count();
                        positions = MeshUtils::expand_indexed_buffer(mesh.get_primitives(), mesh.get_primitive_count(), mesh.get_positions());
                    }

                    { // Upload positions.
                        HRESULT hr = upload_default_buffer(positions, dx_mesh.vertex_count, D3D11_BIND_VERTEX_BUFFER, 
                                                           dx_mesh.positions_address());
                        if (FAILED(hr))
                            printf("Could not upload '%s' position buffer.\n", mesh.get_name().c_str());
                    }

                    { // Upload normals. TODO Encode as float2.
                        Vector3f* normals = mesh.get_normals();

                        if (mesh.get_normals() == nullptr) {
                            // Compute hard normals.
                            normals = new Vector3f[dx_mesh.vertex_count];
                            MeshUtils::compute_hard_normals(positions, positions + dx_mesh.vertex_count, normals);
                        } else if (expand_indexed_buffers)
                            normals = MeshUtils::expand_indexed_buffer(mesh.get_primitives(), mesh.get_primitive_count(), normals);

                        HRESULT hr = upload_default_buffer(normals, dx_mesh.vertex_count, D3D11_BIND_VERTEX_BUFFER,
                                                           dx_mesh.normals_address());
                        if (FAILED(hr))
                            printf("Could not upload '%s' normal buffer.\n", mesh.get_name().c_str());

                        if (normals != mesh.get_normals())
                            delete[] normals;
                    }

                    { // Upload texcoords if present, otherwise upload 'null buffer'.
                        Vector2f* texcoords = mesh.get_texcoords();
                        if (texcoords != nullptr) {

                            if (expand_indexed_buffers)
                                texcoords = MeshUtils::expand_indexed_buffer(mesh.get_primitives(), mesh.get_primitive_count(), texcoords);

                            HRESULT hr = upload_default_buffer(texcoords, dx_mesh.vertex_count, D3D11_BIND_VERTEX_BUFFER,
                                                               dx_mesh.texcoords_address());
                            if (FAILED(hr))
                                printf("Could not upload '%s' texcoord buffer.\n", mesh.get_name().c_str());

                            if (texcoords != mesh.get_texcoords())
                                delete[] texcoords;
                        } else
                            *dx_mesh.texcoords_address() = m_vertex_shading.null_buffer;
                    }

                    dx_mesh.buffer_count = 3; // Positions, normals and texcoords. NOTE We can get away with binding the null texcoord buffer once at the beginning of the rendering pass, as then a semi-valid buffer is always bound.

                    // Delete temporary expanded positions.
                    if (positions != mesh.get_positions())
                        delete[] positions;

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
