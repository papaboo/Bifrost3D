// DirectX 11 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/EnvironmentManager.h>
#include <DX11Renderer/LightManager.h>
#include <DX11Renderer/Renderer.h>
#include <DX11Renderer/TextureManager.h>
#include <DX11Renderer/Types.h>
#include <DX11Renderer/Utils.h>

#define NOMINMAX
#include <D3D11_1.h>
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
#include <codecvt>
#include <vector>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;
using namespace std;

namespace DX11Renderer {

//----------------------------------------------------------------------------
// DirectX 11 renderer implementation.
//----------------------------------------------------------------------------
class Renderer::Implementation {
private:
    ID3D11Device* m_device;
    ID3D11DeviceContext* m_render_context; // Is this the same as the immediate context?
    IDXGISwapChain* m_swap_chain;

    // Backbuffer members.
    Vector2ui m_backbuffer_size;
    ID3D11RenderTargetView* m_backbuffer_view;
    ID3D11Texture2D* m_depth_buffer;
    ID3D11DepthStencilView* m_depth_view;

    // Cogwheel resources
    vector<Dx11Material> m_materials = vector<Dx11Material>(0);
    vector<Dx11Mesh> m_meshes = vector<Dx11Mesh>(0);
    vector<Transform> m_transforms = vector<Transform>(0);

    vector<int> m_model_indices = vector<int>(0); // The models index in the sorted models array.
    vector<Dx11Model> m_sorted_models = vector<Dx11Model>(0);

    EnvironmentManager* m_environments;
    LightManager m_lights;
    TextureManager m_textures;

    struct {
        ID3D11Buffer* null_buffer;
        ID3D11InputLayout* input_layout;
        ID3D11VertexShader* shader;
    } m_vertex_shading;

    struct {
        ID3D11RasterizerState* raster_state;
        ID3D11DepthStencilState* depth_state;
        ID3D11PixelShader* shader;
    } m_opaque;

    struct {
        int first_model_index = 0;
        ID3D11RasterizerState* raster_state;
    } m_cutout;

    struct Transparent {
        struct SortedModel {
            float distance;
            int model_index;
        };

        int first_model_index = 0;
        ID3D11BlendState* blend_state;
        ID3D11DepthStencilState* depth_state;
        ID3D11PixelShader* shader;
        std::vector<SortedModel> sorted_models_pool; // List of sorted transparent models. Created as a pool to minimize runtime memory allocation.
    } m_transparent;

    // Constant buffer for a single material (single material for now!)
    ID3D11Buffer* material_buffer;

    // Scene constants
    struct SceneConstants {
        Matrix4x4f view_projection_matrix;
        Vector4f camera_position;
        Vector4f environment_tint; // .w component is 1 if an environment tex is bound, otherwise 0.
        Matrix4x4f inverse_view_projection_matrix;
    };
    ID3D11Buffer* m_scene_buffer;

    ID3D11Buffer* m_transform_buffer;

    std::wstring m_shader_folder_path;

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

                    D3D_FEATURE_LEVEL feature_level_requested = D3D_FEATURE_LEVEL_11_0;

                    D3D_FEATURE_LEVEL feature_level;
                    hr = D3D11CreateDeviceAndSwapChain(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, &feature_level_requested, 1,
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

        {
            std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
            std::wstring data_folder_path = converter.from_bytes(Engine::get_instance()->data_path());
            m_shader_folder_path = data_folder_path + L"DX11Renderer\\Shaders\\";
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

        { // Setup asset managing.
            Dx11Model dummy_model = { 0, 0, 0, 0, 0 };
            m_sorted_models.push_back(dummy_model);
            m_model_indices.resize(1);
            m_model_indices[0] = 0;

            m_environments = new EnvironmentManager(*m_device, m_shader_folder_path, m_textures);
            m_lights = LightManager(*m_device, LightSources::capacity());
            m_textures = TextureManager(*m_device);
        }

        { // Setup vertex processing.
            ID3D10Blob* vertex_shader_blob = compile_shader(m_shader_folder_path + L"VertexShader.hlsl", "vs_5_0");

            // Create the shader objects.
            HRESULT hr = m_device->CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), NULL, &m_vertex_shading.shader);
            THROW_ON_FAILURE(hr);

            // Create the input layout
            D3D11_INPUT_ELEMENT_DESC input_layout_desc[] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 2, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
            };

            hr = m_device->CreateInputLayout(input_layout_desc, 3, UNPACK_BLOB_ARGS(vertex_shader_blob), &m_vertex_shading.input_layout);
            THROW_ON_FAILURE(hr);

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

        { // Setup opaque rendering.
            CD3D11_RASTERIZER_DESC opaque_raster_state = CD3D11_RASTERIZER_DESC(CD3D11_DEFAULT());
            HRESULT hr = m_device->CreateRasterizerState(&opaque_raster_state, &m_opaque.raster_state);
            THROW_ON_FAILURE(hr);

            D3D11_DEPTH_STENCIL_DESC depth_desc = CD3D11_DEPTH_STENCIL_DESC(CD3D11_DEFAULT());
            depth_desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
            hr = m_device->CreateDepthStencilState(&depth_desc, &m_opaque.depth_state);
            THROW_ON_FAILURE(hr);

            ID3D10Blob* pixel_shader_buffer = compile_shader(m_shader_folder_path + L"FragmentShader.hlsl", "ps_5_0", "opaque");
            hr = m_device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_buffer), NULL, &m_opaque.shader);
            THROW_ON_FAILURE(hr);
        }

        { // Setup cutout rendering. Reuses some of the opaque state.
            CD3D11_RASTERIZER_DESC twosided_raster_state = CD3D11_RASTERIZER_DESC(CD3D11_DEFAULT());
            twosided_raster_state.CullMode = D3D11_CULL_NONE;
            HRESULT hr = m_device->CreateRasterizerState(&twosided_raster_state, &m_cutout.raster_state);
            THROW_ON_FAILURE(hr);
        }

        { // Setup transparent rendering.

            D3D11_BLEND_DESC blend_desc;
            blend_desc.AlphaToCoverageEnable = false;
            blend_desc.IndependentBlendEnable = false;

            D3D11_RENDER_TARGET_BLEND_DESC& rt_blend_desc = blend_desc.RenderTarget[0];
            rt_blend_desc.BlendEnable = true;
            rt_blend_desc.SrcBlend = D3D11_BLEND_SRC_ALPHA;
            rt_blend_desc.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
            rt_blend_desc.BlendOp = D3D11_BLEND_OP_ADD;
            rt_blend_desc.SrcBlendAlpha = D3D11_BLEND_ONE;
            rt_blend_desc.DestBlendAlpha = D3D11_BLEND_ONE;
            rt_blend_desc.BlendOpAlpha = D3D11_BLEND_OP_ADD;
            rt_blend_desc.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

            HRESULT hr = m_device->CreateBlendState(&blend_desc, &m_transparent.blend_state);
            THROW_ON_FAILURE(hr);

            D3D11_DEPTH_STENCIL_DESC depth_desc = CD3D11_DEPTH_STENCIL_DESC(CD3D11_DEFAULT());
            depth_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
            hr = m_device->CreateDepthStencilState(&depth_desc, &m_transparent.depth_state);
            THROW_ON_FAILURE(hr);

            ID3D10Blob* pixel_shader_buffer = compile_shader(m_shader_folder_path + L"FragmentShader.hlsl", "ps_5_0", "transparent");
            hr = m_device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_buffer), NULL, &m_transparent.shader);
            THROW_ON_FAILURE(hr);
        }

        { // Scene constant buffer
            HRESULT hr = create_constant_buffer(*m_device, sizeof(SceneConstants), &m_scene_buffer);
            THROW_ON_FAILURE(hr);
        }

        { // Transform constant buffer.
            HRESULT hr = create_constant_buffer(*m_device, sizeof(Matrix4x4f), &m_transform_buffer);
            THROW_ON_FAILURE(hr);
        }

        { // Material constant buffer.
            HRESULT hr = create_constant_buffer(*m_device, sizeof(Dx11Material), &material_buffer);
            THROW_ON_FAILURE(hr);
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

        safe_release(&m_vertex_shading.input_layout);
        safe_release(&m_vertex_shading.null_buffer);
        safe_release(&m_vertex_shading.shader);

        safe_release(&m_opaque.depth_state);
        safe_release(&m_opaque.raster_state);
        safe_release(&m_opaque.shader);

        safe_release(&m_cutout.raster_state);

        safe_release(&m_transparent.blend_state);
        safe_release(&m_transparent.depth_state);
        safe_release(&m_transparent.shader);

        safe_release(&m_scene_buffer);
        safe_release(&material_buffer);

        for (Dx11Mesh mesh : m_meshes) {
            safe_release(&mesh.indices);
            safe_release(mesh.positions_address());
            safe_release(mesh.normals_address());
            safe_release(mesh.texcoords_address());
        }
        
        delete m_environments;
        m_lights.release();
        m_textures.release();
    }

    void render_model(ID3D11DeviceContext* context, Dx11Model model, Cameras::UID camera_ID) {
        Dx11Mesh mesh = m_meshes[model.mesh_ID];

        { // Set the buffers.
            if (mesh.index_count != 0)
                context->IASetIndexBuffer(mesh.indices, DXGI_FORMAT_R32_UINT, 0);

            // Setup strides and offsets for the buffers.
            // Layout is [positions, normals, texcoords].
            static unsigned int strides[3] = { sizeof(float3), sizeof(float3), sizeof(float2) };
            static unsigned int offsets[3] = { 0, 0, 0 };
            
            context->IASetVertexBuffers(0, mesh.buffer_count, mesh.buffers, strides, offsets);
        }

        { // Upload world transform.
            Matrix4x4f to_world_matrix = to_matrix4x4(m_transforms[model.transform_ID]);
            context->UpdateSubresource(m_transform_buffer, 0, NULL, &to_world_matrix, 0, 0);
            context->VSSetConstantBuffers(2, 1, &m_transform_buffer);
        }

        { // Material parameters

          // Update constant buffer.
            context->UpdateSubresource(material_buffer, 0, NULL, &m_materials[model.material_ID], 0, 0);
            context->PSSetConstantBuffers(3, 1, &material_buffer);

            Dx11Texture colorTexture = m_textures.get_texture(m_materials[model.material_ID].tint_texture_index);
            if (colorTexture.sampler != nullptr) {
                context->PSSetShaderResources(1, 1, &colorTexture.image->srv);
                context->PSSetSamplers(1, 1, &colorTexture.sampler);
            }

            Dx11Texture coverateTexture = m_textures.get_texture(m_materials[model.material_ID].coverage_texture_index);
            if (coverateTexture.sampler != nullptr) {
                context->PSSetShaderResources(2, 1, &coverateTexture.image->srv);
                context->PSSetSamplers(2, 1, &coverateTexture.sampler);
            }
        }

        if (mesh.index_count != 0)
            context->DrawIndexed(mesh.index_count, 0, 0);
        else
            context->Draw(mesh.vertex_count, 0);
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

        m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        m_render_context->OMSetRenderTargets(1, &m_backbuffer_view, m_depth_view);
        m_render_context->OMSetDepthStencilState(m_opaque.depth_state, 0);
        m_render_context->ClearDepthStencilView(m_depth_view, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

        // Opaque state setup.
        m_render_context->OMSetBlendState(0, 0, 0xffffffff);
        m_render_context->OMSetDepthStencilState(m_opaque.depth_state, 1);
        m_render_context->RSSetState(m_opaque.raster_state);

        SceneRoot scene = Cameras::get_scene_ID(camera_ID);
        { // Setup scene constants.
            SceneConstants scene_vars;
            scene_vars.view_projection_matrix = Cameras::get_view_projection_matrix(camera_ID);
            scene_vars.camera_position = Vector4f(Cameras::get_transform(camera_ID).translation, 1.0f);
            RGB env_tint = scene.get_environment_tint();
            float valid_env_map = scene.get_environment_map() == Textures::UID::invalid_UID() ? 0.0f : 1.0f;
            scene_vars.environment_tint = { env_tint.r, env_tint.g, env_tint.b, valid_env_map };
            scene_vars.inverse_view_projection_matrix = Cameras::get_inverse_view_projection_matrix(camera_ID);
            m_render_context->UpdateSubresource(m_scene_buffer, 0, NULL, &scene_vars, 0, 0);
            m_render_context->VSSetConstantBuffers(0, 1, &m_scene_buffer);
            m_render_context->PSSetConstantBuffers(0, 1, &m_scene_buffer);
        }

        m_environments->render(*m_render_context, scene.get_ID());

        { // Render models.

            // Bind light buffer.
            m_render_context->PSSetConstantBuffers(1, 1, m_lights.light_buffer_addr());

            // Set vertex and pixel shaders.
            m_render_context->VSSetShader(m_vertex_shading.shader, 0, 0);
            m_render_context->IASetInputLayout(m_vertex_shading.input_layout);
            m_render_context->PSSetShader(m_opaque.shader, 0, 0);

            unsigned int stride = sizeof(float2);
            unsigned int offset = 0;
            m_render_context->IASetVertexBuffers(2, 1, &m_vertex_shading.null_buffer, &stride, &offset);

            for (int i = 1; i < m_transparent.first_model_index; ++i) {

                // Setup twosided raster state for cutout materials.
                if (i == m_cutout.first_model_index)
                    m_render_context->RSSetState(m_cutout.raster_state);

                Dx11Model model = m_sorted_models[i];
                assert(model.model_ID != 0);
                render_model(m_render_context, model, camera_ID);
            }

            { // Render transparent models

                // Apply used cutout state if not already applied.
                bool no_cutouts_present = m_cutout.first_model_index > m_transparent.first_model_index;
                if (no_cutouts_present)
                    m_render_context->RSSetState(m_cutout.raster_state);

                // Set transparent state.
                m_render_context->OMSetBlendState(m_transparent.blend_state, 0, 0xffffffff);
                m_render_context->OMSetDepthStencilState(m_transparent.depth_state, 1);
                m_render_context->PSSetShader(m_transparent.shader, 0, 0);

                int transparent_model_count = int(m_sorted_models.size()) - m_transparent.first_model_index;
                auto transparent_models = m_transparent.sorted_models_pool; // Alias the pool.
                transparent_models.resize(transparent_model_count);

                { // Sort transparent models. TODO in a separate thread that is waited on when we get to the transparent render index.
                    Vector3f cam_pos = Cameras::get_transform(camera_ID).translation;
                    for (int i = 0; i < transparent_model_count; ++i) {
                        // Calculate the distance to point halfway between the models center and side of the bounding box.
                        int model_index = i + m_transparent.first_model_index;
                        Dx11Mesh& mesh = m_meshes[m_sorted_models[model_index].mesh_ID];
                        Transform transform = m_transforms[m_sorted_models[model_index].transform_ID];
                        Cogwheel::Math::AABB bounds = { Vector3f(mesh.bounds.min.x, mesh.bounds.min.y, mesh.bounds.min.z),
                                                        Vector3f(mesh.bounds.max.x, mesh.bounds.max.y, mesh.bounds.max.z) };
                        float distance_to_cam = alpha_sort_value(cam_pos, transform, bounds);
                        transparent_models[i] = { distance_to_cam, model_index};
                    }

                    std::sort(transparent_models.begin(), transparent_models.end(), 
                        [](Transparent::SortedModel lhs, Transparent::SortedModel rhs) -> bool {
                            return lhs.distance < rhs.distance;
                    });
                }

                for (auto transparent_model : transparent_models) {
                    Dx11Model model = m_sorted_models[transparent_model.model_index];
                    assert(model.model_ID != 0);
                    render_model(m_render_context, model, camera_ID);
                }
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
        m_environments->handle_updates(*m_device, *m_render_context);
        m_lights.handle_updates(*m_render_context);
        m_textures.handle_updates(*m_device, *m_render_context);

        { // Material updates.
            for (Material mat : Materials::get_changed_materials()) {
                unsigned int material_index = mat.get_ID();

                // Just ignore deleted materials. They shouldn't be referenced anyway.
                if (!mat.get_changes().is_set(Materials::Change::Destroyed)) {
                    if (m_materials.size() <= material_index)
                        m_materials.resize(Materials::capacity());

                    Dx11Material& dx11_material = m_materials[material_index];
                    dx11_material.tint.x = mat.get_tint().r;
                    dx11_material.tint.y = mat.get_tint().g;
                    dx11_material.tint.z = mat.get_tint().b;
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
                if (m_meshes.size() <= mesh_ID)
                    m_meshes.resize(Meshes::capacity());

                if (Meshes::get_changes(mesh_ID).any_set(Meshes::Change::Created, Meshes::Change::Destroyed)) {
                    if (m_meshes[mesh_ID].vertex_count != 0) {
                        m_meshes[mesh_ID].index_count = m_meshes[mesh_ID].vertex_count = 0;
                        safe_release(&m_meshes[mesh_ID].indices);
                        safe_release(m_meshes[mesh_ID].positions_address());
                        safe_release(m_meshes[mesh_ID].normals_address());
                        safe_release(m_meshes[mesh_ID].texcoords_address());
                    }
                }

                if (Meshes::get_changes(mesh_ID).is_set(Meshes::Change::Created)) {
                    Cogwheel::Assets::Mesh mesh = mesh_ID;
                    Dx11Mesh dx_mesh = {};

                    Cogwheel::Math::AABB bounds = mesh.get_bounds();
                    dx_mesh.bounds = { make_float3(bounds.minimum), make_float3(bounds.maximum) };

                    // Expand the indexed buffers if an index buffer is used, but no normals are given.
                    // In that case we need to compute hard normals per triangle and we can only store that for non-indexed buffers.
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

                    bool has_texcoords = mesh.get_texcoords() != nullptr;
                    dx_mesh.buffer_count = has_texcoords ? 3 : 2;

                    // Delete temporary expanded positions.
                    if (positions != mesh.get_positions())
                        delete[] positions;

                    m_meshes[mesh_ID] = dx_mesh;
                }
            }
        }

        { // Transform updates.
            for (SceneNodes::UID node_ID : SceneNodes::get_changed_nodes()) {
                if (SceneNodes::get_changes(node_ID).any_set(SceneNodes::Change::Created, SceneNodes::Change::Transform)) {

                    if (m_transforms.size() <= node_ID)
                        m_transforms.resize(SceneNodes::capacity());

                    m_transforms[node_ID] = SceneNodes::get_global_transform(node_ID);
                }
            }
        }

        { // Model updates
            if (!MeshModels::get_changed_models().is_empty()) {
                if (m_sorted_models.size() <= MeshModels::capacity()) {
                    m_sorted_models.reserve(MeshModels::capacity());
                    int old_size = (int)m_model_indices.size();
                    m_model_indices.resize(MeshModels::capacity());
                    std::fill(m_model_indices.begin() + old_size, m_model_indices.end(), 0);
                }

                for (MeshModel model : MeshModels::get_changed_models()) {
                    unsigned int model_index = m_model_indices[model.get_ID()];

                    if (model.get_changes() == MeshModels::Change::Destroyed) {
                        m_sorted_models[model_index].model_ID = 0;
                        m_sorted_models[model_index].material_ID = 0;
                        m_sorted_models[model_index].mesh_ID = 0;
                        m_sorted_models[model_index].transform_ID = 0;
                        m_sorted_models[model_index].properties = Dx11Model::Properties::Destroyed;

                        m_model_indices[model.get_ID()] = 0;
                    }

                    if (model.get_changes() & MeshModels::Change::Created) {
                        Dx11Model dx_model;
                        dx_model.model_ID = model.get_ID();
                        dx_model.material_ID = model.get_material().get_ID();
                        dx_model.mesh_ID = model.get_mesh().get_ID();
                        dx_model.transform_ID = model.get_scene_node().get_ID();

                        Material mat = model.get_material();
                        bool is_transparent = mat.get_coverage_texture_ID() != Textures::UID::invalid_UID() || mat.get_coverage() < 1.0f;
                        bool is_cutout = mat.get_flags().is_set(MaterialFlag::Cutout);
                        unsigned int transparent_type = is_cutout ? Dx11Model::Properties::Cutout : Dx11Model::Properties::Transparent;
                        dx_model.properties = is_transparent ? transparent_type : Dx11Model::Properties::None;

                        if (model_index == 0) {
                            m_model_indices[model.get_ID()] = (int)m_sorted_models.size();
                            m_sorted_models.push_back(dx_model);
                        } else
                            m_sorted_models[model_index] = dx_model;
                    }
                }

                // Sort the models in the order [dummy, opaque, cutout, transparent, destroyed].
                // The new position/index of the model is stored in model_index.
                // Incidently also compacts the list of models by removing any deleted models.
                {
                    // TODO Sort by material ID as well and use the info while rendering.
                    // If the material hasn't changed then don't rebind it or the textures.

                    // The models to be sorted starts at index 1, because the first model is a dummy model.
                    std::sort(m_sorted_models.begin() + 1, m_sorted_models.end(),
                        [](Dx11Model lhs, Dx11Model rhs) -> bool {
                        return lhs.properties < rhs.properties;
                    });

                    { // Register the models new position and find the transition between model buckets.
                        int sorted_models_end = m_cutout.first_model_index = m_transparent.first_model_index =
                            (int)m_sorted_models.size();

                        #pragma omp parallel for
                        for (int i = 1; i < m_sorted_models.size(); ++i) {
                            Dx11Model& model = m_sorted_models[i];
                            m_model_indices[model.model_ID] = i;

                            Dx11Model& prevModel = m_sorted_models[i - 1];
                            if (prevModel.properties != model.properties) {
                                if (!prevModel.is_cutout() && model.is_cutout())
                                    m_cutout.first_model_index = i;
                                if (!prevModel.is_transparent() && model.is_transparent())
                                    m_transparent.first_model_index = i;
                                if (!prevModel.is_destroyed() && model.is_destroyed())
                                    sorted_models_end = i;
                            }
                        }

                        // Correct indices in case no bucket transition was found.
                        if (m_transparent.first_model_index > sorted_models_end)
                            m_transparent.first_model_index = sorted_models_end;
                        if (m_cutout.first_model_index > m_transparent.first_model_index)
                            m_cutout.first_model_index = m_transparent.first_model_index;

                        m_sorted_models.resize(sorted_models_end);
                    }
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
