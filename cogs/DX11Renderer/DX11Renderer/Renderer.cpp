// DirectX 11 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/EnvironmentManager.h>
#include <DX11Renderer/LightManager.h>
#include <DX11Renderer/MaterialManager.h>
#include <DX11Renderer/Renderer.h>
#include <DX11Renderer/SSAO.h>
#include <DX11Renderer/TextureManager.h>
#include <DX11Renderer/TransformManager.h>
#include <DX11Renderer/Types.h>
#include <DX11Renderer/Utils.h>

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Core/Window.h>
#include <Cogwheel/Math/OctahedralNormal.h>
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
    ID3D11Device1& m_device;
    ODeviceContext1 m_render_context;

    Settings m_settings;

    // Cogwheel resources
    vector<Dx11Mesh> m_meshes = vector<Dx11Mesh>(0);

    vector<int> m_model_indices = vector<int>(0); // The models index in the sorted models array.
    vector<Dx11Model> m_sorted_models = vector<Dx11Model>(0);

    EnvironmentManager* m_environments;
    MaterialManager m_materials;
    TextureManager m_textures;
    TransformManager m_transforms;
    SSAO::AlchemyAO m_ssao;

    struct {
        unsigned int width, height;
        OShaderResourceView depth_SRV;
        ODepthStencilView depth_view;
        OShaderResourceView normal_SRV;
        ORenderTargetView normal_RTV;

        OVertexShader vertex_shader;
        OInputLayout vertex_input_layout;
        OPixelShader pixel_shader;
    } m_g_buffer;

    struct {
        OBuffer null_buffer;
        OInputLayout input_layout;
        OVertexShader shader;
    } m_vertex_shading;

    struct {
        ORasterizerState raster_state;
        ODepthStencilState depth_state;
        OPixelShader shader;
    } m_opaque;

    struct {
        int first_model_index = 0;
        ORasterizerState raster_state;
    } m_cutout;

    struct Transparent {
        struct SortedModel {
            float distance;
            int model_index;
        };

        int first_model_index = 0;
        OBlendState blend_state;
        ODepthStencilState depth_state;
        OPixelShader shader;
        std::vector<SortedModel> sorted_models_pool; // List of sorted transparent models. Created as a pool to minimize runtime memory allocation.
    } m_transparent;

    // Scene constants
    struct SceneConstants {
        Matrix4x4f view_projection_matrix;
        Vector4f camera_position;
        Vector4f environment_tint; // .w component is 0 if an environment tex is not bound, otherwise positive.
        Matrix4x4f inverse_view_projection_matrix;
        Matrix4x4f projection_matrix;
        Matrix4x4f inverse_projection_matrix;
        Matrix4x3f world_to_view_matrix;
    };
    OBuffer m_scene_buffer;

    // Lights
    struct {
        LightManager manager;
        OVertexShader vertex_shader;
        OPixelShader pixel_shader;
    } m_lights;

    std::wstring m_shader_folder_path;

public:
    Implementation(ID3D11Device1& device, int width_hint, int height_hint, const std::wstring& data_folder_path)
        : m_device(device), m_settings(Settings::default()), m_shader_folder_path(data_folder_path + L"DX11Renderer\\Shaders\\") {

        device.GetImmediateContext1(&m_render_context);

        { // Setup asset managing.
            Dx11Model dummy_model = { 0, 0, 0, 0, 0 };
            m_sorted_models.push_back(dummy_model);
            m_model_indices.resize(1);
            m_model_indices[0] = 0;

            m_environments = new EnvironmentManager(m_device, m_shader_folder_path, m_textures);

            m_materials = MaterialManager(m_device, *m_render_context);
            m_textures = TextureManager(m_device);
            m_transforms = TransformManager(m_device, *m_render_context);

            // Setup static state.
            m_render_context->PSSetShaderResources(14, 1, m_materials.get_GGX_SPTD_fit_srv_addr());
            m_render_context->PSSetShaderResources(15, 1, m_materials.get_GGX_with_fresnel_rho_srv_addr());

            D3D11_SAMPLER_DESC sampler_desc = {};
            sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
            sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
            sampler_desc.MinLOD = 0;
            sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
            OSamplerState point_sampler;
            THROW_ON_FAILURE(device.CreateSamplerState(&sampler_desc, &point_sampler));

            sampler_desc.Filter = D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
            OSamplerState linear_sampler;
            THROW_ON_FAILURE(device.CreateSamplerState(&sampler_desc, &linear_sampler));

            ID3D11SamplerState* samplers[2] = { point_sampler, linear_sampler };
            m_render_context->PSSetSamplers(14, 2, samplers);
        }

        { // Setup g-buffer
            OBlob vertex_shader_blob = compile_shader(m_shader_folder_path + L"GBuffer.hlsl", "vs_5_0", "main_VS");
            THROW_ON_FAILURE(m_device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_g_buffer.vertex_shader));

            // Create the input layout
            D3D11_INPUT_ELEMENT_DESC input_layout_desc = { "GEOMETRY", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 };
            THROW_ON_FAILURE(m_device.CreateInputLayout(&input_layout_desc, 1, UNPACK_BLOB_ARGS(vertex_shader_blob), &m_g_buffer.vertex_input_layout));

            OBlob pixel_shader_blob = compile_shader(m_shader_folder_path + L"GBuffer.hlsl", "ps_5_0", "main_PS");
            THROW_ON_FAILURE(m_device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_g_buffer.pixel_shader));

            // G-buffer is initialized on demand when the output dimensions are known.
            m_g_buffer.width = m_g_buffer.height = 0u;
            m_g_buffer.depth_SRV = nullptr;
            m_g_buffer.depth_view = nullptr;
            m_g_buffer.normal_SRV = nullptr;
            m_g_buffer.normal_RTV = nullptr;
        }

        m_ssao = SSAO::AlchemyAO(m_device, m_shader_folder_path);

        { // Setup vertex processing.
            OBlob vertex_shader_blob = compile_shader(m_shader_folder_path + L"VertexShader.hlsl", "vs_5_0", "main");

            // Create the shader objects.
            HRESULT hr = m_device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_vertex_shading.shader);
            THROW_ON_FAILURE(hr);

            // Create the input layout
            D3D11_INPUT_ELEMENT_DESC input_layout_desc[] = {
                { "GEOMETRY", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
            };

            hr = m_device.CreateInputLayout(input_layout_desc, 2, UNPACK_BLOB_ARGS(vertex_shader_blob), &m_vertex_shading.input_layout);
            THROW_ON_FAILURE(hr);

            // Create a default emptyish buffer.
            D3D11_BUFFER_DESC empty_desc = {};
            empty_desc.Usage = D3D11_USAGE_IMMUTABLE;
            empty_desc.ByteWidth = sizeof(Vector4f);
            empty_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

            Vector4f lval = Vector4f::zero();
            D3D11_SUBRESOURCE_DATA empty_data = {};
            empty_data.pSysMem = &lval;
            hr = m_device.CreateBuffer(&empty_desc, &empty_data, &m_vertex_shading.null_buffer);
            THROW_ON_FAILURE(hr);
        }

#if SPTD_AREA_LIGHTS
        D3D_SHADER_MACRO fragment_macros[] = { "SPTD_AREA_LIGHTS",  "1", 0, 0 };
#else 
        D3D_SHADER_MACRO fragment_macros[] = { "SPTD_AREA_LIGHTS",  "0", 0, 0 };
#endif

        { // Setup opaque rendering.
            CD3D11_RASTERIZER_DESC opaque_raster_state = CD3D11_RASTERIZER_DESC(CD3D11_DEFAULT());
            HRESULT hr = m_device.CreateRasterizerState(&opaque_raster_state, &m_opaque.raster_state);
            THROW_ON_FAILURE(hr);

            D3D11_DEPTH_STENCIL_DESC depth_desc = CD3D11_DEPTH_STENCIL_DESC(CD3D11_DEFAULT());
            depth_desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
            hr = m_device.CreateDepthStencilState(&depth_desc, &m_opaque.depth_state);
            THROW_ON_FAILURE(hr);

            OBlob pixel_shader_blob = compile_shader(m_shader_folder_path + L"FragmentShader.hlsl", "ps_5_0", "opaque", fragment_macros);
            hr = m_device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_opaque.shader);
            THROW_ON_FAILURE(hr);
        }

        { // Setup cutout rendering. Reuses some of the opaque state.
            CD3D11_RASTERIZER_DESC twosided_raster_state = CD3D11_RASTERIZER_DESC(CD3D11_DEFAULT());
            twosided_raster_state.CullMode = D3D11_CULL_NONE;
            HRESULT hr = m_device.CreateRasterizerState(&twosided_raster_state, &m_cutout.raster_state);
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

            HRESULT hr = m_device.CreateBlendState(&blend_desc, &m_transparent.blend_state);
            THROW_ON_FAILURE(hr);

            D3D11_DEPTH_STENCIL_DESC depth_desc = CD3D11_DEPTH_STENCIL_DESC(CD3D11_DEFAULT());
            depth_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
            hr = m_device.CreateDepthStencilState(&depth_desc, &m_transparent.depth_state);
            THROW_ON_FAILURE(hr);

            OBlob pixel_shader_buffer = compile_shader(m_shader_folder_path + L"FragmentShader.hlsl", "ps_5_0", "transparent", fragment_macros);
            hr = m_device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_buffer), nullptr, &m_transparent.shader);
            THROW_ON_FAILURE(hr);
        }

        { // Scene constant buffer
            HRESULT hr = create_constant_buffer(m_device, sizeof(SceneConstants), &m_scene_buffer);
            THROW_ON_FAILURE(hr);
        }

        { // Setup lights
            m_lights.manager = LightManager(m_device, LightSources::capacity());

            // Sphere light visualization shaders.
            OBlob vertex_shader_blob = compile_shader(m_shader_folder_path + L"SphereLight.hlsl", "vs_5_0", "vs");
            HRESULT hr = m_device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_lights.vertex_shader);
            THROW_ON_FAILURE(hr);
            OBlob pixel_shader_blob = compile_shader(m_shader_folder_path + L"SphereLight.hlsl", "ps_5_0", "ps");
            hr = m_device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_lights.pixel_shader);
            THROW_ON_FAILURE(hr);
        }
    }

    ~Implementation() {
        for (Dx11Mesh mesh : m_meshes) {
            safe_release(&mesh.indices);
            safe_release(mesh.geometry_address());
            safe_release(mesh.texcoords_address());
        }

        delete m_environments;
    }

    void fill_g_buffer() {
        { // Setup state. Opaque render state should already have been set up. TODO Verify implicit state from opaque pass
            // Render target views.
            m_render_context->OMSetRenderTargets(1, &m_g_buffer.normal_RTV, m_g_buffer.depth_view);
            float zero[4] = { 0, 0, 0, 0 };
            m_render_context->ClearRenderTargetView(m_g_buffer.normal_RTV, zero);
            m_render_context->ClearDepthStencilView(m_g_buffer.depth_view, D3D11_CLEAR_DEPTH, 1.0f, 0);

            // Shaders
            m_render_context->VSSetShader(m_g_buffer.vertex_shader, 0, 0);
            m_render_context->IASetInputLayout(m_g_buffer.vertex_input_layout);
            m_render_context->PSSetShader(m_g_buffer.pixel_shader, 0, 0);

            // Set null buffer as default texcoord buffer.
            unsigned int stride = sizeof(float2);
            unsigned int offset = 0;
            m_render_context->IASetVertexBuffers(2, 1, &m_vertex_shading.null_buffer, &stride, &offset);
        }

        // Render opaque objects.
        for (int i = 1; i < m_cutout.first_model_index; ++i) {

            // TODO loop until the first transparent model and update the raster state.
            // Setup twosided raster state for cutout materials.
            // if (i == m_cutout.first_model_index)
            //     m_render_context->RSSetState(m_cutout.raster_state);

            Dx11Model model = m_sorted_models[i];
            assert(model.model_ID != 0);
            render_model<true>(m_render_context, model);
        }
    }

    template <bool geometry_only>
    void render_model(ID3D11DeviceContext1* context, Dx11Model model) {
        Dx11Mesh mesh = m_meshes[model.mesh_ID];

        { // Set the buffers.
            if (mesh.index_count != 0)
                context->IASetIndexBuffer(mesh.indices, DXGI_FORMAT_R32_UINT, 0);

            // Setup strides and offsets for the buffers.
            // Layout is [geometry, texcoords].
            static unsigned int strides[2] = { sizeof(float4), sizeof(float2) };
            static unsigned int offsets[2] = { 0, 0 };

            context->IASetVertexBuffers(0, mesh.buffer_count, mesh.buffers, strides, offsets);
        }

        { // Bind world transform.
            m_transforms.bind_transform(*context, 2, model.transform_ID);
        }

        { // Material parameters

            Dx11MaterialTextures& material_textures = m_materials.get_material_textures(model.material_ID);

            if (!geometry_only) {
                // Bind material constant buffer.
                m_materials.bind_material(*context, 3, model.material_ID);

                Dx11Texture& color_texture = m_textures.get_texture(material_textures.tint_index);
                if (color_texture.sampler != nullptr) {
                    context->PSSetShaderResources(1, 1, &color_texture.image->srv);
                    context->PSSetSamplers(1, 1, &color_texture.sampler);
                }
            }

            Dx11Texture& coverage_texture = m_textures.get_texture(material_textures.coverage_index);
            if (coverage_texture.sampler != nullptr) {
                context->PSSetShaderResources(2, 1, &coverage_texture.image->srv);
                context->PSSetSamplers(2, 1, &coverage_texture.sampler);
            }
        }

        if (mesh.index_count != 0)
            context->DrawIndexed(mesh.index_count, 0, 0);
        else
            context->Draw(mesh.vertex_count, 0);
    } 

    void render(ORenderTargetView& backbuffer_RTV, const Cogwheel::Scene::Cameras::UID camera_ID, int width, int height) {
        // TODO Replace by assert
        m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); // NOTE This should already be the case as we finish a frame by rendering models or post process effects.

        // Opaque state setup.
        m_render_context->OMSetBlendState(0, 0, 0xffffffff);
        m_render_context->OMSetDepthStencilState(m_opaque.depth_state, 0);
        m_render_context->RSSetState(m_opaque.raster_state);

        SceneRoot scene = Cameras::get_scene_ID(camera_ID);
        { // Setup scene constants.
            SceneConstants scene_vars;
            scene_vars.view_projection_matrix = Cameras::get_view_projection_matrix(camera_ID);
            scene_vars.camera_position = Vector4f(Cameras::get_transform(camera_ID).translation, 1.0f);
            RGB env_tint = scene.get_environment_tint();
            scene_vars.environment_tint = { env_tint.r, env_tint.g, env_tint.b, float(scene.get_environment_map().get_index()) };
            scene_vars.inverse_view_projection_matrix = Cameras::get_inverse_view_projection_matrix(camera_ID);
            scene_vars.projection_matrix = Cameras::get_projection_matrix(camera_ID);
            scene_vars.inverse_projection_matrix = Cameras::get_inverse_projection_matrix(camera_ID);
            scene_vars.world_to_view_matrix = to_matrix4x3(Cameras::get_view_transform(camera_ID));
            m_render_context->UpdateSubresource(m_scene_buffer, 0, nullptr, &scene_vars, 0, 0);
            m_render_context->VSSetConstantBuffers(0, 1, &m_scene_buffer);
            m_render_context->PSSetConstantBuffers(0, 1, &m_scene_buffer);
        }

        { // Render G-buffer
            auto g_buffer_marker = PerformanceMarker(*m_render_context, L"G-buffer");

            // Re-allocate buffers if the dimensions have changed.
            if (m_g_buffer.width != width || m_g_buffer.height != height) {
                { // Depth buffer
                    if (m_g_buffer.depth_SRV) m_g_buffer.depth_SRV->Release();
                    if (m_g_buffer.depth_view) m_g_buffer.depth_view->Release();

                    D3D11_TEXTURE2D_DESC depth_desc;
                    depth_desc.Width = width;
                    depth_desc.Height = height;
                    depth_desc.MipLevels = 1;
                    depth_desc.ArraySize = 1;
                    depth_desc.Format = DXGI_FORMAT_R32_TYPELESS; // DXGI_FORMAT_D32_FLOAT for depth view and DXGI_FORMAT_R32_FLOAT for SRV.
                    depth_desc.SampleDesc.Count = 1;
                    depth_desc.SampleDesc.Quality = 0;
                    depth_desc.Usage = D3D11_USAGE_DEFAULT;
                    depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
                    depth_desc.CPUAccessFlags = 0;
                    depth_desc.MiscFlags = 0;

                    OTexture2D depth_buffer;
                    THROW_ON_FAILURE(m_device.CreateTexture2D(&depth_desc, nullptr, &depth_buffer));

                    D3D11_DEPTH_STENCIL_VIEW_DESC depth_view_desc = CD3D11_DEPTH_STENCIL_VIEW_DESC(D3D11_DSV_DIMENSION_TEXTURE2D, DXGI_FORMAT_D32_FLOAT);
                    THROW_ON_FAILURE(m_device.CreateDepthStencilView(depth_buffer, &depth_view_desc, &m_g_buffer.depth_view));
                    
                    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = CD3D11_SHADER_RESOURCE_VIEW_DESC(D3D11_SRV_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_FLOAT);
                    THROW_ON_FAILURE(m_device.CreateShaderResourceView(depth_buffer, &srv_desc, &m_g_buffer.depth_SRV));
                }

                { // Normal buffer
                    if (m_g_buffer.normal_SRV) m_g_buffer.normal_SRV->Release();
                    if (m_g_buffer.normal_RTV) m_g_buffer.normal_RTV->Release();
                    // create_texture_2D(m_device, DXGI_FORMAT_R16G16_SNORM, width, height, &m_g_buffer.normal_SRV, nullptr, &m_g_buffer.normal_RTV);
                    create_texture_2D(m_device, DXGI_FORMAT_R16G16B16A16_FLOAT, width, height, &m_g_buffer.normal_SRV, nullptr, &m_g_buffer.normal_RTV);
                }

                m_g_buffer.width = width;
                m_g_buffer.height = height;
            }

            fill_g_buffer();
        }

        ID3D11ShaderResourceView* ssao_SRV = nullptr;
        { // Pre-render effects on G-buffer.
            auto ssao_marker = PerformanceMarker(*m_render_context, L"SSAO");

            if (m_settings.ssao_enabled)
                ssao_SRV = m_ssao.apply(m_render_context, m_g_buffer.normal_SRV, m_g_buffer.depth_SRV, m_g_buffer.width, m_g_buffer.height).get();
            else
                // A really inefficient way to disable ssao. The application is still part of the material shaders.
                ssao_SRV = m_ssao.apply_none(m_render_context, m_g_buffer.width, m_g_buffer.height).get();
        }

        m_render_context->OMSetRenderTargets(1, &backbuffer_RTV, m_g_buffer.depth_view);
        m_render_context->PSSetShaderResources(13, 1, &ssao_SRV);

        { // Render lights.
            auto lights_marker = PerformanceMarker(*m_render_context, L"Lights");

            m_environments->render(*m_render_context, scene.get_ID());

            // Bind light buffer.
            m_render_context->VSSetConstantBuffers(1, 1, m_lights.manager.light_buffer_addr());
            m_render_context->PSSetConstantBuffers(1, 1, m_lights.manager.light_buffer_addr());

            { // Render sphere lights.
                m_render_context->VSSetShader(m_lights.vertex_shader, 0, 0);
                m_render_context->PSSetShader(m_lights.pixel_shader, 0, 0);

                m_render_context->Draw(m_lights.manager.active_light_count() * 6, 0);
            }
        }

        { // Render models.
            auto opaque_marker = PerformanceMarker(*m_render_context, L"Opaque geometry");

            // Set vertex and pixel shaders.
            m_render_context->VSSetShader(m_vertex_shading.shader, 0, 0);
            m_render_context->IASetInputLayout(m_vertex_shading.input_layout);
            m_render_context->PSSetShader(m_opaque.shader, 0, 0);

            // Set null buffer as default texcoord buffer.
            unsigned int stride = sizeof(float2);
            unsigned int offset = 0;
            m_render_context->IASetVertexBuffers(2, 1, &m_vertex_shading.null_buffer, &stride, &offset);

            for (int i = 1; i < m_transparent.first_model_index; ++i) {

                // Setup twosided raster state for cutout materials.
                if (i == m_cutout.first_model_index)
                    m_render_context->RSSetState(m_cutout.raster_state);

                Dx11Model model = m_sorted_models[i];
                assert(model.model_ID != 0);
                render_model<false>(m_render_context, model);
            }

            opaque_marker.end();

            { // Render transparent models

                auto transparent_marker = PerformanceMarker(*m_render_context, L"Transparent geometry");

                // Apply used cutout state if not already applied.
                bool no_cutouts_present = m_cutout.first_model_index >= m_transparent.first_model_index;
                if (no_cutouts_present)
                    m_render_context->RSSetState(m_cutout.raster_state);

                // Set transparent state.
                m_render_context->OMSetBlendState(m_transparent.blend_state, 0, 0xffffffff);
                m_render_context->OMSetDepthStencilState(m_transparent.depth_state, 0);
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
                        Transform transform = m_transforms.get_transform(m_sorted_models[model_index].transform_ID);
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
                    render_model<false>(m_render_context, model);
                }
            }
        }
    }

    template <typename T>
    HRESULT upload_default_buffer(T* data, int element_count, D3D11_BIND_FLAG flags, ID3D11Buffer** buffer) {
        D3D11_BUFFER_DESC desc = {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = sizeof(T) * element_count;
        desc.BindFlags = flags;

        D3D11_SUBRESOURCE_DATA resource_data = {};
        resource_data.pSysMem = data;
        return m_device.CreateBuffer(&desc, &resource_data, buffer);
    }

    void handle_updates() {
        m_environments->handle_updates(m_device, *m_render_context);
        m_lights.manager.handle_updates(*m_render_context);
        m_materials.handle_updates(m_device, *m_render_context);
        m_textures.handle_updates(m_device, *m_render_context);
        m_transforms.handle_updates(m_device, *m_render_context);

        { // Mesh updates.
            for (Meshes::UID mesh_ID : Meshes::get_changed_meshes()) {
                if (m_meshes.size() <= mesh_ID)
                    m_meshes.resize(Meshes::capacity());

                if (Meshes::get_changes(mesh_ID).any_set(Meshes::Change::Created, Meshes::Change::Destroyed)) {
                    if (m_meshes[mesh_ID].vertex_count != 0) {
                        m_meshes[mesh_ID].index_count = m_meshes[mesh_ID].vertex_count = 0;
                        safe_release(&m_meshes[mesh_ID].indices);
                        safe_release(m_meshes[mesh_ID].geometry_address());
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

                    { // Upload geometry.
                        auto create_vertex_geometry = [](Vector3f p, Vector3f n) -> Dx11VertexGeometry {
                            float3 dx_p = { p.x, p.y, p.z };
                            OctahedralNormal encoded_normal = OctahedralNormal::encode_precise(n);
                            int2 dx_normal = { encoded_normal.encoding.x, encoded_normal.encoding.y };
                            int packed_dx_normal = (dx_normal.x - SHRT_MIN) | (dx_normal.y << 16);
                            Dx11VertexGeometry geometry = { dx_p, packed_dx_normal };
                            return geometry;
                        };

                        Vector3f* normals = mesh.get_normals();

                        Dx11VertexGeometry* geometry = new Dx11VertexGeometry[dx_mesh.vertex_count];
                        if (normals == nullptr) {
                            // Compute hard normals. Positions have already been expanded if there is an index buffer.
                            #pragma omp parallel for
                            for (int i = 0; i < int(dx_mesh.vertex_count); i += 3) {
                                Vector3f p0 = positions[i], p1 = positions[i + 1], p2 = positions[i+2];
                                Vector3f normal = normalize(cross(p1 - p0, p2 - p0));
                                geometry[i] = create_vertex_geometry(p0, normal);
                                geometry[i+1] = create_vertex_geometry(p1, normal);
                                geometry[i+2] = create_vertex_geometry(p2, normal);
                            }
                        } else {
                            // Copy position and normal.
                            #pragma omp parallel for
                            for (int i = 0; i < int(dx_mesh.vertex_count); ++i)
                                geometry[i] = create_vertex_geometry(positions[i], normals[i]);
                        }

                        HRESULT hr = upload_default_buffer(geometry, dx_mesh.vertex_count, D3D11_BIND_VERTEX_BUFFER,
                            dx_mesh.geometry_address());
                        if (FAILED(hr))
                            printf("Could not upload %s's geometry buffer.\n", mesh.get_name().c_str());

                        delete[] geometry;
                    }

                    // Delete temporary expanded positions.
                    if (positions != mesh.get_positions())
                        delete[] positions;

                    { // Upload texcoords if present, otherwise upload 'null buffer'.
                        Vector2f* texcoords = mesh.get_texcoords();
                        if (texcoords != nullptr) {

                            if (expand_indexed_buffers)
                                texcoords = MeshUtils::expand_indexed_buffer(mesh.get_primitives(), mesh.get_primitive_count(), texcoords);

                            HRESULT hr = upload_default_buffer(texcoords, dx_mesh.vertex_count, D3D11_BIND_VERTEX_BUFFER,
                                                               dx_mesh.texcoords_address());
                            if (FAILED(hr))
                                printf("Could not upload %s's texcoord buffer.\n", mesh.get_name().c_str());

                            if (texcoords != mesh.get_texcoords())
                                delete[] texcoords;
                        } else
                            *dx_mesh.texcoords_address() = m_vertex_shading.null_buffer;
                    }

                    bool has_texcoords = mesh.get_texcoords() != nullptr;
                    dx_mesh.buffer_count = has_texcoords ? 2 : 1;

                    m_meshes[mesh_ID] = dx_mesh;
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
                    } else if (model.get_changes() & MeshModels::Change::Material)
                        m_sorted_models[model_index].material_ID = model.get_material().get_ID();
                }

                // Sort the models in the order [dummy, opaque, cutout, transparent, destroyed].
                // The new position/index of the model is stored in model_index.
                // Incidently also compacts the list of models by removing any deleted models.
                {
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

    Renderer::Settings get_settings() const { return m_settings; }
    void set_settings(Settings& settings) { m_settings = settings; }
};

//----------------------------------------------------------------------------
// DirectX 11 renderer.
//----------------------------------------------------------------------------
IRenderer* Renderer::initialize(ID3D11Device1& device, int width_hint, int height_hint, const std::wstring& data_folder_path) {
    return new Renderer(device, width_hint, height_hint, data_folder_path);
}

Renderer::Renderer(ID3D11Device1& device, int width_hint, int height_hint, const std::wstring& data_folder_path) {
    m_impl = new Implementation(device, width_hint, height_hint, data_folder_path);
    m_renderer_ID = Renderers::create("DX11Renderer");
}

Renderer::~Renderer() {
    Cogwheel::Core::Renderers::destroy(m_renderer_ID);
    delete m_impl;
}

void Renderer::handle_updates() {
    m_impl->handle_updates();
}

void Renderer::render(ORenderTargetView& backbuffer_RTV, Cogwheel::Scene::Cameras::UID camera_ID, int width, int height) {
    m_impl->render(backbuffer_RTV, camera_ID, width, height);
}

Renderer::Settings Renderer::get_settings() const { return m_impl->get_settings(); }
void Renderer::set_settings(Settings& settings) { m_impl->set_settings(settings); }

} // NS DX11Renderer
