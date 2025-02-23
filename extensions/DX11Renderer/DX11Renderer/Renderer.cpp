// DirectX 11 renderer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/EnvironmentManager.h>
#include <DX11Renderer/LightManager.h>
#include <DX11Renderer/MaterialManager.h>
#include <DX11Renderer/MeshModelManager.h>
#include <DX11Renderer/Renderer.h>
#include <DX11Renderer/ShaderManager.h>
#include <DX11Renderer/SSAO.h>
#include <DX11Renderer/TextureManager.h>
#include <DX11Renderer/TransformManager.h>
#include <DX11Renderer/Utils.h>

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Core/Engine.h>
#include <Bifrost/Core/Window.h>
#include <Bifrost/Math/OctahedralNormal.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/SceneRoot.h>

#include <algorithm>
#include <vector>

using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;
using namespace std;

namespace DX11Renderer {

//----------------------------------------------------------------------------
// DirectX 11 renderer implementation.
//----------------------------------------------------------------------------
class Renderer::Implementation {
private:
    ODevice1& m_device;
    ODeviceContext1 m_render_context;

    ShaderManager m_shader_manager;

    ORenderTargetView m_backbuffer_RTV;
    OShaderResourceView m_backbuffer_SRV;

    Settings m_settings;
    DebugSettings m_debug_settings;

    // Bifrost resources
    vector<Dx11Mesh> m_meshes = vector<Dx11Mesh>(0);

    EnvironmentManager* m_environments;
    MaterialManager m_materials;
    TextureManager m_textures;
    TransformManager m_transforms;
    MeshModelManager m_mesh_models;
    SSAO::AlchemyAO m_ssao;

    struct {
        unsigned int width, height;
        OShaderResourceView depth_SRV;
        ODepthStencilView depth_view;
        OShaderResourceView normal_SRV;
        ORenderTargetView normal_RTV;
        ODepthStencilState depth_state;

        struct {
            OPixelShader pixel_shader;
        } lights;

        struct {
            ORasterizerState raster_state;
            OVertexShader vertex_shader;
            OInputLayout vertex_input_layout;
            OPixelShader pixel_shader;
        } opaque;

        struct {
            ORasterizerState raster_state;
            OVertexShader vertex_shader;
            OInputLayout vertex_input_layout;
            OPixelShader pixel_shader;
        } thin_walled;
    } m_g_buffer;

    struct {
        OBuffer null_buffer;
        OInputLayout input_layout;
        OVertexShader shader;
    } m_vertex_shading;

    struct {
        ODepthStencilState depth_state;
        vector<OPixelShader> shading_models;
    } m_opaque;

    struct {
        ORasterizerState backface_culled;
        ORasterizerState thin_walled;
        ID3D11RasterizerState* current_state = nullptr;

        void set_raster_state(ORasterizerState& rs_state, ODeviceContext1& render_context) { 
            render_context->RSSetState(rs_state);
            current_state = rs_state;
        }
    } m_raster_state;

    struct Transparent {
        struct SortedModel {
            float distance;
            MeshModelManager::ConstIterator model_iterator;
        };

        OBlendState blend_state;
        ODepthStencilState depth_state;
        vector<OPixelShader> shading_models;
        vector<SortedModel> sorted_models_pool; // List of sorted transparent models. Created as a pool to minimize runtime memory allocation.
    } m_transparent;

    // Mirrored GPU side by SceneVariables in Utils.hlsl
    struct SceneConstants {
        Matrix4x4f view_projection_matrix;
        Vector4f environment_tint; // .w component is 0 if an environment tex is not bound, otherwise positive.
        int2 g_buffer_to_ao_index_offset;
        int2 viewport_size;
        Matrix4x4f inverse_view_projection_matrix;
        Matrix4x4f projection_matrix;
        Matrix4x4f inverse_projection_matrix;
        Matrix3x4f world_to_view_matrix;
        Matrix3x4f view_to_world_matrix;
    };
    OBuffer m_scene_buffer;

    // Lights
    struct {
        LightManager manager;
        OVertexShader vertex_shader;
        OPixelShader pixel_shader;
    } m_lights;

    struct {
        OVertexShader display_vertex_shader;
        OPixelShader display_debug_pixel_shader;
        OPixelShader material_params_shader;
    } m_debug;

    OPixelShader create_shading_model(ODevice1& device, ShaderManager& shader_manager, const char* entry_point) {
        OBlob pixel_shader_blob = shader_manager.compile_shader_from_file("ModelShading.hlsl", "ps_5_0", entry_point);
        OPixelShader pixel_shader;
        THROW_DX11_ERROR(device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &pixel_shader));
        return pixel_shader;
    }

public:
    Implementation(ODevice1& device, const std::filesystem::path& data_directory)
        : m_device(device), m_shader_manager(ShaderManager(data_directory)) {

        device->GetImmediateContext1(&m_render_context);

        { // Setup asset managing.
            m_environments = new EnvironmentManager(m_device, m_textures, m_shader_manager);

            m_materials = MaterialManager(m_device, *m_render_context);
            m_mesh_models = MeshModelManager();
            m_textures = TextureManager(m_device);
            m_transforms = TransformManager(m_device, *m_render_context);

            // Setup static state.
            m_render_context->PSSetShaderResources(15, 1, m_materials.get_GGX_with_fresnel_rho_srv_addr());

            OSamplerState linear_sampler = TextureManager::create_clamped_linear_sampler(device);
            m_render_context->PSSetSamplers(15, 1, &linear_sampler);
            m_render_context->CSSetSamplers(15, 1, &linear_sampler);
        }

        { // Setup g-buffer
            { // Light
                OBlob pixel_shader_blob = m_shader_manager.compile_shader_from_file("SphereLight.hlsl", "ps_5_0", "g_buffer_PS");
                THROW_DX11_ERROR(m_device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_g_buffer.lights.pixel_shader));
            }

            { // Opaque shaders
                D3D11_RASTERIZER_DESC raster_state = CD3D11_RASTERIZER_DESC(CD3D11_DEFAULT());
                THROW_DX11_ERROR(m_device->CreateRasterizerState(&raster_state, &m_g_buffer.opaque.raster_state));

                OBlob vertex_shader_blob = m_shader_manager.compile_shader_from_file("ModelGBuffer.hlsl", "vs_5_0", "opaque_VS");
                THROW_DX11_ERROR(m_device->CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_g_buffer.opaque.vertex_shader));

                // Create the input layout
                D3D11_INPUT_ELEMENT_DESC input_layout_desc = { "GEOMETRY", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 };
                THROW_DX11_ERROR(m_device->CreateInputLayout(&input_layout_desc, 1, UNPACK_BLOB_ARGS(vertex_shader_blob), &m_g_buffer.opaque.vertex_input_layout));

                OBlob pixel_shader_blob = m_shader_manager.compile_shader_from_file("ModelGBuffer.hlsl", "ps_5_0", "opaque_PS");
                THROW_DX11_ERROR(m_device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_g_buffer.opaque.pixel_shader));
            }

            { // Thin walled shaders
                D3D11_RASTERIZER_DESC raster_state = CD3D11_RASTERIZER_DESC(CD3D11_DEFAULT());
                raster_state.CullMode = D3D11_CULL_NONE;
                THROW_DX11_ERROR(m_device->CreateRasterizerState(&raster_state, &m_g_buffer.thin_walled.raster_state));

                OBlob vertex_shader_blob = m_shader_manager.compile_shader_from_file("ModelGBuffer.hlsl", "vs_5_0", "thin_walled_VS");
                THROW_DX11_ERROR(m_device->CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_g_buffer.thin_walled.vertex_shader));

                // Create the input layout
                D3D11_INPUT_ELEMENT_DESC input_layout_desc[] = {
                    { "GEOMETRY", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                    { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
                };
                THROW_DX11_ERROR(m_device->CreateInputLayout(input_layout_desc, 2, UNPACK_BLOB_ARGS(vertex_shader_blob), &m_g_buffer.thin_walled.vertex_input_layout));

                OBlob pixel_shader_blob = m_shader_manager.compile_shader_from_file("ModelGBuffer.hlsl", "ps_5_0", "thin_walled_PS");
                THROW_DX11_ERROR(m_device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_g_buffer.thin_walled.pixel_shader));
            }

            // Depth-stencil state
            D3D11_DEPTH_STENCIL_DESC depth_desc = CD3D11_DEPTH_STENCIL_DESC(CD3D11_DEFAULT());
            depth_desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
            THROW_DX11_ERROR(m_device->CreateDepthStencilState(&depth_desc, &m_g_buffer.depth_state));

            // G-buffer is initialized on demand when the output dimensions are known ...
            m_g_buffer.width = m_g_buffer.height = 0u;
            m_g_buffer.depth_SRV = nullptr;
            m_g_buffer.depth_view = nullptr;
            m_g_buffer.normal_SRV = nullptr;
            m_g_buffer.normal_RTV = nullptr;

            // .. and so is the backbuffer.
            m_backbuffer_RTV = nullptr;
            m_backbuffer_SRV = nullptr;
        }

        m_ssao = SSAO::AlchemyAO(m_device, m_shader_manager);

        { // Setup vertex processing.
            OBlob vertex_shader_blob = m_shader_manager.compile_shader_from_file("ModelShading.hlsl", "vs_5_0", "vs");

            // Create the shader objects.
            THROW_DX11_ERROR(m_device->CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_vertex_shading.shader));

            // Create the input layout
            D3D11_INPUT_ELEMENT_DESC input_layout_desc[] = {
                { "GEOMETRY", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 2, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
            };

            THROW_DX11_ERROR(m_device->CreateInputLayout(input_layout_desc, 3, UNPACK_BLOB_ARGS(vertex_shader_blob), &m_vertex_shading.input_layout));

            // Create a default emptyish buffer.
            D3D11_BUFFER_DESC empty_desc = {};
            empty_desc.Usage = D3D11_USAGE_IMMUTABLE;
            empty_desc.ByteWidth = sizeof(Vector4f);
            empty_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

            Vector4f lval = Vector4f::zero();
            D3D11_SUBRESOURCE_DATA empty_data = {};
            empty_data.pSysMem = &lval;
            THROW_DX11_ERROR(m_device->CreateBuffer(&empty_desc, &empty_data, &m_vertex_shading.null_buffer));
        }

        { // Setup raster states
            // Backface culled
            D3D11_RASTERIZER_DESC raster_state = CD3D11_RASTERIZER_DESC(CD3D11_DEFAULT());
            raster_state.ScissorEnable = true; // Enable scissor rect to disable rendering to the guard band when shading.
            THROW_DX11_ERROR(m_device->CreateRasterizerState(&raster_state, &m_raster_state.backface_culled));

            // Thin walled
            raster_state.CullMode = D3D11_CULL_NONE;
            THROW_DX11_ERROR(m_device->CreateRasterizerState(&raster_state, &m_raster_state.thin_walled));
        }

        { // Setup opaque rendering.
            D3D11_DEPTH_STENCIL_DESC depth_desc = CD3D11_DEPTH_STENCIL_DESC(CD3D11_DEFAULT());
            depth_desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
            depth_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
            THROW_DX11_ERROR(m_device->CreateDepthStencilState(&depth_desc, &m_opaque.depth_state));

            m_opaque.shading_models.resize((int)ShadingModel::Count);
            m_opaque.shading_models[(int)ShadingModel::Default] = create_shading_model(m_device, m_shader_manager, "default_opaque");
            m_opaque.shading_models[(int)ShadingModel::Diffuse] = create_shading_model(m_device, m_shader_manager, "diffuse_opaque");
            m_opaque.shading_models[(int)ShadingModel::Transmissive] = create_shading_model(m_device, m_shader_manager, "error_material"); // Should never be called as transmissive aren't opaque.
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

            THROW_DX11_ERROR(m_device->CreateBlendState(&blend_desc, &m_transparent.blend_state));

            D3D11_DEPTH_STENCIL_DESC depth_desc = CD3D11_DEPTH_STENCIL_DESC(CD3D11_DEFAULT());
            depth_desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
            depth_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
            THROW_DX11_ERROR(m_device->CreateDepthStencilState(&depth_desc, &m_transparent.depth_state));

            m_transparent.shading_models.resize((int)ShadingModel::Count);
            m_transparent.shading_models[(int)ShadingModel::Default] = create_shading_model(m_device, m_shader_manager, "default_transparent");
            m_transparent.shading_models[(int)ShadingModel::Diffuse] = create_shading_model(m_device, m_shader_manager, "diffuse_transparent");
            m_transparent.shading_models[(int)ShadingModel::Transmissive] = create_shading_model(m_device, m_shader_manager, "transmissive_transparent");
        }

        { // Scene constant buffer
            THROW_DX11_ERROR(create_constant_buffer(m_device, sizeof(SceneConstants), &m_scene_buffer));
        }

        { // Setup lights
            m_lights.manager = LightManager(m_device, LightSources::capacity());

            // Sphere light visualization shaders.
            OBlob vertex_shader_blob = m_shader_manager.compile_shader_from_file("SphereLight.hlsl", "vs_5_0", "vs");
            THROW_DX11_ERROR(m_device->CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_lights.vertex_shader));
            OBlob pixel_shader_blob = m_shader_manager.compile_shader_from_file("SphereLight.hlsl", "ps_5_0", "color_PS");
            THROW_DX11_ERROR(m_device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_lights.pixel_shader));
        }

        { // Debug
            OBlob vertex_shader_blob = m_shader_manager.compile_shader_from_file("Debug.hlsl", "vs_5_0", "main_vs");
            THROW_DX11_ERROR(m_device->CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_debug.display_vertex_shader));
            OBlob display_debug_blob = m_shader_manager.compile_shader_from_file("Debug.hlsl", "ps_5_0", "display_debug_ps");
            THROW_DX11_ERROR(m_device->CreatePixelShader(UNPACK_BLOB_ARGS(display_debug_blob), nullptr, &m_debug.display_debug_pixel_shader));
            OBlob material_params_blob = m_shader_manager.compile_shader_from_file("ModelShading.hlsl", "ps_5_0", "visualize_material_params");
            THROW_DX11_ERROR(m_device->CreatePixelShader(UNPACK_BLOB_ARGS(material_params_blob), nullptr, &m_debug.material_params_shader));
        }
    }

    ~Implementation() {
        for (Dx11Mesh mesh : m_meshes) {
            safe_release(&mesh.constant_buffer);
            safe_release(&mesh.indices);
            safe_release(mesh.geometry_address());
            safe_release(mesh.texcoords_address());
            safe_release(mesh.tint_and_roughness_address());
        }

        delete m_environments;
    }

    void fill_g_buffer() {
        { // Setup state. Opaque render state should already have been set up.
            // Render target views.
            m_render_context->OMSetRenderTargets(1, &m_g_buffer.normal_RTV, m_g_buffer.depth_view);
            float zero[4] = { 0, 0, -1, 0 };
            m_render_context->ClearRenderTargetView(m_g_buffer.normal_RTV, zero);
            m_render_context->ClearDepthStencilView(m_g_buffer.depth_view, D3D11_CLEAR_DEPTH, 1.0f, 0);

            // Opaque state
            m_render_context->RSSetState(m_g_buffer.opaque.raster_state);
            m_render_context->IASetInputLayout(m_g_buffer.opaque.vertex_input_layout);
        }

        { // Render sphere lights.
            m_render_context->VSSetShader(m_lights.vertex_shader, 0, 0);
            m_render_context->PSSetShader(m_g_buffer.lights.pixel_shader, 0, 0);
            m_render_context->Draw(m_lights.manager.active_light_count() * 6, 0);
        }

        { // Render opaque objects.
            m_render_context->VSSetShader(m_g_buffer.opaque.vertex_shader, 0, 0);
            m_render_context->PSSetShader(m_g_buffer.opaque.pixel_shader, 0, 0);
            for (auto model_itr = m_mesh_models.cbegin(); model_itr != m_mesh_models.cbegin_transparent_models(); ++model_itr) {
                // Setup two-sided raster state for thin-walled materials.
                if (model_itr == m_mesh_models.cbegin_opaque_thin_walled_models()) {
                    m_render_context->RSSetState(m_g_buffer.thin_walled.raster_state);
                    m_render_context->VSSetShader(m_g_buffer.thin_walled.vertex_shader, 0, 0);
                    m_render_context->IASetInputLayout(m_g_buffer.thin_walled.vertex_input_layout);
                    m_render_context->PSSetShader(m_g_buffer.thin_walled.pixel_shader, 0, 0);
                }

                assert(model_itr->model_ID != 0);
                render_model<true>(m_render_context, *model_itr);
            }
        }
    }

    template <bool geometry_only>
    void render_model(ID3D11DeviceContext1* context, Dx11Model model) {
        Dx11Mesh mesh = m_meshes[model.mesh_ID];

        { // Set the buffers.
            context->VSSetConstantBuffers(4, 1, &mesh.constant_buffer);

            if (mesh.index_count != 0)
                context->IASetIndexBuffer(mesh.indices, DXGI_FORMAT_R32_UINT, 0);

            // Setup strides and offsets for the buffers.
            // Layout is [geometry, texcoords, tint_and_roughness].
            static unsigned int strides[3] = { sizeof(float4), sizeof(float2), sizeof(TintRoughness) };
            static unsigned int offsets[3] = { 0, 0, 0 };

            context->IASetVertexBuffers(0, mesh.vertex_buffer_count, mesh.vertex_buffers, strides, offsets);
        }

        { // Bind world transform.
            m_transforms.bind_transform(*context, 2, model.transform_ID);
        }

        { // Material parameters

          // Bind material constant buffer.
            m_materials.bind_material(*context, 3, model.material_ID);

            Dx11MaterialTextures& material_textures = m_materials.get_material_textures(model.material_ID);

            Dx11Texture& coverage_texture = m_textures.get_texture(material_textures.coverage_index);
            if (coverage_texture.sampler != nullptr) {
                context->PSSetShaderResources(1, 1, &coverage_texture.image->srv);
                context->PSSetSamplers(1, 1, &coverage_texture.sampler);
            }

            if (!geometry_only) {

                Dx11Texture& tint_roughness_texture = m_textures.get_texture(material_textures.tint_roughness_index);
                Dx11Texture& metallic_texture = m_textures.get_texture(material_textures.metallic_index);
                if (metallic_texture.image || tint_roughness_texture.image) {

                    ID3D11ShaderResourceView* SRVs[2] = {
                        tint_roughness_texture.image ? tint_roughness_texture.image->srv.get() : nullptr,
                        metallic_texture.image ? metallic_texture.image->srv.get() : nullptr
                    };
                    context->PSSetShaderResources(2, 2, SRVs);

                    ID3D11SamplerState* samplers[2] = { tint_roughness_texture.sampler, metallic_texture.sampler };
                    context->PSSetSamplers(2, 2, samplers);
                }
            }
        }

        if (mesh.index_count != 0)
            context->DrawIndexed(mesh.index_count, 0, 0);
        else
            context->Draw(mesh.vertex_count, 0);
    } 

    void post_render_cleanup() {
        // Reset scissor rect used to maske out the guard band.
        m_render_context->RSSetScissorRects(0, nullptr);

        // Unbind the rendertarget as the SRV is returned for reading.
        ID3D11RenderTargetView* null_RTV = nullptr;
        m_render_context->OMSetRenderTargets(1, &null_RTV, nullptr);
    }

    RenderedFrame render(const Bifrost::Scene::CameraID camera_ID, Vector2i frame_size) {
        int frame_width = frame_size.x;
        int frame_height = frame_size.y;

        int2 g_buffer_guard_band_size = { int(frame_width * m_settings.g_buffer_guard_band_scale),
                                          int(frame_height * m_settings.g_buffer_guard_band_scale) };
        int g_buffer_width = frame_width + 2 * g_buffer_guard_band_size.x;
        int g_buffer_height = frame_height + 2 * g_buffer_guard_band_size.y;
        Recti backbuffer_viewport = { g_buffer_guard_band_size.x, g_buffer_guard_band_size.y, frame_width, frame_height };

        SceneRoot scene = Cameras::get_scene_ID(camera_ID);
        { // Setup scene constants.
            // Recalculate the guard band scale to account for the buffer dimensions being discrete.
            float2 guard_band_scale = { g_buffer_guard_band_size.x / float(frame_width), g_buffer_guard_band_size.y / float(frame_height) };

            // Scale projection matrix and it's inverse to fit the projection onto the backbuffer + guard band.
            float2 inverse_projection_matrix_scale = { 1.0f + 2.0f * guard_band_scale.x, 1.0f + 2.0f * guard_band_scale.y };
            Matrix4x4f inverse_projection_matrix = Cameras::get_inverse_projection_matrix(camera_ID);
            inverse_projection_matrix.set_row(0, inverse_projection_matrix.get_row(0) * inverse_projection_matrix_scale.x);
            inverse_projection_matrix.set_row(1, inverse_projection_matrix.get_row(1) * inverse_projection_matrix_scale.y);

            float2 projection_matrix_scale = { 1.0f / inverse_projection_matrix_scale.x, 1.0f / inverse_projection_matrix_scale.y };
            Matrix4x4f projection_matrix = Cameras::get_projection_matrix(camera_ID);
            projection_matrix.set_column(0, projection_matrix.get_column(0) * projection_matrix_scale.x);
            projection_matrix.set_column(1, projection_matrix.get_column(1) * projection_matrix_scale.y);
            
            Transform world_to_view_transform = Cameras::get_view_transform(camera_ID);
            Transform view_to_world_transform = Cameras::get_inverse_view_transform(camera_ID);

            SceneConstants scene_vars;
            scene_vars.view_projection_matrix = projection_matrix * to_matrix4x4(world_to_view_transform);
            RGB env_tint = scene.get_environment_tint();
            scene_vars.environment_tint = { env_tint.r, env_tint.g, env_tint.b, float(scene.get_environment_map().get_ID().get_index()) };
            scene_vars.g_buffer_to_ao_index_offset = m_ssao.compute_g_buffer_to_ao_index_offset(m_settings.ssao.settings, backbuffer_viewport);
            scene_vars.viewport_size = { backbuffer_viewport.width, backbuffer_viewport.height };
            scene_vars.inverse_view_projection_matrix = to_matrix4x4(view_to_world_transform) * inverse_projection_matrix;
            scene_vars.projection_matrix = projection_matrix;
            scene_vars.inverse_projection_matrix = inverse_projection_matrix;
            scene_vars.world_to_view_matrix = to_matrix3x4(world_to_view_transform);
            scene_vars.view_to_world_matrix = to_matrix3x4(view_to_world_transform);
            m_render_context->UpdateSubresource(m_scene_buffer, 0, nullptr, &scene_vars, 0, 0);
            m_render_context->VSSetConstantBuffers(13, 1, &m_scene_buffer);
            m_render_context->PSSetConstantBuffers(13, 1, &m_scene_buffer);
            m_render_context->CSSetConstantBuffers(13, 1, &m_scene_buffer);
        }

        { // Bind light buffers
            m_render_context->VSSetConstantBuffers(12, 1, m_lights.manager.light_buffer_addr());
            m_render_context->PSSetConstantBuffers(12, 1, m_lights.manager.light_buffer_addr());
        }

        m_render_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        { // Render G-buffer
            auto g_buffer_marker = PerformanceMarker(*m_render_context, L"G-buffer");

            // Re-allocate buffers if the dimensions have changed.
            if (m_g_buffer.width < unsigned int(g_buffer_width )|| m_g_buffer.height < unsigned int(g_buffer_height)) {
                m_g_buffer.width = std::max(m_g_buffer.width, unsigned int(g_buffer_width));
                m_g_buffer.height = std::max(m_g_buffer.height, unsigned int(g_buffer_height));

                { // Backbuffer.
                    m_backbuffer_RTV.release();
                    m_backbuffer_SRV.release();

                    create_texture_2D(m_device, DXGI_FORMAT_R16G16B16A16_FLOAT, m_g_buffer.width, m_g_buffer.height, &m_backbuffer_SRV, nullptr, &m_backbuffer_RTV);
                }

                { // Depth buffer
                    m_g_buffer.depth_SRV.release();
                    m_g_buffer.depth_view.release();

                    D3D11_TEXTURE2D_DESC depth_desc;
                    depth_desc.Width = m_g_buffer.width;
                    depth_desc.Height = m_g_buffer.height;
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
                    THROW_DX11_ERROR(m_device->CreateTexture2D(&depth_desc, nullptr, &depth_buffer));

                    D3D11_DEPTH_STENCIL_VIEW_DESC depth_view_desc = CD3D11_DEPTH_STENCIL_VIEW_DESC(D3D11_DSV_DIMENSION_TEXTURE2D, DXGI_FORMAT_D32_FLOAT);
                    THROW_DX11_ERROR(m_device->CreateDepthStencilView(depth_buffer, &depth_view_desc, &m_g_buffer.depth_view));

                    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = CD3D11_SHADER_RESOURCE_VIEW_DESC(D3D11_SRV_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_FLOAT);
                    THROW_DX11_ERROR(m_device->CreateShaderResourceView(depth_buffer, &srv_desc, &m_g_buffer.depth_SRV));
                }

                { // Normal buffer
                    m_g_buffer.normal_SRV.release();
                    m_g_buffer.normal_RTV.release();
                    create_texture_2D(m_device, DXGI_FORMAT_R16G16_SNORM, m_g_buffer.width, m_g_buffer.height, &m_g_buffer.normal_SRV, nullptr, &m_g_buffer.normal_RTV);
                }
            }

            { // Set the full g-buffer viewport.
                D3D11_VIEWPORT dx_viewport;
                dx_viewport.TopLeftX = dx_viewport.TopLeftY = 0.0f;
                dx_viewport.Width = float(g_buffer_width);
                dx_viewport.Height = float(g_buffer_height);
                dx_viewport.MinDepth = 0.0f;
                dx_viewport.MaxDepth = 1.0f;
                m_render_context->RSSetViewports(1, &dx_viewport);
            }

            // Opaque state setup.
            m_render_context->OMSetBlendState(nullptr, 0, 0xffffffff);
            m_render_context->OMSetDepthStencilState(m_g_buffer.depth_state, 0);

            fill_g_buffer();
        }

        ID3D11ShaderResourceView* ssao_SRV = nullptr;
        { // Pre-render effects on G-buffer.
            auto ssao_marker = PerformanceMarker(*m_render_context, L"SSAO");

            if (m_settings.ssao.enabled && m_settings.ssao.settings.sample_count > 0) {
                int2 g_buffer_size = { int(m_g_buffer.width),  int(m_g_buffer.height) };
                ssao_SRV = m_ssao.apply(m_render_context, camera_ID.get_index(), m_g_buffer.normal_SRV, m_g_buffer.depth_SRV, 
                                        g_buffer_size, backbuffer_viewport, m_settings.ssao.settings).get();
            } else
                // A really inefficient way to disable ssao. Application is still part of the material shaders.
                ssao_SRV = m_ssao.apply_none(m_render_context, camera_ID.get_index(), backbuffer_viewport).get();
        }

        // Scissor rect to disable rendering to the guard band.
        D3D11_RECT rect = CD3D11_RECT(g_buffer_guard_band_size.x, g_buffer_guard_band_size.y,
                                      frame_width + g_buffer_guard_band_size.x, frame_height + g_buffer_guard_band_size.y);
        m_render_context->RSSetScissorRects(1, &rect);

        // Debug display g-buffer or AO
        if (m_debug_settings.display_mode == DebugSettings::DisplayMode::Depth ||
            m_debug_settings.display_mode == DebugSettings::DisplayMode::Normals ||
            m_debug_settings.display_mode == DebugSettings::DisplayMode::SceneSize ||
            m_debug_settings.display_mode == DebugSettings::DisplayMode::AO) {

            m_render_context->OMSetRenderTargets(1, &m_backbuffer_RTV, nullptr);

            ID3D11ShaderResourceView* srvs[3] = { m_g_buffer.normal_SRV, m_g_buffer.depth_SRV, ssao_SRV };
            m_render_context->PSSetShaderResources(0, 3, srvs);

            int4 display_constants = { int(m_debug_settings.display_mode), 0, 0, 0 };
            OBuffer display_constant_buffer;
            create_constant_buffer(m_device, display_constants, &display_constant_buffer);

            m_render_context->PSSetConstantBuffers(1, 1, &display_constant_buffer);
            m_render_context->VSSetShader(m_debug.display_vertex_shader, 0, 0);
            m_render_context->PSSetShader(m_debug.display_debug_pixel_shader, 0, 0);
            m_render_context->Draw(3, 0);

            display_constant_buffer.release();
            m_render_context->PSSetConstantBuffers(1, 1, &display_constant_buffer); // Reset by setting cleared buffer

            post_render_cleanup();
            return { m_backbuffer_SRV, backbuffer_viewport, std::numeric_limits<unsigned int>::max() };
        }

        m_render_context->OMSetRenderTargets(1, &m_backbuffer_RTV, m_g_buffer.depth_view);
        m_render_context->PSSetShaderResources(13, 1, &ssao_SRV);
        m_raster_state.set_raster_state(m_raster_state.backface_culled, m_render_context);
        m_render_context->OMSetDepthStencilState(m_opaque.depth_state, 0);

        { // Render lights.
            auto lights_marker = PerformanceMarker(*m_render_context, L"Lights");

            m_environments->render(*m_render_context, scene.get_ID());

            { // Render sphere lights.
                m_render_context->VSSetShader(m_lights.vertex_shader, 0, 0);
                m_render_context->PSSetShader(m_lights.pixel_shader, 0, 0);
                m_render_context->Draw(m_lights.manager.active_light_count() * 6, 0);
            }
        }

        { // Render models.
            m_render_context->IASetInputLayout(m_vertex_shading.input_layout);
            m_render_context->VSSetShader(m_vertex_shading.shader, 0, 0);

            // Setup debug material if needed.
            bool debug_material_params = m_debug_settings.display_mode == DebugSettings::DisplayMode::Tint ||
                                         m_debug_settings.display_mode == DebugSettings::DisplayMode::Roughness ||
                                         m_debug_settings.display_mode == DebugSettings::DisplayMode::Metallic ||
                                         m_debug_settings.display_mode == DebugSettings::DisplayMode::Coat ||
                                         m_debug_settings.display_mode == DebugSettings::DisplayMode::CoatRoughness ||
                                         m_debug_settings.display_mode == DebugSettings::DisplayMode::Coverage ||
                                         m_debug_settings.display_mode == DebugSettings::DisplayMode::UV;
            if (debug_material_params) {
                Vector4i debug_material_constants = { m_debug_settings.display_mode, 0, 0, 0 };
                OBuffer debug_material_constant_buffer;
                create_constant_buffer(m_device, debug_material_constants, &debug_material_constant_buffer);
                m_render_context->PSSetConstantBuffers(4, 1, &debug_material_constant_buffer);

                // Clear blend.
                m_render_context->OMSetBlendState(nullptr, 0, 0xffffffff);

                // Setup depth state with LESS_EQUAL depth component to allow rendering transparent geometry as opaque on top of the g-buffer.
                m_render_context->OMSetDepthStencilState(m_g_buffer.depth_state, 0);

                m_render_context->PSSetShader(m_debug.material_params_shader, 0, 0);
                for (auto model_itr = m_mesh_models.cbegin(); model_itr != m_mesh_models.cend(); ++model_itr) {
                    assert(model_itr->model_ID != 0);

                    // Setup two-sided raster state for thin-walled materials.
                    if (model_itr == m_mesh_models.cbegin_opaque_thin_walled_models())
                        m_raster_state.set_raster_state(m_raster_state.thin_walled, m_render_context);

                    render_model<false>(m_render_context, *model_itr);
                }

            } else {

                { // Render opaque models
                    auto opaque_marker = PerformanceMarker(*m_render_context, L"Opaque geometry");

                    unsigned int previous_shading_model = 0;
                    m_render_context->PSSetShader(m_opaque.shading_models[previous_shading_model], 0, 0);
                    for (auto model_itr = m_mesh_models.cbegin(); model_itr != m_mesh_models.cbegin_transparent_models(); ++model_itr) {
                        assert(model_itr->model_ID != 0);

                        // Setup two-sided raster state for thin-walled materials.
                        if (model_itr == m_mesh_models.cbegin_opaque_thin_walled_models())
                            m_raster_state.set_raster_state(m_raster_state.thin_walled, m_render_context);

                        unsigned int current_shading_model = m_materials.get_material(model_itr->material_ID).shading_model;
                        if (previous_shading_model != current_shading_model) {
                            m_render_context->PSSetShader(m_opaque.shading_models[current_shading_model], 0, 0);
                            previous_shading_model = current_shading_model;
                        }

                        render_model<false>(m_render_context, *model_itr);
                    }

                    opaque_marker.end();
                }

                { // Render transparent models
                    auto transparent_marker = PerformanceMarker(*m_render_context, L"Transparent geometry");

                    auto transparent_models = m_transparent.sorted_models_pool; // Alias the pool.
                    { // Sort transparent models. TODO in a separate thread that is waited on when we get to the transparent render index.
                        int transparent_model_count = int(m_mesh_models.cend() - m_mesh_models.cbegin_transparent_models());
                        transparent_models.reserve(transparent_model_count);
                        transparent_models.resize(0);

                        Vector3f cam_pos = Cameras::get_transform(camera_ID).translation;
                        for (auto model_itr = m_mesh_models.cbegin_transparent_models(); model_itr != m_mesh_models.cend(); ++model_itr) {
                            // Calculate the distance to point halfway between the models center and side of the bounding box.
                            Dx11Mesh& mesh = m_meshes[model_itr->mesh_ID];
                            Transform transform = m_transforms.get_transform(model_itr->transform_ID);
                            Bifrost::Math::AABB bounds = { Vector3f(mesh.bounds.min.x, mesh.bounds.min.y, mesh.bounds.min.z),
                                                           Vector3f(mesh.bounds.max.x, mesh.bounds.max.y, mesh.bounds.max.z) };
                            float distance_to_cam = alpha_sort_value(cam_pos, transform, bounds);
                            transparent_models.push_back({ distance_to_cam, model_itr });
                        }

                        std::sort(transparent_models.begin(), transparent_models.end(),
                            [](Transparent::SortedModel lhs, Transparent::SortedModel rhs) -> bool {
                            return lhs.distance < rhs.distance;
                        });
                    }

                    // Set transparent state.
                    m_render_context->OMSetBlendState(m_transparent.blend_state, 0, 0xffffffff);
                    m_render_context->OMSetDepthStencilState(m_transparent.depth_state, 0);

                    unsigned int previous_shading_model = 0;
                    m_render_context->PSSetShader(m_transparent.shading_models[previous_shading_model], 0, 0);
                    for (auto transparent_model : transparent_models) {
                        Dx11Model model = *transparent_model.model_iterator;
                        assert(model.model_ID != 0);

                        unsigned int current_shading_model = m_materials.get_material(model.material_ID).shading_model;
                        if (previous_shading_model != current_shading_model) {
                            m_render_context->PSSetShader(m_transparent.shading_models[current_shading_model], 0, 0);
                            previous_shading_model = current_shading_model;
                        }

                        // Apply raster state
                        if (model.is_thin_walled()) {
                            if (m_raster_state.current_state != m_raster_state.thin_walled.get())
                                m_raster_state.set_raster_state(m_raster_state.thin_walled, m_render_context);
                        } else {
                            if (m_raster_state.current_state != m_raster_state.backface_culled.get())
                                m_raster_state.set_raster_state(m_raster_state.backface_culled, m_render_context);
                        }

                        render_model<false>(m_render_context, model);
                    }
                }
            }
        }

        post_render_cleanup();
        return { m_backbuffer_SRV, backbuffer_viewport, std::numeric_limits<unsigned int>::max() };
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
        m_environments->handle_updates(m_device, *m_render_context);
        m_lights.manager.handle_updates(*m_render_context);
        m_materials.handle_updates(m_device, *m_render_context);
        m_textures.handle_updates(m_device, *m_render_context);
        m_transforms.handle_updates(m_device, *m_render_context);

        { // Camera updates.
            for (CameraID cam_ID : Cameras::get_changed_cameras())
                if (Cameras::get_changes(cam_ID).contains(Cameras::Change::Destroyed))
                    m_ssao.clear_camera_state(cam_ID);
        }

        { // Mesh updates.
            for (MeshID mesh_ID : Meshes::get_changed_meshes()) {
                if (m_meshes.size() <= mesh_ID)
                    m_meshes.resize(Meshes::capacity());

                auto mesh_changes = Meshes::get_changes(mesh_ID);
                if (mesh_changes.any_set(Meshes::Change::Created, Meshes::Change::Destroyed)) {
                    if (m_meshes[mesh_ID].vertex_count != 0) {
                        m_meshes[mesh_ID].index_count = m_meshes[mesh_ID].vertex_count = 0;
                        safe_release(&m_meshes[mesh_ID].constant_buffer);
                        safe_release(&m_meshes[mesh_ID].indices);
                        safe_release(m_meshes[mesh_ID].geometry_address());
                        safe_release(m_meshes[mesh_ID].texcoords_address());
                        safe_release(m_meshes[mesh_ID].tint_and_roughness_address());
                    }
                }

                if (mesh_changes.is_set(Meshes::Change::Created) && !mesh_changes.is_set(Meshes::Change::Destroyed)) {
                    Bifrost::Assets::Mesh mesh = mesh_ID;
                    Dx11Mesh dx_mesh = {};

                    Bifrost::Math::AABB bounds = mesh.get_bounds();
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

                    { // Upload tints and roughness if present, otherwise upload 'null buffer'.
                        TintRoughness* tints = mesh.get_tint_and_roughness();
                        if (tints != nullptr) {

                            if (expand_indexed_buffers)
                                tints = MeshUtils::expand_indexed_buffer(mesh.get_primitives(), mesh.get_primitive_count(), tints);

                            HRESULT hr = upload_default_buffer(tints, dx_mesh.vertex_count, D3D11_BIND_VERTEX_BUFFER, dx_mesh.tint_and_roughness_address());
                            if (FAILED(hr))
                                printf("Could not upload %s's tint and roughness buffer.\n", mesh.get_name().c_str());

                            if (tints != mesh.get_tint_and_roughness())
                                delete[] tints;
                        } else
                            *dx_mesh.tint_and_roughness_address() = m_vertex_shading.null_buffer;
                    }

                    // Constant buffer
                    Dx11MeshConstans mesh_constants;
                    mesh_constants.has_tint_and_roughness = mesh.get_tint_and_roughness() != nullptr;
                    THROW_DX11_ERROR(create_constant_buffer(m_device, mesh_constants, &dx_mesh.constant_buffer));

                    // Set the buffer count to the minimal number of buffers containing data.
                    bool has_texcoords = mesh.get_texcoords() != nullptr;
                    dx_mesh.vertex_buffer_count = has_texcoords ? 2 : 1;
                    bool has_colors = mesh.get_tint_and_roughness() != nullptr;
                    dx_mesh.vertex_buffer_count = has_colors ? 3 : dx_mesh.vertex_buffer_count;

                    m_meshes[mesh_ID] = dx_mesh;
                }
            }
        }

        m_mesh_models.handle_updates();
    }

    Renderer::Settings get_settings() const { return m_settings; }
    void set_settings(Settings& settings) { m_settings = settings; }

    Renderer::DebugSettings get_debug_settings() const { return m_debug_settings; }
    void set_debug_settings(DebugSettings& settings) { m_debug_settings = settings; }
};

//----------------------------------------------------------------------------
// DirectX 11 renderer.
//----------------------------------------------------------------------------
Renderer::Renderer(ODevice1& device, const std::filesystem::path& data_directory) {
    m_impl = new Implementation(device, data_directory);
    m_renderer_ID = Renderers::create("DX11Renderer");
}

Renderer::~Renderer() {
    Bifrost::Core::Renderers::destroy(m_renderer_ID);
    delete m_impl;
}

void Renderer::handle_updates() {
    m_impl->handle_updates();
}

RenderedFrame Renderer::render(Bifrost::Scene::CameraID camera_ID, Vector2i frame_size) {
    return m_impl->render(camera_ID, frame_size);
}

Renderer::Settings Renderer::get_settings() const { return m_impl->get_settings(); }
void Renderer::set_settings(Settings& settings) { m_impl->set_settings(settings); }

Renderer::DebugSettings Renderer::get_debug_settings() const { return m_impl->get_debug_settings(); }
void Renderer::set_debug_settings(DebugSettings& settings) { m_impl->set_debug_settings(settings); }

} // NS DX11Renderer
