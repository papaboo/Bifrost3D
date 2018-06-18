// Cogwheel ImGui DX11 Renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <ImGui/Renderers/DX11Renderer.h>

#include <ImGui/Src/imgui.h>

#include <DX11Renderer/Types.h>
#include <DX11Renderer/Utils.h>

using namespace ::DX11Renderer;

namespace ImGui {
namespace Renderers {

struct DX11Renderer::Implementation {

    // --------------------------------------------------------------------------------------------
    // Members
    // --------------------------------------------------------------------------------------------
    ODevice1& m_device;

    ORasterizerState m_rasterizer_state;
    OBlendState m_blend_state;
    ODepthStencilState m_depth_stencil_state;
    OBuffer m_projection_matrix; // Constant buffer

    OInputLayout m_input_layout;
    OVertexShader m_vertex_shader;
    OPixelShader m_pixel_shader;

    OShaderResourceView m_font_SRV;

    OBuffer m_vertex_buffer;
    int m_vertex_buffer_capacity;
    OBuffer m_index_buffer;
    int m_index_buffer_capacity;

    // --------------------------------------------------------------------------------------------
    // Constructor
    // --------------------------------------------------------------------------------------------
    DX11Renderer::Implementation(ODevice1& device) 
        : m_device(device) {

        { // Pipeline state

            // Rasterizer state
            D3D11_RASTERIZER_DESC rasterizer_desc = {};
            rasterizer_desc.FillMode = D3D11_FILL_SOLID;
            rasterizer_desc.CullMode = D3D11_CULL_NONE;
            rasterizer_desc.ScissorEnable = true;
            rasterizer_desc.DepthClipEnable = true;
            m_device->CreateRasterizerState(&rasterizer_desc, &m_rasterizer_state);

            // Blend state
            D3D11_BLEND_DESC blend_desc = {};
            blend_desc.AlphaToCoverageEnable = false;
            blend_desc.RenderTarget[0].BlendEnable = true;
            blend_desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
            blend_desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
            blend_desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
            blend_desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
            blend_desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
            blend_desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
            blend_desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
            m_device->CreateBlendState(&blend_desc, &m_blend_state);

            // Depth stencil state
            D3D11_DEPTH_STENCIL_DESC depth_stencil_desc = {};
            depth_stencil_desc.DepthEnable = false;
            depth_stencil_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
            depth_stencil_desc.DepthFunc = D3D11_COMPARISON_ALWAYS;
            depth_stencil_desc.StencilEnable = false;
            depth_stencil_desc.FrontFace.StencilFailOp = depth_stencil_desc.FrontFace.StencilDepthFailOp = depth_stencil_desc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
            depth_stencil_desc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
            depth_stencil_desc.BackFace = depth_stencil_desc.FrontFace;
            m_device->CreateDepthStencilState(&depth_stencil_desc, &m_depth_stencil_state);

            // Projection matrix
            create_constant_buffer(m_device, 16 * sizeof(float), &m_projection_matrix);
        }

        { // Shaders
            { // Vertex
                static const char* vertex_shader_src =
                    "cbuffer vertexBuffer : register(b0) { \
                        float4x4 ProjectionMatrix; \
                    };\
                    struct VS_INPUT {\
                        float2 pos : POSITION;\
                        float4 col : COLOR0;\
                        float2 uv  : TEXCOORD0;\
                    };\
                    \
                    struct PS_INPUT {\
                        float4 pos : SV_POSITION;\
                        float4 col : COLOR0;\
                        float2 uv  : TEXCOORD0;\
                    };\
                    \
                    PS_INPUT main(VS_INPUT input) {\
                        PS_INPUT output;\
                        output.pos = mul(ProjectionMatrix, float4(input.pos.xy, 0.f, 1.f));\
                        output.col = input.col;\
                        output.uv  = input.uv;\
                        return output;\
                    }";

                OBlob vertex_shader_blob;
                THROW_ON_FAILURE(D3DCompile(vertex_shader_src, strlen(vertex_shader_src), NULL, NULL, NULL, "main", "vs_5_0", 0, 0, &vertex_shader_blob, NULL));
                THROW_ON_FAILURE(m_device->CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), NULL, &m_vertex_shader));

                // Create the input layout
                D3D11_INPUT_ELEMENT_DESC local_layout[] = {
                    { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,   0, (size_t)(&((ImDrawVert*)0)->pos), D3D11_INPUT_PER_VERTEX_DATA, 0 },
                    { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,   0, (size_t)(&((ImDrawVert*)0)->uv),  D3D11_INPUT_PER_VERTEX_DATA, 0 },
                    { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, (size_t)(&((ImDrawVert*)0)->col), D3D11_INPUT_PER_VERTEX_DATA, 0 },
                };
                THROW_ON_FAILURE(m_device->CreateInputLayout(local_layout, 3, UNPACK_BLOB_ARGS(vertex_shader_blob), &m_input_layout));
            }

            { // Pixel shader
                static const char* pixel_shader_src =
                    "struct PS_INPUT {\
                        float4 pos : SV_POSITION;\
                        float4 col : COLOR0;\
                        float2 uv  : TEXCOORD0;\
                    };\
                    Texture2D texture0 : register(t0);\
                    SamplerState bilinear_sampler : register(s15);\
                    \
                    float4 main(PS_INPUT input) : SV_Target {\
                        return input.col * texture0.Sample(bilinear_sampler, input.uv); \
                    }";

                OBlob pixel_shader_blob;
                THROW_ON_FAILURE(D3DCompile(pixel_shader_src, strlen(pixel_shader_src), NULL, NULL, NULL, "main", "ps_5_0", 0, 0, &pixel_shader_blob, NULL));
                THROW_ON_FAILURE(m_device->CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), NULL, &m_pixel_shader));
            }
        }

        { // Font texture and sampler
            // Build texture atlas
            ImGuiIO& io = ImGui::GetIO();
            unsigned char* pixels;
            int width, height;
            io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

            // Upload texture to graphics system
            D3D11_TEXTURE2D_DESC desc = {};
            desc.Width = width;
            desc.Height = height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            desc.CPUAccessFlags = 0;

            D3D11_SUBRESOURCE_DATA subresource;
            subresource.pSysMem = pixels;
            subresource.SysMemPitch = desc.Width * 4;
            subresource.SysMemSlicePitch = 0;
            
            OTexture2D texture;
            THROW_ON_FAILURE(m_device->CreateTexture2D(&desc, &subresource, &texture));

            // Create texture view
            D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
            srv_desc.Format = desc.Format;
            srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            srv_desc.Texture2D.MipLevels = desc.MipLevels;
            srv_desc.Texture2D.MostDetailedMip = 0;
            THROW_ON_FAILURE(m_device->CreateShaderResourceView(texture, &srv_desc, &m_font_SRV));

            // Store our identifier
            io.Fonts->TexID = (void *)m_font_SRV; // NOTE Apparently needed to flag that a font is set/loaded.
        }

        { // Buffers are initialized on demand
            m_vertex_buffer = nullptr; 
            m_vertex_buffer_capacity = 0;
            m_index_buffer = nullptr;
            m_index_buffer_capacity = 0;
        }
    }

    // --------------------------------------------------------------------------------------------
    // Render GUI
    // --------------------------------------------------------------------------------------------
    void render(ODeviceContext1& context) {

        if (ImGui::GetFrameCount() == 0)
            return;

        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();

        // No GUI
        if (draw_data->TotalVtxCount == 0)
            return;

        { // Resize internal buffers if needed.
            // Vertex buffer
            if (m_vertex_buffer_capacity < draw_data->TotalVtxCount) {
                m_vertex_buffer.release();
                m_vertex_buffer_capacity = draw_data->TotalVtxCount + 5000;
                D3D11_BUFFER_DESC desc = {};
                desc.Usage = D3D11_USAGE_DYNAMIC;
                desc.ByteWidth = m_vertex_buffer_capacity * sizeof(ImDrawVert);
                desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
                desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                THROW_ON_FAILURE(m_device->CreateBuffer(&desc, NULL, &m_vertex_buffer));
            }

            // Index buffer
            if (m_index_buffer_capacity < draw_data->TotalIdxCount) {
                m_index_buffer.release();
                m_index_buffer_capacity = draw_data->TotalIdxCount + 10000;
                D3D11_BUFFER_DESC desc = {};
                desc.Usage = D3D11_USAGE_DYNAMIC;
                desc.ByteWidth = m_index_buffer_capacity * sizeof(ImDrawIdx);
                desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
                desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                THROW_ON_FAILURE(m_device->CreateBuffer(&desc, NULL, &m_index_buffer));
            }
        }

        { // Render ImGui

            D3D11_MAPPED_SUBRESOURCE vertex_resource, index_resource;
            THROW_ON_FAILURE(context->Map(m_vertex_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &vertex_resource));
            THROW_ON_FAILURE(context->Map(m_index_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &index_resource));
            ImDrawVert* vertex_dest = (ImDrawVert*)vertex_resource.pData;
            ImDrawIdx* index_dest = (ImDrawIdx*)index_resource.pData;
            for (int n = 0; n < draw_data->CmdListsCount; ++n) {
                const ImDrawList* cmd_list = draw_data->CmdLists[n];
                memcpy(vertex_dest, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
                memcpy(index_dest, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
                vertex_dest += cmd_list->VtxBuffer.Size;
                index_dest += cmd_list->IdxBuffer.Size;
            }
            context->Unmap(m_vertex_buffer, 0);
            context->Unmap(m_index_buffer, 0);

            { // Setup orthographic projection matrix into our constant buffer
                float L = 0.0f;
                float R = ImGui::GetIO().DisplaySize.x;
                float B = ImGui::GetIO().DisplaySize.y;
                float T = 0.0f;
                float mvp[4][4] = {
                    { 2.0f / (R - L),   0.0f,           0.0f,       0.0f },
                    { 0.0f,         2.0f / (T - B),     0.0f,       0.0f },
                    { 0.0f,         0.0f,           0.5f,       0.0f },
                    { (R + L) / (L - R),  (T + B) / (B - T),    0.5f,       1.0f },
                };
                context->UpdateSubresource(m_projection_matrix, 0, nullptr, mvp, 0u, 0u);
            }

            // Backup DX state that will be modified to restore it afterwards (unfortunately this is very ugly looking and verbose. Close your eyes!)
            struct DX11BackupState {
                unsigned int             scissor_rect_count, viewport_count;
                D3D11_RECT               scissor_rects[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
                D3D11_VIEWPORT           viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
                ORasterizerState         rasterizer_state;
                OBlendState              blend_state;
                float                    blend_factor[4];
                unsigned int             sample_mask;
                unsigned int             stencil_ref;
                ODepthStencilState       depth_stencil_state;
                D3D11_PRIMITIVE_TOPOLOGY primitive_topology;
            } old;
            old.scissor_rect_count = old.viewport_count = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
            context->RSGetScissorRects(&old.scissor_rect_count, old.scissor_rects);
            context->RSGetViewports(&old.viewport_count, old.viewports);
            context->RSGetState(&old.rasterizer_state);
            context->OMGetBlendState(&old.blend_state, old.blend_factor, &old.sample_mask);
            context->OMGetDepthStencilState(&old.depth_stencil_state, &old.stencil_ref);
            context->IAGetPrimitiveTopology(&old.primitive_topology);

            // Setup viewport
            D3D11_VIEWPORT vp = {};
            vp.Width = ImGui::GetIO().DisplaySize.x;
            vp.Height = ImGui::GetIO().DisplaySize.y;
            vp.MinDepth = 0.0f;
            vp.MaxDepth = 1.0f;
            vp.TopLeftX = vp.TopLeftY = 0.0f;
            context->RSSetViewports(1, &vp);

            // Bind shader and vertex buffers
            unsigned int stride = sizeof(ImDrawVert);
            unsigned int offset = 0;
            context->IASetInputLayout(m_input_layout);
            context->IASetVertexBuffers(0, 1, &m_vertex_buffer, &stride, &offset);
            context->IASetIndexBuffer(m_index_buffer, sizeof(ImDrawIdx) == 2 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT, 0);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(m_vertex_shader, nullptr, 0);
            context->VSSetConstantBuffers(0, 1, &m_projection_matrix);
            context->PSSetShader(m_pixel_shader, nullptr, 0);

            // Setup render state
            const float blend_factor[4] = { 0.f, 0.f, 0.f, 0.f };
            context->OMSetBlendState(m_blend_state, blend_factor, 0xffffffff);
            context->OMSetDepthStencilState(m_depth_stencil_state, 0);
            context->RSSetState(m_rasterizer_state);

            // Render command lists
            int vertex_offset = 0;
            int index_offset = 0;
            for (int n = 0; n < draw_data->CmdListsCount; ++n) {
                const ImDrawList* cmd_list = draw_data->CmdLists[n];
                for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; ++cmd_i) {
                    const ImDrawCmd* draw_cmd = &cmd_list->CmdBuffer[cmd_i];
                    if (draw_cmd->UserCallback) {
                        draw_cmd->UserCallback(cmd_list, draw_cmd);
                    } else {
                        const D3D11_RECT r = { (LONG)draw_cmd->ClipRect.x, (LONG)draw_cmd->ClipRect.y, (LONG)draw_cmd->ClipRect.z, (LONG)draw_cmd->ClipRect.w };
                        context->PSSetShaderResources(0, 1, (ID3D11ShaderResourceView**)&draw_cmd->TextureId);
                        context->RSSetScissorRects(1, &r);
                        context->DrawIndexed(draw_cmd->ElemCount, index_offset, vertex_offset);
                    }
                    index_offset += draw_cmd->ElemCount;
                }
                vertex_offset += cmd_list->VtxBuffer.Size;
            }

            // Restore modified DX state
            context->RSSetScissorRects(old.scissor_rect_count, old.scissor_rects);
            context->RSSetViewports(old.viewport_count, old.viewports);
            context->RSSetState(old.rasterizer_state);
            context->OMSetBlendState(old.blend_state, old.blend_factor, old.sample_mask);
            context->OMSetDepthStencilState(old.depth_stencil_state, old.stencil_ref);
            context->IASetPrimitiveTopology(old.primitive_topology);
        }
    }
};

// ------------------------------------------------------------------------------------------------
// Facade
// ------------------------------------------------------------------------------------------------
DX11Renderer::DX11Renderer(ODevice1& device) {
    m_impl = new Implementation(device);
}

void DX11Renderer::render(ODeviceContext1& context) {
    m_impl->render(context);
}

} // NS Renderers
} // NS ImGui