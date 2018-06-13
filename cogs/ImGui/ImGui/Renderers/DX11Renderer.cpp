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
            io.Fonts->TexID = (void *)m_font_SRV; // TODO Needed for anything? Perhaps with multiple fonts?
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

            D3D11_MAPPED_SUBRESOURCE vtx_resource, idx_resource;
            THROW_ON_FAILURE(context->Map(m_vertex_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &vtx_resource));
            THROW_ON_FAILURE(context->Map(m_index_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &idx_resource));
            ImDrawVert* vtx_dst = (ImDrawVert*)vtx_resource.pData;
            ImDrawIdx* idx_dst = (ImDrawIdx*)idx_resource.pData;
            for (int n = 0; n < draw_data->CmdListsCount; ++n) {
                const ImDrawList* cmd_list = draw_data->CmdLists[n];
                memcpy(vtx_dst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
                memcpy(idx_dst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
                vtx_dst += cmd_list->VtxBuffer.Size;
                idx_dst += cmd_list->IdxBuffer.Size;
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
            struct BACKUP_DX11_STATE {
                UINT                        ScissorRectsCount, ViewportsCount;
                D3D11_RECT                  ScissorRects[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
                D3D11_VIEWPORT              Viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
                ID3D11RasterizerState*      RS;
                ID3D11BlendState*           BlendState;
                FLOAT                       BlendFactor[4];
                UINT                        SampleMask;
                UINT                        StencilRef;
                ID3D11DepthStencilState*    DepthStencilState;
                ID3D11ShaderResourceView*   PSShaderResource;
                ID3D11SamplerState*         PSSampler;
                ID3D11PixelShader*          PS;
                ID3D11VertexShader*         VS;
                UINT                        PSInstancesCount, VSInstancesCount;
                ID3D11ClassInstance*        PSInstances[256], *VSInstances[256];   // 256 is max according to PSSetShader documentation
                D3D11_PRIMITIVE_TOPOLOGY    PrimitiveTopology;
                ID3D11Buffer*               IndexBuffer, *VertexBuffer, *VSConstantBuffer;
                UINT                        IndexBufferOffset, VertexBufferStride, VertexBufferOffset;
                DXGI_FORMAT                 IndexBufferFormat;
                ID3D11InputLayout*          InputLayout;
            };
            BACKUP_DX11_STATE old;
            old.ScissorRectsCount = old.ViewportsCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
            context->RSGetScissorRects(&old.ScissorRectsCount, old.ScissorRects);
            context->RSGetViewports(&old.ViewportsCount, old.Viewports);
            context->RSGetState(&old.RS);
            context->OMGetBlendState(&old.BlendState, old.BlendFactor, &old.SampleMask);
            context->OMGetDepthStencilState(&old.DepthStencilState, &old.StencilRef);
            context->PSGetShaderResources(0, 1, &old.PSShaderResource);
            context->PSGetSamplers(0, 1, &old.PSSampler);
            old.PSInstancesCount = old.VSInstancesCount = 256;
            context->PSGetShader(&old.PS, old.PSInstances, &old.PSInstancesCount);
            context->VSGetShader(&old.VS, old.VSInstances, &old.VSInstancesCount);
            context->VSGetConstantBuffers(0, 1, &old.VSConstantBuffer);
            context->IAGetPrimitiveTopology(&old.PrimitiveTopology);
            context->IAGetIndexBuffer(&old.IndexBuffer, &old.IndexBufferFormat, &old.IndexBufferOffset);
            context->IAGetVertexBuffers(0, 1, &old.VertexBuffer, &old.VertexBufferStride, &old.VertexBufferOffset);
            context->IAGetInputLayout(&old.InputLayout);

            // Setup viewport
            D3D11_VIEWPORT vp;
            memset(&vp, 0, sizeof(D3D11_VIEWPORT));
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
            context->VSSetShader(m_vertex_shader, NULL, 0);
            context->VSSetConstantBuffers(0, 1, &m_projection_matrix);
            context->PSSetShader(m_pixel_shader, NULL, 0);

            // Setup render state
            const float blend_factor[4] = { 0.f, 0.f, 0.f, 0.f };
            context->OMSetBlendState(m_blend_state, blend_factor, 0xffffffff);
            context->OMSetDepthStencilState(m_depth_stencil_state, 0);
            context->RSSetState(m_rasterizer_state);

            // Render command lists
            int vtx_offset = 0;
            int idx_offset = 0;
            for (int n = 0; n < draw_data->CmdListsCount; ++n) {
                const ImDrawList* cmd_list = draw_data->CmdLists[n];
                for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; ++cmd_i) {
                    const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
                    if (pcmd->UserCallback) {
                        pcmd->UserCallback(cmd_list, pcmd);
                    } else {
                        const D3D11_RECT r = { (LONG)pcmd->ClipRect.x, (LONG)pcmd->ClipRect.y, (LONG)pcmd->ClipRect.z, (LONG)pcmd->ClipRect.w };
                        context->PSSetShaderResources(0, 1, (ID3D11ShaderResourceView**)&pcmd->TextureId);
                        context->RSSetScissorRects(1, &r);
                        context->DrawIndexed(pcmd->ElemCount, idx_offset, vtx_offset);
                    }
                    idx_offset += pcmd->ElemCount;
                }
                vtx_offset += cmd_list->VtxBuffer.Size;
            }

            // Restore modified DX state
            context->RSSetScissorRects(old.ScissorRectsCount, old.ScissorRects);
            context->RSSetViewports(old.ViewportsCount, old.Viewports);
            context->RSSetState(old.RS); if (old.RS) old.RS->Release();
            context->OMSetBlendState(old.BlendState, old.BlendFactor, old.SampleMask); if (old.BlendState) old.BlendState->Release();
            context->OMSetDepthStencilState(old.DepthStencilState, old.StencilRef); if (old.DepthStencilState) old.DepthStencilState->Release();
            context->PSSetShaderResources(0, 1, &old.PSShaderResource); if (old.PSShaderResource) old.PSShaderResource->Release();
            context->PSSetSamplers(0, 1, &old.PSSampler); if (old.PSSampler) old.PSSampler->Release();
            context->PSSetShader(old.PS, old.PSInstances, old.PSInstancesCount); if (old.PS) old.PS->Release();
            for (UINT i = 0; i < old.PSInstancesCount; i++) if (old.PSInstances[i]) old.PSInstances[i]->Release();
            context->VSSetShader(old.VS, old.VSInstances, old.VSInstancesCount); if (old.VS) old.VS->Release();
            context->VSSetConstantBuffers(0, 1, &old.VSConstantBuffer); if (old.VSConstantBuffer) old.VSConstantBuffer->Release();
            for (UINT i = 0; i < old.VSInstancesCount; i++) if (old.VSInstances[i]) old.VSInstances[i]->Release();
            context->IASetPrimitiveTopology(old.PrimitiveTopology);
            context->IASetIndexBuffer(old.IndexBuffer, old.IndexBufferFormat, old.IndexBufferOffset); if (old.IndexBuffer) old.IndexBuffer->Release();
            context->IASetVertexBuffers(0, 1, &old.VertexBuffer, &old.VertexBufferStride, &old.VertexBufferOffset); if (old.VertexBuffer) old.VertexBuffer->Release();
            context->IASetInputLayout(old.InputLayout); if (old.InputLayout) old.InputLayout->Release();
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