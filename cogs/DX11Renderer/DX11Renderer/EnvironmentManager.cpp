// DirectX 11 environment manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Dx11Renderer/EnvironmentManager.h"
#include "Dx11Renderer/TextureManager.h"
#include "Dx11Renderer/Utils.h"

#include "Cogwheel/Scene/SceneRoot.h"

using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

namespace DX11Renderer {

EnvironmentManager::EnvironmentManager(ID3D11Device1& device, const std::wstring& shader_folder_path, TextureManager& textures)
    : m_textures(textures) {

    ID3D10Blob* vertex_shader_blob = compile_shader(shader_folder_path + L"EnvironmentMap.hlsl", "vs_5_0", "main_vs");
    HRESULT hr = device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), NULL, &m_vertex_shader);
    THROW_ON_FAILURE(hr);

    ID3D10Blob* pixel_shader_blob = compile_shader(shader_folder_path + L"EnvironmentMap.hlsl", "ps_5_0", "main_ps");
    hr = device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), NULL, &m_pixel_shader);
    THROW_ON_FAILURE(hr);
}

EnvironmentManager::~EnvironmentManager() {
    safe_release(&m_vertex_shader);
    safe_release(&m_pixel_shader);
}

bool EnvironmentManager::render(ID3D11DeviceContext1& render_context, int environment_ID) {

#if CHECK_IMPLICIT_STATE
    // Check that the screen space triangle will be rendered correctly.
    D3D11_PRIMITIVE_TOPOLOGY topology;
    render_context.IAGetPrimitiveTopology(&topology);
    always_assert(topology == D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Check that the environment can be rendered on top of the far plane.
    ID3D11DepthStencilState* depth_state;
    unsigned int unused;
    render_context.OMGetDepthStencilState(&depth_state, &unused);
    D3D11_DEPTH_STENCIL_DESC depth_desc;
    depth_state->GetDesc(&depth_desc);
    always_assert(depth_desc.DepthFunc == D3D11_COMPARISON_LESS_EQUAL || depth_desc.DepthFunc == D3D11_COMPARISON_NEVER);
#endif

    Environment& env = m_envs[environment_ID];
    if (env.map_ID != 0) {
        // Set vertex and pixel shaders.
        render_context.VSSetShader(m_vertex_shader, 0, 0);
        render_context.PSSetShader(m_pixel_shader, 0, 0);

        Dx11Texture envTexture = m_textures.get_texture(env.map_ID);
        render_context.PSSetShaderResources(0, 1, &envTexture.image->srv);
        render_context.PSSetSamplers(0, 1, &envTexture.sampler);

        render_context.Draw(3, 0);

        return true;
    } else {
        // Bind white environment instead.
        render_context.PSSetShaderResources(0, 1, &m_textures.white_texture().srv);
        render_context.PSSetSamplers(0, 1, &m_textures.white_texture().sampler);

        ID3D11RenderTargetView* backbuffer;
        ID3D11DepthStencilView* depth;
        render_context.OMGetRenderTargets(1, &backbuffer, &depth);

        render_context.ClearRenderTargetView(backbuffer, &env.tint.x);

        return false;
    }
}

void EnvironmentManager::handle_updates() {
    if (!SceneRoots::get_changed_scenes().is_empty()) {
        if (m_envs.size() < SceneRoots::capacity())
            m_envs.resize(SceneRoots::capacity());

        for (SceneRoot scene : SceneRoots::get_changed_scenes()) {
            m_envs[scene.get_ID()].map_ID = scene.get_environment_map();
            RGBA tint = scene.get_environment_tint();
            m_envs[scene.get_ID()].tint.x = tint.r;
            m_envs[scene.get_ID()].tint.y = tint.g;
            m_envs[scene.get_ID()].tint.z = tint.b;
            m_envs[scene.get_ID()].tint.w = tint.a;
        }
    }
}

} // NS DX11Renderer
