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

struct Constants {
    Matrix4x4f inverse_vp_matrix;
    float4 camera_position;
    float4 tint;
};

EnvironmentManager::EnvironmentManager(ID3D11Device& device, const std::wstring& shader_folder_path, TextureManager* textures)
    : m_textures(textures) {

    ID3D10Blob* vertex_shader_blob = compile_shader(shader_folder_path + L"EnvironmentMap.hlsl", "vs_5_0", "main_vs");
    HRESULT hr = device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), NULL, &m_vertex_shader);
    THROW_ON_FAILURE(hr);

    ID3D10Blob* pixel_shader_blob = compile_shader(shader_folder_path + L"EnvironmentMap.hlsl", "ps_5_0", "main_ps");
    hr = device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), NULL, &m_pixel_shader);
    THROW_ON_FAILURE(hr);

    hr = create_constant_buffer(device, sizeof(Constants), &m_constant_buffer);
    THROW_ON_FAILURE(hr);
}

bool EnvironmentManager::render(ID3D11DeviceContext& render_context, Matrix4x4f inverse_vp_matrix, float4 camera_position, int environment_ID) {

    Environment& env = m_envs[environment_ID];
    if (env.map_ID != 0) {
        // Set vertex and pixel shaders.
        render_context.VSSetShader(m_vertex_shader, 0, 0);
        render_context.IASetInputLayout(nullptr); // TODO Needed? Not on NVIDIA, but in general?
        render_context.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); // TODO Assume that this is the case.

        render_context.PSSetShader(m_pixel_shader, 0, 0);
        Constants constants = { inverse_vp_matrix, camera_position, env.tint };
        render_context.UpdateSubresource(m_constant_buffer, 0, NULL, &constants, 0, 0);
        render_context.PSSetConstantBuffers(0, 1, &m_constant_buffer);

        Dx11Texture envTexture = m_textures->get_texture(env.map_ID);
        render_context.PSSetShaderResources(0, 1, &envTexture.image->srv);
        render_context.PSSetSamplers(0, 1, &envTexture.sampler);

        render_context.Draw(3, 0);

        return true;
    }
    return false;
}

void EnvironmentManager::handle_updates(ID3D11Device& device, ID3D11DeviceContext& device_context) {
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
