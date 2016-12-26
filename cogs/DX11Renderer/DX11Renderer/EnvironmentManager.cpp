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

#define NOMINMAX
#include <D3D11.h>
#undef RGB

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

    ID3D10Blob* vertex_shader_blob = compile_shader(shader_folder_path + L"PostProcess\\VertexShader.hlsl", "vs_5_0");
    HRESULT hr = device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), NULL, &m_vertex_shader);
    THROW_ON_FAILURE(hr);

    // Create the input layout
    D3D11_INPUT_ELEMENT_DESC input_layout_desc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };

    hr = device.CreateInputLayout(input_layout_desc, 1, UNPACK_BLOB_ARGS(vertex_shader_blob), &m_input_layout);
    THROW_ON_FAILURE(hr);

    { // Create a screen space vertex buffer.
        D3D11_BUFFER_DESC position_desc = {};
        position_desc.Usage = D3D11_USAGE_IMMUTABLE;
        position_desc.ByteWidth = sizeof(float2) * 3;
        position_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

        float2 positions[] = { {-1, -3}, {-1, 1}, {3, 1} };
        D3D11_SUBRESOURCE_DATA position_data = {};
        position_data.pSysMem = &positions;
        hr = device.CreateBuffer(&position_desc, &position_data, &m_position_buffer);
        THROW_ON_FAILURE(hr);
    }

    ID3D10Blob* pixel_shader_blob = compile_shader(shader_folder_path + L"PostProcess\\EnvironmentMap.hlsl", "ps_5_0");
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
        render_context.IASetInputLayout(m_input_layout);
        render_context.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); // TODO Assume that this is the case.

        render_context.PSSetShader(m_pixel_shader, 0, 0);
        Constants constants = { inverse_vp_matrix, camera_position, env.tint };
        render_context.UpdateSubresource(m_constant_buffer, 0, NULL, &constants, 0, 0);
        render_context.PSSetConstantBuffers(0, 1, &m_constant_buffer);

        Dx11Texture envTexture = m_textures->get_texture(env.map_ID);
        render_context.PSSetShaderResources(0, 1, &envTexture.image->srv);
        render_context.PSSetSamplers(0, 1, &envTexture.sampler);

        // TODO Hey! We can do this without a vertex buffer as well. Just build it directly into the shader and use the primitive ID.
        unsigned int stride = sizeof(float2);
        unsigned int offset = 0;
        render_context.IASetVertexBuffers(0, 1, &m_position_buffer, &stride, &offset);

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
