// DirectX 11 tone mapper.
// ---------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/ToneMapper.h>
#include <DX11Renderer/Utils.h>

namespace DX11Renderer {

ToneMapper::ToneMapper()
    : m_vertex_shader(nullptr), m_pixel_shader(nullptr), m_sampler(nullptr) { }

ToneMapper::ToneMapper(ID3D11Device1& device, const std::wstring& shader_folder_path) {
    OID3DBlob vertex_shader_blob = compile_shader(shader_folder_path + L"ToneMapping.hlsl", "vs_5_0", "main_vs");
    HRESULT hr = device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_vertex_shader);
    THROW_ON_FAILURE(hr);

    OID3DBlob pixel_shader_blob = compile_shader(shader_folder_path + L"ToneMapping.hlsl", "ps_5_0", "main_ps");
    hr = device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_pixel_shader);
    THROW_ON_FAILURE(hr);

    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

    hr = device.CreateSamplerState(&sampler_desc, &m_sampler);
    THROW_ON_FAILURE(hr);
}

void ToneMapper::tonemap(ID3D11DeviceContext1& render_context, ID3D11ShaderResourceView* pixel_SRV) {
    render_context.VSSetShader(m_vertex_shader, 0, 0);
    render_context.PSSetShader(m_pixel_shader, 0, 0);

    render_context.PSSetShaderResources(0, 1, &pixel_SRV);
    render_context.PSSetSamplers(0, 1, &m_sampler);

    render_context.Draw(3, 0);
}

} // NS DX11Renderer
