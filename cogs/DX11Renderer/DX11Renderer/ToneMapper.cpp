// DirectX 11 tone mapper.
// ---------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/ToneMapper.h>
#include <DX11Renderer/Utils.h>

using namespace Cogwheel::Math;

namespace DX11Renderer {

ToneMapper::ToneMapper()
    : m_fullscreen_VS(nullptr), m_log_luminance_PS(nullptr)
    , m_linear_tonemapping_PS(nullptr), m_simple_tonemapping_PS(nullptr), m_reinhard_tonemapping_PS(nullptr), m_filmic_tonemapping_PS(nullptr)
    , m_width(0), m_height(0), m_log_luminance_RTV(nullptr), m_log_luminance_SRV(nullptr), m_log_luminance_sampler(nullptr){ }

ToneMapper::ToneMapper(ID3D11Device1& device, const std::wstring& shader_folder_path)
    : m_width(0), m_height(0), m_log_luminance_RTV(nullptr), m_log_luminance_SRV(nullptr) {

    { // Setup shaders
        const std::wstring shader_filename = shader_folder_path + L"ToneMapping.hlsl";

        OID3DBlob vertex_shader_blob = compile_shader(shader_filename, "vs_5_0", "fullscreen_vs");
        HRESULT hr = device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_fullscreen_VS);
        THROW_ON_FAILURE(hr);

        auto create_pixel_shader = [&](const char* entry_point) -> OID3D11PixelShader {
            OID3D11PixelShader pixel_shader;
            OID3DBlob pixel_shader_blob = compile_shader(shader_filename, "ps_5_0", entry_point);
            HRESULT hr = device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &pixel_shader);
            THROW_ON_FAILURE(hr);
            return pixel_shader;
        };

        m_log_luminance_PS = create_pixel_shader("log_luminance_ps");
        m_linear_tonemapping_PS = create_pixel_shader("linear_tonemapping_ps");
        m_simple_tonemapping_PS = create_pixel_shader("simple_tonemapping_ps");
        m_reinhard_tonemapping_PS = create_pixel_shader("reinhard_tonemapping_ps");
        m_filmic_tonemapping_PS = create_pixel_shader("filmic_tonemapping_ps");
    }

    { // Setup interpolation sampler for log average image.
        D3D11_SAMPLER_DESC sampler_desc = {};
        sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sampler_desc.MinLOD = 0;
        sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

        HRESULT hr = device.CreateSamplerState(&sampler_desc, &m_log_luminance_sampler);
        THROW_ON_FAILURE(hr);
    }
}

void ToneMapper::tonemap(ID3D11DeviceContext1& context, ToneMapping::Parameters parameters, 
                         ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV, 
                         int width, int height) {

    // Setup general state, such as vertex shader and sampler.
    context.VSSetShader(m_fullscreen_VS, 0, 0);

    bool single_pass = parameters.mapping == ToneMapping::Operator::Linear || parameters.mapping == ToneMapping::Operator::Simple;
    if (single_pass) {
        context.OMSetRenderTargets(1, &backbuffer_RTV, nullptr);
        if (parameters.mapping == ToneMapping::Operator::Linear)
            context.PSSetShader(m_linear_tonemapping_PS, 0, 0);
        else // parameters.mapping == ToneMapping::Operator::Simple
            context.PSSetShader(m_simple_tonemapping_PS, 0, 0);

        ID3D11ShaderResourceView* srvs[2] = { pixel_SRV, m_log_luminance_SRV };
        context.PSSetShaderResources(0, 2, srvs);
        context.PSSetSamplers(1, 1, &m_log_luminance_sampler);

        context.Draw(3, 0);
    
    } else {

        if (m_width != width || m_height != height) {
            // Setup the log luminance backbuffer.
            if (m_log_luminance_RTV) m_log_luminance_RTV->Release();
            if (m_log_luminance_SRV) m_log_luminance_SRV->Release();

            D3D11_TEXTURE2D_DESC buffer_desc;
            buffer_desc.Width = width;
            buffer_desc.Height = height;
            buffer_desc.MipLevels = 0; // 0 Because we want DX11 to generate mipmaps for us.
            buffer_desc.ArraySize = 1;
            buffer_desc.Format = DXGI_FORMAT_R16_FLOAT;
            buffer_desc.SampleDesc.Count = 1;
            buffer_desc.SampleDesc.Quality = 0;
            buffer_desc.Usage = D3D11_USAGE_DEFAULT;
            buffer_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            buffer_desc.CPUAccessFlags = 0;
            buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

            using OID3D11Device = DX11Renderer::OwnedResourcePtr<ID3D11Device>;
            OID3D11Device device;
            context.GetDevice(&device);

            OID3D11Texture2D backbuffer;
            HRESULT hr = device->CreateTexture2D(&buffer_desc, nullptr, &backbuffer);
            THROW_ON_FAILURE(hr);
            hr = device->CreateRenderTargetView(backbuffer, nullptr, &m_log_luminance_RTV);
            THROW_ON_FAILURE(hr);
            hr = device->CreateShaderResourceView(backbuffer, nullptr, &m_log_luminance_SRV);
            THROW_ON_FAILURE(hr);

            m_width = width;
            m_height = height;
        }

        { // Create log average texture.
            context.OMSetRenderTargets(1, &m_log_luminance_RTV, nullptr);
            context.PSSetShader(m_log_luminance_PS, 0, 0);
            context.PSSetShaderResources(0, 1, &pixel_SRV);
            context.Draw(3, 0);

            context.OMSetRenderTargets(1, &backbuffer_RTV, nullptr); // TODO Can I clear the rendertarget? Otherwise keep as is and remove the same setter below.
            context.GenerateMips(m_log_luminance_SRV);
        }

        { // Tonemap and render into backbuffer.
            context.OMSetRenderTargets(1, &backbuffer_RTV, nullptr);
            if (parameters.mapping == ToneMapping::Operator::Reinhard)
                context.PSSetShader(m_reinhard_tonemapping_PS, 0, 0);
            else // parameters.mapping == ToneMapping::Operator::Filmic
                context.PSSetShader(m_filmic_tonemapping_PS, 0, 0);

            ID3D11ShaderResourceView* srvs[2] = { pixel_SRV, m_log_luminance_SRV };
            context.PSSetShaderResources(0, 2, srvs);
            context.PSSetSamplers(1, 1, &m_log_luminance_sampler);

            context.Draw(3, 0);
        }
    }
}

} // NS DX11Renderer
