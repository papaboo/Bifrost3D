// DirectX 11 tone mapper.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_TONE_MAPPER_H_
#define _DX11RENDERER_RENDERER_TONE_MAPPER_H_

#include <Cogwheel/Math/ToneMapping.h>

#include <DX11Renderer/Types.h>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// Tone mapping implementation with support for various tonemappers.
// Sources:
// * https://mynameismjp.wordpress.com/2010/04/30/a-closer-look-at-tone-mapping/
// * http://perso.univ-lyon1.fr/jean-claude.iehl/Public/educ/GAMA/2007/gdc07/Post-Processing_Pipeline.pdf
// ------------------------------------------------------------------------------------------------
class ToneMapper {
public:

    ToneMapper();
    ToneMapper(ID3D11Device1& device, const std::wstring& shader_folder_path);

    ToneMapper& operator=(ToneMapper&& rhs) {
        m_fullscreen_VS = std::move(rhs.m_fullscreen_VS);
        m_log_luminance_PS = std::move(rhs.m_log_luminance_PS);
        m_linear_tonemapping_PS = std::move(rhs.m_linear_tonemapping_PS);
        m_reinhard_tonemapping_PS = std::move(rhs.m_reinhard_tonemapping_PS);
        m_filmic_tonemapping_PS = std::move(rhs.m_filmic_tonemapping_PS);

        m_log_luminance_RTV = std::move(rhs.m_log_luminance_RTV);
        m_log_luminance_SRV = std::move(rhs.m_log_luminance_SRV);
        m_log_luminance_sampler = std::move(rhs.m_log_luminance_sampler);
        return *this;
    }

    // Tonemaps the pixels and stores them in the bound render target.
    void tonemap(ID3D11DeviceContext1& context, Cogwheel::Math::ToneMapping::Parameters parameters,
                 ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV,
                 int width, int height);

private:
    ToneMapper(ToneMapper& other) = delete;
    ToneMapper(ToneMapper&& other) = delete;
    ToneMapper& operator=(ToneMapper& rhs) = delete;

    OID3D11VertexShader m_fullscreen_VS;
    OID3D11PixelShader m_log_luminance_PS;
    OID3D11PixelShader m_linear_tonemapping_PS;
    OID3D11PixelShader m_reinhard_tonemapping_PS;
    OID3D11PixelShader m_filmic_tonemapping_PS;

    int m_width, m_height;
    OID3D11RenderTargetView m_log_luminance_RTV;
    OID3D11ShaderResourceView m_log_luminance_SRV;
    OID3D11SamplerState m_log_luminance_sampler;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_TONE_MAPPER_H_