// DirectX 11 tonemapper.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_TONEMAPPER_H_
#define _DX11RENDERER_RENDERER_TONEMAPPER_H_

#include <Cogwheel/Math/Tonemapping.h>

#include <DX11Renderer/Types.h>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// Tonemapping implementation with support for various tonemappers.
// Sources:
// * https://mynameismjp.wordpress.com/2010/04/30/a-closer-look-at-tone-mapping/
// * http://perso.univ-lyon1.fr/jean-claude.iehl/Public/educ/GAMA/2007/gdc07/Post-Processing_Pipeline.pdf
// ------------------------------------------------------------------------------------------------
class Tonemapper {
public:

    Tonemapper();
    Tonemapper(ID3D11Device1& device, const std::wstring& shader_folder_path);

    Tonemapper& operator=(Tonemapper&& rhs) {
        m_fullscreen_VS = std::move(rhs.m_fullscreen_VS);
        m_log_luminance_PS = std::move(rhs.m_log_luminance_PS);
        m_linear_tonemapping_PS = std::move(rhs.m_linear_tonemapping_PS);
        m_reinhard_tonemapping_PS = std::move(rhs.m_reinhard_tonemapping_PS);
        m_uncharted2_tonemapping_PS = std::move(rhs.m_uncharted2_tonemapping_PS);
        m_filmic_tonemapping_PS = std::move(rhs.m_filmic_tonemapping_PS);

        m_log_luminance_RTV = std::move(rhs.m_log_luminance_RTV);
        m_log_luminance_SRV = std::move(rhs.m_log_luminance_SRV);
        m_log_luminance_sampler = std::move(rhs.m_log_luminance_sampler);
        return *this;
    }

    // Tonemaps the pixels and stores them in the bound render target.
    void tonemap(ID3D11DeviceContext1& context, Cogwheel::Math::Tonemapping::Parameters parameters,
                 ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV,
                 int width, int height);

private:
    Tonemapper(Tonemapper& other) = delete;
    Tonemapper(Tonemapper&& other) = delete;
    Tonemapper& operator=(Tonemapper& rhs) = delete;

    OID3D11VertexShader m_fullscreen_VS;
    OID3D11PixelShader m_log_luminance_PS;
    OID3D11PixelShader m_linear_tonemapping_PS;
    OID3D11PixelShader m_reinhard_tonemapping_PS;
    OID3D11PixelShader m_uncharted2_tonemapping_PS;
    OID3D11PixelShader m_filmic_tonemapping_PS;

    int m_width, m_height;
    OID3D11RenderTargetView m_log_luminance_RTV;
    OID3D11ShaderResourceView m_log_luminance_SRV;
    OID3D11SamplerState m_log_luminance_sampler;
};

// ------------------------------------------------------------------------------------------------
// Functionality for computing a histogram from a list of pixels 
// and dynamically determining the exposure of those pixels.
// ------------------------------------------------------------------------------------------------
class ExposureHistogram {
public:
    static const unsigned int bin_count = 64;
    static const unsigned int group_width = 16u;
    static const unsigned int group_height = 8u;
    static const unsigned int group_size = group_width * group_height;

    ExposureHistogram(ID3D11Device1& device, const std::wstring& shader_folder_path);

    ExposureHistogram& operator=(ExposureHistogram&& rhs) {
        m_histogram_reduction = std::move(rhs.m_histogram_reduction);
        m_histogram_SRV = std::move(rhs.m_histogram_SRV);
        m_histogram_UAV = std::move(rhs.m_histogram_UAV);
        return *this;
    }
    OID3D11ShaderResourceView& reduce_histogram(ID3D11DeviceContext1& context, OID3D11Buffer& constants, 
                                                OID3D11ShaderResourceView& pixels, unsigned int image_width);

    // TODO Compute exposure

private:
    ExposureHistogram(ExposureHistogram& other) = delete;
    ExposureHistogram(ExposureHistogram&& other) = delete;
    ExposureHistogram& operator=(ExposureHistogram& rhs) = delete;

    OID3D11ComputeShader m_histogram_reduction;
    OID3D11ShaderResourceView m_histogram_SRV;
    OID3D11UnorderedAccessView m_histogram_UAV;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_TONEMAPPER_H_