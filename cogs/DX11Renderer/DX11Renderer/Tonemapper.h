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
// Functionality for computing the log average luminance of a list of pixels 
// and dynamically determining the exposure of those pixels.
// ------------------------------------------------------------------------------------------------
class LogAverageLuminance {
public:
    static const unsigned int max_groups_dispatched = 128u;
    static const unsigned int group_width = 8u;
    static const unsigned int group_height = 16u;
    static const unsigned int group_size = group_width * group_height;

    LogAverageLuminance();

    LogAverageLuminance(ID3D11Device1& device, const std::wstring& shader_folder_path);

    LogAverageLuminance& operator=(LogAverageLuminance&& rhs) {

        m_log_average_first_reduction = std::move(rhs.m_log_average_first_reduction);
        m_log_average_second_reduction = std::move(rhs.m_log_average_second_reduction);
        m_log_averages_SRV = std::move(rhs.m_log_averages_SRV);
        m_log_averages_UAV = std::move(rhs.m_log_averages_UAV);

        m_linear_exposure_computation = std::move(rhs.m_linear_exposure_computation);

        return *this;
    }

    void compute_log_average(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                             ID3D11ShaderResourceView* pixels, unsigned int image_width,
                             ID3D11UnorderedAccessView* log_average_UAV);

    void compute_linear_exposure(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                 ID3D11ShaderResourceView* pixels, unsigned int image_width,
                                 ID3D11UnorderedAccessView* linear_exposure_UAV);

private:
    LogAverageLuminance(LogAverageLuminance& other) = delete;
    LogAverageLuminance(LogAverageLuminance&& other) = delete;
    LogAverageLuminance& operator=(LogAverageLuminance& rhs) = delete;

    void compute(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                 ID3D11ShaderResourceView* pixels, unsigned int image_width, 
                 OID3D11ComputeShader& second_reduction, ID3D11UnorderedAccessView* output_UAV);

    OID3D11ComputeShader m_log_average_first_reduction;
    OID3D11ComputeShader m_log_average_second_reduction;
    OID3D11ShaderResourceView m_log_averages_SRV;
    OID3D11UnorderedAccessView m_log_averages_UAV;

    OID3D11ComputeShader m_linear_exposure_computation;
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

    ExposureHistogram();

    ExposureHistogram(ID3D11Device1& device, const std::wstring& shader_folder_path);

    ExposureHistogram& operator=(ExposureHistogram&& rhs) {
        m_histogram_reduction = std::move(rhs.m_histogram_reduction);
        m_histogram_SRV = std::move(rhs.m_histogram_SRV);
        m_histogram_UAV = std::move(rhs.m_histogram_UAV);

        m_linear_exposure_computation = std::move(rhs.m_linear_exposure_computation);

        return *this;
    }

    OID3D11ShaderResourceView& reduce_histogram(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                                ID3D11ShaderResourceView* pixels, unsigned int image_width);

    void compute_linear_exposure(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                 ID3D11ShaderResourceView* pixels, unsigned int image_width,
                                 ID3D11UnorderedAccessView* linear_exposure_UAV);

private:
    ExposureHistogram(ExposureHistogram& other) = delete;
    ExposureHistogram(ExposureHistogram&& other) = delete;
    ExposureHistogram& operator=(ExposureHistogram& rhs) = delete;

    OID3D11ComputeShader m_histogram_reduction;
    OID3D11ShaderResourceView m_histogram_SRV;
    OID3D11UnorderedAccessView m_histogram_UAV;

    OID3D11ComputeShader m_linear_exposure_computation;
};

// ------------------------------------------------------------------------------------------------
// Tonemapping implementation with support for various tonemappers.
// Sources:
// * https://mynameismjp.wordpress.com/2010/04/30/a-closer-look-at-tone-mapping/
// * http://perso.univ-lyon1.fr/jean-claude.iehl/Public/educ/GAMA/2007/gdc07/Post-Processing_Pipeline.pdf
// ------------------------------------------------------------------------------------------------
class Tonemapper {
public:

    struct Constants {
        float min_log_luminance;
        float max_log_luminance;
        float min_histogram_percentage;
        float max_histogram_percentage;
        float log_lumiance_bias;
        float eye_adaptation_brightness;
        float eye_adaptation_darkness;
        float delta_time;
    };

    Tonemapper();
    Tonemapper(ID3D11Device1& device, const std::wstring& shader_folder_path);

    Tonemapper& operator=(Tonemapper&& rhs) {

        m_constant_buffer = std::move(rhs.m_constant_buffer);

        m_linear_exposure_SRV = std::move(rhs.m_linear_exposure_SRV);
        m_linear_exposure_UAV = std::move(rhs.m_linear_exposure_UAV);
        m_log_average_luminance = std::move(rhs.m_log_average_luminance);
        m_exposure_histogram = std::move(rhs.m_exposure_histogram);
        m_linear_exposure_from_bias_shader = std::move(rhs.m_linear_exposure_from_bias_shader);

        m_fullscreen_VS = std::move(rhs.m_fullscreen_VS);
        m_linear_tonemapping_PS = std::move(rhs.m_linear_tonemapping_PS);
        m_uncharted2_tonemapping_PS = std::move(rhs.m_uncharted2_tonemapping_PS);
        m_filmic_tonemapping_PS = std::move(rhs.m_filmic_tonemapping_PS);

        return *this;
    }

    // Tonemaps the pixels and stores them in the bound render target.
    void tonemap(ID3D11DeviceContext1& context, Cogwheel::Math::Tonemapping::Parameters parameters, float delta_time,
                 ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV, int width);

private:
    Tonemapper(Tonemapper& other) = delete;
    Tonemapper(Tonemapper&& other) = delete;
    Tonemapper& operator=(Tonemapper& rhs) = delete;

    OID3D11Buffer m_constant_buffer;

    OID3D11ShaderResourceView m_linear_exposure_SRV;
    OID3D11UnorderedAccessView m_linear_exposure_UAV;
    LogAverageLuminance m_log_average_luminance;
    ExposureHistogram m_exposure_histogram;
    OID3D11ComputeShader m_linear_exposure_from_bias_shader;

    OID3D11VertexShader m_fullscreen_VS;
    OID3D11PixelShader m_linear_tonemapping_PS;
    OID3D11PixelShader m_uncharted2_tonemapping_PS;
    OID3D11PixelShader m_filmic_tonemapping_PS;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_TONEMAPPER_H_