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
// Dual Kawase Bloom.
// https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-26-50/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf
// ------------------------------------------------------------------------------------------------
class DualKawaseBloom {
public:
    DualKawaseBloom() = default;
    DualKawaseBloom(DualKawaseBloom&& other) = default;
    DualKawaseBloom(ID3D11Device1& device, const std::wstring& shader_folder_path);

    DualKawaseBloom& operator=(DualKawaseBloom&& rhs) = default;

    OID3D11ShaderResourceView& filter(ID3D11DeviceContext1& context, ID3D11Buffer& constant_buffer, ID3D11SamplerState& bilinear_sampler, 
        ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height, unsigned int half_passes);

private:
    DualKawaseBloom(DualKawaseBloom& other) = delete;
    DualKawaseBloom& operator=(DualKawaseBloom& rhs) = delete;

    struct {
        unsigned int width, height, mipmap_count;
        OID3D11ShaderResourceView* SRVs;
        OID3D11UnorderedAccessView* UAVs;
    } m_temp;

    OID3D11ComputeShader m_extract_high_intensity;
    OID3D11ComputeShader m_downsample_pattern;
    OID3D11ComputeShader m_upsample_pattern;
};

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

    LogAverageLuminance() = default;
    LogAverageLuminance(LogAverageLuminance&& other) = default;
    LogAverageLuminance(ID3D11Device1& device, const std::wstring& shader_folder_path);

    LogAverageLuminance& operator=(LogAverageLuminance&& rhs) = default;

    void compute_log_average(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                             ID3D11ShaderResourceView* pixels, unsigned int image_width,
                             ID3D11UnorderedAccessView* log_average_UAV);

    void compute_linear_exposure(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                 ID3D11ShaderResourceView* pixels, unsigned int image_width,
                                 ID3D11UnorderedAccessView* linear_exposure_UAV);

private:
    LogAverageLuminance(LogAverageLuminance& other) = delete;
    LogAverageLuminance& operator=(LogAverageLuminance& rhs) = delete;

    void compute(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                 ID3D11ShaderResourceView* pixels, unsigned int image_width, 
                 OID3D11ComputeShader& second_reduction, ID3D11UnorderedAccessView* output_UAV);

    OID3D11ComputeShader m_log_average_first_reduction;
    OID3D11ComputeShader m_log_average_computation;
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

    ExposureHistogram() = default;
    ExposureHistogram(ExposureHistogram&& other) = default;
    ExposureHistogram(ID3D11Device1& device, const std::wstring& shader_folder_path);

    ExposureHistogram& operator=(ExposureHistogram&& rhs) = default;

    OID3D11ShaderResourceView& reduce_histogram(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                                ID3D11ShaderResourceView* pixels, unsigned int image_width);

    void compute_linear_exposure(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                 ID3D11ShaderResourceView* pixels, unsigned int image_width,
                                 ID3D11UnorderedAccessView* linear_exposure_UAV);

private:
    ExposureHistogram(ExposureHistogram& other) = delete;
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
        float bloom_threshold;
        float delta_time;
        float3 _padding;
    };

    Tonemapper() = default;
    Tonemapper(Tonemapper&& other) = default;
    Tonemapper(ID3D11Device1& device, const std::wstring& shader_folder_path);

    Tonemapper& operator=(Tonemapper&& rhs) = default;

    // Tonemaps the pixels and stores them in the bound render target.
    void tonemap(ID3D11DeviceContext1& context, Cogwheel::Math::Tonemapping::Parameters parameters, float delta_time,
                 ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV, int width, int height);

private:
    Tonemapper(Tonemapper& other) = delete;
    Tonemapper& operator=(Tonemapper& rhs) = delete;

    OID3D11Buffer m_constant_buffer;
    OID3D11SamplerState m_bilinear_sampler;

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