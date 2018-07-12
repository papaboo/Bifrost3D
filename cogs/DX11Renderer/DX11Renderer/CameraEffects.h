// DirectX 11 camera effects.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_CAMERA_EFFECTS_H_
#define _DX11RENDERER_RENDERER_CAMERA_EFFECTS_H_

#include <Cogwheel/Math/CameraEffects.h>

#include <DX11Renderer/Types.h>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// Gaussian Bloom.
// ------------------------------------------------------------------------------------------------
class GaussianBloom {
public:
    static const unsigned int group_size = 32u;

    GaussianBloom() = default;
    GaussianBloom(GaussianBloom&& other) = default;
    GaussianBloom(ID3D11Device1& device, const std::wstring& shader_folder_path);

    GaussianBloom& operator=(GaussianBloom&& rhs) = default;

    OShaderResourceView& filter(ID3D11DeviceContext1& context, ID3D11Buffer& constants, ID3D11SamplerState& bilinear_sampler,
                                ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height, int bandwidth);

private:
    GaussianBloom(GaussianBloom& other) = delete;
    GaussianBloom& operator=(GaussianBloom& rhs) = delete;

    struct IntermediateTexture {
        unsigned int width, height;
        OShaderResourceView SRV;
        OUnorderedAccessView UAV;
    };

    struct {
        float std_dev;
        int capacity;
        OBuffer buffer;
        OShaderResourceView SRV;
    } m_gaussian_samples;

    IntermediateTexture m_ping, m_pong;

    OComputeShader m_horizontal_filter;
    OComputeShader m_vertical_filter;
};

// ------------------------------------------------------------------------------------------------
// Dual Kawase Bloom.
// https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-26-50/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf
// ------------------------------------------------------------------------------------------------
class DualKawaseBloom {
public:
    static const unsigned int group_size = 32u;

    DualKawaseBloom() = default;
    DualKawaseBloom(DualKawaseBloom&& other) = default;
    DualKawaseBloom(ID3D11Device1& device, const std::wstring& shader_folder_path);

    DualKawaseBloom& operator=(DualKawaseBloom&& rhs) = default;

    OShaderResourceView& filter(ID3D11DeviceContext1& context, ID3D11Buffer& constants, ID3D11SamplerState& bilinear_sampler,
                                ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height, unsigned int half_passes);

private:
    DualKawaseBloom(DualKawaseBloom& other) = delete;
    DualKawaseBloom& operator=(DualKawaseBloom& rhs) = delete;

    struct {
        unsigned int width, height, mipmap_count;
        OShaderResourceView* SRVs;
        OUnorderedAccessView* UAVs;
    } m_temp;

    OComputeShader m_extract_high_intensity;
    OComputeShader m_downsample_pattern;
    OComputeShader m_upsample_pattern;
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
                 OComputeShader& second_reduction, ID3D11UnorderedAccessView* output_UAV);

    OComputeShader m_log_average_first_reduction;
    OComputeShader m_log_average_computation;
    OShaderResourceView m_log_averages_SRV;
    OUnorderedAccessView m_log_averages_UAV;

    OComputeShader m_linear_exposure_computation;
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

    OShaderResourceView& reduce_histogram(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                          ID3D11ShaderResourceView* pixels, unsigned int image_width);

    void compute_linear_exposure(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                 ID3D11ShaderResourceView* pixels, unsigned int image_width,
                                 ID3D11UnorderedAccessView* linear_exposure_UAV);

private:
    ExposureHistogram(ExposureHistogram& other) = delete;
    ExposureHistogram& operator=(ExposureHistogram& rhs) = delete;

    OComputeShader m_histogram_reduction;
    OShaderResourceView m_histogram_SRV;
    OUnorderedAccessView m_histogram_UAV;

    OComputeShader m_linear_exposure_computation;
};

// ------------------------------------------------------------------------------------------------
// Camera post process effects with support for exposure, bloom, various tonemappers and other effects.
// Sources:
// * https://mynameismjp.wordpress.com/2010/04/30/a-closer-look-at-tone-mapping/
// * http://perso.univ-lyon1.fr/jean-claude.iehl/Public/educ/GAMA/2007/gdc07/Post-Processing_Pipeline.pdf
// ------------------------------------------------------------------------------------------------
class CameraEffects {
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
        int bloom_bandwidth;

        float delta_time;
        float2 _padding;
    };

    CameraEffects() = default;
    CameraEffects(CameraEffects&& other) = default;
    CameraEffects(ID3D11Device1& device, const std::wstring& shader_folder_path);

    CameraEffects& operator=(CameraEffects&& rhs) = default;

    // Processes the pixels and stores them in the bound render target.
    void process(ID3D11DeviceContext1& context, Cogwheel::Math::CameraEffects::Settings settings, float delta_time,
                 ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV, int width, int height);

private:
    CameraEffects(CameraEffects& other) = delete;
    CameraEffects& operator=(CameraEffects& rhs) = delete;

    OBuffer m_constant_buffer;
    OSamplerState m_bilinear_sampler;

    OShaderResourceView m_linear_exposure_SRV;
    OUnorderedAccessView m_linear_exposure_UAV;
    LogAverageLuminance m_log_average_luminance;
    ExposureHistogram m_exposure_histogram;
    OComputeShader m_linear_exposure_from_bias_shader;

    GaussianBloom m_bloom;

    ORasterizerState m_raster_state;
    OVertexShader m_fullscreen_VS;
    OPixelShader m_linear_tonemapping_PS;
    OPixelShader m_uncharted2_tonemapping_PS;
    OPixelShader m_filmic_tonemapping_PS;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_CAMERA_EFFECTS_H_