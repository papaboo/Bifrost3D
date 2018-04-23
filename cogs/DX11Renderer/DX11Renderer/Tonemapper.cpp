// DirectX 11 tonemapper.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <DX11Renderer/Tonemapper.h>
#include <DX11Renderer/Utils.h>

using namespace Cogwheel::Math;

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// Dual Kawase Bloom.
// ------------------------------------------------------------------------------------------------
DualKawaseBloom::DualKawaseBloom(ID3D11Device1& device, const std::wstring& shader_folder_path) {

    m_temp = {};

    const std::wstring shader_filename = shader_folder_path + L"ColorGrading/Bloom.hlsl";

    OID3DBlob m_extract_high_intensity_blob = compile_shader(shader_filename, "cs_5_0", "ColorGrading::extract_high_intensity");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(m_extract_high_intensity_blob), nullptr, &m_extract_high_intensity));

    OID3DBlob downsample_pattern_blob = compile_shader(shader_filename, "cs_5_0", "ColorGrading::dual_kawase_downsample");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(downsample_pattern_blob), nullptr, &m_downsample_pattern));

    OID3DBlob upsample_pattern_blob = compile_shader(shader_filename, "cs_5_0", "ColorGrading::dual_kawase_upsample");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(upsample_pattern_blob), nullptr, &m_upsample_pattern));
}

OID3D11ShaderResourceView& DualKawaseBloom::filter(ID3D11DeviceContext1& context, ID3D11Buffer& constant_buffer, ID3D11SamplerState& bilinear_sampler, 
                                                   ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height, unsigned int half_passes) {
    if (m_temp.width != image_width || m_temp.height != image_height) {
        
        // Release old resources
        for (unsigned int m = 0; m < m_temp.mipmap_count; ++m) {
            m_temp.SRVs[m].release();
            m_temp.UAVs[m].release();
        }
        delete[] m_temp.SRVs;
        delete[] m_temp.UAVs;

        // Grab the device.
        ID3D11Device* basic_device;
        context.GetDevice(&basic_device);
        OID3D11Device1 device;
        THROW_ON_FAILURE(basic_device->QueryInterface(IID_PPV_ARGS(&device)));
        basic_device->Release();

        // Allocate new temporaries
        m_temp.mipmap_count = 1;
        while ((image_width >> m_temp.mipmap_count) > 0 || (image_height >> m_temp.mipmap_count) > 0)
            ++m_temp.mipmap_count;

        D3D11_TEXTURE2D_DESC tex_desc = {};
        tex_desc.Width = image_width;
        tex_desc.Height = image_height;
        tex_desc.MipLevels = m_temp.mipmap_count;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.SampleDesc.Quality = 0;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

        OID3D11Texture2D texture2D;
        HRESULT hr = device->CreateTexture2D(&tex_desc, nullptr, &texture2D);
        THROW_ON_FAILURE(hr);

        // TODO Create views in two device calls.
        m_temp.SRVs = new OID3D11ShaderResourceView[m_temp.mipmap_count];
        m_temp.UAVs = new OID3D11UnorderedAccessView[m_temp.mipmap_count];

        for (unsigned int m = 0; m < m_temp.mipmap_count; ++m) {
            // SRV
            D3D11_SHADER_RESOURCE_VIEW_DESC mip_level_SRV_desc;
            mip_level_SRV_desc.Format = tex_desc.Format;
            mip_level_SRV_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            mip_level_SRV_desc.Texture2D.MipLevels = 1;
            mip_level_SRV_desc.Texture2D.MostDetailedMip = m;
            THROW_ON_FAILURE(device->CreateShaderResourceView(texture2D, &mip_level_SRV_desc, &m_temp.SRVs[m]));

            // UAV
            D3D11_UNORDERED_ACCESS_VIEW_DESC mip_level_UAV_desc = {};
            mip_level_UAV_desc.Format = tex_desc.Format;
            mip_level_UAV_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            mip_level_UAV_desc.Texture2D.MipSlice = m;
            THROW_ON_FAILURE(device->CreateUnorderedAccessView(texture2D, &mip_level_UAV_desc, &m_temp.UAVs[m]));
        }

        m_temp.width = image_width;
        m_temp.height = image_height;
    }

    // Copy high intensity part of image. TODO Upload constant
    context.CSSetShader(m_extract_high_intensity, nullptr, 0u);
    context.CSSetShaderResources(0, 1, &pixels);
    context.CSSetUnorderedAccessViews(0, 1, &m_temp.UAVs[0], 0u);
    context.Dispatch(ceil_divide(image_width, group_size), ceil_divide(image_height, group_size), 1);

    // Downsample
    half_passes = min(half_passes, m_temp.mipmap_count-1);
    context.CSSetShader(m_downsample_pattern, nullptr, 0u);
    for (unsigned int p = 0; p < half_passes; ++p) {
        auto* uav = &m_temp.UAVs[p + 1];
        context.CSSetUnorderedAccessViews(0, 1, uav, 0u);
        auto* srv = &m_temp.SRVs[p];
        context.CSSetShaderResources(0, 1, srv);
        unsigned int thread_count_x = image_width >> (p + 1), thread_count_y = image_height >> (p + 1);
        context.Dispatch(ceil_divide(thread_count_x, group_size), ceil_divide(thread_count_y, group_size), 1);
    }

    // Upsample
    context.CSSetShader(m_upsample_pattern, nullptr, 0u);
    for (unsigned int p = half_passes; p > 0; --p) {
        context.CSSetUnorderedAccessViews(0, 1, &m_temp.UAVs[p - 1], 0u);
        context.CSSetShaderResources(0, 1, &m_temp.SRVs[p]);
        unsigned int thread_count_x = image_width >> (p - 1), thread_count_y = image_height >> (p - 1);
        context.Dispatch(ceil_divide(thread_count_x, group_size), ceil_divide(thread_count_y, group_size), 1);
    }

    ID3D11UnorderedAccessView* null_UAV = nullptr;
    context.CSSetUnorderedAccessViews(0, 1, &null_UAV, 0u);

    return m_temp.SRVs[0];
}

// ------------------------------------------------------------------------------------------------
// Log average luminance reduction.
// ------------------------------------------------------------------------------------------------
LogAverageLuminance::LogAverageLuminance(ID3D11Device1& device, const std::wstring& shader_folder_path) {
    const std::wstring shader_filename = shader_folder_path + L"ColorGrading/ReduceLogAverageLuminance.hlsl";

    // Create shaders.
    OID3DBlob log_average_first_reduction_blob = compile_shader(shader_filename, "cs_5_0", "ColorGrading::first_reduction");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(log_average_first_reduction_blob), nullptr, &m_log_average_first_reduction));

    OID3DBlob log_average_computation_blob = compile_shader(shader_filename, "cs_5_0", "ColorGrading::compute_log_average");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(log_average_computation_blob), nullptr, &m_log_average_computation));

    OID3DBlob linear_exposure_computation_blob = compile_shader(shader_filename, "cs_5_0", "ColorGrading::compute_linear_exposure");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_computation_blob), nullptr, &m_linear_exposure_computation));

    // Create buffers
    create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, max_groups_dispatched, &m_log_averages_SRV, &m_log_averages_UAV);
}

void LogAverageLuminance::compute_log_average(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                              ID3D11ShaderResourceView* pixels, unsigned int image_width,
                                              ID3D11UnorderedAccessView* log_average_UAV) {
    compute(context, constants, pixels, image_width, m_log_average_computation, log_average_UAV);
}

void LogAverageLuminance::compute_linear_exposure(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                                  ID3D11ShaderResourceView* pixels, unsigned int image_width,
                                                  ID3D11UnorderedAccessView* linear_exposure_UAV) {
    compute(context, constants, pixels, image_width, m_linear_exposure_computation, linear_exposure_UAV);
}

void LogAverageLuminance::compute(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                  ID3D11ShaderResourceView* pixels, unsigned int image_width, OID3D11ComputeShader& second_reduction,
                                  ID3D11UnorderedAccessView* output_UAV) {

    context.CSSetConstantBuffers(0, 1, &constants);

    context.CSSetUnorderedAccessViews(0, 1, &m_log_averages_UAV, 0u);
    context.CSSetShaderResources(0, 1, &pixels);
    context.CSSetShader(m_log_average_first_reduction, nullptr, 0u);
    context.Dispatch(max_groups_dispatched, 1, 1); // Always dispatch max groups to ensure that unused elements in m_log_averages_UAV are cleared.

    context.CSSetUnorderedAccessViews(0, 1, &output_UAV, 0u);
    context.CSSetShaderResources(0, 1, &m_log_averages_SRV);
    context.CSSetShader(second_reduction, nullptr, 0u);
    context.Dispatch(1, 1, 1);

    ID3D11UnorderedAccessView* null_UAV = nullptr;
    context.CSSetUnorderedAccessViews(0, 1, &null_UAV, 0u);
}

// ------------------------------------------------------------------------------------------------
// Exposure histogram
// ------------------------------------------------------------------------------------------------
ExposureHistogram::ExposureHistogram(ID3D11Device1& device, const std::wstring& shader_folder_path) {
    const std::wstring shader_filename = shader_folder_path + L"ColorGrading/ReduceExposureHistogram.hlsl";

    // Create shaders.
    OID3DBlob reduce_exposure_histogram_blob = compile_shader(shader_filename, "cs_5_0", "ColorGrading::reduce");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(reduce_exposure_histogram_blob), nullptr, &m_histogram_reduction));

    OID3DBlob linear_exposure_computation_blob = compile_shader(shader_filename, "cs_5_0", "ColorGrading::compute_linear_exposure");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_computation_blob), nullptr, &m_linear_exposure_computation));

    // Create buffers
    create_default_buffer(device, DXGI_FORMAT_R32_UINT, bin_count, &m_histogram_SRV, &m_histogram_UAV);
}

OID3D11ShaderResourceView& ExposureHistogram::reduce_histogram(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                                               ID3D11ShaderResourceView* pixels, unsigned int image_width) {

    const unsigned int zeros[4] = { 0u, 0u, 0u, 0u };
    context.ClearUnorderedAccessViewUint(m_histogram_UAV, zeros);

    context.CSSetShader(m_histogram_reduction, nullptr, 0u);
    context.CSSetConstantBuffers(0, 1, &constants);
    context.CSSetShaderResources(0, 1, &pixels);
    context.CSSetUnorderedAccessViews(0, 1, &m_histogram_UAV, 0u);
    unsigned int group_count_x = ceil_divide(image_width, group_width);
    context.Dispatch(group_count_x, 1, 1);

    ID3D11UnorderedAccessView* null_UAV = nullptr;
    context.CSSetUnorderedAccessViews(0, 1, &null_UAV, 0u);

    return m_histogram_SRV;
}

void ExposureHistogram::compute_linear_exposure(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                                ID3D11ShaderResourceView* pixels, unsigned int image_width,
                                                ID3D11UnorderedAccessView* linear_exposure_UAV) {
    const unsigned int zeros[4] = { 0u, 0u, 0u, 0u };
    context.ClearUnorderedAccessViewUint(m_histogram_UAV, zeros);

    context.CSSetConstantBuffers(0, 1, &constants);

    context.CSSetUnorderedAccessViews(0, 1, &m_histogram_UAV, 0u);
    context.CSSetShaderResources(0, 1, &pixels);
    context.CSSetShader(m_histogram_reduction, nullptr, 0u);
    unsigned int group_count_x = ceil_divide(image_width, group_width);
    context.Dispatch(group_count_x, 1, 1);

    context.CSSetUnorderedAccessViews(0, 1, &linear_exposure_UAV, 0u);
    context.CSSetShaderResources(0, 1, &m_histogram_SRV);
    context.CSSetShader(m_linear_exposure_computation, nullptr, 0u);
    context.Dispatch(1, 1, 1);

    ID3D11UnorderedAccessView* null_UAV = nullptr;
    context.CSSetUnorderedAccessViews(0, 1, &null_UAV, 0u);
}

// ------------------------------------------------------------------------------------------------
// Tonemapper
// ------------------------------------------------------------------------------------------------
Tonemapper::Tonemapper(ID3D11Device1& device, const std::wstring& shader_folder_path) {

    THROW_ON_FAILURE(create_constant_buffer(device, sizeof(Constants), &m_constant_buffer));

    create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, &m_linear_exposure_SRV, &m_linear_exposure_UAV);
    m_log_average_luminance = LogAverageLuminance(device, shader_folder_path);
    m_exposure_histogram = ExposureHistogram(device, shader_folder_path);

    m_bloom = DualKawaseBloom(device, shader_folder_path);

    { // Setup tonemapping shaders
        const std::wstring shader_filename = shader_folder_path + L"ColorGrading/Tonemapping.hlsl";

        OID3DBlob linear_exposure_from_bias_blob = compile_shader(shader_filename, "cs_5_0", "ColorGrading::linear_exposure_from_constant_bias");
        THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_from_bias_blob), nullptr, &m_linear_exposure_from_bias_shader));

        OID3DBlob vertex_shader_blob = compile_shader(shader_filename, "vs_5_0", "ColorGrading::fullscreen_vs");
        HRESULT hr = device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_fullscreen_VS);
        THROW_ON_FAILURE(hr);

        auto create_pixel_shader = [&](const char* entry_point) -> OID3D11PixelShader {
            OID3D11PixelShader pixel_shader;
            OID3DBlob pixel_shader_blob = compile_shader(shader_filename, "ps_5_0", entry_point);
            HRESULT hr = device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &pixel_shader);
            THROW_ON_FAILURE(hr);
            return pixel_shader;
        };

        m_linear_tonemapping_PS = create_pixel_shader("ColorGrading::linear_tonemapping_ps");
        m_uncharted2_tonemapping_PS = create_pixel_shader("ColorGrading::uncharted2_tonemapping_ps");
        m_filmic_tonemapping_PS = create_pixel_shader("ColorGrading::unreal4_tonemapping_ps");
    }

    { // Bilinear sampler
        D3D11_SAMPLER_DESC sampler_desc = {};
        sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sampler_desc.MinLOD = 0;
        sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

        THROW_ON_FAILURE(device.CreateSamplerState(&sampler_desc, &m_bilinear_sampler));
    }
}

void Tonemapper::tonemap(ID3D11DeviceContext1& context, Tonemapping::Parameters parameters, float delta_time,
                         ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV, int width, int height) {

    using namespace Cogwheel::Math::Tonemapping;

    { // Upload constants
        Constants constants;
        constants.min_log_luminance = parameters.exposure.min_log_luminance;
        constants.max_log_luminance = parameters.exposure.max_log_luminance;
        constants.min_histogram_percentage = parameters.exposure.min_histogram_percentage;
        constants.max_histogram_percentage = parameters.exposure.max_histogram_percentage;
        constants.log_lumiance_bias = parameters.exposure.log_lumiance_bias;
        if (parameters.exposure.eye_adaptation_enabled) {
            constants.eye_adaptation_brightness = parameters.exposure.eye_adaptation_brightness;
            constants.eye_adaptation_darkness = parameters.exposure.eye_adaptation_darkness;
        } else
            constants.eye_adaptation_brightness = constants.eye_adaptation_darkness = std::numeric_limits<float>::infinity();

        constants.bloom_threshold = parameters.bloom.receiver_threshold;

        constants.delta_time = delta_time;

        context.UpdateSubresource(m_constant_buffer, 0, nullptr, &constants, 0u, 0u);
    }

    context.OMSetRenderTargets(1, &backbuffer_RTV, nullptr);
    context.PSSetConstantBuffers(0, 1, &m_constant_buffer);
    context.PSSetSamplers(0, 1, &m_bilinear_sampler); // TODO Get rid of when all shaders are compute.
    context.CSSetConstantBuffers(0, 1, &m_constant_buffer);
    context.CSSetSamplers(0, 1, &m_bilinear_sampler);

    { // Determine exposure.
        if (parameters.exposure.mode == ExposureMode::Histogram)
            m_exposure_histogram.compute_linear_exposure(context, m_constant_buffer, pixel_SRV, width, m_linear_exposure_UAV);
        else if (parameters.exposure.mode == ExposureMode::LogAverage)
            m_log_average_luminance.compute_linear_exposure(context, m_constant_buffer, pixel_SRV, width, m_linear_exposure_UAV);
        else { // parameters.exposure.mode == ExposureMode::Fixed
            context.CSSetConstantBuffers(0, 1, &m_constant_buffer);

            context.CSSetUnorderedAccessViews(0, 1, &m_linear_exposure_UAV, 0u);
            context.CSSetShader(m_linear_exposure_from_bias_shader, nullptr, 0u);
            context.Dispatch(1, 1, 1);

            ID3D11UnorderedAccessView* null_UAV = nullptr;
            context.CSSetUnorderedAccessViews(0, 1, &null_UAV, 0u);
        }
    }

    // Bloom filter.
    ID3D11ShaderResourceView* bloom_SRV = nullptr;
    if (parameters.bloom.receiver_threshold < INFINITY)
        bloom_SRV = m_bloom.filter(context, m_constant_buffer, m_bilinear_sampler, pixel_SRV, width, height, 3).get();

    { // Tonemap and render into backbuffer.
        context.VSSetShader(m_fullscreen_VS, 0, 0);

        if (parameters.tonemapping.mode == TonemappingMode::Linear)
            context.PSSetShader(m_linear_tonemapping_PS, 0, 0);
        else if (parameters.tonemapping.mode == TonemappingMode::Uncharted2)
            context.PSSetShader(m_uncharted2_tonemapping_PS, 0, 0);
        else // parameters.tonemapping.mode == TonemappingMode::Filmic
            context.PSSetShader(m_filmic_tonemapping_PS, 0, 0);

        ID3D11ShaderResourceView* srvs[3] = { pixel_SRV, m_linear_exposure_SRV, bloom_SRV };
        context.PSSetShaderResources(0, 3, srvs);

        context.Draw(3, 0);
    }
}

} // NS DX11Renderer
