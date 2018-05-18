// DirectX 11 camera effects.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <DX11Renderer/CameraEffects.h>
#include <DX11Renderer/Utils.h>

using namespace Cogwheel::Math;

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// Gaussian Bloom.
// ------------------------------------------------------------------------------------------------
GaussianBloom::GaussianBloom(ID3D11Device1& device, const std::wstring& shader_folder_path) {

    m_ping = {};
    m_pong = {};

    const std::wstring shader_filename = shader_folder_path + L"CameraEffects/Bloom.hlsl";

    OBlob horizontal_filter_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::sampled_gaussian_horizontal_filter");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(horizontal_filter_blob), nullptr, &m_horizontal_filter));

    OBlob vertical_filter_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::sampled_gaussian_vertical_filter");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(vertical_filter_blob), nullptr, &m_vertical_filter));

    m_gaussian_samples.std_dev = std::numeric_limits<float>::infinity();
    m_gaussian_samples.capacity = 64;
    m_gaussian_samples.buffer = create_default_buffer(device, DXGI_FORMAT_R16G16_FLOAT, m_gaussian_samples.capacity, &m_gaussian_samples.SRV);
}

OShaderResourceView& GaussianBloom::filter(ID3D11DeviceContext1& context, ID3D11Buffer& constants, ID3D11SamplerState& bilinear_sampler,
                                           ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height, int bandwidth) {
#if CHECK_IMPLICIT_STATE
    // Check that the constants and sampler are bound.
    OBuffer bound_constants;
    context.CSGetConstantBuffers(0, 1, &bound_constants);
    always_assert(bound_constants.get() == &constants);
    OSamplerState bound_sampler;
    context.CSGetSamplers(0, 1, &bound_sampler);
    always_assert(bound_sampler.get() == &bilinear_sampler);
#endif

    auto performance_marker = PerformanceMarker(context, L"Gaussian bloom");

    int sample_count = ceil_divide(bandwidth, 2);
    if (sample_count > m_gaussian_samples.capacity) {
        m_gaussian_samples.buffer.release();
        m_gaussian_samples.SRV.release();

        m_gaussian_samples.capacity = next_power_of_two(sample_count);
        ODevice1 device = get_device1(context);
        m_gaussian_samples.buffer = create_default_buffer(device, DXGI_FORMAT_R16G16_FLOAT, m_gaussian_samples.capacity, &m_gaussian_samples.SRV);

        // Set std dev to infinity to indicate that the buffer needs updating.
        m_gaussian_samples.std_dev = std::numeric_limits<float>::infinity();
    }

    float std_dev = bandwidth * 0.25f;
    if (m_gaussian_samples.std_dev != std_dev) {
        sample_count = m_gaussian_samples.capacity;
        Tap* taps = new Tap[sample_count];
        fill_bilinear_gaussian_samples(std_dev, taps, taps + sample_count);

        half2* taps_h = new half2[sample_count];
        for (int i = 0; i < sample_count; ++i)
            taps_h[i] = { half(taps[i].offset), half(taps[i].weight) };

        context.UpdateSubresource(m_gaussian_samples.buffer, 0u, nullptr, taps_h, sizeof(half2) * sample_count, 0u);

        delete[] taps_h;
        delete[] taps;

        m_gaussian_samples.std_dev = std_dev;
    }

    if (m_ping.width != image_width || m_ping.height != image_height) {

        // Release old resources
        m_ping.SRV.release();
        m_ping.UAV.release();
        m_pong.SRV.release();
        m_pong.UAV.release();

        // Grab the device.
        ODevice1 device = get_device1(context);

        auto allocate_texture = [&](IntermediateTexture& tex) {

            D3D11_TEXTURE2D_DESC tex_desc = {};
            tex_desc.Width = tex.width = image_width;
            tex_desc.Height = tex.height = image_height;
            tex_desc.MipLevels = 1;
            tex_desc.ArraySize = 1;
            tex_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            tex_desc.SampleDesc.Count = 1;
            tex_desc.SampleDesc.Quality = 0;
            tex_desc.Usage = D3D11_USAGE_DEFAULT;
            tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

            OTexture2D texture2D;
            THROW_ON_FAILURE(device->CreateTexture2D(&tex_desc, nullptr, &texture2D));

            // SRV
            D3D11_SHADER_RESOURCE_VIEW_DESC mip_level_SRV_desc;
            mip_level_SRV_desc.Format = tex_desc.Format;
            mip_level_SRV_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            mip_level_SRV_desc.Texture2D.MipLevels = 1;
            mip_level_SRV_desc.Texture2D.MostDetailedMip = 0;
            THROW_ON_FAILURE(device->CreateShaderResourceView(texture2D, &mip_level_SRV_desc, &tex.SRV));

            // UAV
            D3D11_UNORDERED_ACCESS_VIEW_DESC mip_level_UAV_desc = {};
            mip_level_UAV_desc.Format = tex_desc.Format;
            mip_level_UAV_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            mip_level_UAV_desc.Texture2D.MipSlice = 0;
            THROW_ON_FAILURE(device->CreateUnorderedAccessView(texture2D, &mip_level_UAV_desc, &tex.UAV));
        };

        allocate_texture(m_ping);
        allocate_texture(m_pong);
    }

    // High intensity pass and horizontal filter.
    context.CSSetShader(m_horizontal_filter, nullptr, 0u);
    context.CSSetUnorderedAccessViews(0, 1, &m_pong.UAV, 0u);
    ID3D11ShaderResourceView* SRVs[] = { pixels, m_gaussian_samples.SRV };
    context.CSSetShaderResources(0, 2, SRVs);
    context.Dispatch(ceil_divide(image_width, group_size), ceil_divide(image_height, group_size), 1);

    // Vertical filter
    context.CSSetShader(m_vertical_filter, nullptr, 0u);
    context.CSSetUnorderedAccessViews(0, 1, &m_ping.UAV, 0u);
    context.CSSetShaderResources(0, 1, &m_pong.SRV);
    context.Dispatch(ceil_divide(image_width, group_size), ceil_divide(image_height, group_size), 1);

    ID3D11UnorderedAccessView* null_UAV = nullptr;
    context.CSSetUnorderedAccessViews(0, 1, &null_UAV, 0u);

    return m_ping.SRV;
}


// ------------------------------------------------------------------------------------------------
// Dual Kawase Bloom.
// ------------------------------------------------------------------------------------------------
DualKawaseBloom::DualKawaseBloom(ID3D11Device1& device, const std::wstring& shader_folder_path) {

    m_temp = {};

    const std::wstring shader_filename = shader_folder_path + L"CameraEffects/Bloom.hlsl";

    OBlob m_extract_high_intensity_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::extract_high_intensity");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(m_extract_high_intensity_blob), nullptr, &m_extract_high_intensity));

    OBlob downsample_pattern_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::dual_kawase_downsample");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(downsample_pattern_blob), nullptr, &m_downsample_pattern));

    OBlob upsample_pattern_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::dual_kawase_upsample");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(upsample_pattern_blob), nullptr, &m_upsample_pattern));
}

OShaderResourceView& DualKawaseBloom::filter(ID3D11DeviceContext1& context, ID3D11Buffer& constants, ID3D11SamplerState& bilinear_sampler,
                                                   ID3D11ShaderResourceView* pixels, unsigned int image_width, unsigned int image_height, unsigned int half_passes) {
#if CHECK_IMPLICIT_STATE
    // Check that the constants and sampler are bound.
    OBuffer bound_constants;
    context.CSGetConstantBuffers(0, 1, &bound_constants);
    always_assert(bound_constants.get() == &constants);
    OSamplerState bound_sampler;
    context.CSGetSamplers(0, 1, &bound_sampler);
    always_assert(bound_sampler.get() == &bilinear_sampler);
#endif

    auto performance_marker = PerformanceMarker(context, L"Dual kawase bloom");

    if (m_temp.width != image_width || m_temp.height != image_height) {
        
        // Release old resources
        for (unsigned int m = 0; m < m_temp.mipmap_count; ++m) {
            m_temp.SRVs[m].release();
            m_temp.UAVs[m].release();
        }
        delete[] m_temp.SRVs;
        delete[] m_temp.UAVs;

        ODevice1 device = get_device1(context);

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

        OTexture2D texture2D;
        THROW_ON_FAILURE(device->CreateTexture2D(&tex_desc, nullptr, &texture2D));

        // TODO Create views in two device calls.
        m_temp.SRVs = new OShaderResourceView[m_temp.mipmap_count];
        m_temp.UAVs = new OUnorderedAccessView[m_temp.mipmap_count];

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
    const std::wstring shader_filename = shader_folder_path + L"CameraEffects/ReduceLogAverageLuminance.hlsl";

    // Create shaders.
    OBlob log_average_first_reduction_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::first_reduction");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(log_average_first_reduction_blob), nullptr, &m_log_average_first_reduction));

    OBlob log_average_computation_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::compute_log_average");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(log_average_computation_blob), nullptr, &m_log_average_computation));

    OBlob linear_exposure_computation_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::compute_linear_exposure");
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
                                  ID3D11ShaderResourceView* pixels, unsigned int image_width, OComputeShader& second_reduction,
                                  ID3D11UnorderedAccessView* output_UAV) {
#if CHECK_IMPLICIT_STATE
    // Check that the constants and sampler are bound.
    OBuffer bound_constants;
    context.CSGetConstantBuffers(0, 1, &bound_constants);
    always_assert(bound_constants.get() == constants);
#endif

    auto performance_marker = PerformanceMarker(context, L"Log average luminance");

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
    const std::wstring shader_filename = shader_folder_path + L"CameraEffects/ReduceExposureHistogram.hlsl";

    // Create shaders.
    OBlob reduce_exposure_histogram_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::reduce");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(reduce_exposure_histogram_blob), nullptr, &m_histogram_reduction));

    OBlob linear_exposure_computation_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::compute_linear_exposure");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_computation_blob), nullptr, &m_linear_exposure_computation));

    // Create buffers
    create_default_buffer(device, DXGI_FORMAT_R32_UINT, bin_count, &m_histogram_SRV, &m_histogram_UAV);
}

OShaderResourceView& ExposureHistogram::reduce_histogram(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                                               ID3D11ShaderResourceView* pixels, unsigned int image_width) {
#if CHECK_IMPLICIT_STATE
    // Check that the constants and sampler are bound.
    OBuffer bound_constants;
    context.CSGetConstantBuffers(0, 1, &bound_constants);
    always_assert(bound_constants.get() == constants);
#endif

    const unsigned int zeros[4] = { 0u, 0u, 0u, 0u };
    context.ClearUnorderedAccessViewUint(m_histogram_UAV, zeros);

    context.CSSetShader(m_histogram_reduction, nullptr, 0u);
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
#if CHECK_IMPLICIT_STATE
    // Check that the constants and sampler are bound.
    OBuffer bound_constants;
    context.CSGetConstantBuffers(0, 1, &bound_constants);
    always_assert(bound_constants.get() == constants);
#endif

    auto performance_marker = PerformanceMarker(context, L"Exposure from histogram");

    const unsigned int zeros[4] = { 0u, 0u, 0u, 0u };
    context.ClearUnorderedAccessViewUint(m_histogram_UAV, zeros);

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
// Camera post processing
// ------------------------------------------------------------------------------------------------
CameraEffects::CameraEffects(ID3D11Device1& device, const std::wstring& shader_folder_path) {

    THROW_ON_FAILURE(create_constant_buffer(device, sizeof(Constants), &m_constant_buffer));

    create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, &m_linear_exposure_SRV, &m_linear_exposure_UAV);
    m_log_average_luminance = LogAverageLuminance(device, shader_folder_path);
    m_exposure_histogram = ExposureHistogram(device, shader_folder_path);

    m_bloom = GaussianBloom(device, shader_folder_path);

    { // Setup tonemapping shaders
        const std::wstring shader_filename = shader_folder_path + L"CameraEffects/Tonemapping.hlsl";

        OBlob linear_exposure_from_bias_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::linear_exposure_from_constant_bias");
        THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_from_bias_blob), nullptr, &m_linear_exposure_from_bias_shader));

        OBlob vertex_shader_blob = compile_shader(shader_filename, "vs_5_0", "CameraEffects::fullscreen_vs");
        HRESULT hr = device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_fullscreen_VS);
        THROW_ON_FAILURE(hr);

        auto create_pixel_shader = [&](const char* entry_point) -> OPixelShader {
            OPixelShader pixel_shader;
            OBlob pixel_shader_blob = compile_shader(shader_filename, "ps_5_0", entry_point);
            HRESULT hr = device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &pixel_shader);
            THROW_ON_FAILURE(hr);
            return pixel_shader;
        };

        m_linear_tonemapping_PS = create_pixel_shader("CameraEffects::linear_tonemapping_ps");
        m_uncharted2_tonemapping_PS = create_pixel_shader("CameraEffects::uncharted2_tonemapping_ps");
        m_filmic_tonemapping_PS = create_pixel_shader("CameraEffects::unreal4_tonemapping_ps");
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

void CameraEffects::process(ID3D11DeviceContext1& context, Cogwheel::Math::CameraEffects::Settings settings, float delta_time,
                            ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV, int width, int height) {

    using namespace Cogwheel::Math::CameraEffects;

    auto performance_marker = PerformanceMarker(context, L"Camera effects");

    { // Upload constants
        Constants constants;
        constants.min_log_luminance = settings.exposure.min_log_luminance;
        constants.max_log_luminance = settings.exposure.max_log_luminance;
        constants.min_histogram_percentage = settings.exposure.min_histogram_percentage;
        constants.max_histogram_percentage = settings.exposure.max_histogram_percentage;
        constants.log_lumiance_bias = settings.exposure.log_lumiance_bias;
        if (settings.exposure.eye_adaptation_enabled) {
            constants.eye_adaptation_brightness = settings.exposure.eye_adaptation_brightness;
            constants.eye_adaptation_darkness = settings.exposure.eye_adaptation_darkness;
        } else
            constants.eye_adaptation_brightness = constants.eye_adaptation_darkness = std::numeric_limits<float>::infinity();

        constants.bloom_threshold = settings.bloom.threshold;
        constants.bloom_bandwidth = int(settings.bloom.bandwidth * height);

        constants.delta_time = delta_time;

        context.UpdateSubresource(m_constant_buffer, 0, nullptr, &constants, 0u, 0u);
    }

    context.OMSetRenderTargets(1, &backbuffer_RTV, nullptr);
    context.PSSetConstantBuffers(0, 1, &m_constant_buffer);
    context.PSSetSamplers(0, 1, &m_bilinear_sampler); // TODO Get rid of PS bindings when all shaders are compute.
    context.CSSetConstantBuffers(0, 1, &m_constant_buffer);
    context.CSSetSamplers(0, 1, &m_bilinear_sampler);

    { // Determine exposure.
        if (settings.exposure.mode == ExposureMode::Histogram)
            m_exposure_histogram.compute_linear_exposure(context, m_constant_buffer, pixel_SRV, width, m_linear_exposure_UAV);
        else if (settings.exposure.mode == ExposureMode::LogAverage)
            m_log_average_luminance.compute_linear_exposure(context, m_constant_buffer, pixel_SRV, width, m_linear_exposure_UAV);
        else { // settings.exposure.mode == ExposureMode::Fixed
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
    if (settings.bloom.threshold < INFINITY) {
        int bandwidth = int(settings.bloom.bandwidth * height);
        bloom_SRV = m_bloom.filter(context, m_constant_buffer, m_bilinear_sampler, pixel_SRV, width, height, bandwidth).get();
    }

    { // Tonemap and render into backbuffer.
        context.VSSetShader(m_fullscreen_VS, 0, 0);

        if (settings.tonemapping.mode == TonemappingMode::Linear)
            context.PSSetShader(m_linear_tonemapping_PS, 0, 0);
        else if (settings.tonemapping.mode == TonemappingMode::Uncharted2)
            context.PSSetShader(m_uncharted2_tonemapping_PS, 0, 0);
        else // settings.tonemapping.mode == TonemappingMode::Filmic
            context.PSSetShader(m_filmic_tonemapping_PS, 0, 0);

        ID3D11ShaderResourceView* srvs[3] = { pixel_SRV, m_linear_exposure_SRV, bloom_SRV };
        context.PSSetShaderResources(0, 3, srvs);

        context.Draw(3, 0);
    }
}

} // NS DX11Renderer
