// DirectX 11 camera effects.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <DX11Renderer/CameraEffects.h>
#include <DX11Renderer/Utils.h>

using namespace Bifrost::Math;

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// Gaussian Bloom.
// ------------------------------------------------------------------------------------------------
GaussianBloom::GaussianBloom(ID3D11Device1& device, const std::wstring& shader_folder_path) {

    m_ping = {};
    m_pong = {};

    const std::wstring shader_filename = shader_folder_path + L"CameraEffects/Bloom.hlsl";

    OBlob horizontal_filter_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::sampled_gaussian_horizontal_filter");
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(horizontal_filter_blob), nullptr, &m_horizontal_filter));

    OBlob vertical_filter_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::sampled_gaussian_vertical_filter");
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(vertical_filter_blob), nullptr, &m_vertical_filter));

    m_gaussian_samples.std_dev = std::numeric_limits<float>::infinity();
    m_gaussian_samples.capacity = 64;
    m_gaussian_samples.buffer = create_default_buffer(device, DXGI_FORMAT_R16G16_FLOAT, m_gaussian_samples.capacity, &m_gaussian_samples.SRV);
}

OShaderResourceView& GaussianBloom::filter(ID3D11DeviceContext1& context, ID3D11Buffer& constants, ID3D11ShaderResourceView* pixels, 
                                           unsigned int image_width, unsigned int image_height, int support) {
#if CHECK_IMPLICIT_STATE
    // Check that the constants and sampler are bound.
    OBuffer bound_constants;
    context.CSGetConstantBuffers(0, 1, &bound_constants);
    always_assert(bound_constants.get() == &constants);
#endif

    auto performance_marker = PerformanceMarker(context, L"Gaussian bloom");

    int sample_count = ceil_divide(support, 2);
    if (sample_count > m_gaussian_samples.capacity) {
        m_gaussian_samples.buffer.release();
        m_gaussian_samples.SRV.release();

        m_gaussian_samples.capacity = next_power_of_two(sample_count);
        ODevice1 device = get_device1(context);
        m_gaussian_samples.buffer = create_default_buffer(device, DXGI_FORMAT_R16G16_FLOAT, m_gaussian_samples.capacity, &m_gaussian_samples.SRV);

        // Set std dev to infinity to indicate that the buffer needs updating.
        m_gaussian_samples.std_dev = std::numeric_limits<float>::infinity();
    }

    float std_dev = support * 0.25f;
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

    if (m_ping.width < image_width || m_ping.height < image_height) {

        int buffer_width = max(m_ping.width, image_width);
        int buffer_height = max(m_ping.height, image_height);

        // Release old resources
        m_ping.SRV.release();
        m_ping.UAV.release();
        m_pong.SRV.release();
        m_pong.UAV.release();

        // Grab the device.
        ODevice1 device = get_device1(context);

        m_ping.width = m_pong.width = buffer_width;
        m_ping.height = m_pong.height = buffer_height;
        create_texture_2D(device, DXGI_FORMAT_R16G16B16A16_FLOAT, image_width, buffer_height, &m_ping.SRV, &m_ping.UAV);
        create_texture_2D(device, DXGI_FORMAT_R16G16B16A16_FLOAT, image_width, buffer_height, &m_pong.SRV, &m_pong.UAV);
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
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(m_extract_high_intensity_blob), nullptr, &m_extract_high_intensity));

    OBlob downsample_pattern_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::dual_kawase_downsample");
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(downsample_pattern_blob), nullptr, &m_downsample_pattern));

    OBlob upsample_pattern_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::dual_kawase_upsample");
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(upsample_pattern_blob), nullptr, &m_upsample_pattern));
}

OShaderResourceView& DualKawaseBloom::filter(ID3D11DeviceContext1& context, ID3D11Buffer& constants, ID3D11ShaderResourceView* pixels, 
                                             unsigned int image_width, unsigned int image_height, unsigned int half_passes) {
#if CHECK_IMPLICIT_STATE
    // Check that the constants and sampler are bound.
    OBuffer bound_constants;
    context.CSGetConstantBuffers(0, 1, &bound_constants);
    always_assert(bound_constants.get() == &constants);
#endif

    auto performance_marker = PerformanceMarker(context, L"Dual kawase bloom");

    if (m_temp.width < image_width || m_temp.height < image_height) {

        int buffer_width = max(m_temp.width, image_width);
        int buffer_height = max(m_temp.height, image_height);

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
        while ((buffer_width >> m_temp.mipmap_count) > 0 || (buffer_height >> m_temp.mipmap_count) > 0)
            ++m_temp.mipmap_count;

        D3D11_TEXTURE2D_DESC tex_desc = {};
        tex_desc.Width = buffer_width;
        tex_desc.Height = buffer_height;
        tex_desc.MipLevels = m_temp.mipmap_count;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.SampleDesc.Quality = 0;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

        OTexture2D texture2D;
        THROW_DX11_ERROR(device->CreateTexture2D(&tex_desc, nullptr, &texture2D));

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
            THROW_DX11_ERROR(device->CreateShaderResourceView(texture2D, &mip_level_SRV_desc, &m_temp.SRVs[m]));

            // UAV
            D3D11_UNORDERED_ACCESS_VIEW_DESC mip_level_UAV_desc = {};
            mip_level_UAV_desc.Format = tex_desc.Format;
            mip_level_UAV_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            mip_level_UAV_desc.Texture2D.MipSlice = m;
            THROW_DX11_ERROR(device->CreateUnorderedAccessView(texture2D, &mip_level_UAV_desc, &m_temp.UAVs[m]));
        }

        m_temp.width = buffer_width;
        m_temp.height = buffer_height;
    }

    // Copy high intensity part of image.
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
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(log_average_first_reduction_blob), nullptr, &m_log_average_first_reduction));

    OBlob log_average_computation_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::compute_log_average");
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(log_average_computation_blob), nullptr, &m_log_average_computation));

    OBlob linear_exposure_computation_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::compute_linear_exposure");
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_computation_blob), nullptr, &m_linear_exposure_computation));

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
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(reduce_exposure_histogram_blob), nullptr, &m_histogram_reduction));

    OBlob linear_exposure_computation_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::compute_linear_exposure");
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_computation_blob), nullptr, &m_linear_exposure_computation));

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

    THROW_DX11_ERROR(create_constant_buffer(device, sizeof(Constants), &m_constant_buffer));

    create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, &m_linear_exposure_SRV, &m_linear_exposure_UAV);
    m_log_average_luminance = LogAverageLuminance(device, shader_folder_path);
    m_exposure_histogram = ExposureHistogram(device, shader_folder_path);

    m_bloom = GaussianBloom(device, shader_folder_path);

    D3D11_RASTERIZER_DESC raster_desc = CD3D11_RASTERIZER_DESC(CD3D11_DEFAULT());
    THROW_DX11_ERROR(device.CreateRasterizerState(&raster_desc, &m_raster_state));

    { // Setup tonemapping shaders
        const std::wstring shader_filename = shader_folder_path + L"CameraEffects/Tonemapping.hlsl";

        OBlob linear_exposure_from_bias_blob = compile_shader(shader_filename, "cs_5_0", "CameraEffects::linear_exposure_from_constant_bias");
        THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_from_bias_blob), nullptr, &m_linear_exposure_from_bias_shader));

        OBlob vertex_shader_blob = compile_shader(shader_filename, "vs_5_0", "CameraEffects::fullscreen_vs");
        HRESULT hr = device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_fullscreen_VS);
        THROW_DX11_ERROR(hr);

        auto create_pixel_shader = [&](const char* entry_point) -> OPixelShader {
            OPixelShader pixel_shader;
            OBlob pixel_shader_blob = compile_shader(shader_filename, "ps_5_0", entry_point);
            HRESULT hr = device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &pixel_shader);
            THROW_DX11_ERROR(hr);
            return pixel_shader;
        };

        m_linear_tonemapping_PS = create_pixel_shader("CameraEffects::linear_tonemapping_ps");
        m_uncharted2_tonemapping_PS = create_pixel_shader("CameraEffects::uncharted2_tonemapping_ps");
        m_filmic_tonemapping_PS = create_pixel_shader("CameraEffects::unreal4_tonemapping_ps");
    }
}

void CameraEffects::process(ID3D11DeviceContext1& context, Bifrost::Math::CameraEffects::Settings settings, float delta_time,
                            ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV, 
                            Bifrost::Math::Rect<int> input_viewport, Bifrost::Math::Rect<int> output_viewport) {

    using namespace Bifrost::Math::CameraEffects;

    auto performance_marker = PerformanceMarker(context, L"Camera effects");

    { // Upload constants
        Constants constants;
        constants.input_viewport = Bifrost::Math::Rect<float>(input_viewport);
        constants.output_viewport_offset = { output_viewport.x, output_viewport.y };
        constants.output_pixel_offset = { input_viewport.x - output_viewport.x, input_viewport.y - output_viewport.y };
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
        constants.bloom_support = int(settings.bloom.support * input_viewport.height);

        constants.delta_time = delta_time;

        if (settings.tonemapping.mode == TonemappingMode::Filmic) {
            constants.tonemapping[0] = settings.tonemapping.filmic.black_clip;
            constants.tonemapping[1] = settings.tonemapping.filmic.toe;
            constants.tonemapping[2] = settings.tonemapping.filmic.slope;
            constants.tonemapping[3] = settings.tonemapping.filmic.shoulder;
            constants.tonemapping[4] = settings.tonemapping.filmic.white_clip;
        } else if (settings.tonemapping.mode == TonemappingMode::Uncharted2) {
            constants.tonemapping[0] = settings.tonemapping.uncharted2.shoulder_strength;
            constants.tonemapping[1] = settings.tonemapping.uncharted2.linear_strength;
            constants.tonemapping[2] = settings.tonemapping.uncharted2.linear_angle;
            constants.tonemapping[3] = settings.tonemapping.uncharted2.toe_strength;
            constants.tonemapping[4] = settings.tonemapping.uncharted2.toe_numerator;
            constants.tonemapping[5] = settings.tonemapping.uncharted2.toe_denominator;
            constants.tonemapping[6] = settings.tonemapping.uncharted2.linear_white;
        }

        context.UpdateSubresource(m_constant_buffer, 0, nullptr, &constants, 0u, 0u);
    }

    // Setup state
    D3D11_VIEWPORT dx_viewport;
    dx_viewport.TopLeftX = float(output_viewport.x);
    dx_viewport.TopLeftY = float(output_viewport.y);
    dx_viewport.Width = float(output_viewport.width);
    dx_viewport.Height = float(output_viewport.height);
    dx_viewport.MinDepth = 0.0f;
    dx_viewport.MaxDepth = 1.0f;
    context.RSSetViewports(1, &dx_viewport);
    context.RSSetState(m_raster_state);
    context.OMSetRenderTargets(1, &backbuffer_RTV, nullptr);
    context.PSSetConstantBuffers(0, 1, &m_constant_buffer);
    context.CSSetConstantBuffers(0, 1, &m_constant_buffer);

    { // Determine exposure.
        if (settings.exposure.mode == ExposureMode::Histogram)
            m_exposure_histogram.compute_linear_exposure(context, m_constant_buffer, pixel_SRV, input_viewport.width, m_linear_exposure_UAV);
        else if (settings.exposure.mode == ExposureMode::LogAverage)
            m_log_average_luminance.compute_linear_exposure(context, m_constant_buffer, pixel_SRV, input_viewport.width, m_linear_exposure_UAV);
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
        int support = int(settings.bloom.support * input_viewport.height);
        bloom_SRV = m_bloom.filter(context, m_constant_buffer, pixel_SRV, input_viewport.width, input_viewport.height, support).get();
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
