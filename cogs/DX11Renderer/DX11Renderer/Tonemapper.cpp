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
// Log average luminance reduction.
// ------------------------------------------------------------------------------------------------
LogAverageLuminance::LogAverageLuminance()
    : m_log_average_first_reduction(nullptr), m_log_average_second_reduction(nullptr), m_log_averages_SRV(nullptr), m_log_averages_UAV(nullptr)
    , m_linear_exposure_computation(nullptr), m_linear_exposure_SRV(nullptr), m_linear_exposure_UAV(nullptr) {}

LogAverageLuminance::LogAverageLuminance(ID3D11Device1& device, const std::wstring& shader_folder_path) {
    // Create shaders.
    OID3DBlob log_average_first_reduction_blob = compile_shader(shader_folder_path + L"Compute\\ReduceLogAverageLuminance.hlsl", "cs_5_0", "first_reduction");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(log_average_first_reduction_blob), nullptr, &m_log_average_first_reduction));

    OID3DBlob log_average_second_reduction_blob = compile_shader(shader_folder_path + L"Compute\\ReduceLogAverageLuminance.hlsl", "cs_5_0", "second_reduction");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(log_average_second_reduction_blob), nullptr, &m_log_average_second_reduction));

    OID3DBlob linear_exposure_computation_blob = compile_shader(shader_folder_path + L"Compute\\ReduceLogAverageLuminance.hlsl", "cs_5_0", "compute_linear_exposure");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_computation_blob), nullptr, &m_linear_exposure_computation));

    // Create buffers
    create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, max_groups_dispatched, &m_log_averages_SRV, &m_log_averages_UAV);
    create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, &m_linear_exposure_SRV, &m_linear_exposure_UAV);
}

OID3D11ShaderResourceView& LogAverageLuminance::compute_log_average(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                                                    ID3D11ShaderResourceView* pixels, unsigned int image_width) {
    return compute(context, constants, pixels, image_width, m_log_average_second_reduction);
}

OID3D11ShaderResourceView& LogAverageLuminance::compute_linear_exposure(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                                                        ID3D11ShaderResourceView* pixels, unsigned int image_width) {
    return compute(context, constants, pixels, image_width, m_linear_exposure_computation);
}

OID3D11ShaderResourceView& LogAverageLuminance::compute(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
    ID3D11ShaderResourceView* pixels, unsigned int image_width, OID3D11ComputeShader& second_reduction) {

    context.CSSetConstantBuffers(0, 1, &constants);

    context.CSSetUnorderedAccessViews(0, 1, &m_log_averages_UAV, 0u);
    context.CSSetShaderResources(0, 1, &pixels);
    context.CSSetShader(m_log_average_first_reduction, nullptr, 0u);
    unsigned int group_count_x = ceil_divide(image_width, group_width);
    context.Dispatch(min(group_count_x, max_groups_dispatched), 1, 1);

    context.CSSetUnorderedAccessViews(0, 1, &m_linear_exposure_UAV, 0u);
    context.CSSetShaderResources(0, 1, &m_log_averages_SRV);
    context.CSSetShader(second_reduction, nullptr, 0u);
    context.Dispatch(1, 1, 1);

    ID3D11UnorderedAccessView* null_UAV = nullptr;
    context.CSSetUnorderedAccessViews(0, 1, &null_UAV, 0u);

    return m_linear_exposure_SRV;
}

// ------------------------------------------------------------------------------------------------
// Exposure histogram
// ------------------------------------------------------------------------------------------------
ExposureHistogram::ExposureHistogram()
    : m_histogram_reduction(nullptr), m_histogram_SRV(nullptr), m_histogram_UAV(nullptr)
    , m_linear_exposure_computation(nullptr), m_linear_exposure_SRV(nullptr), m_linear_exposure_UAV(nullptr) { }

ExposureHistogram::ExposureHistogram(ID3D11Device1& device, const std::wstring& shader_folder_path) {
    // Create shaders.
    OID3DBlob reduce_exposure_histogram_blob = compile_shader(shader_folder_path + L"Compute\\ReduceExposureHistogram.hlsl", "cs_5_0", "reduce");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(reduce_exposure_histogram_blob), nullptr, &m_histogram_reduction));

    OID3DBlob linear_exposure_computation_blob = compile_shader(shader_folder_path + L"Compute\\ReduceExposureHistogram.hlsl", "cs_5_0", "compute_linear_exposure");
    THROW_ON_FAILURE(device.CreateComputeShader(UNPACK_BLOB_ARGS(linear_exposure_computation_blob), nullptr, &m_linear_exposure_computation));

    // Create buffers
    create_default_buffer(device, DXGI_FORMAT_R32_UINT, bin_count, &m_histogram_SRV, &m_histogram_UAV);
    create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, 1, &m_linear_exposure_SRV, &m_linear_exposure_UAV);
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

    ID3D11UnorderedAccessView* null_UAV = nullptr ;
    context.CSSetUnorderedAccessViews(0, 1, &null_UAV, 0u);

    return m_histogram_SRV;
}

OID3D11ShaderResourceView& ExposureHistogram::compute_linear_exposure(ID3D11DeviceContext1& context, ID3D11Buffer* constants,
                                                                      ID3D11ShaderResourceView* pixels, unsigned int image_width) {
    const unsigned int zeros[4] = { 0u, 0u, 0u, 0u };
    context.ClearUnorderedAccessViewUint(m_histogram_UAV, zeros);

    context.CSSetConstantBuffers(0, 1, &constants);
    context.CSSetShaderResources(0, 1, &pixels);
    ID3D11UnorderedAccessView* UAVs[2] = { m_histogram_UAV, m_linear_exposure_UAV };
    context.CSSetUnorderedAccessViews(0, 2, UAVs, 0u);

    context.CSSetShader(m_histogram_reduction, nullptr, 0u);
    unsigned int group_count_x = ceil_divide(image_width, group_width);
    context.Dispatch(group_count_x, 1, 1);

    // TODO Use histogram SRV instead of UAV.
    context.CSSetShader(m_linear_exposure_computation, nullptr, 0u);
    context.Dispatch(1, 1, 1);

    ID3D11UnorderedAccessView* null_UAVs[2] = { nullptr, nullptr };
    context.CSSetUnorderedAccessViews(0, 2, null_UAVs, 0u);

    return m_linear_exposure_SRV;
}

// ------------------------------------------------------------------------------------------------
// Tonemapper
// ------------------------------------------------------------------------------------------------
Tonemapper::Tonemapper()
    : m_fullscreen_VS(nullptr)
    , m_linear_tonemapping_PS(nullptr), m_uncharted2_tonemapping_PS(nullptr), m_filmic_tonemapping_PS(nullptr) { }

Tonemapper::Tonemapper(ID3D11Device1& device, const std::wstring& shader_folder_path) {

    m_host_constants.min_log_luminance = -4.0f;
    m_host_constants.max_log_luminance = 4.0f;
    m_host_constants.min_percentage = 0.7f;
    m_host_constants.max_percentage = 0.95f;
    THROW_ON_FAILURE(create_constant_buffer(device, m_host_constants, &m_constants /*, D3D11_USAGE_DEFAULT*/));

    m_log_average_luminance = LogAverageLuminance(device, shader_folder_path);
    m_exposure_histogram = ExposureHistogram(device, shader_folder_path);

    { // Setup shaders
        const std::wstring shader_filename = shader_folder_path + L"Tonemapping.hlsl";

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

        m_linear_tonemapping_PS = create_pixel_shader("linear_tonemapping_ps");
        m_uncharted2_tonemapping_PS = create_pixel_shader("uncharted2_tonemapping_ps");
        m_filmic_tonemapping_PS = create_pixel_shader("unreal4_tonemapping_ps");
    }
}

void Tonemapper::tonemap(ID3D11DeviceContext1& context, Tonemapping::Parameters parameters,
                         ID3D11ShaderResourceView* pixel_SRV, ID3D11RenderTargetView* backbuffer_RTV, 
                         int width, int height) {

    context.OMSetRenderTargets(1, &backbuffer_RTV, nullptr);

    // Compute exposure.
    auto& linear_exposure_SRV = m_exposure_histogram.compute_linear_exposure(context, m_constants, pixel_SRV, width);

    { // Tonemap and render into backbuffer.
        context.VSSetShader(m_fullscreen_VS, 0, 0);

        if (parameters.tonemapping.mode == Tonemapping::TonemappingMode::Linear)
            context.PSSetShader(m_linear_tonemapping_PS, 0, 0);
        else if (parameters.tonemapping.mode == Tonemapping::TonemappingMode::Uncharted2)
            context.PSSetShader(m_uncharted2_tonemapping_PS, 0, 0);
        else // parameters.tonemapping.mode == Tonemapping::TonemappingMode::Filmic
            context.PSSetShader(m_filmic_tonemapping_PS, 0, 0);

        ID3D11ShaderResourceView* srvs[2] = { pixel_SRV, linear_exposure_SRV };
        context.PSSetShaderResources(0, 2, srvs);

        context.Draw(3, 0);
    }
}

} // NS DX11Renderer
