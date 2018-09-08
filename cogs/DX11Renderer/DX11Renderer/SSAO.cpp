// DirectX 11 renderer screen space ambient occlusion implementations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <DX11Renderer/SSAO.h>
#include <DX11Renderer/Utils.h>

#include <Cogwheel/Math/RNG.h>

namespace DX11Renderer {
namespace SSAO {

struct FilterConstants {
    int pixel_offset;
    int __padding;
    int2 axis;
};

inline void create_box_filter_constants(ID3D11Device1& device, OBuffer* constants) {
    FilterConstants host_constants = { 5 };
    THROW_DX11_ERROR(create_constant_buffer(device, host_constants, &constants[0]));
    host_constants.pixel_offset = 3;
    THROW_DX11_ERROR(create_constant_buffer(device, host_constants, &constants[1]));
    host_constants.pixel_offset = 1;
    THROW_DX11_ERROR(create_constant_buffer(device, host_constants, &constants[2]));
}

// ------------------------------------------------------------------------------------------------
// Bilateral blur for SSAO.
// ------------------------------------------------------------------------------------------------
BilateralBlur::BilateralBlur(ID3D11Device1& device, const std::wstring& shader_folder_path, FilterType type)
    : m_type(type), m_width(0), m_height(0), m_intermediate_RTV(nullptr), m_intermediate_SRV(nullptr) {

    OBlob vertex_shader_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "vs_5_0", "main_vs");
    THROW_DX11_ERROR(device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_vertex_shader));

    if (type == FilterType::Cross) {
        m_support = 0;
        OBlob filter_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "ps_5_0", "BilateralBlur::cross_filter_ps");
        THROW_DX11_ERROR(device.CreatePixelShader(UNPACK_BLOB_ARGS(filter_blob), nullptr, &m_filter_shader));
        create_constant_buffer(device, sizeof(FilterConstants), &m_constants[0]);
        create_constant_buffer(device, sizeof(FilterConstants), &m_constants[1]);
    } else {
        m_support = 9;
        OBlob filter_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "ps_5_0", "BilateralBlur::box_filter_ps");
        THROW_DX11_ERROR(device.CreatePixelShader(UNPACK_BLOB_ARGS(filter_blob), nullptr, &m_filter_shader));
        create_box_filter_constants(device, m_constants);
    }
}

OShaderResourceView& BilateralBlur::apply(ID3D11DeviceContext1& context, ORenderTargetView& ao_RTV, OShaderResourceView& ao_SRV, int width, int height, int support) {
    if (m_width < width || m_height < height) {
        m_width = std::max(m_width, width);
        m_height = std::max(m_height, height);

        m_intermediate_SRV.release();
        m_intermediate_RTV.release();

        // Resize backbuffer
        ODevice1 device = get_device1(context);
        create_texture_2D(device, DXGI_FORMAT_R16G16B16A16_FLOAT, m_width, m_height, &m_intermediate_SRV, nullptr, &m_intermediate_RTV);
    }

    if (m_support != support && m_type == FilterType::Cross) {
        m_support = support;
        FilterConstants pass1_constants = { support, 0, 0, 1 };
        context.UpdateSubresource(m_constants[0], 0u, nullptr, &pass1_constants, sizeof(FilterConstants), 0u);
        FilterConstants pass2_constants = { support, 0, 1, 0 };
        context.UpdateSubresource(m_constants[1], 0u, nullptr, &pass2_constants, sizeof(FilterConstants), 0u);
    }

    // Need to grab the normal and depth buffers before OMSetRenderTarget clears them.
    ID3D11ShaderResourceView* srvs[2];
    context.PSGetShaderResources(0, 2, srvs);

    context.VSSetShader(m_vertex_shader, 0, 0);
    context.PSSetShader(m_filter_shader, 0, 0);

    int passes = m_type == FilterType::Box ? MAX_PASSES : 2;
    int i;
    for (i = 0; i < passes; ++i) {
        auto& rtv = (i % 2) == 0 ? m_intermediate_RTV : ao_RTV;
        context.OMSetRenderTargets(1, &rtv, nullptr);
        context.PSSetShaderResources(0, 2, srvs);
        auto& srv = (i % 2) == 0 ? ao_SRV : m_intermediate_SRV;
        context.PSSetShaderResources(2, 1, &srv);
        context.PSSetConstantBuffers(2, 1, &m_constants[i]);
        context.Draw(3, 0);
    }

    srvs[0]->Release();
    srvs[1]->Release();

    return (i % 2) == 0 ? ao_SRV : m_intermediate_SRV;
}

// ------------------------------------------------------------------------------------------------
// The Alchemy screen-space ambient obscurance algorithm.
// http://casual-effects.com/research/McGuire2011AlchemyAO/index.html
// ------------------------------------------------------------------------------------------------
struct SsaoConstants {
    SsaoSettings settings;
    float __padding;
    float2 g_buffer_size;
    float2 recip_g_buffer_viewport_size;
    float2 g_buffer_max_uv;
    int2 g_buffer_to_ao_index_offset;
    float2 ao_buffer_size;
};

const float AlchemyAO::max_screen_space_radius = 0.25f;

AlchemyAO::AlchemyAO(ID3D11Device1& device, const std::wstring& shader_folder_path)
    : m_width(0), m_height(0), m_SSAO_RTV(nullptr), m_SSAO_SRV(nullptr) {

    using namespace Cogwheel::Math;

    THROW_DX11_ERROR(create_constant_buffer(device, sizeof(SsaoConstants), &m_constants));
    THROW_DX11_ERROR(create_constant_buffer(device, sizeof(Vector2f) * m_samples.capacity, &m_samples.buffer));

    OBlob vertex_shader_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "vs_5_0", "main_vs");
    THROW_DX11_ERROR(device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_vertex_shader));

    OBlob linearize_depth_shader_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "ps_5_0", "linearize_depth_ps");
    THROW_DX11_ERROR(device.CreatePixelShader(UNPACK_BLOB_ARGS(linearize_depth_shader_blob), nullptr, &m_depth.pixel_shader));

    OBlob ao_shader_blob = compile_shader(shader_folder_path + L"SSAO.hlsl", "ps_5_0", "alchemy_ps");
    THROW_DX11_ERROR(device.CreatePixelShader(UNPACK_BLOB_ARGS(ao_shader_blob), nullptr, &m_pixel_shader));

    m_filter = BilateralBlur(device, shader_folder_path, BilateralBlur::FilterType::Cross);

    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
    THROW_DX11_ERROR(device.CreateSamplerState(&sampler_desc, &m_trilinear_sampler));
}

int2 AlchemyAO::compute_g_buffer_to_ao_index_offset(Cogwheel::Math::Recti viewport) const {
    return { get_margin() - viewport.x, get_margin() - viewport.y };
}

void AlchemyAO::resize_ao_buffer(ID3D11DeviceContext1& context, int ssao_width, int ssao_height) {
    if (m_width < ssao_width || m_height < ssao_height) {
        m_width = std::max(m_width, ssao_width);
        m_height = std::max(m_height, ssao_height);

        ODevice1 device = get_device1(context);

        // Resize backbuffer
        m_SSAO_SRV.release();
        m_SSAO_RTV.release();
        create_texture_2D(device, DXGI_FORMAT_R16G16B16A16_FLOAT, m_width, m_height, &m_SSAO_SRV, nullptr, &m_SSAO_RTV);
    }
}

void AlchemyAO::resize_depth_buffer(ID3D11DeviceContext1& context, unsigned int camera_ID, int depth_width, int depth_height) {
    if (m_depth.per_camera.size() <= camera_ID + 1)
        m_depth.per_camera.resize(camera_ID + 1);
    auto& camera_depth = m_depth.per_camera[camera_ID];

    if (camera_depth.width != depth_width || camera_depth.height != depth_height) {
        camera_depth.width = depth_width;
        camera_depth.height = depth_height;
        
        ODevice1 device = get_device1(context);

        camera_depth.SRV.release();
        camera_depth.RTV.release();

        D3D11_TEXTURE2D_DESC tex_desc = {};
        tex_desc.Width = camera_depth.width;
        tex_desc.Height = camera_depth.height;
        tex_desc.MipLevels = 0;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_R16_FLOAT;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.SampleDesc.Quality = 0;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        tex_desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

        OTexture2D texture;
        THROW_DX11_ERROR(device->CreateTexture2D(&tex_desc, nullptr, &texture));

        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
        srv_desc.Format = tex_desc.Format;
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = -1;
        srv_desc.Texture2D.MostDetailedMip = 0;
        THROW_DX11_ERROR(device->CreateShaderResourceView(texture, &srv_desc, &camera_depth.SRV));

        THROW_DX11_ERROR(device->CreateRenderTargetView(texture, nullptr, &camera_depth.RTV));

        camera_depth.mip_count = 1;
        while (depth_width >> camera_depth.mip_count > 0 || depth_height >> camera_depth.mip_count > 0)
            ++camera_depth.mip_count;
    }
}

OShaderResourceView& AlchemyAO::apply(ID3D11DeviceContext1& context, unsigned int camera_ID, OShaderResourceView& normals, OShaderResourceView& depth,
                                      int2 g_buffer_size, Cogwheel::Math::Recti viewport, SsaoSettings settings) {

    int ssao_width = viewport.width + 2 * get_margin();
    int ssao_height = viewport.height + 2 * get_margin();
    unsigned int occlusion_sample_count = std::min(settings.sample_count, m_samples.capacity);
    int2 g_buffer_viewport_size = { viewport.width + 2 * viewport.x, viewport.height + 2 * viewport.y };

    resize_ao_buffer(context, ssao_width, ssao_height);

    resize_depth_buffer(context, camera_ID, g_buffer_viewport_size.x, g_buffer_viewport_size.y);
    auto& camera_depth = m_depth.per_camera[camera_ID];

    { // Contants 
        SsaoConstants constants;
        constants.settings = settings;
        constants.settings.sample_count = occlusion_sample_count;
        constants.settings.intensity_scale *= 2.0f / constants.settings.sample_count;
        constants.settings.depth_filtering_percentage *= camera_depth.mip_count / max_screen_space_radius; // Convert filtering percentage to mip level scale.

        constants.g_buffer_size = { float(g_buffer_size.x), float(g_buffer_size.y) };
        constants.recip_g_buffer_viewport_size = { 1.0f / g_buffer_viewport_size.x, 1.0f / g_buffer_viewport_size.y };
        constants.g_buffer_max_uv = { float(g_buffer_viewport_size.x) / g_buffer_size.x, float(g_buffer_viewport_size.y) / g_buffer_size.y };
        constants.g_buffer_to_ao_index_offset = compute_g_buffer_to_ao_index_offset(viewport);
        constants.ao_buffer_size = { float(ssao_width), float(ssao_height) };
        constants.settings.normal_std_dev = 0.5f / (constants.settings.normal_std_dev * constants.settings.normal_std_dev);
        constants.settings.plane_std_dev = 0.5f / (constants.settings.plane_std_dev * constants.settings.plane_std_dev);
        context.UpdateSubresource(m_constants, 0u, nullptr, &constants, 0u, 0u);
    }

    if (m_samples.size != occlusion_sample_count || m_samples.falloff != settings.falloff) {
        // Update the samples buffer.
        using namespace Cogwheel::Math;

        m_samples.size = occlusion_sample_count;
        m_samples.falloff = settings.falloff;

        static auto cosine_disk_sampling = [](Vector2f sample_uv) -> Vector2f {
            float r = sample_uv.x;
            float theta = 2.0f * PI<float>() * sample_uv.y;
            return r * Vector2f(cos(theta), sin(theta));
        };

        Vector2f samples[Samples::capacity];
        for (unsigned int i = 0; i < m_samples.size; ++i)
            samples[i] = cosine_disk_sampling(RNG::sample02(i + 1)); // Drop the first sample as it is (0, 0), e.g centered on the pixel being sampled.

        // Sort samples based on distance from center pixel to improve mipmap cache coherence.
        std::sort(samples, samples + m_samples.size, [](Vector2f lhs, Vector2f rhs) -> bool { return magnitude(lhs) < magnitude(rhs); });

        context.UpdateSubresource(m_samples.buffer, 0u, nullptr, samples, 0u, 0u);
    }

    ID3D11Buffer* constant_buffers[] = { m_constants, m_samples.buffer };
    context.VSSetConstantBuffers(0, 1, constant_buffers);
    context.PSSetConstantBuffers(0, 2, constant_buffers);

    // Compute linear depth and filter.
    // Assumes that the G-buffer viewport is set // TODO Assert on viewport dimensions!
    context.OMSetRenderTargets(1, &camera_depth.RTV, nullptr);
    context.PSSetShaderResources(1, 1, &depth);
    context.VSSetShader(m_vertex_shader, 0, 0);
    context.PSSetShader(m_depth.pixel_shader, 0, 0);
    context.Draw(3, 0);

    if (settings.depth_filtering_percentage != 0.0f)
        context.GenerateMips(camera_depth.SRV);

    // Grab old viewport.
    // Assumes only one viewport is used. If we start using more then it may just be easier to bite the bullet and move to compute (which turned out to be slower than pixel shaders at first try)
    unsigned int previous_viewport_count = 1u;
    D3D11_VIEWPORT previous_viewport;
    context.RSGetViewports(&previous_viewport_count, &previous_viewport);

    // Setup viewport to fit the SSAO margins.
    D3D11_VIEWPORT ao_viewport;
    ao_viewport.TopLeftX = ao_viewport.TopLeftY = 0.0f;
    ao_viewport.Width = float(ssao_width);
    ao_viewport.Height = float(ssao_height);
    ao_viewport.MinDepth = 0.0f;
    ao_viewport.MaxDepth = 1.0f;
    context.RSSetViewports(1, &ao_viewport);

    // Compute SSAO.
    context.OMSetRenderTargets(1, &m_SSAO_RTV, nullptr);
    ID3D11ShaderResourceView* SRVs[2] = { normals, camera_depth.SRV };
    context.PSSetSamplers(1, 1, &m_trilinear_sampler);
    context.PSSetShaderResources(0, 2, SRVs);
    context.PSSetShader(m_pixel_shader, 0, 0);
    context.Draw(3, 0);

    // Filter
    OShaderResourceView& ao_SRV = settings.filter_support > 0 ?
        m_filter.apply(context, m_SSAO_RTV, m_SSAO_SRV, ssao_width, ssao_height, settings.filter_support) :
        m_SSAO_SRV;

    // Unbind SSAO_RTV
    ID3D11RenderTargetView* null_RTV = nullptr;
    context.OMSetRenderTargets(1, &null_RTV, nullptr);

    // Reset the viewport.
    context.RSSetViewports(1, &previous_viewport);

    return ao_SRV;
}

OShaderResourceView& AlchemyAO::apply_none(ID3D11DeviceContext1& context, Cogwheel::Math::Recti viewport) {
    int ssao_width = viewport.width + 2 * get_margin();
    int ssao_height = viewport.height + 2 * get_margin();

    resize_ao_buffer(context, ssao_width, ssao_height);

    float cleared_ssao[4] = { 1, 0, 0, 0 };
    context.ClearView(m_SSAO_RTV, cleared_ssao, nullptr, 0);

    return m_SSAO_SRV;
}

} // NS SSAO
} // NS DX11Renderer