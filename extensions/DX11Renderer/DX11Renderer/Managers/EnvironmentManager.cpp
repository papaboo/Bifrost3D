// DirectX 11 environment manager.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Dx11Renderer/Managers/EnvironmentManager.h"
#include "Dx11Renderer/Managers/ShaderManager.h"
#include "Dx11Renderer/Managers/TextureManager.h"
#include "Dx11Renderer/Utils.h"

#include "Bifrost/Assets/InfiniteAreaLight.h"
#include "Bifrost/Math/RNG.h"
#include "Bifrost/Scene/SceneRoot.h"

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace DX11Renderer::Managers {

//=================================================================================================
// Environment manager.
//=================================================================================================
EnvironmentManager::EnvironmentManager(ID3D11Device1& device, TextureManager& textures, const ShaderManager& shader_manager)
    : m_textures(textures) {

    OBlob vertex_shader_blob = shader_manager.compile_shader_from_file("EnvironmentMap.hlsl", "vs_5_0", "main_vs");
    THROW_DX11_ERROR(device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), nullptr, &m_vertex_shader));

    OBlob pixel_shader_blob = shader_manager.compile_shader_from_file("EnvironmentMap.hlsl", "ps_5_0", "main_ps");
    THROW_DX11_ERROR(device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), nullptr, &m_pixel_shader));

    OBlob convolution_shader_blob = shader_manager.compile_shader_from_file("IBLConvolution.hlsl", "cs_5_0", "MIS_convolute");
    THROW_DX11_ERROR(device.CreateComputeShader(UNPACK_BLOB_ARGS(convolution_shader_blob), nullptr, &m_convolution_shader));

    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

    THROW_DX11_ERROR(device.CreateSamplerState(&sampler_desc, &m_sampler));
}

bool EnvironmentManager::render(ID3D11DeviceContext1& render_context, int environment_ID) {

#if CHECK_IMPLICIT_STATE
    // Check that the screen space triangle will be rendered correctly.
    D3D11_PRIMITIVE_TOPOLOGY topology;
    render_context.IAGetPrimitiveTopology(&topology);
    always_assert(topology == D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Check that the environment can be rendered on top of the far plane.
    ID3D11DepthStencilState* depth_state;
    unsigned int unused;
    render_context.OMGetDepthStencilState(&depth_state, &unused);
    D3D11_DEPTH_STENCIL_DESC depth_desc;
    depth_state->GetDesc(&depth_desc);
    always_assert(depth_desc.DepthFunc == D3D11_COMPARISON_LESS_EQUAL || depth_desc.DepthFunc == D3D11_COMPARISON_NEVER);
#endif

    Environment& env = m_envs[environment_ID];
    if (env.texture_ID != 0) {
        // Set vertex and pixel shaders.
        render_context.VSSetShader(m_vertex_shader, 0, 0);
        render_context.PSSetShader(m_pixel_shader, 0, 0);

        render_context.PSSetShaderResources(0, 1, &env.srv);
        render_context.PSSetSamplers(0, 1, &m_sampler);

        render_context.Draw(3, 0);

        return true;
    } else {
        // Bind white environment instead.
        render_context.PSSetShaderResources(0, 1, &m_textures.white_texture().srv);
        render_context.PSSetSamplers(0, 1, &m_textures.white_texture().sampler);

        ORenderTargetView backbuffer;
        ODepthStencilView depth;
        render_context.OMGetRenderTargets(1, &backbuffer, &depth);

        render_context.ClearRenderTargetView(backbuffer, &env.tint.x);

        return false;
    }
}

void EnvironmentManager::handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& device_context) {
    if (!SceneRoots::get_changed_scenes().is_empty()) {
        if (m_envs.size() < SceneRoots::capacity())
            m_envs.resize(SceneRoots::capacity());

        for (SceneRoot scene : SceneRoots::get_changed_scenes()) {
            Environment& env = m_envs[scene.get_ID()];

            RGBA tint = scene.get_environment_tint();
            env.tint = { tint.r, tint.g, tint.b, tint.a };

            env.texture_ID = scene.get_environment_map().get_ID();

            if (env.texture_ID != 0) {

                InfiniteAreaLight& light = *scene.get_environment_light();

                int env_width = max(light.get_width(), 256u);
                int env_height = max(light.get_height(), 128u);

                // Compute mipmap count.
                int mipmap_count = 0;
                int total_pixel_count = 0;
                while (env_width >> mipmap_count > 16 || env_height >> mipmap_count > 16) {
                    total_pixel_count += (env_width >> mipmap_count) * (env_height >> mipmap_count);
                    ++mipmap_count;
                }

                bool cpu_estimation = false;
                if (cpu_estimation)
                {
                    R11G11B10_Float* pixel_data = new R11G11B10_Float[total_pixel_count];

                    { // Compute mipmap pixels.
                        using namespace InfiniteAreaLightUtils;
                        R11G11B10_Float* next_pixels = pixel_data;
                        IBLConvolution<R11G11B10_Float>* convolutions = new IBLConvolution<R11G11B10_Float>[mipmap_count];
                        for (int m = 0; m < mipmap_count; ++m) {
                            convolutions[m].Width = env_width >> m;
                            convolutions[m].Height = env_height >> m;
                            convolutions[m].Roughness = m / (mipmap_count - 1.0f);
                            convolutions[m].sample_count = next_power_of_two(unsigned int(256 * convolutions[m].Roughness));
                            convolutions[m].Pixels = next_pixels;
                            next_pixels += convolutions[m].Width * convolutions[m].Height;
                        }

                        convolute(light, convolutions, convolutions + mipmap_count,
                            [](RGB c) -> R11G11B10_Float { return R11G11B10_Float(c.r, c.g, c.b); });

                        delete[] convolutions;
                    }

                    { // Generate texture and srv.
                        D3D11_TEXTURE2D_DESC tex_desc = {};
                        tex_desc.Width = env_width;
                        tex_desc.Height = env_height;
                        tex_desc.MipLevels = mipmap_count;
                        tex_desc.ArraySize = 1;
                        tex_desc.Format = DXGI_FORMAT_R11G11B10_FLOAT;
                        tex_desc.SampleDesc.Count = 1;
                        tex_desc.SampleDesc.Quality = 0;
                        tex_desc.Usage = D3D11_USAGE_IMMUTABLE;
                        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

                        D3D11_SUBRESOURCE_DATA* tex_data = new D3D11_SUBRESOURCE_DATA[tex_desc.MipLevels];

                        R11G11B10_Float* next_pixels = pixel_data;
                        for (unsigned int m = 0; m < tex_desc.MipLevels; ++m) {
                            int width = tex_desc.Width >> m, height = tex_desc.Height >> m;

                            tex_data[m].SysMemPitch = sizeof_dx_format(tex_desc.Format) * width;
                            tex_data[m].SysMemSlicePitch = tex_data[m].SysMemPitch * height;
                            tex_data[m].pSysMem = next_pixels;

                            next_pixels += width * height;
                        }

                        THROW_DX11_ERROR(device.CreateTexture2D(&tex_desc, tex_data, &env.texture2D));
                    }

                    delete[] pixel_data;
                } else {
                    // GPU convolution.

                    // Create and upload light samples and PDF for MIS.
                    OShaderResourceView per_pixel_PDF_SRV = nullptr;
                    OShaderResourceView light_samples_SRV = nullptr;
                    {
                        { // Per pixel PDF.

                            float* per_pixel_PDF_data = new float[light.get_width() * light.get_height()];
                            InfiniteAreaLightUtils::reconstruct_solid_angle_PDF_sans_sin_theta(light, per_pixel_PDF_data);

                            create_texture_2D(device, DXGI_FORMAT_R32_FLOAT, per_pixel_PDF_data, light.get_width(), light.get_height(), D3D11_USAGE_IMMUTABLE, &per_pixel_PDF_SRV);

                            delete[] per_pixel_PDF_data;
                        }

                        { // Light samples.
                            const unsigned int light_sample_count = 2048;

                            D3D11_BUFFER_DESC sample_buffer_desc = {};
                            sample_buffer_desc.Usage = D3D11_USAGE_IMMUTABLE;
                            sample_buffer_desc.ByteWidth = sizeof(LightSample) * light_sample_count;
                            sample_buffer_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                            sample_buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                            sample_buffer_desc.StructureByteStride = sizeof(LightSample);

                            LightSample* light_samples = new LightSample[light_sample_count];
                            #pragma omp parallel for schedule(dynamic, 16)
                            for (int i = 0; i < light_sample_count; ++i)
                                light_samples[i] = light.sample(RNG::sample02(i));

                            D3D11_SUBRESOURCE_DATA sample_resource_data = {};
                            sample_resource_data.pSysMem = light_samples;
                            sample_resource_data.SysMemPitch = sizeof(LightSample) * light_sample_count;
                            OBuffer light_samples_buffer = nullptr;
                            THROW_DX11_ERROR(device.CreateBuffer(&sample_buffer_desc, &sample_resource_data, &light_samples_buffer));

                            delete[] light_samples;

                            THROW_DX11_ERROR(device.CreateShaderResourceView(light_samples_buffer, nullptr, &light_samples_SRV));
                        }
                    }

                    { // Create environment texture and upload specular base map.

                        D3D11_TEXTURE2D_DESC tex_desc = {};
                        tex_desc.Width = env_width;
                        tex_desc.Height = env_height;
                        tex_desc.MipLevels = mipmap_count;
                        tex_desc.ArraySize = 1;
                        tex_desc.Format = DXGI_FORMAT_R11G11B10_FLOAT;
                        tex_desc.SampleDesc.Count = 1;
                        tex_desc.SampleDesc.Quality = 0;
                        tex_desc.Usage = D3D11_USAGE_DEFAULT;
                        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

                        THROW_DX11_ERROR(device.CreateTexture2D(&tex_desc, nullptr, &env.texture2D));

                        R11G11B10_Float* pixels = new R11G11B10_Float[env_width* env_height];
                        #pragma omp parallel for schedule(dynamic, 16)
                        for (int i = 0; i < env_width * env_height; ++i) {
                            int x = i % env_width, y = i / env_width;
                            Vector2f uv = Vector2f((x + 0.5f) / env_width, (y + 0.5f) / env_height);
                            RGB c = sample2D(light.get_texture_ID(), uv).rgb();
                            pixels[x + y * env_width] = R11G11B10_Float(c.r, c.g, c.b);
                        }

                        device_context.UpdateSubresource(env.texture2D, 0, nullptr, pixels, sizeof(R11G11B10_Float) * env_width, 0);
                        delete[] pixels;
                    }

                    { // IBL mip level convolution.

                        // Constant buffer.
                        struct ConvolutionConstants {
                            float rcp_mipmap_count_minus_one; // 1.0 / (mipmap_count - 1)
                            float rcp_smallest_width; // 1.0 / width of the smallest mipmap
                            unsigned int max_sample_count;
                            float __padding;
                        };

                        int smallest_width = env_width >> (mipmap_count - 1);
                        ConvolutionConstants constants = { 1.0f / float(mipmap_count - 1), 1.0f / smallest_width, 1024u };
                        OBuffer constant_buffer;
                        THROW_DX11_ERROR(create_constant_buffer(device, constants, &constant_buffer));

                        // Create UAVs for the mip levels.
                        OUnorderedAccessView* mip_level_UAVs = new OUnorderedAccessView[mipmap_count - 1];

                        for (int m = 1; m < mipmap_count; ++m) {
                            D3D11_UNORDERED_ACCESS_VIEW_DESC mip_level_UAV_desc = {};
                            mip_level_UAV_desc.Format = DXGI_FORMAT_R11G11B10_FLOAT;
                            mip_level_UAV_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
                            mip_level_UAV_desc.Texture2D.MipSlice = m;

                            THROW_DX11_ERROR(device.CreateUnorderedAccessView(env.texture2D, &mip_level_UAV_desc, &mip_level_UAVs[m - 1]));
                        }

                        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
                        srv_desc.Format = DXGI_FORMAT_R11G11B10_FLOAT;
                        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                        srv_desc.Texture2D.MipLevels = 1;
                        srv_desc.Texture2D.MostDetailedMip = 0;
                        OShaderResourceView env_SRV;
                        THROW_DX11_ERROR(device.CreateShaderResourceView(env.texture2D, &srv_desc, &env_SRV));

                        // Launch kernels.
                        device_context.CSSetShader(m_convolution_shader, nullptr, 0);
                        device_context.CSSetConstantBuffers(0, 1, &constant_buffer);
                        ID3D11ShaderResourceView* SRVs[] = { env_SRV, per_pixel_PDF_SRV, light_samples_SRV };
                        device_context.CSSetShaderResources(0, 3, SRVs);
                        device_context.CSSetSamplers(0, 1, &m_sampler);

                        for (int m = 1; m < mipmap_count; ++m) {
                            device_context.CSSetUnorderedAccessViews(0, 1, &mip_level_UAVs[m - 1], nullptr);
                            int width = env_width >> m, height = env_height >> m;
                            device_context.Dispatch(ceil_divide(width, 16), ceil_divide(height, 16), 1);
                        }

                        { // Cleanup.
                            // Unbind resource UAVs and other resources, so they can be released and the texture can be used as input.
                            ID3D11ShaderResourceView* null_SRVs[] = { nullptr, nullptr, nullptr };
                            device_context.CSSetShaderResources(0, 3, null_SRVs);
                            ID3D11UnorderedAccessView* null_UAV = nullptr;
                            device_context.CSSetUnorderedAccessViews(0, 1, &null_UAV, nullptr);
                            ID3D11Buffer* null_buffer = nullptr;
                            device_context.CSSetConstantBuffers(0, 1, &null_buffer);

                            // Release UAV for mip levels.
                            delete[] mip_level_UAVs;
                        }
                    }
                }

                THROW_DX11_ERROR(device.CreateShaderResourceView(env.texture2D, nullptr, &env.srv));
            }
        }
    }
}

} // NS DX11Renderer::Managers
