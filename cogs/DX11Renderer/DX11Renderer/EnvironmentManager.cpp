// DirectX 11 environment manager.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Dx11Renderer/EnvironmentManager.h"
#include "Dx11Renderer/TextureManager.h"
#include "Dx11Renderer/Utils.h"

#include "Cogwheel/Assets/InfiniteAreaLight.h"
#include "Cogwheel/Math/RNG.h"
#include "Cogwheel/Scene/SceneRoot.h"

using namespace Cogwheel::Assets;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

namespace DX11Renderer {

//=================================================================================================
// Environment manager.
//=================================================================================================
EnvironmentManager::EnvironmentManager(ID3D11Device1& device, const std::wstring& shader_folder_path, TextureManager& textures)
    : m_textures(textures) {

    ID3D10Blob* vertex_shader_blob = compile_shader(shader_folder_path + L"EnvironmentMap.hlsl", "vs_5_0", "main_vs");
    HRESULT hr = device.CreateVertexShader(UNPACK_BLOB_ARGS(vertex_shader_blob), NULL, &m_vertex_shader);
    THROW_ON_FAILURE(hr);

    ID3D10Blob* pixel_shader_blob = compile_shader(shader_folder_path + L"EnvironmentMap.hlsl", "ps_5_0", "main_ps");
    hr = device.CreatePixelShader(UNPACK_BLOB_ARGS(pixel_shader_blob), NULL, &m_pixel_shader);
    THROW_ON_FAILURE(hr);

    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

    hr = device.CreateSamplerState(&sampler_desc, &m_sampler);
    THROW_ON_FAILURE(hr);
}

EnvironmentManager::~EnvironmentManager() {
    safe_release(&m_vertex_shader);
    safe_release(&m_pixel_shader);
    safe_release(&m_sampler);
    for (Environment env : m_envs) {
        safe_release(&env.srv);
        safe_release(&env.texture2D);
    }
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

        ID3D11RenderTargetView* backbuffer;
        ID3D11DepthStencilView* depth;
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

            env.texture_ID = scene.get_environment_map();

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
                RGB* pixel_data = new RGB[total_pixel_count];

                { // Compute mipmap pixels.
                    using namespace InfiniteAreaLightUtils;
                    RGB* next_pixels = pixel_data;
                    IBLConvolution* convolutions = new IBLConvolution[mipmap_count];
                    for (int m = 0; m < mipmap_count; ++m) {
                        convolutions[m].Width = env_width >> m;
                        convolutions[m].Height = env_height >> m;
                        convolutions[m].Roughness = m / (mipmap_count - 1.0f);
                        convolutions[m].sample_count = next_power_of_two(unsigned int (256 * convolutions[m].Roughness));
                        convolutions[m].Pixels = next_pixels;
                        next_pixels += convolutions[m].Width * convolutions[m].Height;
                    }

                    Convolute(light, convolutions, convolutions + mipmap_count);

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

                    R11G11B10_Float* compressed_pixels = new R11G11B10_Float[total_pixel_count];

                    RGB* next_rgb_pixels = pixel_data;
                    R11G11B10_Float* next_compressed_pixels = compressed_pixels;
                    for (unsigned int m = 0; m < tex_desc.MipLevels; ++m) {
                        int width = tex_desc.Width >> m, height = tex_desc.Height >> m;

                        tex_data[m].SysMemPitch = sizeof_dx_format(tex_desc.Format) * width;
                        tex_data[m].SysMemSlicePitch = tex_data[m].SysMemPitch * height;
                        tex_data[m].pSysMem = next_compressed_pixels;

                        RGB* rgb_pixels_end = next_rgb_pixels + width * height;
                        while (next_rgb_pixels < rgb_pixels_end) {
                            RGB p = *next_rgb_pixels++;
                            *next_compressed_pixels++ = R11G11B10_Float(p.r, p.g, p.b);
                        }
                    }

                    HRESULT hr = device.CreateTexture2D(&tex_desc, tex_data, &env.texture2D);
                    THROW_ON_FAILURE(hr);

                    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
                    srv_desc.Format = tex_desc.Format;
                    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                    srv_desc.Texture2D.MipLevels = tex_desc.MipLevels;
                    srv_desc.Texture2D.MostDetailedMip = 0;
                    hr = device.CreateShaderResourceView(env.texture2D, &srv_desc, &env.srv);
                    THROW_ON_FAILURE(hr);

                    delete[] compressed_pixels;
                }

                delete[] pixel_data;
            }
        }
    }
}

} // NS DX11Renderer
