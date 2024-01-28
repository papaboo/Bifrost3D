// DirectX 11 texture manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/TextureManager.h>
#include <DX11Renderer/Utils.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Texture.h>

using namespace Bifrost::Assets;

namespace DX11Renderer {

TextureManager::TextureManager(ID3D11Device1& device) {
    // Initialize null image and texture.
    m_images.resize(Images::capacity());
    m_textures.resize(Textures::capacity());

    // Create default textures.
    static auto create_color_texture = [](ID3D11Device1& device, unsigned char pixel[4]) -> DefaultTexture {
        DefaultTexture tex;
        create_texture_2D(device, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, pixel, 1, 1, &tex.srv);

        D3D11_SAMPLER_DESC sampler_desc = {};
        sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sampler_desc.MinLOD = 0;
        sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

        THROW_DX11_ERROR(device.CreateSamplerState(&sampler_desc, &tex.sampler));

        return tex;
    };

    unsigned char white[4] = { 255, 255, 255, 255 };
    m_white_texture = create_color_texture(device, white);
}

OSamplerState TextureManager::create_clamped_linear_sampler(ID3D11Device1& device) {
    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;

    OSamplerState linear_sampler;
    THROW_DX11_ERROR(device.CreateSamplerState(&sampler_desc, &linear_sampler));

    return linear_sampler;
}

void TextureManager::handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& device_context) {
    { // Image updates.
        if (!Images::get_changed_images().is_empty()) {
            if (m_images.size() < Images::capacity())
                m_images.resize(Images::capacity());

            static auto to_DX_format = [](PixelFormat format) -> DXGI_FORMAT {
                switch (format) {
                case PixelFormat::Alpha8:
                    return DXGI_FORMAT_A8_UNORM;
                case PixelFormat::RGB24:
                    return DXGI_FORMAT_UNKNOWN;
                case PixelFormat::RGBA32:
                    return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                case PixelFormat::RGB_Float:
                    return DXGI_FORMAT_R32G32B32_FLOAT;
                case PixelFormat::RGBA_Float:
                    return DXGI_FORMAT_R32G32B32A32_FLOAT;
                case PixelFormat::Unknown:
                default:
                    return DXGI_FORMAT_UNKNOWN;
                }
            };

            static auto rgb24_to_rgba32 = [](unsigned char* pixels, int pixel_count) -> unsigned char* {
                unsigned char* new_pixels = new unsigned char[pixel_count * 4];
                unsigned char* pixel_end = pixels + pixel_count * 3;
                while (pixels < pixel_end) {
                    *new_pixels++ = *pixels++;
                    *new_pixels++ = *pixels++;
                    *new_pixels++ = *pixels++;
                    *new_pixels++ = 255;
                }
                return new_pixels - pixel_count * 4;;
            };

            for (ImageID image_ID : Images::get_changed_images()) {
                Dx11Image& dx_image = m_images[image_ID];

                if (Images::get_changes(image_ID).is_set(Images::Change::Destroyed)) {
                    dx_image.srv.release();

                } else if (Images::get_changes(image_ID).is_set(Images::Change::Created)) {
                    dx_image.srv.release(); // Explicit release because the resource pointer is directly modified below.

                    Image image = image_ID;
                    D3D11_TEXTURE2D_DESC tex_desc = {};
                    tex_desc.Width = image.get_width();
                    tex_desc.Height = image.get_height();
                    tex_desc.MipLevels = image.get_mipmap_count();
                    tex_desc.ArraySize = 1;
                    tex_desc.Format = to_DX_format(image.get_pixel_format());
                    tex_desc.SampleDesc.Count = 1;
                    tex_desc.SampleDesc.Quality = 0;
                    tex_desc.Usage = D3D11_USAGE_DEFAULT;
                    tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

                    D3D11_SUBRESOURCE_DATA resource_data;
                    resource_data.pSysMem = image.get_pixels();

                    // RGB24 not supported. Instead convert it to RGBA32.
                    if (image.get_pixel_format() == PixelFormat::RGB24) {
                        tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                        resource_data.pSysMem = rgb24_to_rgba32((unsigned char*)resource_data.pSysMem, image.get_pixel_count());
                    }

                    if (tex_desc.Format != DXGI_FORMAT_UNKNOWN) {
                        resource_data.SysMemPitch = sizeof_dx_format(tex_desc.Format) *  image.get_width();

                        bool generate_mipmaps = image.is_mipmapable();

                        OTexture2D texture;
                        HRESULT hr;
                        if (generate_mipmaps) {
                            // Additional mipmap generation settings.
                            tex_desc.MipLevels = 0;
                            tex_desc.BindFlags |= D3D11_BIND_RENDER_TARGET;
                            tex_desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

                            // If a texture has autogenerated mipmaps, then CreateTexture2D expects the data to contain the full mipchain, 
                            // so we have to upload the data separately.
                            hr = device.CreateTexture2D(&tex_desc, nullptr, &texture);
                            device_context.UpdateSubresource(texture, 0, nullptr, resource_data.pSysMem, resource_data.SysMemPitch, 0);
                        } else
                            hr = device.CreateTexture2D(&tex_desc, &resource_data, &texture);

                        if (FAILED(hr))
                            printf("Could not create the texture '%s'.\n", image.get_name().c_str());

                        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
                        srv_desc.Format = tex_desc.Format;
                        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                        srv_desc.Texture2D.MipLevels = generate_mipmaps ? -1 : tex_desc.MipLevels;
                        srv_desc.Texture2D.MostDetailedMip = 0;
                        hr = device.CreateShaderResourceView(texture, &srv_desc, &dx_image.srv);
                        if (FAILED(hr))
                            printf("Could not create the shader resource view for texture '%s'.\n", image.get_name().c_str());

                        if (generate_mipmaps)
                            device_context.GenerateMips(dx_image.srv);
                    }

                    // Cleanup temporary pixel data.
                    if (resource_data.pSysMem != image.get_pixels())
                        delete[] resource_data.pSysMem;

                } else if (Images::get_changes(image_ID).is_set(Images::Change::PixelsUpdated))
                    assert(!"Pixel update not implemented yet.\n");
            }
        }
    }

    { // Texture/sampler updates.
        if (!Textures::get_changed_textures().is_empty()) {
            if (m_textures.size() < Textures::capacity())
                m_textures.resize(Textures::capacity());

            for (TextureID tex_ID : Textures::get_changed_textures()) {
                Dx11Texture& dx_tex = m_textures[tex_ID];

                if (Textures::get_changes(tex_ID) == Textures::Change::Destroyed) {
                    dx_tex.sampler.release();

                } else if (Textures::get_changes(tex_ID) & Textures::Change::Created) {
                    dx_tex.sampler.release(); // Explicit release, because the resource will be overwritten below.

                    Texture texture = tex_ID;

                    static auto to_DX_filtermode = [](MagnificationFilter mag_filter, MinificationFilter min_filter) -> D3D11_FILTER {
                        if (mag_filter == MagnificationFilter::None) {
                            switch (min_filter) {
                            case MinificationFilter::None:
                                return D3D11_FILTER_MIN_MAG_MIP_POINT;
                            case MinificationFilter::Linear:
                                return D3D11_FILTER_MIN_LINEAR_MAG_MIP_POINT;
                            case MinificationFilter::Trilinear:
                                return D3D11_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR;
                            }
                        } else { // mag_filter == MagnificationFilter::Linear
                            switch (min_filter) {
                            case MinificationFilter::None:
                                return D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
                            case MinificationFilter::Linear:
                                return D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                            case MinificationFilter::Trilinear:
                                return D3D11_FILTER_MIN_MAG_MIP_LINEAR;
                            }
                        }
                        return D3D11_FILTER_MIN_MAG_MIP_LINEAR;
                    };

                    static auto to_DX_wrapmode = [](WrapMode wm) -> D3D11_TEXTURE_ADDRESS_MODE {
                        switch (wm) {
                        case WrapMode::Clamp:
                            return D3D11_TEXTURE_ADDRESS_CLAMP;
                        case WrapMode::Repeat:
                            return D3D11_TEXTURE_ADDRESS_WRAP;
                        }
                        return D3D11_TEXTURE_ADDRESS_MIRROR;
                    };

                    D3D11_SAMPLER_DESC desc = {};
                    desc.Filter = to_DX_filtermode(texture.get_magnification_filter(), texture.get_minification_filter());
                    desc.AddressU = to_DX_wrapmode(texture.get_wrapmode_U());
                    desc.AddressV = to_DX_wrapmode(texture.get_wrapmode_V());
                    desc.AddressW = to_DX_wrapmode(texture.get_wrapmode_W());
                    desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
                    desc.MinLOD = 0;
                    desc.MaxLOD = D3D11_FLOAT32_MAX;

                    HRESULT hr = device.CreateSamplerState(&desc, &dx_tex.sampler);
                    if (FAILED(hr))
                        printf("Could not create the sampler.\n");

                    dx_tex.image = &m_images[texture.get_image().get_ID()];
                }
            }
        }
    }
}

} // NS DX11Renderer