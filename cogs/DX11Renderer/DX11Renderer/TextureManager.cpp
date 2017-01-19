// DirectX 11 texture manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <DX11Renderer/TextureManager.h>
#include <DX11Renderer/Utils.h>

#include <Cogwheel/Assets/Image.h>
#include <Cogwheel/Assets/Texture.h>

using namespace Cogwheel::Assets;

namespace DX11Renderer {

TextureManager::TextureManager(ID3D11Device& device) {
    // Initialize null image and texture.
    m_images.resize(Images::capacity());
    m_textures.resize(Textures::capacity());
    m_images[0] = {};
    m_textures[0] = {};

    // Create default textures.
    static auto create_color_texture = [](ID3D11Device& device, unsigned char pixel[4]) -> DefaultTexture {
        DefaultTexture tex;

        D3D11_TEXTURE2D_DESC tex_desc = {};
        tex_desc.Width = 1;
        tex_desc.Height = 1;
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.SampleDesc.Quality = 0;
        tex_desc.Usage = D3D11_USAGE_IMMUTABLE;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        D3D11_SUBRESOURCE_DATA resource_data;
        resource_data.pSysMem = pixel;
        resource_data.SysMemPitch = sizeof(unsigned char) * 4;

        HRESULT hr = device.CreateTexture2D(&tex_desc, &resource_data, &tex.texture2D);

        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
        srv_desc.Format = tex_desc.Format;
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = tex_desc.MipLevels;
        srv_desc.Texture2D.MostDetailedMip = 0;
        hr = device.CreateShaderResourceView(tex.texture2D, &srv_desc, &tex.srv);

        D3D11_SAMPLER_DESC desc = {};
        desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        desc.MinLOD = 0;
        desc.MaxLOD = D3D11_FLOAT32_MAX;

        hr = device.CreateSamplerState(&desc, &tex.sampler);

        return tex;
    };

    unsigned char white[4] = { 255, 255, 255, 255 };
    m_white_texture = create_color_texture(device, white);
}

void TextureManager::release() {
    printf("bye bye tex man\n");
    safe_release(&m_white_texture.sampler);
    safe_release(&m_white_texture.srv);
    safe_release(&m_white_texture.texture2D);

    for (Dx11Image image : m_images) {
        safe_release(&image.srv);
        safe_release(&image.texture2D);
    }

    for (Dx11Texture tex : m_textures)
        safe_release(&tex.sampler);
}

void TextureManager::handle_updates(ID3D11Device& device, ID3D11DeviceContext& device_context) {
    { // Image updates.
        if (!Images::get_changed_images().is_empty()) {
            if (m_images.size() < Images::capacity())
                m_images.resize(Images::capacity());

            static auto to_DX_format = [](PixelFormat format) -> DXGI_FORMAT {
                switch (format) {
                case PixelFormat::I8:
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

            static auto dx_format_size = [](DXGI_FORMAT format) -> int {
                switch (format) {
                case DXGI_FORMAT_A8_UNORM:
                    return 1;
                case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
                    return 4;
                case DXGI_FORMAT_R32G32B32_FLOAT:
                    return 12;
                case DXGI_FORMAT_R32G32B32A32_FLOAT:
                    return 16;
                default:
                    return 0;
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

            for (Images::UID image_ID : Images::get_changed_images()) {
                Dx11Image& dx_image = m_images[image_ID];

                if (Images::get_changes(image_ID) == Images::Change::Destroyed &&
                    dx_image.texture2D != nullptr) {
                    safe_release(&dx_image.texture2D);
                    safe_release(&dx_image.srv);

                } else if (Images::get_changes(image_ID).is_set(Images::Change::Created)) {
                    safe_release(&dx_image.texture2D);
                    safe_release(&dx_image.srv);

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
                        resource_data.SysMemPitch = dx_format_size(tex_desc.Format) *  image.get_width();

                        bool generate_mipmaps = image.is_mipmapable();

                        HRESULT hr;
                        if (generate_mipmaps) {
                            // Additional mipmap generation settings.
                            tex_desc.MipLevels = 0;
                            tex_desc.BindFlags |= D3D11_BIND_RENDER_TARGET;
                            tex_desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

                            // If a texture has autogenerated mipmaps, then CreateTexture2D expects the data to contain the full mipchain, 
                            // so we have to upload the data separately.
                            hr = device.CreateTexture2D(&tex_desc, nullptr, &dx_image.texture2D);
                            device_context.UpdateSubresource(dx_image.texture2D, 0, 0, resource_data.pSysMem, resource_data.SysMemPitch, 0);
                        } else
                            hr = device.CreateTexture2D(&tex_desc, &resource_data, &dx_image.texture2D);

                        if (FAILED(hr))
                            printf("Could not create the texture '%s'.\n", image.get_name().c_str());

                        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
                        srv_desc.Format = tex_desc.Format;
                        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                        srv_desc.Texture2D.MipLevels = generate_mipmaps ? -1 : tex_desc.MipLevels;
                        srv_desc.Texture2D.MostDetailedMip = 0;
                        hr = device.CreateShaderResourceView(dx_image.texture2D, &srv_desc, &dx_image.srv);
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

            for (Textures::UID tex_ID : Textures::get_changed_textures()) {
                Dx11Texture& dx_tex = m_textures[tex_ID];

                if (Textures::get_changes(tex_ID) == Textures::Change::Destroyed &&
                    dx_tex.sampler != nullptr) {
                    safe_release(&dx_tex.sampler);
                } else if (Textures::get_changes(tex_ID) & Textures::Change::Created) {
                    TextureND texture = tex_ID;

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