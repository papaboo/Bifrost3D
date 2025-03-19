// DirectX 11 texture manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_MANAGERS_TEXTURE_MANAGER_H_
#define _DX11RENDERER_MANAGERS_TEXTURE_MANAGER_H_

#include "Dx11Renderer/Types.h"

#include <vector>

namespace DX11Renderer::Managers {

//----------------------------------------------------------------------------
// Texture manager.
// Uploads and manages all images and textures.
// Future work:
// * Non SRGB texture support.
//----------------------------------------------------------------------------
class TextureManager {
private:
    std::vector<Dx11Texture> m_textures = std::vector<Dx11Texture>(0);
    std::vector<Dx11Image> m_images = std::vector<Dx11Image>(0);
    std::vector<Bifrost::Assets::ImageID> m_newly_referenced_images = std::vector<Bifrost::Assets::ImageID>(0);

    // Default textures.
    struct DefaultTexture {
        OShaderResourceView srv;
        OSamplerState sampler;
    };

    DefaultTexture m_white_texture;

public:

    TextureManager() = default;
    TextureManager(ID3D11Device1& device);

    static OSamplerState create_clamped_linear_sampler(ID3D11Device1& device);

    inline const DefaultTexture& white_texture() { return m_white_texture; }

    inline Dx11Image& get_image(int id) { return m_images[id]; }
    inline Dx11Texture& get_texture(int id) { return m_textures[id]; }

    void handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& device_context);
};

} // NS DX11Renderer::Managers

#endif // _DX11RENDERER_MANAGERS_TEXTURE_MANAGER_H_