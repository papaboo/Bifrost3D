// DirectX 11 texture manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_TEXTURE_MANAGER_H_
#define _DX11RENDERER_RENDERER_TEXTURE_MANAGER_H_

#include "Dx11Renderer/Types.h"

#include <vector>

namespace DX11Renderer {

//----------------------------------------------------------------------------
// Texture manager.
// Uploads and manages all images and textures.
// Future work:
// * Non SRGB texture support.
//----------------------------------------------------------------------------
class TextureManager {
private:
    std::vector<Dx11Image> m_images = std::vector<Dx11Image>(0);
    std::vector<Dx11Texture> m_textures = std::vector<Dx11Texture>(0);

    // Default textures.
    struct DefaultTexture {
        ID3D11Texture2D* texture2D;
        ID3D11ShaderResourceView* srv;
        ID3D11SamplerState* sampler;
    };

    DefaultTexture m_white_texture;

public:

    TextureManager() {}
    TextureManager(ID3D11Device& device);
    void release();

    inline const DefaultTexture& white_texture() { return m_white_texture; }

    inline Dx11Image& get_image(int id) { return m_images[id]; }
    inline Dx11Texture& get_texture(int id) { return m_textures[id]; }

    void handle_updates(ID3D11Device& device, ID3D11DeviceContext& device_context);
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_TEXTURE_MANAGER_H_