// DirectX 11 tone mapper.
// ---------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_TONE_MAPPER_H_
#define _DX11RENDERER_RENDERER_TONE_MAPPER_H_

#include <DX11Renderer/Types.h>

namespace DX11Renderer {

class ToneMapper {
public:
    ToneMapper();
    ToneMapper(ID3D11Device1& device, const std::wstring& shader_folder_path);

    ToneMapper& operator=(ToneMapper&& rhs) {
        m_vertex_shader = std::move(rhs.m_vertex_shader);
        m_pixel_shader = std::move(rhs.m_pixel_shader);
        m_sampler = std::move(rhs.m_sampler);
        return *this;
    }

    // Tonemaps the pixels and stores them in the bound render target.
    void tonemap(ID3D11DeviceContext1& render_context, ID3D11ShaderResourceView* pixel_SRV);

private:
    ToneMapper(ToneMapper& other) = delete;
    ToneMapper(ToneMapper&& other) = delete;
    ToneMapper& operator=(ToneMapper& rhs) = delete;

    OID3D11VertexShader m_vertex_shader;
    OID3D11PixelShader m_pixel_shader;
    OID3D11SamplerState m_sampler;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_TONE_MAPPER_H_