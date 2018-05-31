// DirectX 11 renderer screen space ambient occlusion implementations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_SSAO_H_
#define _DX11RENDERER_RENDERER_SSAO_H_

#include <DX11Renderer/Types.h>

namespace DX11Renderer {
namespace SSAO {

// ------------------------------------------------------------------------------------------------
// The Alchemy screen-space ambient obscurance algorithm.
// http://casual-effects.com/research/McGuire2011AlchemyAO/index.html
// ------------------------------------------------------------------------------------------------
class AlchemyAO {
public:
    AlchemyAO() = default;
    AlchemyAO(AlchemyAO&& other) = default;
    AlchemyAO(AlchemyAO& other) = delete;
    AlchemyAO(ID3D11Device1& device, const std::wstring& shader_folder_path);

    AlchemyAO& operator=(AlchemyAO&& rhs) = default;
    AlchemyAO& operator=(AlchemyAO& rhs) = delete;

    OShaderResourceView& apply(ID3D11DeviceContext1& context, OShaderResourceView& normals, OShaderResourceView& depth, int width, int height);

private:
    OVertexShader m_vertex_shader;
    OPixelShader m_pixel_shader;

    OSamplerState m_point_sampler;

    int m_width, m_height;
    ORenderTargetView m_SSAO_RTV;
    OShaderResourceView m_SSAO_SRV;
};

} // NS SSAO
} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_SSAO_H_