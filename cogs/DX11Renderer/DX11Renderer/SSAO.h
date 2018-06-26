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

class BilateralBlur {
public:
    static const int max_passes = 3;

    BilateralBlur() = default;
    BilateralBlur(BilateralBlur&& other) = default;
    BilateralBlur(BilateralBlur& other) = delete;
    BilateralBlur(ID3D11Device1& device, const std::wstring& shader_folder_path);

    BilateralBlur& operator=(BilateralBlur&& rhs) = default;
    BilateralBlur& operator=(BilateralBlur& rhs) = delete;

    OShaderResourceView& apply(ID3D11DeviceContext1& context, ORenderTargetView& ao_RTV, OShaderResourceView& ao_SRV, int width, int height);

private:
    OVertexShader m_vertex_shader;
    OPixelShader m_filter_shader;

    struct Constants {
        float pixel_offset;
        float3 _padding;
    };
    OBuffer m_constants[max_passes];

    int m_width, m_height;
    ORenderTargetView m_intermediate_RTV;
    OShaderResourceView m_intermediate_SRV;
};

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

    OShaderResourceView& apply(ID3D11DeviceContext1& context, OShaderResourceView& normals, OShaderResourceView& depth, int width, int height, SsaoSettings settings);

    OShaderResourceView& apply_none(ID3D11DeviceContext1& context, int width, int height);

private:
    OBuffer m_constants;
    OVertexShader m_vertex_shader;
    OPixelShader m_pixel_shader;

    int m_width, m_height;
    ORenderTargetView m_SSAO_RTV;
    OShaderResourceView m_SSAO_SRV;

    BilateralBlur m_filter;
};

} // NS SSAO
} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_SSAO_H_