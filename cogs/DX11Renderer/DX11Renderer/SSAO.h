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

#include <Cogwheel/Math/Rect.h>

namespace DX11Renderer {
namespace SSAO {

class BilateralBlur {
public:
    enum class FilterType { Cross, Box };

    static const int MAX_PASSES = 3;

    BilateralBlur() = default;
    BilateralBlur(BilateralBlur&& other) = default;
    BilateralBlur(BilateralBlur& other) = delete;
    BilateralBlur(ID3D11Device1& device, const std::wstring& shader_folder_path, FilterType type);

    BilateralBlur& operator=(BilateralBlur&& rhs) = default;
    BilateralBlur& operator=(BilateralBlur& rhs) = delete;

    inline int get_support() const { return m_support; }

    OShaderResourceView& apply(ID3D11DeviceContext1& context, ORenderTargetView& ao_RTV, OShaderResourceView& ao_SRV, int width, int height, int support);

private:
    FilterType m_type;
    int m_support;

    OVertexShader m_vertex_shader;
    OPixelShader m_filter_shader;

    OBuffer m_constants[MAX_PASSES];

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
    static const unsigned int max_sample_count = 256;

    AlchemyAO() = default;
    AlchemyAO(AlchemyAO&& other) = default;
    AlchemyAO(AlchemyAO& other) = delete;
    AlchemyAO(ID3D11Device1& device, const std::wstring& shader_folder_path);

    AlchemyAO& operator=(AlchemyAO&& rhs) = default;
    AlchemyAO& operator=(AlchemyAO& rhs) = delete;

    int2 compute_g_buffer_to_ao_index_offset(Cogwheel::Math::Recti viewport) const;

    OShaderResourceView& apply(ID3D11DeviceContext1& context, OShaderResourceView& normals, OShaderResourceView& depth, int2 g_buffer_size, Cogwheel::Math::Recti viewport, SsaoSettings settings);

    OShaderResourceView& apply_none(ID3D11DeviceContext1& context, Cogwheel::Math::Recti viewport);

    Cogwheel::Math::Recti get_ssao_viewport() const { return Cogwheel::Math::Recti(get_margin(), get_margin(), m_width, m_height); }

private:
    void conditional_buffer_resize(ID3D11DeviceContext1& context, int ssao_width, int ssao_height);
    inline int get_margin() const { return m_filter.get_support(); }

    OBuffer m_constants;
    OBuffer m_samples;
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