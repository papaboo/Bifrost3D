// DirectX 11 renderer screen space ambient occlusion implementations.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_SSAO_H_
#define _DX11RENDERER_RENDERER_SSAO_H_

#include <DX11Renderer/Types.h>

#include <Bifrost/Math/Rect.h>

#include <vector>

namespace DX11Renderer {
namespace SSAO {

class BilateralBlur {
public:
    static const int MAX_PASSES = 3;

    BilateralBlur() = default;
    BilateralBlur(BilateralBlur&& other) = default;
    BilateralBlur(BilateralBlur& other) = delete;
    BilateralBlur(ID3D11Device1& device, const std::wstring& shader_folder_path, SsaoFilter type);

    BilateralBlur& operator=(BilateralBlur&& rhs) = default;
    BilateralBlur& operator=(BilateralBlur& rhs) = delete;

    inline SsaoFilter get_type() const { return m_type; }
    void set_support(ID3D11DeviceContext1& context, int support);
    inline int get_support() const { return get_support(m_type, m_support); }
    inline static int get_support(SsaoFilter type, int expected_support) { return type == SsaoFilter::Box ? 9 : expected_support; }

    OShaderResourceView& apply(ID3D11DeviceContext1& context, ORenderTargetView& ao_RTV, OShaderResourceView& ao_SRV, OShaderResourceView& normal_SRV, OShaderResourceView& depth_SRV, int width, int height, int support);

private:
    SsaoFilter m_type;
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
// Future work:
// * HBAO/GTAO: https://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf
// ------------------------------------------------------------------------------------------------
class AlchemyAO {
public:
    static const float max_screen_space_radius;

    AlchemyAO() = default;
    AlchemyAO(AlchemyAO&& other) = default;
    AlchemyAO(AlchemyAO& other) = delete;
    AlchemyAO(ID3D11Device1& device, const std::wstring& shader_folder_path);

    AlchemyAO& operator=(AlchemyAO&& rhs) = default;
    AlchemyAO& operator=(AlchemyAO& rhs) = delete;

    inline static int2 compute_g_buffer_to_ao_index_offset(SsaoSettings settings, Bifrost::Math::Recti viewport) {
        int support = BilateralBlur::get_support(settings.filter_type, settings.filter_support);
        return { support - viewport.x, support - viewport.y };
    }

    OShaderResourceView& apply(ID3D11DeviceContext1& context, unsigned int camera_ID, OShaderResourceView& normals, OShaderResourceView& depth,
                               int2 g_buffer_size, Bifrost::Math::Recti viewport, SsaoSettings settings);

    OShaderResourceView& apply_none(ID3D11DeviceContext1& context, unsigned int camera_ID, Bifrost::Math::Recti viewport);

    inline void clear_camera_state(unsigned int camera_ID) { if (camera_ID < m_depth.per_camera.size()) m_depth.per_camera[camera_ID].clear(); }

private:
    inline int2 compute_g_buffer_to_ao_index_offset(Bifrost::Math::Recti viewport) const {
        return { get_margin() - viewport.x, get_margin() - viewport.y };
    }
    inline Bifrost::Math::Recti get_ssao_viewport() const { return Bifrost::Math::Recti(get_margin(), get_margin(), m_width, m_height); }
    inline int get_margin() const { return m_filter.get_support(); }

    void resize_ao_buffer(ID3D11DeviceContext1& context, int ssao_width, int ssao_height);
    void resize_depth_buffer(ID3D11DeviceContext1& context, unsigned int camera_ID, int depth_width, int depth_height);

    std::wstring m_shader_folder_path;

    struct Samples {
        static const unsigned int capacity = 256;
        unsigned int size;
        OBuffer buffer;
    } m_samples;
    OBuffer m_constants;
    OVertexShader m_vertex_shader;
    OPixelShader m_pixel_shader;
    OSamplerState m_trilinear_sampler;

    int m_width, m_height;
    ORenderTargetView m_SSAO_RTV;
    OShaderResourceView m_SSAO_SRV;

    struct Depth {
        struct PerCamera {
            int width, height, mip_count;
            ORenderTargetView RTV;
            OShaderResourceView SRV;

            inline void clear() {
                width = height = mip_count = 0;
                RTV = nullptr; SRV = nullptr;
            }
        };
        std::vector<PerCamera> per_camera;
        OPixelShader pixel_shader;
    } m_depth;

    BilateralBlur m_filter;
};

} // NS SSAO
} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_SSAO_H_