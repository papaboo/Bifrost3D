// OptiX renderer tangent, bitangent and normal container.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_TBN_H_
#define _OPTIXRENDERER_TBN_H_

#include <OptiXRenderer/Defines.h>
#include <OptiXRenderer/Utils.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {

//==============================================================================
// Tangent, bitangent and normal container.
// Can be used as an othonormal basis in stead of optix::Onb, since Onb only 
// supports transforming in one direction.
// Future work
//   * Reduce the members to tangent, normal and a bitangent directional sign.
//     And align struct. Check if it improves performance and reduces register pressure.
//==============================================================================
class TBN {
private:
    optix::float3 m_tangent;
    optix::float3 m_bitangent;
    optix::float3 m_normal;

public:

    __inline_all__ explicit TBN(const optix::float3& normal)
    : m_normal(normal) {
        compute_tangents(m_normal, m_tangent, m_bitangent);
    }

    __inline_all__ TBN(const optix::float3& tangent, const optix::float3& bitangent, const optix::float3& normal)
        : m_tangent(tangent), m_bitangent(bitangent), m_normal(normal) {}

    __inline_all__ const optix::float3& get_tangent() const { return m_tangent; }
    __inline_all__ const optix::float3& get_bitangent() const { return m_bitangent; }
    __inline_all__ const optix::float3& get_normal() const { return m_normal; }

    __inline_all__ optix::float3 operator*(const optix::float3& rhs) const {
		return optix::make_float3(optix::dot(m_tangent, rhs),
                                  optix::dot(m_bitangent, rhs),
                                  optix::dot(m_normal, rhs));
	}
};

} // NS OptiXRenderer

__inline_all__ optix::float3 operator*(const optix::float3& lhs, const OptiXRenderer::TBN& rhs) {
    return lhs.x * rhs.get_tangent() + lhs.y * rhs.get_bitangent() + lhs.z * rhs.get_normal();
}

#endif // _OPTIXRENDERER_TBN_H_