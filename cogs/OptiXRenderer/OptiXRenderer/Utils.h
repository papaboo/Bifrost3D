// OptiX shading utilities.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_UTILS_H_
#define _OPTIXRENDERER_SHADING_UTILS_H_

#include <OptiXRenderer/Defines.h>

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

namespace OptiXRenderer {

__inline_dev__ optix::float3 project_ray_direction(optix::float2 viewport_pos, 
                                                   const optix::float3& camera_position, 
                                                   const optix::Matrix4x4& inverted_view_projection_matrix) {
    using namespace optix;

    // Generate rays.
    float4 normalized_projected_pos = make_float4(viewport_pos.x * 2.0f - 1.0f,
                                                  1.0f - viewport_pos.y * 2.0f, // Inline flipping of the viewport's y.
                                                  -1.0f, 1.0f);

    float4 projected_world_pos = inverted_view_projection_matrix * normalized_projected_pos;

    float3 ray_origin = make_float3(projected_world_pos) / projected_world_pos.w;

    return normalize(ray_origin - camera_position);
}

__inline_dev__ optix::float3 gammacorrect(const optix::float3& color, float gamma) {
    return optix::make_float3(pow(color.x, gamma),
                              pow(color.y, gamma),
                              pow(color.z, gamma));
}

__inline_dev__ optix::float4 gammacorrect(const optix::float4& color, float gamma) {
    return optix::make_float4(pow(color.x, gamma),
                              pow(color.y, gamma),
                              pow(color.z, gamma),
                              color.w);
}

// Robert Jenkins hash function.
// https://gist.github.com/badboy/6267743
__inline_all__ unsigned int hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// Computes a tangent and bitangent that together with the normal creates an orthonormal bases.
// Consider using TBN to wrap the tangents.
__inline_all__ static void compute_tangents(const optix::float3& normal,
                                            optix::float3& tangent, optix::float3& bitangent) {
    using namespace optix;

    float3 a0;
    if (abs(normal.x) < abs(normal.y)) {
        const float zup = abs(normal.z) < abs(normal.x) ? 0.0f : 1.0f;
        a0 = make_float3(zup, 0.0f, 1.0f - zup);
    } else {
        const float zup = (abs(normal.z) < abs(normal.y)) ? 0.0f : 1.0f;
        a0 = make_float3(0.0f, zup, 1.0f - zup);
    }

    bitangent = normalize(cross(normal, a0));
    tangent = normalize(cross(bitangent, normal));
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_UTILS_H_