// Spherical pivot transform distribution utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_SPTD_H_
#define _DX11_RENDERER_SHADERS_SPTD_H_

#include "Utils.hlsl"

namespace SPTD {

    float2 pivot_transform(float2 r, float pivot) {
        float2 tmp1 = float2(r.x - pivot, r.y);
        float2 tmp2 = pivot * r - float2(1, 0);
        float x = dot(tmp1, tmp2);
        float y = tmp1.y * tmp2.x - tmp1.x * tmp2.y;
        float qf = dot(tmp2, tmp2);

        return float2(x, y) / qf;
    }

    // Equation 2 in SPDT, Dupuy et al. 17.
    float3 pivot_transform(float3 r, float3 pivot) {
        float3 numerator = (dot(r, pivot) - 1.0f) * (r - pivot) - cross(r - pivot, cross(r, pivot));
        float denominator = pow2(dot(r, pivot) - 1.0f) + length_squared(cross(r, pivot));
        return numerator / denominator;
    }

    Cone pivot_transform(Cone cone, float3 pivot) {
        // extract pivot length and direction
        float pivot_mag = length(pivot);
        // special case: the pivot is at the origin
        if (pivot_mag < 0.001f)
            return Cone::make(-cone.direction, cone.cos_theta);
        float3 pivot_dir = pivot / pivot_mag;

        // 2D cap dir
        float cos_phi = dot(cone.direction, pivot_dir);
        float sin_phi = sqrt(1.0f - cos_phi * cos_phi);

        // 2D basis = (pivotDir, PivotOrthogonalDirection)
        float3 pivot_ortho_dir;
        if (abs(cos_phi) < 0.9999f)
            pivot_ortho_dir = (cone.direction - cos_phi * pivot_dir) / sin_phi;
        else
            pivot_ortho_dir = float3(0, 0, 0);

        // compute cap 2D end points
        float sin_theta_sqrd = sqrt(1.0f - cone.cos_theta * cone.cos_theta);
        float a1 = cos_phi * cone.cos_theta;
        float a2 = sin_phi * sin_theta_sqrd;
        float a3 = sin_phi * cone.cos_theta;
        float a4 = cos_phi * sin_theta_sqrd;
        float2 dir1 = float2(a1 + a2, a3 - a4);
        float2 dir2 = float2(a1 - a2, a3 + a4);

        // project in 2D
        float2 dir1_xf = pivot_transform(dir1, pivot_mag);
        float2 dir2_xf = pivot_transform(dir2, pivot_mag);

        // compute the cap 2D direction
        float area = dir1_xf.x * dir2_xf.y - dir1_xf.y * dir2_xf.x;
        float s = area > 0.0f ? 1.0f : -1.0f;
        float2 dir_xf = s * normalize(dir1_xf + dir2_xf);

        return Cone::make(dir_xf.x * pivot_dir + dir_xf.y * pivot_ortho_dir,
                          dot(dir_xf, dir1_xf));
    }
}

#endif // _DX11_RENDERER_SHADERS_SPTD_H_