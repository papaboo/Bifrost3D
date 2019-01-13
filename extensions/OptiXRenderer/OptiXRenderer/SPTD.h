// A Spherical Cap Preserving Parameterization for Spherical Distributions
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SPTD_FIT_H_
#define _OPTIXRENDERER_SPTD_FIT_H_

#include <OptiXRenderer/Utils.h>

#include <optixu/optixpp_namespace.h>
#undef RGB

namespace OptiXRenderer {
namespace SPTD {
using namespace optix;

__inline_all__ float2 pivot_transform(float2 r, float pivot) {
    float2 tmp1 = make_float2(r.x - pivot, r.y);
    float2 tmp2 = pivot * r - make_float2(1, 0);
    float x = dot(tmp1, tmp2);
    float y = tmp1.y * tmp2.x - tmp1.x * tmp2.y;
    float qf = dot(tmp2, tmp2);

    return make_float2(x, y) / qf;
}

// Equation 2 in SPDT, Dupuy et al. 17.
__inline_all__ float3 pivot_transform(const float3& r, const float3& pivot) {
    float3 numerator = (dot(r, pivot) - 1.0f) * (r - pivot) - cross(r - pivot, cross(r, pivot));
    float denominator = pow2(dot(r, pivot) - 1.0f) + length_squared(cross(r, pivot));
    return numerator / denominator;
}

__inline_all__ OptiXRenderer::Cone pivot_transform(const Cone& cone, const float3& pivot) {
    // extract pivot length and direction
    float pivot_mag = length(pivot);
    // special case: the pivot is at the origin
    if (pivot_mag < 0.001f)
        return OptiXRenderer::Cone::make(-cone.direction, cone.cos_theta);
    float3 pivot_dir = pivot / pivot_mag;

    // 2D cap dir
    float cos_phi = dot(cone.direction, pivot_dir);
    float sin_phi = sqrt(1.0f - cos_phi * cos_phi);

    // 2D basis = (pivotDir, PivotOrthogonalDirection)
    float3 pivot_ortho_dir;
    if (abs(cos_phi) < 0.9999f)
        pivot_ortho_dir = (cone.direction - cos_phi * pivot_dir) / sin_phi;
    else
        pivot_ortho_dir = make_float3(0, 0, 0);

    // compute cap 2D end points
    float sin_theta_sqrd = sqrt(1.0f - cone.cos_theta * cone.cos_theta);
    float a1 = cos_phi * cone.cos_theta;
    float a2 = sin_phi * sin_theta_sqrd;
    float a3 = sin_phi * cone.cos_theta;
    float a4 = cos_phi * sin_theta_sqrd;
    float2 dir1 = make_float2(a1 + a2, a3 - a4);
    float2 dir2 = make_float2(a1 - a2, a3 + a4);

    // project in 2D
    float2 dir1_xf = pivot_transform(dir1, pivot_mag);
    float2 dir2_xf = pivot_transform(dir2, pivot_mag);

    // compute the cap 2D direction
    float area = dir1_xf.x * dir2_xf.y - dir1_xf.y * dir2_xf.x;
    float s = area > 0.0f ? 1.0f : -1.0f;
    float2 dir_xf = s * normalize(dir1_xf + dir2_xf);

    return OptiXRenderer::Cone::make(dir_xf.x * pivot_dir + dir_xf.y * pivot_ortho_dir,
        dot(dir_xf, dir1_xf));
}

// Map a sphere to the spherical cap at origo.
__inline_all__ Cone sphere_to_sphere_cap(const float3& position, float radius) {
    float sin_theta_sqrd = clamp(radius * radius / dot(position, position), 0.0f, 1.0f);
    return Cone::make(normalize(position), sqrt(1.0f - sin_theta_sqrd));
}

__inline_all__ float solidangle(const Cone& c) { return TWO_PIf - TWO_PIf * c.cos_theta; }

// Based on Oat and Sander's 2007 technique in Ambient aperture lighting.
__inline_all__ float solidangle_of_union(const Cone& c1, const Cone& c2) {
    float r1 = acos(c1.cos_theta);
    float r2 = acos(c2.cos_theta);
    float rd = acos(dot(c1.direction, c2.direction));

    if (rd <= abs(r2 - r1))
        // One cone is completely inside the other
        return TWO_PIf - TWO_PIf * fmaxf(c1.cos_theta, c2.cos_theta);
    else if (rd >= r1 + r2)
        // No intersection exists
        return 0.0f;
    else {
        float diff = abs(r2 - r1);
        float den = r1 + r2 - diff;
        float x = 1.0f - (rd - diff) / den;
        return smoothstep(0.0f, 1.0f, x) * (TWO_PIf - TWO_PIf * fmaxf(c1.cos_theta, c2.cos_theta));
    }
}

// The centroid of the intersection of the two cones.
// See Ambient aperture lighting, 2007, section 3.3.
__inline_all__ float3 centroid_of_union(const Cone& c1, const Cone& c2) {
    float r1 = acos(c1.cos_theta);
    float r2 = acos(c2.cos_theta);
    float d = acos(dot(c1.direction, c2.direction));
    
    if (d <= abs(r2 - r1))
        // One cone is completely inside the other
        return c1.cos_theta > c2.cos_theta ? c1.direction : c2.direction;
    else {
        float w = (r2 - r1 + d) / (2.0f * d);
        return normalize(lerp(c2.direction, c1.direction, clamp(w, 0.0f, 1.0f)));
    }
}

struct  __align__(16) CentroidAndSolidangle {
    float3 centroid;
    float solidangle;
};

// Computes the centroid and solidangle of the intersection from the cone with the hemisphere.
// Assumes that the cone has a maximum angle of 90 degrees (positive cos theta).
__inline_all__ CentroidAndSolidangle centroid_and_solidangle_on_hemisphere(const Cone& cone) {
    const Cone hemipshere = { make_float3(0.0f, 0.0f, 1.0f), 0.0f };

    float r1 = acos(cone.cos_theta);
    float r2 = 1.57079637f;
    float rd = acos(cone.direction.z);

    if (rd <= r2 - r1) {
        // One cone is completely inside the other
        float3 centroid = cone.cos_theta > hemipshere.cos_theta ? cone.direction : hemipshere.direction;
        float solidangle = TWO_PIf - TWO_PIf * cone.cos_theta;
        return { centroid, solidangle };
    } else {
        float w = (r2 - r1 + rd) / (2.0f * rd);
        float3 centroid = normalize(lerp(hemipshere.direction, cone.direction, w));

        if (rd >= r1 + r2)
            // No intersection exists
            return { centroid, 0.0f };
        else {
            float diff = r2 - r1;
            float den = 2.0f * r1;
            float x = 1.0f - (rd - diff) / den;
            float solidangle = smoothstep(0.0f, 1.0f, x) * (TWO_PIf - TWO_PIf * cone.cos_theta);
            return { centroid, solidangle };
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Fitted Spherical pivot.
// TODO Parameterize by distribution, fx uniform or cosine.
// ------------------------------------------------------------------------------------------------
struct Pivot {

    // lobe amplitude
    float amplitude;

    // parameterization 
    float distance;
    float theta;

    // pivot position
    inline optix::float3 position() const { return distance * optix::make_float3(sinf(theta), 0.0f, cosf(theta)); }

    float eval(const optix::float3& wi) const {
        optix::float3 xi = position();
        float num = 1.0f - dot(xi, xi);
        optix::float3 tmp = wi - xi;
        float den = dot(tmp, tmp);
        float p = num / den;
        float jacobian = p * p;
        float pdf = jacobian / (4.0f * PIf);
        return amplitude * pdf;
    }

    optix::float3 sample(const float U1, const float U2) const {
        const float sphere_theta = acosf(-1.0f + 2.0f * U1);
        const float sphere_phi = 2.0f * 3.14159f * U2;
        const optix::float3 sphere_sample = optix::make_float3(sinf(sphere_theta) * cosf(sphere_phi), sinf(sphere_theta) * sinf(sphere_phi), -cosf(sphere_theta));
        return pivot_transform(sphere_sample, position());
    }
};

optix::float4 GGX_fit_lookup(float cos_theta, float ggx_alpha);

optix::TextureSampler GGX_fit_texture(optix::Context& context);

} // NS SPTD
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SPTD_FIT_H_