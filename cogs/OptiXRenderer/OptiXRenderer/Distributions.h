// OptiX distributions for monte carlo integration.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_DISTRIBUTIONS_H_
#define _OPTIXRENDERER_DISTRIBUTIONS_H_

#include <OptiXRenderer/Defines.h>
#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Utils.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {
namespace Distributions {

struct __align__(16) DirectionalSample {
    optix::float3 direction;
    float PDF;
};

//=================================================================================================
// Uniform cone distribution.
//=================================================================================================
namespace Cone {

    __inline_all__ float PDF(float cos_theta_max) {
        return 1.0f / (2.0f * PIf * (1.0f - cos_theta_max));
    }

    __inline_all__ DirectionalSample sample(float cos_theta_max, optix::float2 random_sample) {
        float cos_theta = (1.0f - random_sample.x) + random_sample.x * cos_theta_max;
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
        float phi = 2.0f * PIf * random_sample.y;

        DirectionalSample res;
        res.direction = optix::make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        res.PDF = PDF(cos_theta_max);
        return res;
    }

} // NS Cone

//=================================================================================================
// Cosine distribution.
//=================================================================================================
namespace Cosine {

    __inline_all__ float PDF(float abs_cos_theta) {
        return abs_cos_theta * RECIP_PIf;
    }

    __inline_all__ DirectionalSample sample(optix::float2 random_sample) {
        float phi = 2.0f * PIf * random_sample.x;
        float r2 = random_sample.y;
        float r = sqrt(1.0f - r2);
        float z = sqrt(r2);

        DirectionalSample res;
        res.direction = optix::make_float3(cos(phi) * r, sin(phi) * r, z);
        res.PDF = z * RECIP_PIf;
        return res;
    }

} // NS Cosine

//=================================================================================================
// GGX distribution.
// Future work
// * Reference equations in Walter07.
//=================================================================================================
namespace GGX {

    __inline_all__ float D(float alpha, float abs_cos_theta) {
        float alpha_sqrd = alpha * alpha;
        float cos_theta_sqrd = abs_cos_theta * abs_cos_theta;
        float tan_theta_sqrd = optix::fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
        float cos_theta_cubed = cos_theta_sqrd * cos_theta_sqrd;
        float foo = alpha_sqrd + tan_theta_sqrd; // No idea what to call this.
        return alpha_sqrd / (PIf * cos_theta_cubed * foo * foo);
    }

    __inline_all__ float PDF(float alpha, float abs_cos_theta) {
        return D(alpha, abs_cos_theta) * abs_cos_theta;
    }

    __inline_all__ DirectionalSample sample(float alpha, optix::float2 random_sample) {
        float phi = random_sample.y * (2.0f * PIf);

        float tan_theta_sqrd = alpha * alpha * random_sample.x / (1.0f - random_sample.x);
        float cos_theta = 1.0f / sqrt(1.0f + tan_theta_sqrd);

        float r = sqrt(optix::fmaxf(1.0f - cos_theta * cos_theta, 0.0f));

        DirectionalSample res;
        res.direction = optix::make_float3(cos(phi) * r, sin(phi) * r, cos_theta);
        res.PDF = PDF(alpha, cos_theta); // We have to be able to inline this to reuse some temporaries.
        return res;
    }

} // NS GGX

//=================================================================================================
// Sampling of the visible normal distribution function for GGX.
// Importance Sampling Microfacet-Based BSDFs with the Distribution of Visible Normals, Heitz 14.
// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14.
//=================================================================================================
namespace VNDF_GGX {
    // Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Equation 72.
    __inline_all__ float lambda(float alpha, float cos_theta) {
        // NOTE At grazing angles the lambda function is going to return very low values. 
        //      This is generally not a problem, unless alpha is low as well, 
        //      meaning that the in and out vectors will always be at grazing angles 
        //      and therefore the masking is consistently underestimated. 
        //      We could fix this be adding a very specific scaling for this one case.
        //      Check the GGX Rho computation for validity.
        float cos_theta_sqrd = cos_theta * cos_theta;
        float tan_theta_sqrd = fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
        float a_sqrd = 1.0f / (alpha * alpha * tan_theta_sqrd);
        return (-1.0f + sqrt(1.0f + 1.0f / a_sqrd)) / 2.0f;
    }

    __inline_all__ float masking(float alpha, float cos_theta) {
        return 1.0f / (1.0f + lambda(alpha, cos_theta));
    }

    // Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, supplemental 1, page 19.
    __inline_all__ optix::float2 sample11(float cos_theta, optix::float2 random_sample) {
        using namespace optix;

        float cos_theta_sqrd = cos_theta * cos_theta;
        // Special case (normal incidence)
        if (cos_theta_sqrd > 0.99999f) {
            float r = sqrt(random_sample.x / (1 - random_sample.x));
            float phi = 6.28318530718f * random_sample.y;
            return make_float2(r * cos(phi), r * sin(phi));
        }

        float U1 = random_sample.x;
        float U2 = random_sample.y;

        // GGX masking term with alpha of 1.0.
        float tan_theta_sqrd = fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
        float tan_theta = sqrt(tan_theta_sqrd);
        float a = 1.0f / tan_theta;
        float G1 = 2.0f / (1.0f + sqrt(1.0f + 1.0f / (a*a)));

        // Sample slope_x
        float A = 2.0f * U1 / G1 - 1.0f;
        float tmp = 1.0f / (A * A - 1.0f);
        float B = tan_theta;
        float D = sqrt(B * B * tmp * tmp - (A * A - B * B) * tmp);
        float slope_x_1 = B * tmp - D;
        float slope_x_2 = B * tmp + D;
        float2 slope;
        slope.x = (A < 0.0f || slope_x_2 > 1.0f / tan_theta) ? slope_x_1 : slope_x_2;

        // Sample slope_y
        // TODO Simplifiable??
        float S;
        if (U2 > 0.5f) {
            S = 1.0f;
            U2 = 2.0f * (U2 - 0.5f);
        } else {
            S = -1.0f;
            U2 = 2.0f * (0.5f - U2);
        }

        float z = (U2*(U2*(U2*0.27385f - 0.73369f) + 0.46341f)) / (U2*(U2*(U2*0.093073f + 0.309420f) - 1.0f) + 0.597999f);
        slope.y = S * z * sqrt(1.0f + slope.x * slope.x);
        return slope;
    }

    // Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, supplemental 1, page 4.
    __inline_all__ optix::float3 sample_halfway(float alpha, optix::float3 wo, optix::float2 random_sample) {
        using namespace optix;

        // Stretch wo
        float3 stretched_wo = normalize(make_float3(alpha * wo.x, alpha * wo.y, wo.z));

        // Sample P22_{wo}(x_slope, y_slope, 1, 1)
        float2 slope = sample11(stretched_wo.z, random_sample);

        // Rotate
        float _cos_phi = cos_phi(stretched_wo);
        float _sin_phi = sin_phi(stretched_wo);
        float tmp = _cos_phi * slope.x - _sin_phi * slope.y;
        slope.y = _sin_phi * slope.x + _cos_phi * slope.y;
        slope.x = tmp;

        // Unstretch and compute normal.
        return normalize(make_float3(-slope.x * alpha, -slope.y * alpha, 1.0f));
    }

    // Importance Sampling Microfacet-Based BSDFs with the Distribution of Visible Normals, equation 2
    __inline_all__ float PDF(float alpha, optix::float3 wo, optix::float3 halfway) {
#if _DEBUG
        if (wo.z < 0.0f)
            THROW(OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION);
#endif

        float G1 = masking(alpha, wo.z);
        float D = Distributions::GGX::D(alpha, abs(halfway.z));
        return G1 * abs(optix::dot(wo, halfway)) * D / wo.z;
    }

    __inline_all__ Distributions::DirectionalSample sample(float alpha, optix::float3 wo, optix::float2 random_sample) {
        Distributions::DirectionalSample sample;
        sample.direction = sample_halfway(alpha, wo, random_sample);;
        sample.PDF = PDF(alpha, wo, sample.direction);
        return sample;
    }

} // NS VNDF_GGX

//=================================================================================================
// A Spherical Cap Preserving Parameterization for Spherical Distributions
//=================================================================================================
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

    // Equation 2 in SPDT, Heitz et al. 17.
    __inline_all__ float3 pivot_transform(const float3& r, const float3& pivot) {
        float3 numerator = (dot(r, pivot) - 1.0f) * (r - pivot) - cross(r - pivot, cross(r, pivot));
        float denominator = pow2(dot(r, pivot) - 1.0f) + length_squared(cross(r, pivot));
        return numerator / denominator;
    }

    __inline_all__ OptiXRenderer::Cone pivot_transform(const OptiXRenderer::Cone& cone, const float3& pivot) {
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

    __inline_all__ float solidangle(const OptiXRenderer::Cone& c) { return TWO_PIf - TWO_PIf * c.cos_theta; }

    // Based on Oat and Sander's 2007 technique in Ambient aperture lighting.
    __inline_all__ float solidangle_of_union(const OptiXRenderer::Cone& c1, const OptiXRenderer::Cone& c2) {
        float r1 = acos(c1.cos_theta);
        float r2 = acos(c2.cos_theta);
        float rd = acos(dot(c1.direction, c2.direction));

        if (rd <= fmaxf(r1, r2) - fminf(r1, r2))
            // One cap is completely inside the other
            return TWO_PIf - TWO_PIf * fmaxf(c1.cos_theta, c2.cos_theta);
        else if (rd >= r1 + r2)
            // No intersection exists
            return 0.0f;
        else {
            float diff = abs(r1 - r2);
            float den = r1 + r2 - diff;
            float x = 1.0f - clamp((rd - diff) / den, 0.0f, 1.0f); // TODO smoothstep clamps, so clamping here shouldn't be needed.
            return smoothstep(0.0f, 1.0f, x) * (TWO_PIf - TWO_PIf * fmaxf(c1.cos_theta, c2.cos_theta));
        }
    }
} // NS SPTD

} // NS Distributions
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_DISTRIBUTIONS_H_