// OptiX distributions for monte carlo integration.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
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
        float foo = alpha_sqrd + tan_theta_sqrd; // No idea what to call this.
        return alpha_sqrd / (PIf * pow2(cos_theta_sqrd * foo));
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
// Sampling the GGX Distribution of Visible Normals, Heitz, 2018
// Importance Sampling Microfacet-Based BSDFs with the Distribution of Visible Normals, Heitz, 2014.
// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz, 2014.
//=================================================================================================
namespace GGX_VNDF {
    using namespace optix;

    // Sampling the GGX Distribution of Visible Normals, equation 1.
    __inline_all__ float D(float alpha_x, float alpha_y, const float3& halfway) {
        float m = pow2(halfway.x / alpha_x) + pow2(halfway.y / alpha_y) + pow2(halfway.z);
        return 1 / (PIf * alpha_x * alpha_y * pow2(m));
    }
    __inline_all__ float D(float alpha, const float3& halfway) { return D(alpha, alpha, halfway); }

    // Sampling the GGX Distribution of Visible Normals, equation 2.
    __inline_all__ float lambda(float alpha_x, float alpha_y, const float3& w) {
        return 0.5f * (-1 + sqrt(1 + (pow2(alpha_x * w.x) + pow2(alpha_y * w.y)) / pow2(w.z)));
    }
    __inline_all__ float lambda(float alpha, const float3& w) { return lambda(alpha, alpha, w); }

    // Sampling the GGX Distribution of Visible Normals, listing 1.
    __inline_all__ float3 sample_halfway(float alpha_x, float alpha_y, const float3& wo, float2 random_sample) {
        // Section 3.2: transforming the view direction to the hemisphere configuration
        float3 Vh = normalize(make_float3(alpha_x * wo.x, alpha_y * wo.y, wo.z));

        // Section 4.1: orthonormal basis
        float3 T1 = (Vh.z < 0.9999f) ? normalize(cross(make_float3(0, 0, 1), Vh)) : make_float3(1, 0, 0);
        float3 T2 = cross(Vh, T1);

        // Section 4.2: parameterization of the projected area
        float r = sqrt(random_sample.x);
        float phi = 2.0f * PIf * random_sample.y;
        float t1 = r * cos(phi);
        float t2 = r * sin(phi);
        float s = 0.5f * (1.0f + Vh.z);
        t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;

        // Section 4.3: reprojection onto hemisphere
        float3 Nh = t1 * T1 + t2 * T2 + sqrt(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

        // Section 3.4: transforming the normal back to the ellipsoid configuration
        return normalize(make_float3(alpha_x * Nh.x, alpha_y * Nh.y, fmaxf(0.0f, Nh.z)));
    }
    __inline_all__ float3 sample_halfway(float alpha, const float3& wo, float2 random_sample) { return sample_halfway(alpha, alpha, wo, random_sample); }

    // Sampling the GGX Distribution of Visible Normals, equation 3.
    __inline_all__ float PDF(float alpha, const float3& wo, const float3& halfway) {
#if _DEBUG
        if (wo.z < 0.0f || dot(wo, halfway) < 0.0f)
            THROW(OPTIX_GGX_WRONG_HEMISPHERE_EXCEPTION);
#endif

        float recip_G1 = 1.0f + lambda(alpha, wo);
        float D = Distributions::GGX_VNDF::D(alpha, halfway);
        return dot(wo, halfway) * D / (recip_G1 * wo.z);
    }

    __inline_all__ Distributions::DirectionalSample sample(float alpha, const float3& wo, float2 random_sample) {
        Distributions::DirectionalSample sample;
        sample.direction = sample_halfway(alpha, wo, random_sample);
        sample.PDF = PDF(alpha, wo, sample.direction);
        return sample;
    }

} // NS GGX_VNDF

} // NS Distributions
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_DISTRIBUTIONS_H_