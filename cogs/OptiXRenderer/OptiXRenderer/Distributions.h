// OptiX distributions for monte carlo integration.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_DISTRIBUTIONS_H_
#define _OPTIXRENDERER_DISTRIBUTIONS_H_

#include <OptiXRenderer/Defines.h>

#include <optixu/optixu_math_namespace.h>

namespace OptiXRenderer {
namespace Distributions {

struct __align__(16) DirectionalSample {
    optix::float3 direction;
    float PDF;
};

//==============================================================================
// Uniform cone distribution.
//==============================================================================
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

//==============================================================================
// Cosine distribution.
//==============================================================================
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

//==============================================================================
// GGX distribution.
// Future work
// * Reference equations in Walter07.
//==============================================================================
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

} // NS Distributions
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_DISTRIBUTIONS_H_