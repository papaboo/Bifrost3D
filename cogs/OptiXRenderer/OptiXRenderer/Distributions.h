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
// Uniform cone Distribution.
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
// Cosine Distribution.
//==============================================================================
namespace Cosine {

__inline_all__ float PDF(float cos_theta) {
    return cos_theta / PIf;
}

__inline_all__ DirectionalSample sample(optix::float2 random_sample) {
    float phi = 2.0f * PIf * random_sample.x;
    float r2 = random_sample.y;
    float r = sqrt(1.0f - r2);
    float z = sqrt(r2);

    DirectionalSample res;
    res.direction = optix::make_float3(cos(phi) * r, sin(phi) * r, z);
    res.PDF = z / PIf;
    return res;
}

} // NS Cosine

} // NS Distributions
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_DISTRIBUTIONS_H_