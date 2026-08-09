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

struct PositionalSample2D {
    optix::float2 position;
    float PDF;
};

struct __align__(16) DirectionalSample {
    optix::float3 direction;
    float PDF;
};

//=================================================================================================
// Disk distribution.
//=================================================================================================
namespace Disk {

    __inline_all__ float PDF(float radius) {
        return 1.0f / (PIf * pow2(radius));
    }

    __inline_all__ PositionalSample2D sample(float radius, optix::float2 random_sample) {
        float r = sqrtf(random_sample.x) * radius;
        float phi = 2.0f * PIf * random_sample.y;
        PositionalSample2D res;
        res.position = optix::make_float2(r * cosf(phi), r * sinf(phi));
        res.PDF = PDF(radius);
        return res;
    }

    // Concentric mapping sampling from Ray Tracing Gems 16.5.1.2. Supposed to better preserve stratification across samples.
    __inline_all__ PositionalSample2D sample_concentric_mapping(float radius, optix::float2 random_sample) {
        float a = 2 * random_sample.x - 1;
        float b = 2 * random_sample.y - 1;
        if (b == 0) b = 1;

        float r, phi;
        if (a * a > b * b) {
            r = radius * a;
            phi = (PIf / 4) * (b / a);
        } else {
            r = radius * b;
            phi = (PIf / 2) - (PIf / 4) * (a / b);
        }

        float sin_phi, cos_phi;
        sincos(phi, sin_phi, cos_phi);

        PositionalSample2D res;
        res.position = optix::make_float2(r * cos_phi, r * sin_phi);
        res.PDF = PDF(radius);
        return res;
    }

} // NS Disk

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
        float sin_phi, cos_phi;
        sincos(phi, sin_phi, cos_phi);

        DirectionalSample res;
        res.direction = optix::make_float3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
        res.PDF = PDF(cos_theta_max);
        return res;
    }

} // NS Cone

//=================================================================================================
// Uniform sphere distribution.
// Using octahedral concentric map as in Ray Tracing Gems 16.5.4.2.
//=================================================================================================
namespace UniformSphere {

    __inline_all__ float PDF() {
        return 0.25f * RECIP_PIf;
    }

    __inline_all__ DirectionalSample sample(optix::float2 random_sample) {
        // Compute radius r
        optix::float2 u = 2 * random_sample - 1;
        float d = 1 - (abs(u.x) + abs(u.y));
        float r = 1 - abs(d);

        // Compute phi in the first quadrant (branchless, except for the division-by-zero test),
        // using sign(u) to map the result to the correct quadrant below.
        float phi = (r == 0) ? 0 : (PIf / 4) * ((abs(u.x) - abs(u.y)) / r + 1);
        float sin_phi, cos_phi;
        sincos(phi, sin_phi, cos_phi);
        float f = r * sqrt(2 - r * r);
        float x = f * sign(u.x) * cos_phi;
        float y = f * sign(u.y) * sin_phi;
        float z = sign(d) * (1 - r * r);

        optix::float3 direction = { x, y, z };
        return { direction, PDF() };
    }

} // NS Uniform sphere

//=================================================================================================
// Uniform hemisphere distribution.
//=================================================================================================
namespace UniformHemisphere {

    __inline_all__ float PDF() {
        return 0.5f * RECIP_PIf;
    }

    __inline_all__ DirectionalSample sample(optix::float2 random_sample) {
        float z = random_sample.x;
        float r = sqrt(fmaxf(0.0f, 1.0f - z * z));

        float phi = TWO_PIf * random_sample.y;
        float sin_phi, cos_phi;
        sincos(phi, sin_phi, cos_phi);

        DirectionalSample res;
        res.direction = optix::make_float3(r * cos_phi, r * sin_phi, z);
        res.PDF = PDF();
        return res;
    }

} // NS Uniform hemisphere

//=================================================================================================
// Cosine distribution.
//=================================================================================================
namespace Cosine {

    __inline_all__ float PDF(float abs_cos_theta) {
        return abs_cos_theta * RECIP_PIf;
    }

    __inline_all__ DirectionalSample sample(optix::float2 random_sample) {
        float r2 = random_sample.x;
        float r = sqrt(1.0f - r2);
        float z = sqrt(r2);

        float phi = 2.0f * PIf * random_sample.y;
        float sin_phi, cos_phi;
        sincos(phi, sin_phi, cos_phi);

        DirectionalSample res;
        res.direction = optix::make_float3(r * cos_phi, r * sin_phi, z);
        res.PDF = z * RECIP_PIf;
        return res;
    }

} // NS Cosine

} // NS Distributions
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_DISTRIBUTIONS_H_