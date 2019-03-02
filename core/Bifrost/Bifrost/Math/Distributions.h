// Bifrost distributions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_DISTRIBUTIONS_H_
#define _BIFROST_MATH_DISTRIBUTIONS_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/Vector.h>

namespace Bifrost {
namespace Math {
namespace Distributions {

//=================================================================================================
// GGX distribution.
//=================================================================================================
namespace GGX {

struct Sample {
    Vector3f direction;
    float PDF;
};

__always_inline__ float D(float alpha, float abs_cos_theta) {
    float alpha_sqrd = alpha * alpha;
    float cos_theta_sqrd = abs_cos_theta * abs_cos_theta;
    float tan_theta_sqrd = fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
    float cos_theta_cubed = cos_theta_sqrd * cos_theta_sqrd;
    float foo = alpha_sqrd + tan_theta_sqrd; // No idea what to call this.
    return alpha_sqrd / (PI<float>() * cos_theta_cubed * foo * foo);
}

__always_inline__ float PDF(float alpha, float abs_cos_theta) {
    return D(alpha, abs_cos_theta) * abs_cos_theta;
}

__always_inline__ Sample sample(float alpha, Vector2f random_sample) {
    float phi = random_sample.y * (2.0f * PI<float>());

    float tan_theta_sqrd = alpha * alpha * random_sample.x / (1.0f - random_sample.x);
    float cos_theta = 1.0f / sqrt(1.0f + tan_theta_sqrd);

    float r = sqrt(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));

    Sample res;
    res.direction = Vector3f(cos(phi) * r, sin(phi) * r, cos_theta);
    res.PDF = PDF(alpha, cos_theta); // We have to be able to inline this to reuse some temporaries.
    return res;
}

} // NS GGX


//=================================================================================================
// Uniform sphere distribution.
//=================================================================================================
namespace Sphere {

__always_inline__ float PDF() { return 0.5f / PI<float>(); }

__always_inline__ Vector3f Sample(Vector2f random_sample) {
    float z = 1.0f - 2.0f * random_sample.x;
    float r = sqrt(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2.0f * PI<float>() * random_sample.y;
    return Vector3f(r * cos(phi), r * sin(phi), z);
}

} // NS Sphere

} // NS Distributions
} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_DISTRIBUTIONS_H_
