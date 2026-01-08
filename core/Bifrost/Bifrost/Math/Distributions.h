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
#include <Bifrost/Math/Utils.h>

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

__always_inline__ float PDF() { return 1.0f / (4.0f * PI<float>()); }

__always_inline__ Vector3f sample(Vector2f random_sample) {
    float z = 1.0f - 2.0f * random_sample.x;
    float r = sqrt(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2.0f * PI<float>() * random_sample.y;
    return Vector3f(r * cos(phi), r * sin(phi), z);
}

} // NS Sphere

//=================================================================================================
// Exponential distribution.
//=================================================================================================
namespace Exponential {

struct Sample {
    float distance;
    float PDF;
};

__always_inline__ float PDF(float sigma, float distance) { return sigma * std::expf(-sigma * distance); }

__always_inline__ float sample_distance(float sigma, float random_sample) { return -std::logf(1 - random_sample) / sigma; }

__always_inline__ Sample sample(float sigma, float random_sample) {
    float distance = sample_distance(sigma, random_sample);
    float pdf = PDF(sigma, distance);
    return { distance, pdf };
}

} // NS Exponential

//=================================================================================================
// Henyey-Greenstein distribution.
// Physically Based Rendering version 4, section 11.3.1
// https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions#TheHenyeyndashGreensteinPhaseFunction
//=================================================================================================
namespace HenyeyGreenstein {

struct Sample {
    Vector3f direction;
    float PDF;
};

// When g approximates -1 and random_sample approximates 0 or when g approximates 1 and random_sample approximates 1,
// the computation of cos_theta below is unstable and can give 0, leading to NaNs.
// For now we limit g to the range where it is stable.
__always_inline__ float safe_g(float g) {
    return clamp(g, -.99f, .99f);
}

__always_inline__ float evaluate(float g, float cos_theta) {
    g = safe_g(g);
    float denominator = 1 + pow2(g) + 2 * g * cos_theta;
    constexpr float recip_4_pi = 1.0f / (4 * PI<float>());
    return recip_4_pi * (1 - pow2(g)) / (denominator * sqrt(max(0.0f, denominator)));
}

__always_inline__ float evaluate(float g, Vector3f wo, Vector3f wi) {
    float cos_theta = dot(wo, wi);
    return evaluate(g, cos_theta);
}

// Sample the cosine of the angle for the distribution.
__always_inline__ float sample_cos_theta(float g, float random_sample) {
    g = safe_g(g);

    if (abs(g) < 1e-3f)
        return 1 - 2 * random_sample; // Use spherical distribution directly when g is close to 0.
    else
        return -1 / (2 * g) * (1 + pow2(g) - pow2((1 - pow2(g)) / (1 + g - 2 * g * random_sample)));
}

// Sample a direction in the distribution wrt [0,0,1] as wo.
__always_inline__ Vector3f sample_direction(float g, Vector2f random_sample) {
    float cos_theta = sample_cos_theta(g, random_sample.x);

    float sin_theta = sqrt(fmaxf(0.0f, 1.0f - pow2(cos_theta)));
    float phi = 2.0f * PI<float>() * random_sample.y;
    return Vector3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Sample the distribution wrt [0,0,1] as wo.
__always_inline__ Sample sample(float g, Vector2f random_sample) {
    Vector3f wi = sample_direction(g, random_sample);
    float pdf = evaluate(g, wi.z);
    return { wi, pdf };
}

// Sample a direction in the distribution.
__always_inline__ Vector3f sample_direction(float g, Vector3f wo, Vector2f random_sample) {
    Vector3f local_wi = sample_direction(g, random_sample);

    Vector3f tangent, bitangent;
    compute_tangents(wo, tangent, bitangent);
    return tangent * local_wi.x + bitangent * local_wi.y + wo * local_wi.z;
}

// Sample the distribution.
__always_inline__ Sample sample(float g, Vector3f wo, Vector2f random_sample) {
    Vector3f wi = sample_direction(g, wo, random_sample);
    float pdf = evaluate(g, dot(wo, wi));
    return { wi, pdf };
}

} // NS HenyeyGreenstein

} // NS Distributions
} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_DISTRIBUTIONS_H_
