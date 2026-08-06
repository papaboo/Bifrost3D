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
#include <Bifrost/Math/Matrix.h>
#include <Bifrost/Math/Vector.h>
#include <Bifrost/Math/Utils.h>

namespace Bifrost::Math::Distributions {

struct DirectionalSample {
    Vector3f direction;
    float PDF;
};

//=================================================================================================
// Triangle distribution.
// A Low-Distortion Map Between Triangle and Square, Heitz, 2019
//=================================================================================================
namespace Triangle {

__always_inline__ GPU_ENABLED  float PDF(float triangle_area) {
    return 1.0f / triangle_area;
}

__always_inline__ GPU_ENABLED  Vector3f sample_barycentric_coords(Vector2f random_sample) {
    float b0, b1;
    if (random_sample.x < random_sample.y) {
        b0 = random_sample.x * 0.5f;
        b1 = random_sample.y - b0;
    } else {
        b1 = random_sample.y * 0.5f;
        b0 = random_sample.x - b1;
    }
    return { b0, b1, 1 - b0 - b1 };
}

}//=================================================================================================
// Uniform sphere distribution.
//=================================================================================================
namespace Sphere {

__always_inline__ GPU_ENABLED Vector3f create_direction(float phi, float cos_theta) {
    float radius = sqrt(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    return Vector3f(radius * cos_phi, radius * sin_phi, cos_theta);
}

constexpr __always_inline__ GPU_ENABLED float PDF() { return 0.25f / PI<float>(); }

__always_inline__ GPU_ENABLED Vector3f sample_direction(Vector2f random_sample) {
    float cos_theta = 1.0f - 2.0f * random_sample.x;
    float phi = 2.0f * PI<float>() * random_sample.y;
    return create_direction(phi, cos_theta);
}

__always_inline__ GPU_ENABLED DirectionalSample sample(Vector2f random_sample) {
    return { sample_direction(random_sample), PDF() };
}

} // NS Sphere

//=================================================================================================
// Uniform hemisphere distribution.
//=================================================================================================
namespace UniformHemisphere {

constexpr __always_inline__ GPU_ENABLED float PDF() {
    return 0.5f / PI<float>();
}

__always_inline__ GPU_ENABLED DirectionalSample sample(Vector2f random_sample) {
    float cos_theta = random_sample.x;
    float phi = 2.0f * PI<float>() * random_sample.y;

    DirectionalSample res;
    res.direction = Sphere::create_direction(phi, cos_theta);;
    res.PDF = PDF();
    return res;
}

} // NS Uniform hemisphere

//=================================================================================================
// Cosine distribution.
//=================================================================================================
namespace Cosine {

__always_inline__ GPU_ENABLED float PDF(float abs_cos_theta) {
    return abs_cos_theta / PI<float>();
}

__always_inline__ GPU_ENABLED DirectionalSample sample(Vector2f random_sample) {
    float r2 = random_sample.x;
    float cos_theta = sqrt(r2);
    float r = sqrt(1.0f - r2);

    float phi = 2.0f * PI<float>() * random_sample.y;
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);

    DirectionalSample res;
    res.direction = Vector3f(r * cos_phi, r * sin_phi, cos_theta);
    res.PDF = cos_theta / PI<float>();
    return res;
}

} // NS Cosine

//=================================================================================================
// Clipped linearly transformed cosine distribution for OrenNayar.
//=================================================================================================
namespace OrenNayerCLTC {

using namespace Bifrost::Math;

__always_inline__ GPU_ENABLED Vector3f apply_tangent_basis(Matrix2x2f tangents, Vector3f w) {
    Vector2f xy = tangents * Vector2f(w.x, w.y);
    return Vector3f(xy, w.z);
}

// Unlike Listing 3 in the paper, we here limit the basis to a 2x2 matrix,
// as the rest of the entries in the 3x3 are just from the identity matrix and holds no value.
// This reduces the number of registers used.
__always_inline__ GPU_ENABLED Matrix2x2f orthonormal_tangents_ltc(Vector3f w) {
    Vector2f wh = Vector2f(w.x, w.y);
    float lenSqr = dot(wh, wh);
    Vector2f X = lenSqr > 0.0f ? wh / sqrt(lenSqr) : Vector2f(1, 0);
    Vector2f Y = Vector2f(-X.y, X.x); // cross(Z, X)

    Matrix2x2f res;
    res.set_column(0, X);
    res.set_column(1, Y);
    return res;
}

__always_inline__ GPU_ENABLED void oren_nayar_LTC_coefficients(float cos_theta, float roughness,
    float& a, float& b, float& c, float& d) {
    a = 1.0f + roughness * (0.303392f + (-0.518982f + 0.111709f*cos_theta)*cos_theta + (-0.276266f + 0.335918f*cos_theta)*roughness);
    b = roughness * (-1.16407f + 1.15859f*cos_theta + (0.150815f - 0.150105f*cos_theta)*roughness) / (cos_theta*cos_theta*cos_theta - 1.43545f);
    c = 1.0f + (0.20013f + (-0.506373f + 0.261777f*cos_theta)*cos_theta)*roughness;
    d = ((0.540852f + (-1.01625f + 0.475392f*cos_theta)*cos_theta)*roughness) / (-1.0743f + cos_theta * (0.0725628f + cos_theta));
}

__always_inline__ GPU_ENABLED DirectionalSample sample(float roughness, Vector3f wo, Vector2f random_sample) {
    float a, b, c, d;
    oren_nayar_LTC_coefficients(wo.z, roughness, a, b, c, d); // coeffs of LTC M

    float radius = sqrt(random_sample.x);
    float phi = 2.0f * PI<float>() * random_sample.y; // CLTC sampling
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    float x = radius * cos_phi;
    float y = radius * sin_phi; // CLTC sampling

    float vz = 1.0f / sqrt(d*d + 1.0f); // CLTC sampling factors
    float s = 0.5f * (1.0f + vz); // CLTC sampling factors
    x = -lerp(sqrt(1.0f - y * y), x, s); // CLTC sampling
    Vector3f wh = Vector3f(x, y, sqrt(fmaxf(1.0f - (x*x + y * y), 0.0f))); // wH sample via CLTC
    float pdf_wh = wh.z / (PI<float>() * s); // PDF of wH sample
    Vector3f wi = Vector3f(a*wh.x + b * wh.z, c*wh.y, d*wh.x + wh.z); // M wH (unnormalized)
    float wi_magnitude = magnitude(wi); // ||M wH|| = 1 / ||M^-1 wh||
    float determinant_M = c * (a - b * d); // |M|
    float pdf_wi = pdf_wh * wi_magnitude * wi_magnitude * wi_magnitude / determinant_M; // wi sample PDF
    auto from_LTC = orthonormal_tangents_ltc(wo); // wi -> local space
    wi = normalize(apply_tangent_basis(from_LTC, wi)); // wi -> local space

    DirectionalSample res;
    res.direction = wi;
    res.PDF = pdf_wi;
    return res;
}

__always_inline__ GPU_ENABLED float PDF(float roughness, Vector3f wo_shading, Vector3f wi_shading) {
    auto to_LTC = transpose(orthonormal_tangents_ltc(wo_shading)); // wi -> LTC space
    Vector3f wi = apply_tangent_basis(to_LTC, wi_shading); // wi -> LTC space

    float a, b, c, d;
    oren_nayar_LTC_coefficients(wo_shading.z, roughness, a, b, c, d); // coeffs of LTC M

    float determinant_M = c * (a - b * d); // |M|
    Vector3f wh = Vector3f(c*(wi.x - b * wi.z), (a - b * d)*wi.y, -c * (d*wi.x - a * wi.z)); // adj(M) wi
    float wh_magnitude_squared = dot(wh, wh); // |M| ||M^-1 wi||
    float vz = 1.0f / sqrt(d*d + 1.0f); // CLTC sampling factors
    float s = 0.5f * (1.0f + vz); // CLTC sampling factors
    return determinant_M * determinant_M / pow2(wh_magnitude_squared) * fmaxf(wh.z, 0.0f) / (PI<float>() * s); // wi sample PDF
}

} // NS OrenNayerCLTC

//=================================================================================================
// GGX distribution.
//=================================================================================================
namespace GGX {

__always_inline__ GPU_ENABLED float D(float alpha, float abs_cos_theta) {
    float alpha_sqrd = alpha * alpha;
    float cos_theta_sqrd = abs_cos_theta * abs_cos_theta;
    float tan_theta_sqrd = fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
    float cos_theta_cubed = cos_theta_sqrd * cos_theta_sqrd;
    float foo = alpha_sqrd + tan_theta_sqrd; // No idea what to call this.
    return alpha_sqrd / (PI<float>() * cos_theta_cubed * foo * foo);
}

__always_inline__ GPU_ENABLED float PDF(float alpha, float abs_cos_theta) {
    return D(alpha, abs_cos_theta) * abs_cos_theta;
}

__always_inline__ GPU_ENABLED DirectionalSample sample(float alpha, Vector2f random_sample) {
    float phi = random_sample.y * (2.0f * PI<float>());

    float tan_theta_sqrd = alpha * alpha * random_sample.x / (1.0f - random_sample.x);
    float cos_theta = 1.0f / sqrt(1.0f + tan_theta_sqrd);

    DirectionalSample res;
    res.direction = Sphere::create_direction(phi, cos_theta);
    res.PDF = PDF(alpha, cos_theta);
    return res;
}

} // NS GGX

//=================================================================================================
// Sampling the visible normal distribution function for GGX.
// Sampling Visible GGX Normals with Spherical Caps, Dupuy et al, 2023.
// Sampling the GGX Distribution of Visible Normals, Heitz, 2018.
// Importance Sampling Microfacet-Based BSDFs with the Distribution of Visible Normals, Heitz, 2014.
// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz, 2014.
//=================================================================================================
namespace GGX_VNDF {

using namespace Bifrost::Math;

// Sampling the GGX Distribution of Visible Normals, equation 1.
__always_inline__ GPU_ENABLED float D(float alpha_x, float alpha_y, Vector3f halfway) {
    float m = pow2(halfway.x / alpha_x) + pow2(halfway.y / alpha_y) + pow2(halfway.z);
    return 1 / (PI<float>() * alpha_x * alpha_y * pow2(m));
}
__always_inline__ GPU_ENABLED float D(float alpha, Vector3f halfway) { return D(alpha, alpha, halfway); }

// Sampling the GGX Distribution of Visible Normals, equation 2.
__always_inline__ GPU_ENABLED float lambda(float alpha_x, float alpha_y, Vector3f w) {
    return 0.5f * (-1 + sqrt(1 + (pow2(alpha_x * w.x) + pow2(alpha_y * w.y)) / pow2(w.z)));
}
__always_inline__ GPU_ENABLED float lambda(float alpha, Vector3f w) { return lambda(alpha, alpha, w); }

// Sampling the GGX Distribution of Visible Normals, listing 1.
__always_inline__ GPU_ENABLED Vector3f sample_halfway_heitz(float alpha_x, float alpha_y, Vector3f wo, Vector2f random_sample) {
    // Section 3.2: transforming the view direction to the hemisphere configuration
    Vector3f Vh = normalize(Vector3f(alpha_x * wo.x, alpha_y * wo.y, wo.z));

    // Section 4.1: orthonormal basis
    Vector3f T1 = (Vh.z < 0.9999f) ? normalize(Vector3f(-Vh.y, Vh.x, 0)) : Vector3f(1, 0, 0);
    Vector3f T2 = cross(Vh, T1);

    // Section 4.2: parameterization of the projected area
    float r = sqrt(random_sample.x);
    float phi = 2.0f * PI<float>() * random_sample.y;
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    float t1 = r * cos_phi;
    float t2 = r * sin_phi;
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    Vector3f Nh = T1 * t1 + T2 * t2 + Vh * sqrt(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    return normalize(Vector3f(alpha_x * Nh.x, alpha_y * Nh.y, fmaxf(0.0f, Nh.z)));
}

// Sampling Visible GGX Normals with Spherical Caps, listing 1 and 3.
__always_inline__ GPU_ENABLED Vector3f sample_halfway(float alpha_x, float alpha_y, Vector3f wo, Vector2f random_sample) {
    // Section 3.2: transforming the view direction to the hemisphere configuration
    Vector3f wo_std = normalize(Vector3f(alpha_x * wo.x, alpha_y * wo.y, wo.z));

    // sample a spherical cap in (-wi.z, 1]
    float phi = 2.0f * PI<float>() * random_sample.y;
    float z = fma(1.0f - random_sample.x, 1.0f + wo_std.z, -wo_std.z);
    float sin_theta = sqrt(clamp(1.0f - z * z, 0.0f, 1.0f));
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    Vector3f c = Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, z);

    // compute halfway direction;
    Vector3f wi_std = c + wo_std;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    return normalize(Vector3f(alpha_x * wi_std.x, alpha_y * wi_std.y, fmaxf(0.0f, wi_std.z)));
}

__always_inline__ GPU_ENABLED Vector3f sample_halfway(float alpha, Vector3f wo, Vector2f random_sample) { return sample_halfway(alpha, alpha, wo, random_sample); }

// Sampling the GGX Distribution of Visible Normals, equation 3.
__always_inline__ GPU_ENABLED float PDF(float alpha, Vector3f wo, Vector3f halfway) {
    float recip_G1 = 1.0f + lambda(alpha, wo);
    float D = Distributions::GGX_VNDF::D(alpha, halfway);
    return dot(wo, halfway) * D / (recip_G1 * abs(wo.z));
}

__always_inline__ GPU_ENABLED Distributions::DirectionalSample sample(float alpha, Vector3f wo, Vector2f random_sample) {
    Distributions::DirectionalSample sample;
    sample.direction = sample_halfway(alpha, wo, random_sample);
    sample.PDF = PDF(alpha, wo, sample.direction);
    return sample;
}

} // NS GGX_VNDF

//=================================================================================================
// Sampling a tighter bound of the visible normal distribution function for GGX.
// Bounded VNDF Sampling for Smith–GGX Reflections, Eto et al, 2023.
// Sampling Visible GGX Normals with Spherical Caps, Dupuy et al, 2023.
// Sampling the GGX Distribution of Visible Normals, Heitz, 2018.
// Importance Sampling Microfacet-Based BSDFs with the Distribution of Visible Normals, Heitz, 2014.
// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz, 2014.
//=================================================================================================
namespace GGX_Bounded_VNDF {

using namespace Bifrost::Math;

// Sampling the GGX Distribution of Visible Normals, equation 1.
__always_inline__ GPU_ENABLED float D(Vector2f alpha, Vector3f halfway) {
    float m = pow2(halfway.x / alpha.x) + pow2(halfway.y / alpha.y) + pow2(halfway.z);
    return 1 / (PI<float>() * alpha.x * alpha.y * pow2(m));
}
__always_inline__ GPU_ENABLED float D(float alpha, Vector3f halfway) { return D(Vector2f(alpha, alpha), halfway); }

// Bounded VNDF Sampling for Smith–GGX Reflections, listing 1.
__always_inline__ GPU_ENABLED Vector3f sample_reflection(Vector2f alpha, Vector3f wo, Vector2f random_sample) {
    Vector3f wo_std = normalize(Vector3f(wo.x * alpha.x, wo.y * alpha.y, wo.z));

    // Sample a spherical cap
    float phi = 2.0f * PI<float>() * random_sample.y;
    float a = fminf(alpha.x, alpha.y); // Eq. 6
    float s = 1.0f + magnitude(Vector2f(wo.x, wo.y)); // Omit sign for a <=1
    float a2 = a * a; float s2 = s * s;
    float k = (1.0f - a2) * s2 / (s2 + a2 * wo.z * wo.z); // Eq. 5
    float b = wo.z >= 0 ? k * wo_std.z : wo_std.z;
    float z = fma(1.0f - random_sample.x, 1.0f + b, -b);
    float sin_theta = sqrt(fmaxf(1.0f - z * z, 0.0f));
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    Vector3f o_std = { sin_theta * cos_phi, sin_theta * sin_phi, z };

    // Compute the microfacet normal m
    Vector3f halfway_std = wo_std + o_std;
    Vector3f halfway = normalize(Vector3f(halfway_std.x * alpha.x, halfway_std.y * alpha.y, halfway_std.z));

    // Return the reflection vector o
    return reflect(-wo, halfway);
}

__always_inline__ GPU_ENABLED Vector3f sample_reflection(float alpha, Vector3f wo, Vector2f random_sample) { return sample_reflection(Vector2f(alpha), wo, random_sample); }

// Bounded VNDF Sampling for Smith–GGX Reflections, listing 2.
__always_inline__ GPU_ENABLED float reflection_PDF(Vector2f alpha, Vector3f wo, Vector3f wi) {
    Vector3f halfway = normalize(wo + wi);
    float ndf = D(alpha, halfway);
    Vector2f ao = alpha * Vector2f(wo.x, wo.y);
    float len2 = dot(ao, ao);
    float t = sqrt(len2 + wo.z * wo.z);
    if (wo.z >= 0.0f) {
        float min_alpha = fminf(alpha.x, alpha.y); // Eq. 6
        float s = 1.0f + magnitude(Vector2f(wo.x, wo.y)); // Omit sign for a <=1
        float min_alpha_squared = min_alpha * min_alpha; float s2 = s * s;
        float k = (1.0f - min_alpha_squared) * s2 / (s2 + min_alpha_squared * wo.z * wo.z); // Eq. 5
        return ndf / (2.0f * (k * wo.z + t)); // Eq. 8 * || dm/do ||
    }

    // Numerically stable form of the previous PDF for wo.z < 0
    return ndf * (t - wo.z) / (2.0f * len2); // = Eq. 7 * || dm/do ||
}

__always_inline__ GPU_ENABLED float reflection_PDF(float alpha, Vector3f wo, Vector3f wi) { return reflection_PDF(Vector2f(alpha, alpha), wo, wi); }

__always_inline__ GPU_ENABLED Distributions::DirectionalSample sample(Vector2f alpha, Vector3f wo, Vector2f random_sample) {
    Distributions::DirectionalSample sample;
    sample.direction = sample_reflection(alpha, wo, random_sample);
    sample.PDF = reflection_PDF(alpha, wo, sample.direction);
    return sample;
}

__always_inline__ GPU_ENABLED Distributions::DirectionalSample sample(float alpha, Vector3f wo, Vector2f random_sample) {
    return sample(Vector2f(alpha, alpha), wo, random_sample);
}

} // NS GGX_Bounded_VNDF

//=================================================================================================
// Exponential distribution.
//=================================================================================================
namespace Exponential {

struct Sample {
    float distance;
    float PDF;
};

__always_inline__ GPU_ENABLED float PDF(float sigma, float distance) { return sigma * exp(-sigma * distance); }

__always_inline__ GPU_ENABLED float sample_distance(float sigma, float random_sample) { return -log(1 - random_sample) / sigma; }

__always_inline__ GPU_ENABLED Sample sample(float sigma, float random_sample) {
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

// When g approximates -1 and random_sample approximates 0 or when g approximates 1 and random_sample approximates 1,
// the computation of cos_theta below is unstable and can give 0, leading to NaNs.
// For now we limit g to the range where it is stable.
__always_inline__ GPU_ENABLED float safe_g(float g) { return clamp(g, -.99f, .99f); }

__always_inline__ GPU_ENABLED float evaluate(float g, float cos_theta) {
    g = safe_g(g);
    float denominator = 1 + pow2(g) + 2 * g * cos_theta;
    constexpr float recip_4_pi = 1.0f / (4 * PI<float>());
    return recip_4_pi * (1 - pow2(g)) / (denominator * sqrt(max(0.0f, denominator)));
}

__always_inline__ GPU_ENABLED float evaluate(float g, Vector3f wo, Vector3f wi) {
    float cos_theta = dot(wo, wi);
    return evaluate(g, cos_theta);
}

// Sample the cosine of the angle for the distribution.
__always_inline__ GPU_ENABLED float sample_cos_theta(float g, float random_sample) {
    g = safe_g(g);

    if (abs(g) < 1e-3f)
        return 1 - 2 * random_sample; // Use spherical distribution directly when g is close to 0.
    else
        return -1 / (2 * g) * (1 + pow2(g) - pow2((1 - pow2(g)) / (1 + g - 2 * g * random_sample)));
}

// Sample a direction in the distribution wrt [0,0,1] as wo.
__always_inline__ GPU_ENABLED Vector3f sample_direction(float g, Vector2f random_sample) {
    float cos_theta = sample_cos_theta(g, random_sample.x);

    float sin_theta = sqrt(fmaxf(0.0f, 1.0f - pow2(cos_theta)));
    float phi = 2.0f * PI<float>() * random_sample.y;
    return Vector3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Sample the distribution wrt [0,0,1] as wo.
__always_inline__ GPU_ENABLED DirectionalSample sample(float g, Vector2f random_sample) {
    Vector3f wi = sample_direction(g, random_sample);
    float pdf = evaluate(g, wi.z);
    return { wi, pdf };
}

// Sample a direction in the distribution.
__always_inline__ GPU_ENABLED Vector3f sample_direction(float g, Vector3f wo, Vector2f random_sample) {
    Vector3f local_wi = sample_direction(g, random_sample);

    Vector3f tangent, bitangent;
    compute_tangents(wo, tangent, bitangent);
    return tangent * local_wi.x + bitangent * local_wi.y + wo * local_wi.z;
}

// Sample the distribution.
__always_inline__ GPU_ENABLED DirectionalSample sample(float g, Vector3f wo, Vector2f random_sample) {
    Vector3f wi = sample_direction(g, wo, random_sample);
    float pdf = evaluate(g, dot(wo, wi));
    return { wi, pdf };
}

} // NS HenyeyGreenstein

} // NS Bifrost::Math::Distributions

#endif // _BIFROST_MATH_DISTRIBUTIONS_H_
