// Bifrost shading utilities
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_UTILS_H_
#define _BIFROST_ASSETS_SHADING_UTILS_H_

#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Vector.h>

namespace Bifrost::Assets::Shading {

// ------------------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------------------
constexpr float PIf = 3.14159265358979323846f;
constexpr float TWO_PIf = 6.28318530717958647692f;
constexpr float RECIP_PIf = 0.31830988618379067153776752674503f;

// ------------------------------------------------------------------------------------------------
// General shading utils
// ------------------------------------------------------------------------------------------------

_inline_all_archs_ bool same_hemisphere(Math::Vector3f wo, Math::Vector3f wi) {
    return wo.z * wi.z >= 0.0f;
}

// Local helpers prefixed with _ to not clash with the definitions in math utils.
constexpr _inline_all_archs_ float _pow2(float x) { return x * x; }
constexpr _inline_all_archs_ float _pow5(float x) { return _pow2(x * x) * x; }

_inline_all_archs_ float schlick_fresnel(float incident_specular, float abs_cos_theta) {
    return incident_specular + (1.0f - incident_specular) * _pow5(1.0f - abs_cos_theta);
}
_inline_all_archs_ Math::RGB schlick_fresnel(Math::RGB incident_specular, float abs_cos_theta) {
    float schlick_t = _pow5(1.0f - abs_cos_theta);
    return incident_specular + (1.0f - incident_specular) * schlick_t;
}

_inline_all_archs_ float dielectric_schlick_fresnel(float incident_specular, float abs_cos_theta, float ior_i_over_o) {
    // Return 1.0 for full reflection in case of total internal reflection.
    // cos(theta) is expected to be absolute and ior_i_over_o to have been preadjusted to fit the side hit.
    // Sources:
    // * PBRT's FrDielectric in scattering.h
    // * https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/totalinternalreflection
    float sin2_theta = 1 - _pow2(abs_cos_theta);
    if (sin2_theta >= _pow2(ior_i_over_o))
        return 1.0f;

    float t = _pow5(1.0f - abs_cos_theta);
    return (1.0f - t) * incident_specular + t;
}

// Specularity of dielectrics at normal incidence, where the ray is leaving a medium with index of refraction ior_o
// and entering a medium with index of refraction, ior_i.
// Ray Tracing Gems 2, Chapter 9, The Schlick Fresnel Approximation, page 110 footnote.
constexpr _inline_all_archs_ float dielectric_specularity(float ior_o, float ior_i) {
    return _pow2((ior_o - ior_i) / (ior_o + ior_i));
}

// Specularity of dielectrics at normal incidence, where the ray is leaving a dielectric medium with index of refraction ior_o
// and entering a conductor medium with index of refraction, ior_i, and extinction coefficient, ext_i.
_inline_all_archs_ Math::RGB conductor_specularity(Math::RGB ior_o, Math::RGB ior_i, Math::RGB ext_i) {
    Math::RGB ext_i_sqrd = pow2(ext_i);
    return (pow2(ior_o - ior_i) + ext_i_sqrd) / (pow2(ior_o + ior_i) + ext_i_sqrd);
}

// Estimates a dielectric's index of refraction from specularity.
// It is assumed that the specularity describes the specularity of the material when bordering air, i.e ior_o is 1.0.
// Finding the index of refraction requires solving a second degree polynomial with two solutions.
// For dielectrics the solution with the largest value is the correct one.
// The whole thing can be reduced to the expression below.
// Source: Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering, section 3.2, Burley, 2015
_inline_all_archs_ float dielectric_ior_from_specularity(float specularity) {
    return 2.0f / (1.0f - sqrt(specularity)) - 1.0f;
}

// Estimates a conductor's index of refraction from specularity.
// It is assumed that the specularity describes the specularity of the material when bordering air, i.e ior_o is 1.0.
// Finding the index of refraction requires solving a second degree polynomial with two solutions.
// For dielectrics the solution with the lowest value is the correct one.
_inline_all_archs_ Math::RGB conductor_ior_from_specularity(Math::RGB specularity, Math::RGB ext_i) {
    Math::RGB a = specularity - 1;
    Math::RGB b = 2 * specularity + 2;
    Math::RGB c = a + (specularity - 1) * pow2(ext_i);
    Math::RGB d = b * b - 4 * a * c;
    Math::RGB sqrt_d = { sqrt(d.r), sqrt(d.g), sqrt(d.b) };
    return (b * -1.0f + sqrt_d) / (2 * a);
}

// Adjust the specularity of a dielectric material, which is set with the assumption that the material is seen through air,
// to the specularity that the material would have as seen through a volume with the ior defined by the exterior ior.
_inline_all_archs_ float adjust_dielectric_specularity_to_exterior_medium(float exterior_ior, float specularity_through_air) {
    // Convert specularity to base_ior
    float base_ior = dielectric_ior_from_specularity(specularity_through_air);

    // Compute new base specularity
    return dielectric_specularity(exterior_ior, base_ior);
}

// Adjust the specularity of a conductor material, which is set with the assumption that the material is seen through air,
// to the specularity that the material would have as seen through a volume with the ior defined by the exterior ior.
_inline_all_archs_ Math::RGB adjust_conductor_specularity_to_exterior_medium(Math::RGB exterior_ior, Math::RGB specularity_through_air, Math::RGB extinction_coefficient) {
    // Convert specularity to base_ior
    Math::RGB base_ior = conductor_ior_from_specularity(specularity_through_air, extinction_coefficient);

    // Compute new base specularity
    return conductor_specularity(exterior_ior, base_ior, extinction_coefficient);
}

// Copy of OptiX' refract implementation, but with normal set to (0, 0, 1).
_inline_all_archs_ bool refract(Math::Vector3f& refraction_direction, Math::Vector3f wi, Math::Vector3f n, const float ior) {
    Math::Vector3f nn = n;
    float negNdotV = dot(wi, nn);
    float ior_i_over_o;

    if (negNdotV > 0.0f)
    {
        ior_i_over_o = ior;
        nn = -n;
        negNdotV = -negNdotV;
    }
    else
        ior_i_over_o = 1.f / ior;

    const float k = 1.f - ior_i_over_o * ior_i_over_o * (1.f - negNdotV * negNdotV);

    refraction_direction = normalize(ior_i_over_o * wi - (ior_i_over_o * negNdotV + sqrtf(k)) * nn);
    return k >= 0.0f;
}

// Copy of OptiX' refract implementation, but with normal set to (0, 0, 1).
_inline_all_archs_ bool refract(Math::Vector3f& refraction_direction, Math::Vector3f wi, float ior_i_over_o) {
    float normal_z = 1;
    float cos_theta_i = wi.z;

    if (cos_theta_i > 0.0f) {
        normal_z = -1;
        cos_theta_i = -cos_theta_i;
    } else
        ior_i_over_o = 1.f / ior_i_over_o;

    float k = 1.0f - ior_i_over_o * ior_i_over_o * (1.0f - cos_theta_i * cos_theta_i);

    refraction_direction = ior_i_over_o * wi - Math::Vector3f(0, 0, (ior_i_over_o * cos_theta_i + sqrtf(k)) * normal_z);
    return k >= 0.0f;
}

_inline_all_archs_ bool refract(float& refraction_cos_theta, float cos_theta_i, float ior_i_over_o) {
    float normal_z = 1;
    float adjusted_cos_theta_i = cos_theta_i;

    if (cos_theta_i > 0.0f) {
        normal_z = -1;
        adjusted_cos_theta_i = -adjusted_cos_theta_i;
    } else
        ior_i_over_o = 1.f / ior_i_over_o;

    float k = 1.0f - _pow2(ior_i_over_o) * (1.0f - _pow2(adjusted_cos_theta_i));

    refraction_cos_theta = ior_i_over_o * cos_theta_i - (ior_i_over_o * adjusted_cos_theta_i + sqrtf(k)) * normal_z;
    return k >= 0.0f;
}

// ------------------------------------------------------------------------------------------------
// Volume scattering utilities
// ------------------------------------------------------------------------------------------------

// Compute the attenuation of a beam passing through a medium.
// https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law
_inline_all_archs_ float beers_law(float optical_density, float distance) { return expf(-optical_density * distance); }

// ------------------------------------------------------------------------------------------------
// BSDF sampling utils
// ------------------------------------------------------------------------------------------------

struct alignas(8) BSDFResponse {
    Math::RGB reflectance;
    Math::MonteCarlo::PDF PDF;

    _inline_all_archs_ static BSDFResponse none() {
        BSDFResponse evaluation = {};
        return evaluation;
    }
};

struct alignas(16) BSDFSample {
    Math::RGB reflectance;
    Math::MonteCarlo::PDF PDF;
    Math::Vector3f direction;
    float __padding;

    _inline_all_archs_ static BSDFSample none() {
        BSDFSample sample = {};
        return sample;
    }
};

struct alignas(16) SeparableBSSRDFPositionSample {
    Math::RGB reflectance;
    Math::MonteCarlo::PDF PDF;
    Math::Vector3f position;
    float __padding;

    _inline_all_archs_ static SeparableBSSRDFPositionSample none() {
        SeparableBSSRDFPositionSample sample = {};
        return sample;
    }
};

} // NS Bifrost::Assets::Shading

#endif // _BIFROST_ASSETS_SHADING_UTILS_H_