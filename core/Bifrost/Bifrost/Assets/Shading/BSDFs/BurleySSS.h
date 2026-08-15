// Bifrost functions for the Burley normalized subsurface scattering.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_SSS_H_
#define _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_SSS_H_

#include <Bifrost/Assets/Shading/Utils.h>
#include <Bifrost/Math/Distributions.h>

namespace Bifrost::Assets::Shading::BSDFs {

//----------------------------------------------------------------------------
// Implementation of the Burley normalized subsurface scattering model.
// Sources:
// * Approximate Reflectance Profiles for Efficient Subsurface Scattering, Christensen et al., 15.
// * Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering. Burley et at., 15.
// * Efficient Screen Space Subsurface Scattering Using Burley's Normalized Diffusion in Realtime, Golubev et al, 2018,
//   https://advances.realtimerendering.com/s2018/Efficient%20screen%20space%20subsurface%20scattering%20Siggraph%202018.pdf
// * Sampling Burley's Normalized Diffusion Profiles, Golubev, 2019, https://zero-radiance.github.io/post/sampling-diffusion/
//----------------------------------------------------------------------------
namespace BurleySSS {

using namespace Bifrost::Math;

// Container for parameters for the Burley's subsurface scattering approximation.
// We precompute the parameters and store them to avoid redoing the same transformations multiple times.
struct Parameters {
    enum class LightConfig { Search, Diffuse };

    RGB diffuse_albedo;

    // l / s from equation 3 in Approximate Reflectance Profiles for Efficient Subsurface Scattering
    Vector3f diffuse_mean_free_path;

    // Equation 5 in Approximate Reflectance Profiles for Efficient Subsurface Scattering.
    _inline_all_archs_ static Vector3f search_light_mean_free_path_scaling_term(RGB diffuse_albedo) {
        RGB a = diffuse_albedo - 0.8f;
        RGB aaa = a * a * a;
        RGB s = 1.85f - diffuse_albedo + RGB(abs(aaa.r), abs(aaa.g), abs(aaa.b)) * 7.0f;
        return { s.r, s.g, s.b };
    }

    // Equation 6 in Approximate Reflectance Profiles for Efficient Subsurface Scattering.
    _inline_all_archs_ static Vector3f diffuse_light_mean_free_path_scaling_term(RGB diffuse_albedo) {
        RGB s = 1.9f - diffuse_albedo + pow2(diffuse_albedo - 0.8f) * 3.5f;
        return { s.r, s.g, s.b };
    }

    _inline_all_archs_ static Parameters create(RGB diffuse_albedo, Vector3f mean_free_path, LightConfig light_config = LightConfig::Diffuse) {
        Vector3f s = light_config == LightConfig::Diffuse ?
            diffuse_light_mean_free_path_scaling_term(diffuse_albedo) :
            search_light_mean_free_path_scaling_term(diffuse_albedo);
        return { diffuse_albedo, mean_free_path / s };
    }
};

// Equation 2 in Approximate Reflectance Profiles for Efficient Subsurface Scattering.
_inline_all_archs_ float evaluate(float entry_and_exit_distance, float diffuse_mean_free_path) {
    // Note: These are not actually derived as single and multi scattering terms, but can be viewed as such and it makes the code names more readable.
    float single_scattering_term = expf(-entry_and_exit_distance / (3.0f * diffuse_mean_free_path));
    float multi_scattering_term = single_scattering_term * single_scattering_term * single_scattering_term;
    float normalizer = 8 * PIf * diffuse_mean_free_path * entry_and_exit_distance;

    return (single_scattering_term + multi_scattering_term) / normalizer;
}

_inline_all_archs_ RGB evaluate(float entry_and_exit_distance, Vector3f diffuse_mean_free_path) {
    return { evaluate(entry_and_exit_distance, diffuse_mean_free_path.x),
             evaluate(entry_and_exit_distance, diffuse_mean_free_path.y),
             evaluate(entry_and_exit_distance, diffuse_mean_free_path.z) };
}

// Equation 3 in Approximate Reflectance Profiles for Efficient Subsurface Scattering.
_inline_all_archs_ RGB evaluate(Parameters params, Vector3f po, Vector3f pi) {
    float r = distance(po, pi);
    return params.diffuse_albedo * evaluate(r, params.diffuse_mean_free_path);
}

// Performs sampling of a Normalized Burley diffusion profile in polar coordinates.
// 'u' is the random number (the value of the CDF): [0, 1).
// rcp(s) = 1 / ShapeParam = diffuse_mean_free_path.
// 'radius' is the sampled radial distance, s.t. (u = 0 -> r = 0) and (u = 1 -> r = Inf).
// rcp(Pdf) is the reciprocal of the corresponding PDF value.
// Derivation and source: https://zero-radiance.github.io/post/sampling-diffusion/
// The PDF has been changed to include 'radius' to convert from polar to cartesian coordinates.
_inline_all_archs_ void sample_diffusion_profile(float u, float diffuse_mean_free_path, float& radius, float& rcpPdf) {
    const float LOG2_E = 1.44269504089f;

    u = 1 - u; // Convert CDF to CCDF; the resulting value of (u != 0)

    float g = 1 + (4 * u) * (2 * u + sqrt(1 + (4 * u) * u));
    float n = exp2(log2(g) * (-1.0f / 3.0f));                // g^(-1/3)
    float p = (g * n) * n;                                   // g^(+1/3)
    float c = 1 + p + n;                                     // 1 + g^(+1/3) + g^(-1/3)
    float x = (3 / LOG2_E) * log2(c / (4 * u));              // 3 * Log[c / (4 * u)]

    // x      = s * r
    // exp_13 = Exp[-x/3] = Exp[-1/3 * 3 * Log[c / (4 * u)]]
    // exp_13 = Exp[-Log[c / (4 * u)]] = (4 * u) / c
    // exp_1  = Exp[-x] = exp_13 * exp_13 * exp_13
    // expSum = exp_1 + exp_13 = exp_13 * (1 + exp_13 * exp_13)
    // rcpExp = rcp(expSum) = c^3 / ((4 * u) * (c^2 + 16 * u^2))
    float cc = c * c;
    float four_u = 4 * u;
    float rcpExp = (cc * c) / (four_u * (cc + pow2(four_u)));

    radius = x * diffuse_mean_free_path;
    rcpPdf = (8 * PIf * radius * diffuse_mean_free_path) * rcpExp; // ((8 * Pi * r) / s) / (Exp[-s * r / 3] + Exp[-s * r])
}

// Performs sampling of a Normalized Burley diffusion profile in polar coordinates.
// 'u' is the random number (the value of the CDF): [0, 1).
// rcp(s) = 1 / ShapeParam = diffuse_mean_free_path.
// Returns the sampled radial distance, s.t. (u = 0 -> r = 0) and (u = 1 -> r = Inf).
// Derivation and source: https://zero-radiance.github.io/post/sampling-diffusion/
_inline_all_archs_ float sample_diffusion_profile(float u, float diffuse_mean_free_path) {
    const float LOG2_E = 1.44269504089f;

    u = 1 - u; // Convert CDF to CCDF; the resulting value of (u != 0)

    float g = 1 + (4 * u) * (2 * u + sqrt(1 + (4 * u) * u));
    float n = exp2(log2(g) * (-1.0f / 3.0f));                // g^(-1/3)
    float p = (g * n) * n;                                   // g^(+1/3)
    float c = 1 + p + n;                                     // 1 + g^(+1/3) + g^(-1/3)
    float x = (3 / LOG2_E) * log2(c / (4 * u));              // 3 * Log[c / (4 * u)]
    return x * diffuse_mean_free_path;
}

// Brian Karis' approximation to sampling the radius from Burley's diffusion profile.
// Real-time subsurface scattering with single pass variance-guided adaptive importance sampling, Xie et al., 2020.
_inline_all_archs_ float sample_diffusion_profile_approximation(float u, float diffuse_mean_free_path) {
    // const float c = 2.5715f; // c value proposed in the paper.
    const float c = 2.6f; // c value used in Unreal Engine 5, November 2025. See RadiusRootFindByApproximation in BurleyNormalizedSSSCommon.ush
    return diffuse_mean_free_path * ((2 - c) * u - 2) * log(1 - u);
}

// Sample the distribution based on the channel with the largest scattering distance,
// as suggested in Efficient Screen Space Subsurface Scattering Using Burley's Normalized Diffusion in Realtime, slide 26.
namespace SampleMostScattering {

_inline_all_archs_ MonteCarlo::PDF pdf(Parameters params, Vector3f po, Vector3f pi) {
    float r = distance(po, pi);
    float sampled_diffuse_mean_free_path = fmax(params.diffuse_mean_free_path.x, fmax(params.diffuse_mean_free_path.y, params.diffuse_mean_free_path.z));
    return evaluate(r, sampled_diffuse_mean_free_path);
}

_inline_all_archs_ BSDFResponse evaluate_with_PDF(Parameters params, Vector3f po, Vector3f pi) {
    auto reflectance = evaluate(params, po, pi);
    auto PDF = SampleMostScattering::pdf(params, po, pi);
    return { reflectance, PDF };
}

// Sample the Burley normalized diffusion profile.
_inline_all_archs_ SeparableBSSRDFPositionSample sample(Parameters params, Vector3f po, Vector3f random_sample) {
    float sample_diffuse_mean_free_path = fmax(params.diffuse_mean_free_path.x, fmax(params.diffuse_mean_free_path.y, params.diffuse_mean_free_path.z));

    float radius, reciprocal_PDF;
    sample_diffusion_profile(random_sample.x, sample_diffuse_mean_free_path, radius, reciprocal_PDF);

    // Sample an angle and turn into an offset from origo.
    float phi = 2.0f * PIf * random_sample.y;
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    Vector3f pi_offset = { radius * cos_phi, radius * sin_phi, 0.0f };

    SeparableBSSRDFPositionSample bssrdf_sample;
    bssrdf_sample.position = po + pi_offset;
    bssrdf_sample.PDF = 1 / reciprocal_PDF;
    bssrdf_sample.reflectance = params.diffuse_albedo * evaluate(radius, params.diffuse_mean_free_path);
    return bssrdf_sample;
}

} // NS SampleMaxChannel

namespace AlbedoMIS {

_inline_all_archs_ MonteCarlo::PDF pdf(Parameters params, Vector3f po, Vector3f pi) {
    float r = distance(po, pi);
    RGB per_channel_PDF = evaluate(r, params.diffuse_mean_free_path);

    RGB per_channel_probability = params.diffuse_albedo / sum(params.diffuse_albedo);

    return sum(per_channel_PDF * per_channel_probability);
}

_inline_all_archs_ BSDFResponse evaluate_with_PDF(Parameters params, Vector3f po, Vector3f pi) {
    auto reflectance = evaluate(params, po, pi);
    auto PDF = AlbedoMIS::pdf(params, po, pi);
    return { reflectance, PDF };
}

// Sample the Burley normalized diffusion profile.
_inline_all_archs_ SeparableBSSRDFPositionSample sample(Parameters params, Vector3f po, Vector3f random_sample) {
    RGB per_channel_probability = params.diffuse_albedo / sum(params.diffuse_albedo);

    float sample_scattering_distance;
    if (random_sample.z < per_channel_probability.r)
        sample_scattering_distance = params.diffuse_mean_free_path.x;
    else if (random_sample.z < (per_channel_probability.r + per_channel_probability.g))
        sample_scattering_distance = params.diffuse_mean_free_path.y;
    else
        sample_scattering_distance = params.diffuse_mean_free_path.z;

    float radius = sample_diffusion_profile(random_sample.x, sample_scattering_distance);
    RGB per_channel_PDF = evaluate(radius, params.diffuse_mean_free_path);

    // Sample an angle and turn into an offset from origo.
    float phi = 2.0f * PIf * random_sample.y;
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    Vector3f pi_offset = { radius * cos_phi, radius * sin_phi, 0.0f };

    SeparableBSSRDFPositionSample bssrdf_sample;
    bssrdf_sample.position = po + pi_offset;
    bssrdf_sample.PDF = sum(per_channel_PDF * per_channel_probability);
    bssrdf_sample.reflectance = params.diffuse_albedo * per_channel_PDF;
    return bssrdf_sample;
}

} // NS AlbedoMIS

namespace ApproximateSampling {

_inline_all_archs_ MonteCarlo::PDF pdf(Parameters params, Vector3f po, Vector3f pi) {
    float r = distance(po, pi);
    float sampled_diffuse_mean_free_path = fmax(params.diffuse_mean_free_path.x, fmax(params.diffuse_mean_free_path.y, params.diffuse_mean_free_path.z));
    return evaluate(r, sampled_diffuse_mean_free_path);
}

_inline_all_archs_ BSDFResponse evaluate_with_PDF(Parameters params, Vector3f po, Vector3f pi) {
    auto reflectance = evaluate(params, po, pi);
    auto PDF = SampleMostScattering::pdf(params, po, pi);
    return { reflectance, PDF };
}

// Sample the Burley normalized diffusion profile.
_inline_all_archs_ SeparableBSSRDFPositionSample sample(Parameters params, Vector3f po, Vector3f random_sample) {
    float sample_scattering_distance = fmax(params.diffuse_mean_free_path.x, fmax(params.diffuse_mean_free_path.y, params.diffuse_mean_free_path.z));

    float radius = sample_diffusion_profile_approximation(random_sample.x, sample_scattering_distance);

    // Sample an angle and turn into an offset from origo.
    float phi = 2.0f * PIf * random_sample.y;
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);
    Vector3f pi_offset = { radius * cos_phi, radius * sin_phi, 0.0f };

    SeparableBSSRDFPositionSample bssrdf_sample;
    bssrdf_sample.position = po + pi_offset;
    bssrdf_sample.PDF = evaluate(radius, sample_scattering_distance);
    bssrdf_sample.reflectance = params.diffuse_albedo * evaluate(radius, params.diffuse_mean_free_path);
    return bssrdf_sample;
}

} // NS ApproximateSampling

} // NS BurleySSS
} // NS Bifrost::Assets::Shading::BSDFs

#endif // _BIFROST_ASSETS_SHADING_BSDFS_BURLEY_SSS_H_