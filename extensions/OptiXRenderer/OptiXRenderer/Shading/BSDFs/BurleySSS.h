// OptiX renderer functions for the Burley normalized subsurface scattering.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_BURLEY_SSS_H_
#define _OPTIXRENDERER_BSDFS_BURLEY_SSS_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace Shading {
namespace BSDFs {

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

using namespace optix;

// Container for parameters for the Burley's subsurface scattering approximation.
// We precompute the parameters and store them to avoid redoing the same transformations multiple times.
struct Parameters {
    const float3 albedo;

    // l / s from equation 3 in Approximate Reflectance Profiles for Efficient Subsurface Scattering
    const float3 scattering_distance;

    // Equation 6 in Approximate Reflectance Profiles for Efficient Subsurface Scattering.
    __inline_all__ static float3 mean_free_path_scaling_term(float3 albedo) {
        return 1.9f - albedo + 3.5f * pow2(albedo - 0.8f);
    }

    __inline_all__ static Parameters create(float3 albedo, float3 mean_free_path) {
        float3 s = mean_free_path_scaling_term(albedo);
        return { albedo, mean_free_path / s };
    }
};

using namespace optix;

// Equation 2 in Approximate Reflectance Profiles for Efficient Subsurface Scattering.
__inline_all__ float evaluate(float entry_and_exit_distance, float scattering_distance) {
    // Note: These are not actually derived as direct and indirect terms, but can be viewed as such and it makes the code names more readable.
    float direct_term = expf(-entry_and_exit_distance / (3.0f * scattering_distance));
    float indirect_term = pow3(direct_term);
    float normalizer = 8 * PIf * scattering_distance * entry_and_exit_distance;

    return (direct_term + indirect_term) / normalizer;
}

__inline_all__ float3 evaluate(float entry_and_exit_distance, float3 scattering_distance) {
    return { evaluate(entry_and_exit_distance, scattering_distance.x),
             evaluate(entry_and_exit_distance, scattering_distance.y),
             evaluate(entry_and_exit_distance, scattering_distance.z) };
}

// Equation 3 in Approximate Reflectance Profiles for Efficient Subsurface Scattering.
__inline_all__ float3 evaluate(Parameters params, float3 po, float3 pi) {
    float r = distance(po, pi);
    return params.albedo * evaluate(r, params.scattering_distance);
}

// Performs sampling of a Normalized Burley diffusion profile in polar coordinates.
// 'u' is the random number (the value of the CDF): [0, 1).
// rcp(s) = 1 / ShapeParam = ScatteringDistance.
// 'r' is the sampled radial distance, s.t. (u = 0 -> r = 0) and (u = 1 -> r = Inf).
// rcp(Pdf) is the reciprocal of the corresponding PDF value.
// Derivation and source: https://zero-radiance.github.io/post/sampling-diffusion/
// The PDF has been changed to include 'radius' to convert from polar to cartesian coordinates.
__inline_all__ void sample_diffusion_profile(float u, float scattering_distance, float& radius, float& rcpPdf) {
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

    radius = x * scattering_distance;
    rcpPdf = (8 * PIf * radius * scattering_distance) * rcpExp; // ((8 * Pi * r) / s) / (Exp[-s * r / 3] + Exp[-s * r])
}

// Performs sampling of a Normalized Burley diffusion profile in polar coordinates.
// 'u' is the random number (the value of the CDF): [0, 1).
// rcp(s) = 1 / ShapeParam = ScatteringDistance.
// Returns the sampled radial distance, s.t. (u = 0 -> r = 0) and (u = 1 -> r = Inf).
// Derivation and source: https://zero-radiance.github.io/post/sampling-diffusion/
// The PDF has been changed to include 'radius' to convert from polar to cartesian coordinates.
__inline_all__ float sample_diffusion_profile(float u, float scattering_distance) {
    const float LOG2_E = 1.44269504089f;

    u = 1 - u; // Convert CDF to CCDF; the resulting value of (u != 0)

    float g = 1 + (4 * u) * (2 * u + sqrt(1 + (4 * u) * u));
    float n = exp2(log2(g) * (-1.0f / 3.0f));                // g^(-1/3)
    float p = (g * n) * n;                                   // g^(+1/3)
    float c = 1 + p + n;                                     // 1 + g^(+1/3) + g^(-1/3)
    float x = (3 / LOG2_E) * log2(c / (4 * u));              // 3 * Log[c / (4 * u)]
    return x * scattering_distance;
}

// Sample the distribution based on the channel with the largest scattering distance,
// as suggested in Efficient Screen Space Subsurface Scattering Using Burley's Normalized Diffusion in Realtime, slide 26.
namespace SampleMostScattering {

__inline_all__ PDF pdf(Parameters params, float3 po, float3 pi) {
    float r = distance(po, pi);
    float scattering_distance = fmax(params.scattering_distance.x, fmax(params.scattering_distance.y, params.scattering_distance.z));
    return evaluate(r, scattering_distance);
}

__inline_all__ BSDFResponse evaluate_with_PDF(Parameters params, float3 po, float3 pi) {
    auto reflectance = evaluate(params, po, pi);
    auto PDF = SampleMostScattering::pdf(params, po, pi);
    return { reflectance, PDF };
}

// Sample the Burley normalized diffusion profile.
__inline_all__ BSSRDFSample sample(Parameters params, float3 po, float3 random_sample) {
    float scattering_distance = fmax(params.scattering_distance.x, fmax(params.scattering_distance.y, params.scattering_distance.z));

    float radius, reciprocal_PDF;
    sample_diffusion_profile(random_sample.x, scattering_distance, radius, reciprocal_PDF);

    // Sample an angle and turn into an offset from origo.
    float phi = 2.0f * PIf * random_sample.y;
    float3 pi_offset = { radius * cosf(phi), radius * sinf(phi), 0.0f };

    BSSRDFSample bssrdf_sample;
    bssrdf_sample.pi = po + pi_offset;
    bssrdf_sample.direction = { 0, 0, -1 };
    bssrdf_sample.PDF = (1 / reciprocal_PDF);
    bssrdf_sample.reflectance = evaluate(params, po, bssrdf_sample.pi);
    return bssrdf_sample;
}

} // NS SampleMaxChannel

namespace ImportanceSampleChannels {

__inline_all__ PDF pdf(Parameters params, float3 po, float3 pi) {
    float r = distance(po, pi);
    float3 per_channel_PDF = evaluate(r, params.scattering_distance);

    float recip_summed_albedo = 1.0f / sum(params.albedo);
    float3 per_channel_probability = params.albedo * recip_summed_albedo;

    return sum(per_channel_PDF * per_channel_probability);
}

__inline_all__ BSDFResponse evaluate_with_PDF(Parameters params, float3 po, float3 pi) {
    auto reflectance = evaluate(params, po, pi);
    auto PDF = ImportanceSampleChannels::pdf(params, po, pi);
    return { reflectance, PDF };
}

// Sample the Burley normalized diffusion profile.
__inline_all__ BSSRDFSample sample(Parameters params, float3 po, float3 random_sample) {
    float recip_summed_albedo = 1.0f / sum(params.albedo);
    float3 per_channel_probability = params.albedo * recip_summed_albedo;

    float scattering_distance;
    if (per_channel_probability.x < random_sample.z)
        scattering_distance = params.scattering_distance.x;
    else if ((per_channel_probability.x + per_channel_probability.y) < random_sample.z)
        scattering_distance = params.scattering_distance.y;
    else
        scattering_distance = params.scattering_distance.z;

    float radius = sample_diffusion_profile(random_sample.x, scattering_distance);
    float3 per_channel_PDF = evaluate(radius, params.scattering_distance);

    // Sample an angle and turn into an offset from origo.
    float phi = 2.0f * PIf * random_sample.y;
    float3 pi_offset = { radius * cosf(phi), radius * sinf(phi), 0.0f };

    BSSRDFSample bssrdf_sample;
    bssrdf_sample.pi = po + pi_offset;
    bssrdf_sample.direction = { 0, 0, -1 };
    bssrdf_sample.PDF = sum(per_channel_PDF * per_channel_probability);
    bssrdf_sample.reflectance = params.albedo * per_channel_PDF;
    return bssrdf_sample;
}

} // NS ImportanceSampleChannels

} // NS BurleySSS
} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_BURLEY_SSS_H_