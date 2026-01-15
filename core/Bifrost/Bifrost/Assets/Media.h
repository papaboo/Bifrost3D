// Bifrost media.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_MEDIA_H_
#define _BIFROST_ASSETS_MEDIA_H_

#include <Bifrost/Math/Color.h>

namespace Bifrost::Assets::Media {

struct ArtisticScatteringParameters;

//----------------------------------------------------------------------------
// Measured media scattering parameters.
//----------------------------------------------------------------------------
struct MeasuredScatteringParameters {
    Math::RGB scattering_coefficient;
    Math::RGB absorption_coefficient;

    static MeasuredScatteringParameters from_artistic_parameters(const ArtisticScatteringParameters& artistic_parameters);

    // Get the attenuation coefficient, or extinction coefficient, which is the sum of scattering and absorption along the path.
    // PBRT v4, section 11.1.3
    // https://pbr-book.org/4ed/Volume_Scattering/Volume_Scattering_Processes#OutScatteringandAttenuation
    __always_inline__ Math::RGB get_attenuation_coefficient() const { return scattering_coefficient + absorption_coefficient; }

    // The mean free path is the reciprocal of the attenuation coefficient.
    // PBRT v4, section 11.1.3
    // https://pbr-book.org/4ed/Volume_Scattering/Volume_Scattering_Processes#OutScatteringandAttenuation
    __always_inline__ Math::RGB get_mean_free_path() const {
        Math::RGB sigma_t = get_attenuation_coefficient();
        return { 1.0f / sigma_t.r, 1.0f / sigma_t.g, 1.0f / sigma_t.b };
    }

    __always_inline__ Math::RGB get_single_scattering_albedo() const { return scattering_coefficient / (scattering_coefficient + absorption_coefficient); }

    // Implementation based on the diffuse reflectance computation in A Practical Model for Subsurface Light Transport, Jensen et al., 2001
    __always_inline__ Math::RGB get_diffuse_albedo(float medium_ior = 1.3f) const {
        Math::RGB alpha = get_single_scattering_albedo();

        // Diffuse fresnel, Fdr, and internal reflection parameter, A, computed in section 2.1.
        float Fdr = -1.44f / (medium_ior * medium_ior) + 0.71f / medium_ior + 0.668f + 0.0636f * medium_ior;
        float A = (1 + Fdr) / (1 - Fdr);

        // First equation in section 2.4.
        auto single_channel_diffuse_reflectance = [=](float alpha_c) -> float {
            float exponent2 = -sqrt(3 * (1 - alpha_c));
            float exponent1 = 4.0f / 3.0f * A * exponent2;
            return 0.5f * alpha_c * (1 + expf(exponent1)) * expf(exponent2);
        };

        return { single_channel_diffuse_reflectance(alpha.r),
                 single_channel_diffuse_reflectance(alpha.g),
                 single_channel_diffuse_reflectance(alpha.b) };
    }

    // Measured parameters from "A Practical Model for Subsurface Light Transport", Jensen et al., 2001
    static MeasuredScatteringParameters apple() { return { { 2.29f, 2.39f, 1.97f }, { 0.003f, 0.0034f, 0.046f } }; }
    static MeasuredScatteringParameters chicken1() { return { { 0.15f, 0.21f, 0.38f }, { 0.015f, 0.077f, 0.19f } }; }
    static MeasuredScatteringParameters chicken2() { return { { 0.19f, 0.25f, 0.32f }, { 0.018f, 0.088f, 0.2f } }; }
    static MeasuredScatteringParameters cream() { return { { 7.38f, 5.47f, 3.15f }, { 0.0002f, 0.0028f, 0.0163f } }; }
    static MeasuredScatteringParameters ketchup() { return { { 0.18f, 0.07f, 0.03f }, { 0.061f, 0.97f, 1.45f } }; }
    static MeasuredScatteringParameters marble() { return { { 2.19f, 2.62f, 3.00f }, { 0.0021f, 0.0041f, 0.0071f } }; }
    static MeasuredScatteringParameters potato() { return { { 0.68f, 0.70f, 0.55f }, { 0.0024f, 0.0090f, 0.12f } }; }
    static MeasuredScatteringParameters skimmilk() { return { { 0.70f, 1.22f, 1.90f }, { 0.0014f, 0.0025f, 0.0142f } }; }
    static MeasuredScatteringParameters skin1() { return { { 0.74f, 0.88f, 1.01f }, { 0.032f, 0.17f, 0.48f } }; }
    static MeasuredScatteringParameters skin2() { return { { 1.09f, 1.59f, 1.79f }, { 0.013f, 0.070f, 0.145f } }; }
    static MeasuredScatteringParameters wholemilk() { return { { 2.55f, 3.21f, 3.77f }, { 0.0011f, 0.0024f, 0.014f } }; }
};

//----------------------------------------------------------------------------
// Artistic media scattering parameters to enable easy artistic parameter tweaking.
// Maps one to one to Disney and Pixar's Normalized Burley SSS approximation.
//----------------------------------------------------------------------------
struct ArtisticScatteringParameters {
    Math::RGB diffuse_albedo;
    Math::RGB mean_free_path;

    static ArtisticScatteringParameters from_measured_parameters(const MeasuredScatteringParameters& measured_parameters, float medium_ior = 1.3f);

    // Artistic parameters from Pixar's Renderman documentation
    // https://rmanwiki-25.pixar.com/space/REN25/20416088/Subsurface+Scattering+Parameters
    // The albedos match the results in "A Practical Model for Subsurface Light Transport", Jensen et al., 2001,
    // but Pixar's mean free paths are diffuse mean free paths and artistic and don't match the measured parameters.
    // Instead we've used the mean free paths computed from Jensen's measured data.
    static ArtisticScatteringParameters apple() { return { { 0.846f, 0.841f, 0.528f }, { 0.43611f, 0.41782f, 0.49603f } }; }
    static ArtisticScatteringParameters chicken1() { return { { 0.314f, 0.156f, 0.126f }, { 6.06061f, 3.48432f, 1.75439f } }; }
    static ArtisticScatteringParameters chicken2() { return { { 0.321f, 0.160f, 0.108f }, { 4.80769f, 2.95858f, 1.92308f } }; }
    static ArtisticScatteringParameters cream() { return { { 0.976f, 0.900f, 0.725f }, { 0.13550f, 0.18272f, 0.31583f } }; }
    static ArtisticScatteringParameters ketchup() { return { { 0.164f, 0.006f, 0.002f }, { 4.14938f, 0.96154f, 0.67568f } }; }
    static ArtisticScatteringParameters marble() { return { { 0.830f, 0.791f, 0.753f }, { 0.45618f, 0.38108f, 0.33255f } }; }
    static ArtisticScatteringParameters potato() { return { { 0.764f, 0.613f, 0.213f }, { 1.46542f, 1.41044f, 1.49254f } }; }
    static ArtisticScatteringParameters skimmilk() { return { { 0.815f, 0.813f, 0.682f }, { 1.42572f, 0.81800f, 0.52241f } }; }
    static ArtisticScatteringParameters skin1() { return { { 0.436f, 0.227f, 0.131f }, { 1.29534f, 0.95238f, 0.67114f } }; }
    static ArtisticScatteringParameters skin2() { return { { 0.623f, 0.433f, 0.343f }, { 1.29534f, 0.95238f, 0.67114f } }; }
    static ArtisticScatteringParameters wholemilk() { return { { 0.908f, 0.881f, 0.759f }, { 0.39199f, 0.31129f, 0.26427f } }; }
};

//----------------------------------------------------------------------------
// Implement converters between scattering parameter definitions.
//----------------------------------------------------------------------------

// Implementation based on
// Practical and Controllable Subsurface Scattering for Production Path Tracing, Chiang et all., 2016.
// 's' is left out as our artistic parameters are based on the physical mean free path and not Burley's diffuse mean free path.
MeasuredScatteringParameters MeasuredScatteringParameters::from_artistic_parameters(const ArtisticScatteringParameters& artistic_parameters) {
    Math::RGB a = artistic_parameters.diffuse_albedo;
    Math::RGB exponent = -5.09406f * a + 2.61188f * a * a - 4.31805f * a * a * a;
    Math::RGB single_scattering_albedo = { 1 - expf(exponent.r), 1 - expf(exponent.g), 1 - expf(exponent.b) };
    Math::RGB mpf = artistic_parameters.mean_free_path;
    Math::RGB attenuation_coefficient = { 1 / mpf.r, 1 / mpf.g, 1 / mpf.b };

    Math::RGB scattering_coefficient = single_scattering_albedo * attenuation_coefficient;
    Math::RGB absorption_coefficient = attenuation_coefficient - scattering_coefficient;

    return { scattering_coefficient, absorption_coefficient };
}

// Implementation based on the diffuse reflectance computation in A Practical Model for Subsurface Light Transport, Jensen et al., 2001
ArtisticScatteringParameters ArtisticScatteringParameters::from_measured_parameters(const MeasuredScatteringParameters& measured_parameters, float medium_ior) {
    return { measured_parameters.get_diffuse_albedo(medium_ior), measured_parameters.get_mean_free_path() };
}

} // NS Bifrost::Assets::Media

#endif // _BIFROST_ASSETS_MEDIA_H_