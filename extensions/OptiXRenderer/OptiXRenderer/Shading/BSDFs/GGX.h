// OptiX renderer functions for microfacet models.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_MICROFACET_H_
#define _OPTIXRENDERER_BSDFS_MICROFACET_H_

#include <OptiXRenderer/Distributions.h>
#include <OptiXRenderer/Types.h>
#include <OptiXRenderer/Utils.h>

namespace OptiXRenderer {
namespace Shading {
namespace BSDFs {

//----------------------------------------------------------------------------
// Implementation of GGX's D and G term for the Microfacet model.
// Sources:
// * Walter et al 07.
// * Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz 14.
// * https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/Microfacet.hpp
//----------------------------------------------------------------------------
namespace GGX {

using namespace optix;

// Minimum GGX alpha as defined in PBRT V4 source code
// Found in TrowbridgeReitzDistribution constructor in scattering.h
__constant_all__ float MIN_ALPHA = 1e-4f;

__inline_all__ float alpha_from_roughness(float roughness) {
    return fmaxf(MIN_ALPHA, roughness * roughness);
}

__inline_all__ float roughness_from_alpha(float alpha) {
    return sqrt(alpha);
}

__inline_all__ bool effectively_smooth(float alpha) { return alpha <= MIN_ALPHA; }

// Height correlated masking and shadowing term.
// PBRT V3, section 8.4.3, page 544.
__inline_all__ float height_correlated_G(float alpha, float3 wo, float3 wi) {
    return 1.0f / (1.0f + Distributions::GGX_VNDF::lambda(alpha, wo) + Distributions::GGX_VNDF::lambda(alpha, wi));
}

} // NS GGX

//----------------------------------------------------------------------------
// GGX BSDF, Walter et al 07.
// Here we have seperated the BRDF from the BTDF. 
// This makes sampling slightly less performant, but allows for a full 
// overview of the individual components, which can then later be composed 
// in a new material/BSDF.
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Reflection part of GGX.
//----------------------------------------------------------------------------

namespace GGX_R {

using namespace optix;

__inline_all__ float3 evaluate(float alpha, float3 specularity, float3 wo, float3 wi) {
    if (GGX::effectively_smooth(alpha))
        return make_float3(0.0f);

    // Avoid dividing by zero at grazing angles and reflection is only defined if the directions are in the same hemisphere.
    if (wo.z * wi.z <= 0.0f)
        return make_float3(0.0f);

    float3 halfway = normalize(wo + wi);
    float G = GGX::height_correlated_G(alpha, wo, wi);
    float D = Distributions::GGX_VNDF::D(alpha, halfway);
    float3 F = schlick_fresnel(specularity, dot(wo, halfway));
    return F * (D * G / (4.0f * wo.z * wi.z));
}

// Helper evaluate function for testing and rho computation.
__inline_all__ float evaluate(float alpha, float specularity, float3 wo, float3 wi) {
    return evaluate(alpha, make_float3(specularity), wo, wi).x;
}

__inline_all__ PDF pdf(float alpha, float3 wo, float3 wi) {
    if (GGX::effectively_smooth(alpha))
        return PDF::invalid();
    return Distributions::GGX_Bounded_VNDF::reflection_PDF(alpha, wo, wi);
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float3 specularity, float3 wo, float3 wi) {
    auto reflectance = evaluate(alpha, specularity, wo, wi);
    auto PDF = pdf(alpha, wo, wi);
    return { reflectance, PDF };
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float specularity, float3 wo, float3 wi) {
    return evaluate_with_PDF(alpha, make_float3(specularity), wo, wi);
}

__inline_all__ float3 approx_off_specular_peak(float alpha, float3 wo) {
    float3 reflection = make_float3(-wo.x, -wo.y, wo.z);
    // reflection = lerp(make_float3(0, 0, 1), reflection, (1 - alpha) * (sqrt(1 - alpha) + alpha)); // UE4 implementation
    reflection = lerp(reflection, make_float3(0, 0, 1), alpha);
    return normalize(reflection);
}

__inline_all__ BSDFSample sample(float alpha, float3 specularity, float3 wo, float2 random_sample) {
    if (GGX::effectively_smooth(alpha)) {
        // Sample perfectly specular BRDF
        BSDFSample bsdf_sample;
        bsdf_sample.direction = { -wo.x, -wo.y, wo.z };
        bsdf_sample.PDF = PDF::delta_dirac(1);
        bsdf_sample.reflectance = schlick_fresnel(specularity, abs(wo.z)) / abs(bsdf_sample.direction.z);
        return bsdf_sample;
    } else {
        // Sample rough BRDF
        auto reflection_sample = Distributions::GGX_Bounded_VNDF::sample(alpha, wo, random_sample);

        BSDFSample bsdf_sample;
        bsdf_sample.direction = reflection_sample.direction;
        bsdf_sample.PDF = reflection_sample.PDF;
        bsdf_sample.reflectance = evaluate(alpha, specularity, wo, bsdf_sample.direction);

        // Discard samples if the new direction points into the surface (energy loss).
        bool energyloss = bsdf_sample.direction.z < 0.0f;
        return energyloss ? BSDFSample::none() : bsdf_sample;
    }
}
__inline_all__ BSDFSample sample(float alpha, float specularity, float3 wo, float2 random_sample) { return sample(alpha, make_float3(specularity), wo, random_sample); }

} // NS GGX_R

//----------------------------------------------------------------------------
// Transmission part of GGX.
//----------------------------------------------------------------------------

namespace GGX_T {
using namespace optix;

// Compute change of variables dwh_dwi for microfacet transmission. See MicrofacetTransmission::Pdf in reflection.cpp of PBRT v3.
__inline_all__ float transmission_PDF_scale(float ior_i_over_o, float3 wo, float3 wi, float3 halfway) {
    float sqrt_denom = dot(wo, halfway) + ior_i_over_o * dot(wi, halfway);
    return pow2(ior_i_over_o / sqrt_denom) * abs(dot(wi, halfway));
}

// Compute the halfway vector from wo, wi and their relative index of refracion
__inline_all__ float3 compute_halfway_vector(float ior_i_over_o, float3 wo, float3 wi) {
    float3 halfway = normalize(wo + ior_i_over_o * wi);
    if (halfway.z < 0.0f)
        halfway = -halfway;
    return halfway;
}

__inline_all__ float evaluate(float alpha, float3 wo, float3 wi, float ior_i_over_o, float3 halfway) {
    if (GGX::effectively_smooth(alpha))
        return 0.0f;

    // Reflection evaluates to 0.
    if (sign(wo.z) == sign(wi.z))
        return 0.0f;

    // Discard backfacing microfacets. Equation 9.35 in PBRT4.
    if (dot(wi, halfway) * wi.z <= 0 || dot(wo, halfway) * wo.z <= 0)
        return 0.0f;

    float G = BSDFs::GGX::height_correlated_G(alpha, wo, wi);
    float D = Distributions::GGX_VNDF::D(alpha, halfway);
    float F = 1.0f; // No Fresnel for the pure transmission. It is removed to be able to test 100% transmission. If it is needed it can be added outside the BTDF.

    // Walter et al 07 equation 21. Similar to PBRT V3, equation 8.20, page 548
    float f1 = abs(dot(wo, halfway) * dot(wi, halfway) / (wo.z * wi.z));
    float f2 = pow2(ior_i_over_o) * G * F * D / pow2(dot(wo, halfway) + ior_i_over_o * dot(wi, halfway));
    return f1 * f2;
}

__inline_all__ float evaluate(float alpha, float ior_i_over_o, float3 wo, float3 wi) {
    float3 halfway = compute_halfway_vector(ior_i_over_o, wo, wi);
    return evaluate(alpha, wo, wi, ior_i_over_o, halfway);
}

__inline_all__ PDF pdf(float alpha, float ior_i_over_o, float3 wo, float3 wi) {
    if (GGX::effectively_smooth(alpha))
        return PDF::invalid();

    // Cannot sample reflection.
    if (sign(wo.z) == sign(wi.z))
        return PDF::invalid();

    bool entering = wo.z >= 0.0f;
    if (!entering) {
        wo.z = -wo.z;
        wi.z = -wi.z;
    }

    float3 halfway = compute_halfway_vector(ior_i_over_o, wo, wi);

    // Discard backfacing microfacets. Equation 9.35 in PBRT4.
    if (dot(wo, halfway) < 0.0f || dot(wi, halfway) >= 0.0f)
        return PDF::invalid();

    return Distributions::GGX_VNDF::PDF(alpha, wo, halfway) * transmission_PDF_scale(ior_i_over_o, wo, wi, halfway);
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float ior_i_over_o, float3 wo, float3 wi) {
    auto reflectance = make_float3(1) * evaluate(alpha, ior_i_over_o, wo, wi);
    auto PDF = pdf(alpha, ior_i_over_o, wo, wi);
    return { reflectance, PDF };
}

__inline_all__ BSDFSample sample(float alpha, float ior_i_over_o, float3 wo, float2 random_sample) {
    BSDFSample bsdf_sample;

    // Sample GGX
    bool entering = wo.z >= 0.0f;
    if (!entering)
        wo.z = -wo.z;

    if (GGX::effectively_smooth(alpha)) {
        // Sample perfectly specular BTDF
        if (!refract(bsdf_sample.direction, -wo, make_float3(0, 0, 1), ior_i_over_o))
            return BSDFSample::none();

        float reflectance = 1.0f / abs(bsdf_sample.direction.z);
        bsdf_sample.reflectance = { reflectance, reflectance, reflectance };
        bsdf_sample.PDF = PDF::delta_dirac(1);
    } else {
        // Sample rough BTDF
        float3 halfway = Distributions::GGX_VNDF::sample_halfway(alpha, wo, random_sample);
        bsdf_sample.PDF = Distributions::GGX_VNDF::PDF(alpha, wo, halfway);

        if (!refract(bsdf_sample.direction, -wo, halfway, ior_i_over_o))
            return BSDFSample::none();

        bsdf_sample.PDF = bsdf_sample.PDF * transmission_PDF_scale(ior_i_over_o, wo, bsdf_sample.direction, halfway);

        // Discard samples if wi is on the same side as wo, which is interpreted as energyloss.
        bool energyloss = bsdf_sample.direction.z >= -0.0f;
        if (energyloss)
            return BSDFSample::none();

        float f = evaluate(alpha, wo, bsdf_sample.direction, ior_i_over_o, halfway);
        bsdf_sample.reflectance = make_float3(f);
    }

    if (!entering)
        bsdf_sample.direction.z = -bsdf_sample.direction.z;

    return bsdf_sample;
}

} // NS GGX_T

//----------------------------------------------------------------------------
// Combined reflection and transmission GGX.
// TODO
// * Turn ray refraction on or off.
// ** transmission_PDF_scale evaluates to inf if the ior's are equal. Floating point precision could be generally unstable if the ior's are almost equal.
// ** Should it degenerate to reflected GGX flipped to the other hemisphere?
// * Configurable reciprocity by basing Fresnel on both wi and wo. Only needed if refraction is on. Requires injecting the Fresnel computation into GGX
//----------------------------------------------------------------------------

namespace GGX {
using namespace optix;

__inline_all__ float normalize_reflection_probability(float reflection_probability, float3 transmission_tint) {
    float transmission_probability = 1.0f - reflection_probability;
    float scaled_transmission_probability = sum(transmission_tint) * transmission_probability;
    float scaled_reflection_probability = 3 * reflection_probability;
    return scaled_reflection_probability / (scaled_reflection_probability + scaled_transmission_probability);
}

__inline_all__ float evaluate(float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    if (GGX::effectively_smooth(alpha) || wo.z == 0.0f || wi.z == 0.0f)
        return 0.0f;

    bool entering = wo.z >= 0.0f;
    if (!entering) {
        wo.z = -wo.z;
        wi.z = -wi.z;
    }

    bool is_reflection = same_hemisphere(wo, wi);

    float halfway_ior = is_reflection ? 1.0f : ior_i_over_o;
    float3 halfway = GGX_T::compute_halfway_vector(halfway_ior, wo, wi);

    float G = GGX::height_correlated_G(alpha, wo, wi);
    float D = Distributions::GGX_VNDF::D(alpha, halfway);
    float F = dielectric_schlick_fresnel(specularity, dot(wo, halfway), ior_i_over_o);

    if (is_reflection)
        return F * D * G / (4.0f * wo.z * wi.z);
    else {
        // Discard backfacing microfacets. Equation 9.35 in PBRT4.
        if (dot(wi, halfway) * wi.z <= 0 || dot(wo, halfway) * wo.z <= 0)
            return 0.0f;

        // Walter et al 07 equation 21. Similar to PBRT V3, equation 8.20, page 548
        float f1 = abs(dot(wo, halfway) * dot(wi, halfway) / (wo.z * wi.z));
        float f2 = (1 - F) * G * D * pow2(ior_i_over_o / (dot(wo, halfway) + ior_i_over_o * dot(wi, halfway)));
        return f1 * f2;
    }
}

__inline_all__ float3 evaluate(float3 transmission_tint, float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    float f = evaluate(alpha, specularity, ior_i_over_o, wo, wi);
    bool is_transmission = sign(wo.z) != sign(wi.z);
    return f * (is_transmission ? transmission_tint : make_float3(1));
}

__inline_all__ PDF pdf(float3 transmission_tint, float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    if (GGX::effectively_smooth(alpha))
        return PDF::invalid();

    bool entering = wo.z >= 0.0f;
    if (!entering) {
        wo.z = -wo.z;
        wi.z = -wi.z;
    }

    bool is_reflection = same_hemisphere(wo, wi);

    float halfway_ior = is_reflection ? 1.0f : ior_i_over_o;
    float3 halfway = GGX_T::compute_halfway_vector(halfway_ior, wo, wi);

    // Discard backfacing microfacets. Equation 9.35 in PBRT4.
    bool backfacing_microfacet = !is_reflection && (dot(wo, halfway) < 0.0f || dot(wi, halfway) >= 0.0f);
    if (backfacing_microfacet)
        return PDF::invalid();

    PDF PDF = Distributions::GGX_VNDF::PDF(alpha, wo, halfway);

    // Scale the PDF by the probability to reflect or refract.
    float reflection_probability = dielectric_schlick_fresnel(specularity, dot(wo, halfway), ior_i_over_o);
    float normalized_reflection_probability = normalize_reflection_probability(reflection_probability, transmission_tint);
    PDF *= is_reflection ? normalized_reflection_probability : (1 - normalized_reflection_probability);

    // Change of variable.
    if (is_reflection)
        PDF *= 1 / (4.0f * dot(wo, halfway));
    else
        PDF *= GGX_T::transmission_PDF_scale(ior_i_over_o, wo, wi, halfway);

    return PDF;
}

__inline_all__ PDF pdf(float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    return pdf(make_float3(1), alpha, specularity, ior_i_over_o, wo, wi);
}

__inline_all__ BSDFResponse evaluate_with_PDF(float3 transmission_tint, float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    auto reflectance = evaluate(transmission_tint, alpha, specularity, ior_i_over_o, wo, wi);
    auto PDF = pdf(transmission_tint, alpha, specularity, ior_i_over_o, wo, wi);
    return { reflectance, PDF };
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    return evaluate_with_PDF(make_float3(1), alpha, specularity, ior_i_over_o, wo, wi);
    auto reflectance = make_float3(1) * evaluate(alpha, specularity, ior_i_over_o, wo, wi);
    auto PDF = pdf(alpha, specularity, ior_i_over_o, wo, wi);
    return { reflectance, PDF };
}

__inline_all__ BSDFSample sample(float3 transmission_tint, float alpha, float specularity, float ior_i_over_o, float3 wo, float3 random_sample) {
    BSDFSample bsdf_sample;

    // Sample GGX
    bool entering = wo.z >= 0.0f;
    if (!entering)
        wo.z = -wo.z;

    if (GGX::effectively_smooth(alpha)) {
        // Sample perfectly specular BTDF
        float reflection_probability = dielectric_schlick_fresnel(specularity, abs(wo.z), ior_i_over_o);
        float normalized_reflection_probability = normalize_reflection_probability(reflection_probability, transmission_tint);
        bool is_reflection = random_sample.z < normalized_reflection_probability;

        if (is_reflection) {
            // Sample perfectly specular BRDF
            bsdf_sample.PDF = PDF::delta_dirac(normalized_reflection_probability);
            bsdf_sample.direction = { -wo.x, -wo.y, wo.z };
        } else {
            bsdf_sample.PDF = PDF::delta_dirac(1.0f - normalized_reflection_probability);
            // Sample perfectly specular BTDF
            if (!refract(bsdf_sample.direction, -wo, make_float3(0, 0, 1), ior_i_over_o))
                return BSDFSample::none(); // Should practically never happen, as total internal reflection is included in the Fresnel computation.
        }

        // Reflectance is proportional to Fresnel.
        float reflectance = (is_reflection ? reflection_probability : (1.0f - reflection_probability)) / abs(bsdf_sample.direction.z);
        bsdf_sample.reflectance = { reflectance, reflectance, reflectance };
    } else {
        auto halfway_sample = Distributions::GGX_VNDF::sample(alpha, wo, make_float2(random_sample));
        float3 halfway = halfway_sample.direction;
        bsdf_sample.PDF = halfway_sample.PDF;

        // Reflect or refract based on Fresnel
        float reflection_probability = dielectric_schlick_fresnel(specularity, dot(wo, halfway), ior_i_over_o);
        float normalized_reflection_probability = normalize_reflection_probability(reflection_probability, transmission_tint);
        bool is_reflection = random_sample.z < normalized_reflection_probability;

        // Reflect or refract
        if (is_reflection) {
            bsdf_sample.direction = reflect(-wo, halfway);

            bsdf_sample.PDF *= normalized_reflection_probability / (4.0f * dot(wo, halfway));
        } else {
            if (!refract(bsdf_sample.direction, -wo, halfway, ior_i_over_o))
                return BSDFSample::none(); // Should practically never happen, as total internal reflection is included in the Fresnel computation.

            bsdf_sample.PDF *= 1 - normalized_reflection_probability;
            bsdf_sample.PDF *= GGX_T::transmission_PDF_scale(ior_i_over_o, wo, bsdf_sample.direction, halfway);
        }

        // Discard samples if the direction points into the surface (energy loss).
        bool energyloss = is_reflection ? bsdf_sample.direction.z < 0.0f : bsdf_sample.direction.z >= 0.0f;
        if (energyloss)
            return BSDFSample::none();

        float f = evaluate(alpha, specularity, ior_i_over_o, wo, bsdf_sample.direction);
        bsdf_sample.reflectance = make_float3(f);
    }

    bool is_transmission = sign(wo.z) != sign(bsdf_sample.direction.z);
    if (is_transmission)
        bsdf_sample.reflectance *= transmission_tint;

    if (!entering)
        bsdf_sample.direction.z = -bsdf_sample.direction.z;

    return bsdf_sample;
}

__inline_all__ BSDFSample sample(float alpha, float specularity, float ior_i_over_o, float3 wo, float3 random_sample) {
    return sample(make_float3(1), alpha, specularity, ior_i_over_o, wo, random_sample);
}

} // NS GGX

} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_MICROFACET_H_