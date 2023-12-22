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

__inline_all__ float alpha_from_roughness(float roughness) {
    return fmaxf(0.00000000001f, roughness * roughness);
}

__inline_all__ float roughness_from_alpha(float alpha) {
    return sqrt(alpha);
}

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

__inline_all__ float evaluate(float alpha, float specularity, float3 wo, float3 wi) {
    float3 halfway = normalize(wo + wi);
    float G = GGX::height_correlated_G(alpha, wo, wi);
    float D = Distributions::GGX_VNDF::D(alpha, halfway);
    float F = schlick_fresnel(specularity, dot(wo, halfway));
    return (D * F * G) / (4.0f * wo.z * wi.z);
}

__inline_all__ float3 evaluate(float alpha, float3 specularity, float3 wo, float3 wi, float3 halfway) {
    float G = GGX::height_correlated_G(alpha, wo, wi);
    float D = Distributions::GGX_VNDF::D(alpha, halfway);
    float3 F = schlick_fresnel(specularity, dot(wo, halfway));
    return F * (D * G / (4.0f * wo.z * wi.z));
}

__inline_all__ float3 evaluate(float alpha, float3 specularity, float3 wo, float3 wi) {
    float3 halfway = normalize(wo + wi);
    return evaluate(alpha, specularity, wo, wi, halfway);
}

__inline_all__ float PDF(float alpha, float3 wo, float3 wi) {
    float3 halfway = normalize(wo + wi);
    return Distributions::GGX_VNDF::PDF(alpha, wo, halfway) / (4.0f * dot(wo, halfway));
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float3 specularity, float3 wo, float3 wi, float3 halfway) {
    float lambda_wo = Distributions::GGX_VNDF::lambda(alpha, wo);
    float lambda_wi = Distributions::GGX_VNDF::lambda(alpha, wi);

    float quater_D = 0.25f * Distributions::GGX_VNDF::D(alpha, halfway);

    float3 F = schlick_fresnel(specularity, dot(wo, halfway));

    float recip_G2 = 1.0f + lambda_wo + lambda_wi; // reciprocal height_correlated_G
    float3 reflectance = F * (quater_D / (recip_G2 * wo.z * wi.z));

    BSDFResponse res;
    res.reflectance = reflectance;
    float recip_G1 = 1.0f + lambda_wo;
    res.PDF = quater_D / (recip_G1 * wo.z);
    return res;
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float3 specularity, float3 wo, float3 wi) {
    float3 halfway = normalize(wo + wi);
    return evaluate_with_PDF(alpha, specularity, wo, wi, halfway);
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
    BSDFSample bsdf_sample;

    float3 halfway = Distributions::GGX_VNDF::sample_halfway(alpha, wo, random_sample);
    bsdf_sample.direction = reflect(-wo, halfway);

    BSDFResponse response = evaluate_with_PDF(alpha, specularity, wo, bsdf_sample.direction, halfway);
    bsdf_sample.PDF = response.PDF;
    bsdf_sample.reflectance = response.reflectance;

    bool discardSample = bsdf_sample.PDF < 0.00001f || bsdf_sample.direction.z < 0.00001f; // Discard samples if the pdf is too low (precision issues) or if the new direction points into the surface (energy loss).
    return discardSample ? BSDFSample::none() : bsdf_sample;
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
    if (halfway.z < 0)
        halfway = -halfway;
    return halfway;
}

__inline_all__ float evaluate(float alpha, float3 wo, float3 wi, float ior_i_over_o, float3 halfway) {
    // Reflection evaluates to 0.
    if (sign(wo.z) == sign(wi.z))
        return 0;

    // Same side? PBRT has this check, but is it needed if the halfway vector is correct? And why is it more extensive than the is_reflection check above?
    if (dot(wo, halfway) * dot(wi, halfway) > 0)
        return 0;

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

__inline_all__ float PDF(float alpha, float ior_i_over_o, float3 wo, float3 wi) {
    // Cannot sample reflection.
    if (sign(wo.z) == sign(wi.z))
        return 0;

    bool entering = wo.z >= 0.0;
    if (!entering) {
        wo.z = -wo.z;
        wi.z = -wi.z;
    }

    float3 halfway = compute_halfway_vector(ior_i_over_o, wo, wi);

    // Same side? PBRT has this check, but is it needed and why is it more extensive than the is_reflection check above?
    // Could it be that it captures if a wi lies outside the refracted lobe?
    if (dot(wo, halfway) * dot(wi, halfway) > 0)
        return 0;

    return Distributions::GGX_VNDF::PDF(alpha, wo, halfway) * transmission_PDF_scale(ior_i_over_o, wo, wi, halfway);
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float ior_i_over_o, float3 wo, float3 wi) {
    BSDFResponse response;
    response.reflectance = make_float3(1) * evaluate(alpha, ior_i_over_o, wo, wi);
    response.PDF = PDF(alpha, ior_i_over_o, wo, wi);
    return response;
}

__inline_all__ BSDFSample sample(float alpha, float ior_i_over_o, float3 wo, float2 random_sample) {
    BSDFSample bsdf_sample;

    // Sample GGX
    bool entering = wo.z >= 0.0;
    if (!entering)
        wo.z = -wo.z;
    float3 halfway = Distributions::GGX_VNDF::sample_halfway(alpha, wo, random_sample);
    bsdf_sample.PDF = Distributions::GGX_VNDF::PDF(alpha, wo, halfway);

    if (!refract(bsdf_sample.direction, -wo, halfway, ior_i_over_o))
        return BSDFSample::none(); // Remove the contribution for now.

    bsdf_sample.PDF *= transmission_PDF_scale(ior_i_over_o, wo, bsdf_sample.direction, halfway);

    // Discard samples if the pdf is too low (precision issues) or if wi is on the same side as wo.
    bool discardSample = bsdf_sample.direction.z > -0.00001f;
    discardSample |= bsdf_sample.PDF < 0.00001f;
    if (discardSample)
        return BSDFSample::none();

    // Same side?
    if (dot(wo, halfway) * dot(bsdf_sample.direction, halfway) > 0)
        return BSDFSample::none();

    float f = evaluate(alpha, wo, bsdf_sample.direction, ior_i_over_o, halfway);
    bsdf_sample.reflectance = make_float3(f);

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

__inline_all__ float evaluate(float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    bool is_reflection = sign(wo.z) == sign(wi.z);
    if (is_reflection)
        return BSDFs::GGX_R::evaluate(alpha, specularity, wo, wi);
    else {
        float3 halfway = GGX_T::compute_halfway_vector(ior_i_over_o, wo, wi);

        // Same side? PBRT has this check, but is it needed and why is it more extensive than the is_reflection check above?
        if (dot(wo, halfway) * dot(wi, halfway) > 0)
            return 0;

        float G = BSDFs::GGX::height_correlated_G(alpha, wo, wi);
        float D = Distributions::GGX_VNDF::D(alpha, halfway);
        float F = 1.0f - schlick_fresnel(specularity, abs(dot(wo, halfway)));

        // Walter et al 07 equation 21. Similar to PBRT V3, equation 8.20, page 548
        float f1 = abs(dot(wo, halfway) * dot(wi, halfway) / (wo.z * wi.z));
        float f2 = G * F * D * pow2(ior_i_over_o / (dot(wo, halfway) + ior_i_over_o * dot(wi, halfway)));
        return f1 * f2;
    }
}

__inline_all__ float3 evaluate(float3 tint, float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    float f = evaluate(alpha, specularity, ior_i_over_o, wo, wi);
    bool is_transmission = sign(wo.z) != sign(wi.z);
    return f * (is_transmission ? tint : make_float3(1));
}

__inline_all__ float PDF(float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    bool entering = wo.z >= 0.0;
    if (!entering) {
        wo.z = -wo.z;
        wi.z = -wi.z;
    }

    bool is_refraction = sign(wo.z) != sign(wi.z);
    ior_i_over_o = is_refraction ? ior_i_over_o : 1.0f;
    float3 halfway = GGX_T::compute_halfway_vector(ior_i_over_o, wo, wi);

    // Discard if the directions are close to perpendicular to the surface or if the halfvector is ill-defined.
    // Same side? PBRT has this check, but is it needed and why is it more extensive than the is_reflection check above?
    // Could it be that it captures if a wi lies outside the refracted lobe?
    bool energy_loss = is_refraction ? wi.z > -0.00001f : wi.z < 0.00001f;
    bool same_side = is_refraction && dot(wo, halfway) * dot(wi, halfway) > 0;
    if (energy_loss ||same_side)
        return 0;

    float PDF = Distributions::GGX_VNDF::PDF(alpha, wo, halfway);

    // Change of variable.
    if (is_refraction)
        PDF *= GGX_T::transmission_PDF_scale(ior_i_over_o, wo, wi, halfway);
    else
        PDF /= (4.0f * dot(wo, halfway));

    // Scale the PDF by the probability to reflect or refract.
    float reflection_propability = schlick_fresnel(specularity, dot(wo, halfway));
    PDF *= is_refraction ? (1 - reflection_propability) : reflection_propability;

    // // Discard samples if the pdf is too low (precision issues).
    bool discardSample = PDF < 0.00001f;
    return discardSample ? 0.0f : PDF;
}

__inline_all__ BSDFResponse evaluate_with_PDF(float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    BSDFResponse response;
    response.reflectance = make_float3(1) * evaluate(alpha, specularity, ior_i_over_o, wo, wi);
    response.PDF = PDF(alpha, specularity, ior_i_over_o, wo, wi);
    return response;
}

__inline_all__ BSDFResponse evaluate_with_PDF(float3 tint, float alpha, float specularity, float ior_i_over_o, float3 wo, float3 wi) {
    BSDFResponse response = evaluate_with_PDF(alpha, specularity, ior_i_over_o, wo, wi);
    bool is_transmission = sign(wo.z) != sign(wi.z);
    response.reflectance *= is_transmission ? tint : make_float3(1);
    return response;
}

__inline_all__ BSDFSample sample(float alpha, float specularity, float ior_i_over_o, float3 wo, float3 random_sample) {
    BSDFSample bsdf_sample;

    // Sample GGX
    bool entering = wo.z >= 0.0;
    if (!entering)
        wo.z = -wo.z;

    float3 halfway = Distributions::GGX_VNDF::sample_halfway(alpha, wo, make_float2(random_sample));
    bsdf_sample.PDF = Distributions::GGX_VNDF::PDF(alpha, wo, halfway);

    // Reflect or refract based on Fresnel
    float reflection_propability = schlick_fresnel(specularity, dot(wo, halfway));
    bool is_reflection = random_sample.z < reflection_propability;

    // Reflect or refract
    if (is_reflection) {
        bsdf_sample.PDF *= reflection_propability;
        bsdf_sample.PDF /= (4.0f * dot(wo, halfway));

        bsdf_sample.direction = reflect(-wo, halfway);
    } else {
        if (!refract(bsdf_sample.direction, -wo, halfway, ior_i_over_o))
            return BSDFSample::none(); // Remove the contribution for now.

        bsdf_sample.PDF *= GGX_T::transmission_PDF_scale(ior_i_over_o, wo, bsdf_sample.direction, halfway);
        bsdf_sample.PDF *= 1 - reflection_propability;
    }

    // Discard samples if the pdf is too low (precision issues) or if the direction points into the surface (energy loss).
    bool discardSample = is_reflection ? bsdf_sample.direction.z < 0.00001f : bsdf_sample.direction.z > -0.00001f;
    discardSample |= bsdf_sample.PDF < 0.00001f;
    if (discardSample)
        return BSDFSample::none();

    // Same side?
    if (!is_reflection && dot(wo, halfway) * dot(bsdf_sample.direction, halfway) > 0)
        return BSDFSample::none();

    float f = evaluate(alpha, specularity, ior_i_over_o, wo, bsdf_sample.direction);
    bsdf_sample.reflectance = make_float3(f);

    if (!entering)
        bsdf_sample.direction.z = -bsdf_sample.direction.z;

    return bsdf_sample;
}

__inline_all__ BSDFSample sample(float3 tint, float alpha, float specularity, float ior_i_over_o, float3 wo, float3 random_sample) {
    BSDFSample bsdf_sample = sample(alpha, specularity, ior_i_over_o, wo, random_sample);
    if (sign(wo.z) != sign(bsdf_sample.direction.z))
        bsdf_sample.reflectance *= tint;
    return bsdf_sample;
}
} // NS GGX

} // NS BSDFs
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_MICROFACET_H_