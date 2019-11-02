// OptiX renderer default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_
#define _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_

#include <OptiXRenderer/Shading/BSDFs/Lambert.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Utils.h>

#if GPU_DEVICE
rtTextureSampler<float, 2> estimate_GGX_alpha_texture;
rtTextureSampler<float2, 2> ggx_with_fresnel_rho_texture;
#else
#include <Bifrost/Assets/Shading/Fittings.h>
#endif

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ---------------------------------------------------------------------------
// The default shading material.
// Default shading consist of a diffuse base with a specular layer on top.
// The diffuse tint is weighted by contribution, or rho, of the specular term.
// Future work:
// * Reintroduce the Helmholtz reciprocity.
//   Basically the diffuse tint should depend on rho from both wo.z and wi.z. Perhaps average specular rho before multiplying onto diffuse tint.
// ---------------------------------------------------------------------------
#define COAT_SPECULARITY 0.04f
#define USHORT_MAX 65535.0f

class DefaultShading {
private:
    optix::float3 m_diffuse_tint;
    float m_roughness;
    optix::float3 m_specularity;
    float m_coat_transmission; // 1 - coat_specular_rho - coat_interreflection_rho, the contribution from the BSDFs under the coat.
    float m_coat_scale;
    float m_coat_roughness;
    unsigned short m_specular_probability;
    unsigned short m_coat_probability;

    struct PrecomputedSpecularRho { 
        float base, full;
        __inline_all__ float rho(float specularity) { return optix::lerp(base, full, specularity); }
        __inline_all__ optix::float3 rho(optix::float3 specularity) { 
            return { rho(specularity.x), rho(specularity.y), rho(specularity.z) };
        }
    };

    __inline_all__ static PrecomputedSpecularRho fetch_specular_rho(float abs_cos_theta, float roughness) {
#if GPU_DEVICE
        float2 specular_rho = tex2D(ggx_with_fresnel_rho_texture, abs_cos_theta, roughness);
        return { specular_rho.x, specular_rho.y };
#else
        return { Bifrost::Assets::Shading::Rho::sample_GGX_with_fresnel(abs_cos_theta, roughness),
                 Bifrost::Assets::Shading::Rho::sample_GGX(abs_cos_theta, roughness) };
#endif
    }

    __inline_all__ static optix::float3 compute_specular_rho(optix::float3 specularity, float abs_cos_theta, float roughness) {
        auto precomputed_specular_rho = fetch_specular_rho(abs_cos_theta, roughness);
        return precomputed_specular_rho.rho(specularity);
    }

    __inline_all__ static float compute_specular_probability(optix::float3 diffuse_rho, optix::float3 specular_rho) {
        float diffuse_weight = sum(diffuse_rho);
        float specular_weight = sum(specular_rho);
        return specular_weight / (diffuse_weight + specular_weight);
    }

    __inline_all__ static float compute_coat_probability(optix::float3 base_rho, float coat_rho) {
        float base_weight = average(base_rho);
        float coat_weight = coat_rho;
        return coat_weight / (base_weight + coat_weight);
    }

    // Computes the specularity of the specular microfacet and the tint of the diffuse reflection.
    // The directional-hemispherical reflectance and transmission of the coat is computed as well if the coat is enabled.
    // * The specularity is a linear interpolation of the dielectric specularity and the conductor/metal specularity.
    //   The dielectric specularity is found by multiplying the materials specularity by 0.08, as is described on page 8 of Physically-Based Shading at Disney.
    //   The conductor specularity is simply the tint of the material, as the tint describes the color of the metal when viewed head on.
    // * Microfacet interreflection is approximated from the principle that white-is-white and energy lost 
    //   from not simulating multiple scattering events can be computed as 1 - white_specular_rho = energy_lost. 
    //   The light scattered from the 2nd, 3rd and so on scattering event is assumed to be diffuse and its rho/tint is energy_lost * specularity.
    // * The metallic parameter defines an interpolation between a dielectric material and a conductive material. 
    //   As both materials are described as an independent linear combination of diffuse and microfacet BRDFs, 
    //   the interpolation between the materials can be computed by simply interpolating the material parameters, 
    //   ie. lerp(evaluate(dielectric, ...), evaluate(conductor, ...), metallic) can be expressed as evaluate(lerp(dielectric, conductor, metallic), ...)
    // * The dielectric diffuse tint is computed as tint * (1 - specular_rho) and the metallic diffuse tint is black. 
    //   The diffuse tint of the materials is found by a linear interpolation of the dieliectric and metallic tint based on metalness.
    // * The coat is another microfacet layer on top of the diffuse + specular base. We output the single-bounce directional-hemispherical reflectance for computing importance sampling and the transmission factor.
    //   The diffuse tint is prescaled by the transmission and interreflection from the coat is added to the diffuse tint.
    //   The coat transmission cannot be baked into the specularity of the base material, due to how Fresnel is handled, and is instead applied when the specular layer is evaluated or sampled.
    __inline_all__ static void compute_tints(optix::float3 tint, float roughness, float specularity, float metallic, float coat, float coat_roughness, float abs_cos_theta,
                                             float& coat_single_bounce_rho, float& coat_transmission, optix::float3& base_specularity, optix::float3& diffuse_tint) {
        using namespace optix;

        // Clear coat with fixed index of refraction of 1.5 / specularity of 0.04, representative of polyurethane and glass.
        PrecomputedSpecularRho precomputed_coat_rho = fetch_specular_rho(abs_cos_theta, coat_roughness);
        coat_single_bounce_rho = coat * precomputed_coat_rho.rho(COAT_SPECULARITY);
        float lossless_coat_rho = coat_single_bounce_rho / precomputed_coat_rho.full;
        float coat_interreflection_rho = lossless_coat_rho - coat_single_bounce_rho;
        coat_transmission = (1 - lossless_coat_rho);

        // Specularity
        float dielectric_specularity = specularity * 0.08f; // See Physically-Based Shading at Disney bottom of page 8.
        float3 conductor_specularity = tint;
        base_specularity = lerp(make_float3(dielectric_specularity), conductor_specularity, metallic);

        // Specular directional-hemispherical reflectance function.
        auto precomputed_specular_rho = fetch_specular_rho(abs_cos_theta, roughness);
        float dielectric_specular_rho = precomputed_specular_rho.rho(dielectric_specularity);
        float3 conductor_specular_rho = precomputed_specular_rho.rho(conductor_specularity);

        // Dielectric tint.
        float dielectric_lossless_specular_rho = dielectric_specular_rho / precomputed_specular_rho.full;
        float3 dielectric_tint = tint * (1.0f - dielectric_lossless_specular_rho);

        // Microfacet specular interreflection.
        float3 single_bounce_specular_rho = lerp(make_float3(dielectric_specular_rho), conductor_specular_rho, metallic);
        float3 lossless_specular_rho = single_bounce_specular_rho / precomputed_specular_rho.full;
        float3 specular_interreflection_rho = lossless_specular_rho - single_bounce_specular_rho;

        // Combining dielectric diffuse tint with specular scattering.
        diffuse_tint = dielectric_tint * (1.0f - metallic) + specular_interreflection_rho; // Base tint and specular interreflection
        diffuse_tint *= coat_transmission; // Scale by coat transmission
        diffuse_tint += make_float3(coat_interreflection_rho); // Add contribution from coat interreflection
    }

public:

    __inline_all__ DefaultShading(const Material& material, float abs_cos_theta)
        : m_roughness(material.roughness), m_coat_scale(material.coat), m_coat_roughness(material.coat_roughness) {
        float coat_single_bounce_rho;
        compute_tints(material.tint, m_roughness, material.specularity, material.metallic, m_coat_scale, m_coat_roughness, abs_cos_theta,
                      coat_single_bounce_rho, m_coat_transmission, m_specularity, m_diffuse_tint);

        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness) * m_coat_transmission;
        float specular_probability = compute_specular_probability(m_diffuse_tint, specular_rho);
        m_specular_probability = unsigned short(specular_probability * USHORT_MAX + 0.5f);
        float coat_probability = compute_coat_probability(m_diffuse_tint + specular_rho, coat_single_bounce_rho);
        m_coat_probability = unsigned short(coat_probability * USHORT_MAX + 0.5f);
    }

#if GPU_DEVICE
    __inline_all__ DefaultShading(const Material& material, float abs_cos_theta, optix::float2 texcoord, float min_roughness = 0.0f) {
        using namespace optix;

        // Coat
        m_coat_scale = material.coat;
        m_coat_roughness = material.coat_roughness;

        // Metallic
        float metallic = material.metallic;
        if (material.metallic_texture_ID)
            metallic *= rtTex2D<float>(material.metallic_texture_ID, texcoord.x, texcoord.y);

        // Tint and roughness
        float4 tint_roughness = make_float4(material.tint, material.roughness);
        if (material.tint_roughness_texture_ID)
            tint_roughness *= rtTex2D<float4>(material.tint_roughness_texture_ID, texcoord.x, texcoord.y);
        else if (material.roughness_texture_ID)
            tint_roughness.w *= rtTex2D<float>(material.roughness_texture_ID, texcoord.x, texcoord.y);
        float3 tint = make_float3(tint_roughness);
        m_roughness = tint_roughness.w;
        m_roughness = max(m_roughness, min_roughness);
        float coat_scaled_roughness = BSDFs::GGX::roughness_from_alpha(1.0f - (1.0f - BSDFs::GGX::alpha_from_roughness(m_roughness)) * (1.0f - BSDFs::GGX::alpha_from_roughness(m_coat_roughness)));
        m_roughness = lerp(m_roughness, coat_scaled_roughness, m_coat_scale);

        float coat_single_bounce_rho;
        compute_tints(tint, m_roughness, material.specularity, metallic, m_coat_scale, m_coat_roughness, abs_cos_theta,
                      coat_single_bounce_rho, m_coat_transmission, m_specularity, m_diffuse_tint);

        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness) * m_coat_transmission;
        float specular_probability = compute_specular_probability(m_diffuse_tint, specular_rho);
        m_specular_probability = unsigned short(specular_probability * USHORT_MAX + 0.5f);
        float coat_probability = compute_coat_probability(m_diffuse_tint + specular_rho, coat_single_bounce_rho);
        m_coat_probability = unsigned short(coat_probability * USHORT_MAX + 0.5f);
    }

    __inline_all__ static DefaultShading initialize_with_max_PDF_hint(const Material& material, float abs_cos_theta, optix::float2 texcoord, float max_PDF_hint) {
        // Regularize the material by using a maximally allowed PDF hint and cos(theta) to estimate a minimally allowed GGX alpha / roughness.
        float max_PDF = fminf(max_PDF_hint, 1e37f);
        float2 uv = { max_PDF / (1.0f + max_PDF), abs_cos_theta };
        uv = uv * 62.0f / 64.0f + 1.0f / 64.0f; // Rescale uv from [0, 1] to [1/64, 63/64]. The image is 32x32 and the center of the pixels correspond to the boundary values of estimate_GGX_alpha(max_PDF, cos_theta).
        float min_alpha = tex2D(estimate_GGX_alpha_texture, uv.x, uv.y);
        min_alpha = max_PDF_hint == 0 ? 0.0f : min_alpha; // Set min_alpha to 0.0f if PDF is invalid. (The texture lookup will set it to 1.0f)
        float min_roughness = BSDFs::GGX::roughness_from_alpha(min_alpha);
        return DefaultShading(material, abs_cos_theta, texcoord, min_roughness);
    }
#endif

    __inline_all__ float get_roughness() const { return m_roughness; }

    __inline_all__ static float coverage(const Material& material, optix::float2 texcoord) {
        float coverage = material.coverage;
#if GPU_DEVICE
        if (material.coverage_texture_ID)
            coverage *= optix::rtTex2D<float>(material.coverage_texture_ID, texcoord.x, texcoord.y);
#endif
        return coverage;
    }

    __inline_all__ optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        using namespace optix;

        bool is_same_hemisphere = wi.z * wo.z >= 0.00000001f;
        if (!is_same_hemisphere)
            return make_float3(0.0f);

        // Flip directions if on the backside of the material.
        if (wo.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        optix::float3 reflectance = BSDFs::Lambert::evaluate(m_diffuse_tint, wo, wi);
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        reflectance += BSDFs::GGX::evaluate(ggx_alpha, m_specularity, wo, wi) * m_coat_transmission;

        if (m_coat_scale > 0)
        {
            float coat_alpha = BSDFs::GGX::alpha_from_roughness(m_coat_roughness);
            float coat_f = m_coat_scale * BSDFs::GGX::evaluate(coat_alpha, COAT_SPECULARITY, wo, wi);
            reflectance += make_float3(coat_f, coat_f, coat_f);
        }

        return reflectance;
    }

    __inline_all__ float PDF(optix::float3 wo, optix::float3 wi) const {
        using namespace optix;

        float specular_probability = m_specular_probability / USHORT_MAX;

        // Merge base PDFs based on the specular probability.
        float diffuse_PDF = BSDFs::Lambert::PDF(wo, wi);
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        float specular_PDF = BSDFs::GGX::PDF(ggx_alpha, wo, normalize(wo + wi));
        float PDF = lerp(diffuse_PDF, specular_PDF, specular_probability);
        
        if (m_coat_scale > 0) {
            float coat_probability = m_coat_probability / USHORT_MAX;
            float coat_alpha = BSDFs::GGX::alpha_from_roughness(m_coat_roughness);
            float coat_PDF = BSDFs::GGX::PDF(coat_alpha, wo, normalize(wo + wi));
            PDF = lerp(PDF, coat_PDF, coat_probability);
        }
        
        return PDF;
    }

    __inline_all__ BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        using namespace optix;

        bool is_same_hemisphere = wi.z * wo.z >= 0.00000001f;
        if (!is_same_hemisphere)
            return BSDFResponse::none();

        // Flip directions if on the backside of the material.
        if (wo.z < 0.0f) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        const float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        BSDFResponse specular_eval = BSDFs::GGX::evaluate_with_PDF(ggx_alpha, m_specularity, wo, wi);
        BSDFResponse diffuse_eval = BSDFs::Lambert::evaluate_with_PDF(m_diffuse_tint, wo, wi);

        BSDFResponse res;
        res.reflectance = diffuse_eval.reflectance + specular_eval.reflectance * m_coat_transmission;

        float specular_probability = m_specular_probability / USHORT_MAX;
        res.PDF = lerp(diffuse_eval.PDF, specular_eval.PDF, specular_probability);

        if (m_coat_scale > 0) {
            float coat_probability = m_coat_probability / USHORT_MAX;
            float coat_alpha = BSDFs::GGX::alpha_from_roughness(m_coat_roughness);
            BSDFResponse coat_eval = BSDFs::GGX::evaluate_with_PDF(coat_alpha, COAT_SPECULARITY, wo, wi);
            res.reflectance += m_coat_scale * coat_eval.reflectance;
            res.PDF = lerp(res.PDF, coat_eval.PDF, coat_probability);
        }

        return res;
    }

    __inline_all__ BSDFSample sample_one(optix::float3 wo, optix::float3 random_sample) const {
        using namespace optix;

        // Sample BSDFs based on the contribution of each BRDF.

        float coat_probability = m_coat_probability / USHORT_MAX;
        if (coat_probability > 0) {
            bool sample_coat = random_sample.z < coat_probability;
            if (sample_coat) {
                float coat_alpha = BSDFs::GGX::alpha_from_roughness(m_coat_roughness);
                BSDFSample coat_sample = BSDFs::GGX::sample(coat_alpha, COAT_SPECULARITY, wo, make_float2(random_sample));
                coat_sample.PDF *= coat_probability;
                return coat_sample;
            }

            // Rescale the random BSDF selection sample.
            random_sample.z = (random_sample.z - coat_probability) / (1 - coat_probability);
        }

        // Sample selected BRDF.
        float specular_probability = m_specular_probability / USHORT_MAX;
        bool sample_specular = random_sample.z < specular_probability;
        BSDFSample bsdf_sample;
        if (sample_specular) {
            float alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
            bsdf_sample = BSDFs::GGX::sample(alpha, m_specularity, wo, make_float2(random_sample));
            bsdf_sample.reflectance *= m_coat_transmission;
            bsdf_sample.PDF *= specular_probability;
        } else {
            bsdf_sample = BSDFs::Lambert::sample(m_diffuse_tint, make_float2(random_sample));
            bsdf_sample.PDF *= (1 - specular_probability);
        }

        // Scale diffuse and specular sample probability by coat probability if coat is active.
        if (m_coat_scale > 0)
            bsdf_sample.PDF *= (1 - coat_probability);

        return bsdf_sample;
    }

    __inline_all__ BSDFSample sample_all(optix::float3 wo, optix::float3 random_sample) const {
        using namespace optix;

        // Sample BSDFs based on the contribution of each BRDF.
        float specular_probability = m_specular_probability / USHORT_MAX;
        float coat_probability = m_coat_probability / USHORT_MAX;

        // Pick a BRDF
        bool sample_coat = random_sample.z < coat_probability;
        bool sample_specular = !sample_coat && (random_sample.z - coat_probability) / (1 - coat_probability) < specular_probability;
        bool sample_diffuse = !sample_coat && !sample_specular;

        const float coat_alpha = BSDFs::GGX::alpha_from_roughness(m_coat_roughness);
        const float specular_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);

        // Sample selected BRDF.
        BSDFSample bsdf_sample;
        if (sample_diffuse) {
            bsdf_sample = BSDFs::Lambert::sample(m_diffuse_tint, make_float2(random_sample));
            bsdf_sample.PDF *= (1 - coat_probability) * (1 - specular_probability);
        } else if (sample_specular) {
            bsdf_sample = BSDFs::GGX::sample(specular_alpha, m_specularity, wo, make_float2(random_sample));
            bsdf_sample.reflectance *= m_coat_transmission;
            bsdf_sample.PDF *= (1 - coat_probability) * specular_probability;
        } else {
            bsdf_sample = BSDFs::GGX::sample(coat_alpha, COAT_SPECULARITY, wo, make_float2(random_sample));
            bsdf_sample.reflectance *= m_coat_scale;
            bsdf_sample.PDF *= coat_probability;
        }

        // Break if an invalid sample is produced.
        if (!is_PDF_valid(bsdf_sample.PDF))
            return bsdf_sample;

        // Apply fresnel and compute contribution of the material components not sampled.
        if (!sample_diffuse) {
            // Evaluate diffuse layer as well.
            BSDFResponse diffuse_response = BSDFs::Lambert::evaluate_with_PDF(m_diffuse_tint, wo, bsdf_sample.direction);
            bsdf_sample.reflectance += diffuse_response.reflectance;
            bsdf_sample.PDF += (1 - coat_probability) * (1 - specular_probability) * diffuse_response.PDF;
        }
        if (!sample_specular) {
            // Evaluate specular layer as well.
            BSDFResponse specular_response = BSDFs::GGX::evaluate_with_PDF(specular_alpha, m_specularity, wo, bsdf_sample.direction);
            bsdf_sample.reflectance += specular_response.reflectance * m_coat_transmission;
            bsdf_sample.PDF += (1 - coat_probability) * specular_probability * specular_response.PDF;
        }
        if (!sample_coat && m_coat_scale > 0) {
            // Evaluate coat layer as well.
            BSDFResponse coat_response = BSDFs::GGX::evaluate_with_PDF(coat_alpha, COAT_SPECULARITY, wo, bsdf_sample.direction);
            bsdf_sample.reflectance += m_coat_scale * coat_response.reflectance;
            bsdf_sample.PDF += coat_probability * coat_response.PDF;
        }

        return bsdf_sample;
    }

    // Estimate the directional-hemispherical reflectance function.
    __inline_dev__ optix::float3 rho(float abs_cos_theta) const {
        optix::float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness) * m_coat_transmission ;
        float single_bounce_coat_rho = m_coat_scale * fetch_specular_rho(abs_cos_theta, m_coat_roughness).rho(COAT_SPECULARITY);
        return m_diffuse_tint + specular_rho + single_bounce_coat_rho;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_SHADING_H_