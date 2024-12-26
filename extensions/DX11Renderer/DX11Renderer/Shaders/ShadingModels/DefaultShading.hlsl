// Default shading.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_SHADING_MODELS_DEFAULT_SHADING_H_
#define _DX11_RENDERER_SHADERS_SHADING_MODELS_DEFAULT_SHADING_H_

#include <BSDFs/Diffuse.hlsl>
#include <BSDFs/GGX.hlsl>
#include <ShadingModels/IShadingModel.hlsl>
#include <ShadingModels/Utils.hlsl>

namespace ShadingModels {

// ------------------------------------------------------------------------------------------------
// Default shading consist of a diffuse base with a specular layer on top. It can optionally be coated.
// * The metallic parameter defines an interpolation between the evaluation of a dielectric material and the evaluation of a conductive material with the given parameters.
//   However, the same effect can be achieved by interpolating the two materials' diffuse tint and specularity, saving one material evaluation.
// * The specularity is a linear interpolation of the dielectric specularity and the conductor/metal specularity.
//   The dielectric specularity is defined by the material's specularity.
//   The conductor specularity is simply the tint of the material, as the tint describes the color of the metal when viewed head on.
// The diffuse and specular component of the base layer are set up in the following way.
// * The dielectric diffuse tint is computed as tint * dielectric_specular_transmission and the metallic diffuse tint is black. 
//   The diffuse tint of the materials is found by a linear interpolation of the dielectric and metallic tint based on metalness.
// If the coat is enabled then a microfacet coat on top of the base layer is initialized as well.
// * The base roughness is scaled by the coat roughness to simulate how light would scatter through the coat and 
//   perceptually widen the highlight of the underlying material.
// * The transmission through the coat is computed and baked into the diffuse tint and the specular scale
// ------------------------------------------------------------------------------------------------
struct DefaultShading : IShadingModel {
    float3 m_diffuse_tint;
    float m_roughness;
    float3 m_specularity;
    float m_specular_scale;
    float m_coat_scale;
    float m_coat_alpha;

    static const float COAT_SPECULARITY = 0.04f;

    static void compute_specular_properties(float roughness, float specularity, float scale, float cos_theta_o,
        out float alpha, out float reflection_scale, out float transmission_scale) {
        alpha = BSDFs::GGX::alpha_from_roughness(roughness);
        SpecularRho rho_computation = SpecularRho::fetch(cos_theta_o, roughness);

        // Compensate for lost energy due to multi-scattering and scale by the strength of the specular reflection.
        reflection_scale = scale * rho_computation.energy_loss_adjustment();
        float specular_rho = rho_computation.rho(specularity) * reflection_scale;

        // The amount of energy not reflected by the specular reflection is transmitted further into the material.
        transmission_scale = 1.0f - specular_rho;
    }

    static DefaultShading create(float3 tint, float roughness, float dielectric_specularity, float metallic, float coat_scale, float coat_roughness, float abs_cos_theta_o) {

        DefaultShading shading;

        // Adjust specular reflection roughness
        float coat_modulated_roughness = modulate_roughness_under_coat(roughness, coat_roughness);
        shading.m_roughness = lerp(roughness, coat_modulated_roughness, coat_scale);

        // Dielectric material parameters
        float specular_alpha, dielectric_specular_transmission;
        compute_specular_properties(shading.m_roughness, dielectric_specularity, 1.0f, abs_cos_theta_o,
            specular_alpha, shading.m_specular_scale, dielectric_specular_transmission);
        float3 dielectric_tint = tint * dielectric_specular_transmission;

        // Interpolate between dieletric and conductor parameters based on the metallic parameter.
        // Conductor diffuse component is black, so interpolation amounts to scaling.
        float3 conductor_specularity = tint;
        shading.m_specularity = lerp(dielectric_specularity, conductor_specularity, metallic);
        shading.m_diffuse_tint = dielectric_tint * (1.0f - metallic);

        if (coat_scale > 0) {
            // Clear coat with fixed index of refraction of 1.5 / specularity of 0.04, representative of polyurethane and glass.
            float coat_transmission;
            compute_specular_properties(coat_roughness, COAT_SPECULARITY, coat_scale, abs_cos_theta_o,
                shading.m_coat_alpha, shading.m_coat_scale, coat_transmission);

            // Scale specular and diffuse component by the coat transmission.
            shading.m_specular_scale *= coat_transmission;
            shading.m_diffuse_tint *= coat_transmission;
        } else {
            // No coat
            shading.m_coat_scale = 0;
            shading.m_coat_alpha = 0;
        }

        return shading;
    }

    // --------------------------------------------------------------------------------------------
    // Getters
    // --------------------------------------------------------------------------------------------
    float roughness() { return m_roughness; }
    float specular_alpha() { return BSDFs::GGX::alpha_from_roughness(m_roughness); }
    float coat_scale() { return m_coat_scale; }
    float coat_roughness() { return BSDFs::GGX::roughness_from_alpha(m_coat_alpha); }

    // --------------------------------------------------------------------------------------------
    // Evaluations.
    // --------------------------------------------------------------------------------------------
    float3 evaluate(float3 wo, float3 wi) {
        // Return no contribution if seen or lit from the backside.
        if (wo.z <= 0.0f || wi.z <= 0.0f)
            return float3(0, 0, 0);

        float3 reflectance = m_diffuse_tint * BSDFs::OrenNayar::evaluate(m_roughness, wo, wi);
        reflectance += m_specular_scale * BSDFs::GGX::evaluate(specular_alpha(), m_specularity, wo, wi);
        if (m_coat_scale > 0)
            reflectance += m_coat_scale * BSDFs::GGX::evaluate(m_coat_alpha, COAT_SPECULARITY, wo, wi);
        return reflectance;
    }

    // Evaluate the material lit by a sphere light.
    // Uses evaluation by most representative point internally.
    float3 evaluate_sphere_light(float3 wo, SphereLight light, float ambient_visibility) {
        // Return no contribution if seen or lit from the backside.
        bool light_below_surface = light.position.z < -light.radius;
        if (wo.z <= 0.0f || light_below_surface)
            return float3(0, 0, 0);

        float3 light_radiance = light.power * rcp(4.0f * PI * dot(light.position, light.position));

        float3 radiance = evaluate_sphere_light_lambert(light, light_radiance, wo, m_diffuse_tint, ambient_visibility);
        radiance += m_specular_scale * evaluate_sphere_light_GGX(light, light_radiance, wo, m_specularity, specular_alpha(), ambient_visibility);
        if (m_coat_scale > 0)
            radiance += m_coat_scale * evaluate_sphere_light_GGX(light, light_radiance, wo, COAT_SPECULARITY, m_coat_alpha, ambient_visibility);

        return radiance;
    }

    // Evaluate the material lit by an IBL.
    float3 evaluate_IBL(float3 wo, float3 normal, float ambient_visibility) {
        float3 radiance = evaluate_IBL_lambert(wo, normal, m_diffuse_tint, ambient_visibility);
        radiance += m_specular_scale * evaluate_IBL_GGX(wo, normal, specular_alpha(), m_specularity, ambient_visibility);
        if (m_coat_scale > 0)
            radiance += m_coat_scale * evaluate_IBL_GGX(wo, normal, m_coat_alpha, COAT_SPECULARITY, ambient_visibility);

        return radiance;
    }
};

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_DEFAULT_SHADING_H_