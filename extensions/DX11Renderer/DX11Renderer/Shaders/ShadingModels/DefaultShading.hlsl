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
#include <ShadingModels/Parameters.hlsl>

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
    float m_specular_alpha;
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
        roughness = lerp(roughness, coat_modulated_roughness, coat_scale);

        // Dielectric material parameters
        float dielectric_specular_transmission;
        compute_specular_properties(roughness, dielectric_specularity, 1.0f, abs_cos_theta_o,
            shading.m_specular_alpha, shading.m_specular_scale, dielectric_specular_transmission);
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
    float roughness() { return BSDFs::GGX::roughness_from_alpha(m_specular_alpha); }
    float coat_scale() { return m_coat_scale; }
    float coat_roughness() { return BSDFs::GGX::roughness_from_alpha(m_coat_alpha); }

    // --------------------------------------------------------------------------------------------
    // Evaluations.
    // --------------------------------------------------------------------------------------------
    float3 evaluate(float3 wo, float3 wi) {
        // Return no contribution if seen or lit from the backside.
        if (wo.z <= 0.0f || wi.z <= 0.0f)
            return float3(0, 0, 0);

        float3 reflectance = m_diffuse_tint * BSDFs::Lambert::evaluate();
        reflectance += m_specular_scale * BSDFs::GGX::evaluate(m_specular_alpha, m_specularity, wo, wi);
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

        // Scale ambient visibility.
        Cone light_sphere_cap = light.get_sphere_cap();
        float scaled_ambient_visibility = SphereLight::scale_ambient_visibility(light_sphere_cap, ambient_visibility);

        // Evaluate Lambert.
        CentroidAndSolidangle centroid_and_solidangle = centroid_and_solidangle_on_hemisphere(light_sphere_cap);
        float light_radiance_scale = centroid_and_solidangle.solidangle / solidangle(light_sphere_cap);
        float3 diffuse_f = m_diffuse_tint * BSDFs::Lambert::evaluate();
        float3 diffuse_light_contribution = light_radiance * centroid_and_solidangle.centroid_direction.z * light_radiance_scale * scaled_ambient_visibility;
        float3 radiance = diffuse_f * diffuse_light_contribution;

        radiance += m_specular_scale * evaluate_sphere_light_GGX(light, light_radiance, wo, m_specularity, m_specular_alpha, scaled_ambient_visibility);

        if (m_coat_scale > 0)
            radiance += m_coat_scale * evaluate_sphere_light_GGX(light, light_radiance, wo, COAT_SPECULARITY, m_coat_alpha, scaled_ambient_visibility);

        return radiance;
    }

    // GGX sphere light evaluation by most representative point, heavily inspired by Real Shading in Unreal Engine 4.
    // For UE4 reference see the function AreaLightSpecular() in DeferredLightingCommon.usf. (15/1 -2018)
    float3 evaluate_sphere_light_GGX(SphereLight light, float3 light_radiance, float3 wo, float3 specularity, float ggx_alpha, float ambient_visibility) {
        // Closest point on sphere to ray. Equation 11 in Real Shading in Unreal Engine 4, 2013.
        float3 peak_reflection = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo);
        float3 closest_point_on_ray = dot(light.position, peak_reflection) * peak_reflection;
        float3 center_to_ray = closest_point_on_ray - light.position;
        float3 most_representative_point = light.position + center_to_ray * saturate(light.radius * reciprocal_length(center_to_ray));
        float3 wi = normalize(most_representative_point);

        // Return no contribution if lit from the backside.
        // Due to floating point precision this can happen even if the light is not found to be on the backside.
        if (wi.z <= 0.0f)
            return float3(0, 0, 0);

        // Adjust ambient visibility such that a rough surface has full ambient occlusion applied, while a mirror reflection has none.
        float specular_ambient_visibility = lerp(1, ambient_visibility, ggx_alpha);

        // Limit GGX alpha as nearly specular surfaces produce artifacts.
        ggx_alpha = max(0.0005, ggx_alpha);
        float cos_theta_i = wi.z;
        float sin_theta_squared = pow2(light.radius) / dot(most_representative_point, most_representative_point);
        float a2 = pow2(ggx_alpha);
        float area_light_normalization_term = a2 / (a2 + sin_theta_squared / (cos_theta_i * 3.6 + 0.4));

        return BSDFs::GGX::evaluate(ggx_alpha, specularity, wo, wi) * cos_theta_i * light_radiance * area_light_normalization_term * specular_ambient_visibility;
    }

    // Apply the shading model to the IBL.
    // TODO Take the current LOD and pixel density into account before choosing sample LOD.
    //      See http://casual-effects.blogspot.dk/2011/08/plausible-environment-lighting-in-two.html 
    //      for how to derive the LOD level for cubemaps.
    float3 evaluate_IBL(float3 wo, float3 normal) {
        float width, height, mip_count;
        environment_tex.GetDimensions(0, width, height, mip_count);

        float2 diffuse_tc = direction_to_latlong_texcoord(normal);
        float3 radiance = m_diffuse_tint * environment_tex.SampleLevel(environment_sampler, diffuse_tc, mip_count).rgb;

        float abs_cos_theta_o = abs(dot(wo, normal));

        radiance += m_specular_scale * evaluate_IBL_GGX(wo, normal, abs_cos_theta_o, m_specular_alpha, m_specularity, mip_count);
        if (m_coat_scale > 0)
            radiance += m_coat_scale * evaluate_IBL_GGX(wo, normal, abs_cos_theta_o, m_coat_alpha, COAT_SPECULARITY, mip_count);

        return radiance;
    }

    // Evaluate GGX lit by an IBL.
    float3 evaluate_IBL_GGX(float3 wo, float3 normal, float abs_cos_theta_o, float ggx_alpha, float3 specularity, int mip_count) {
        float roughness = BSDFs::GGX::roughness_from_alpha(ggx_alpha);
        float3 wi = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo, normal);
        float3 rho = compute_specular_rho(specularity, abs_cos_theta_o, roughness);
        float2 ibl_tc = direction_to_latlong_texcoord(wi);
        return rho * environment_tex.SampleLevel(environment_sampler, ibl_tc, mip_count * roughness).rgb;
    }
};

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_DEFAULT_SHADING_H_