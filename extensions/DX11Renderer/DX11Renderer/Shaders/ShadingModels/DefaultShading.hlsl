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
// Default shading.
// ------------------------------------------------------------------------------------------------
struct DefaultShading : IShadingModel {
    float3 m_diffuse_tint;
    float m_roughness;
    float3 m_specularity;
    float m_coat_transmission; // 1 - coat_rho, the contribution from the BSDFs under the coat.
    float m_coat_scale;
    float m_coat_roughness;

    static const float COAT_SPECULARITY = 0.04f;

    // Sets up the diffuse and specular component on the base layer.
    // * The dielectric diffuse tint is computed as tint * dielectric_specular_transmission and the metallic diffuse tint is black. 
    //   The diffuse tint of the materials is found by a linear interpolation of the dielectric and metallic tint based on metalness.
    // * The metallic parameter defines an interpolation between a dielectric material and a conductive material with the given parameters.
    //   As both materials are described as an independent linear combination of diffuse and microfacet BRDFs, 
    //   the interpolation between the materials can be computed by simply interpolating the material parameters, 
    //   ie. lerp(dielectric.evaluate(...), conductor.evaluate(...), metallic) can be expressed as lerp_params(dielectric, conductor, metallic)evaluate(...)
    // * The specularity is a linear interpolation of the dielectric specularity and the conductor/metal specularity.
    //   The dielectric specularity is defined by the material's specularity.
    //   The conductor specularity is simply the tint of the material, as the tint describes the color of the metal when viewed head on.
    // * Microfacet interreflection is approximated from the principle that white-is-white and energy lost 
    //   from not simulating multiple scattering events can be computed as 1 - white_specular_rho = full_specular_interreflection.
    //   The specular interreflection with the given specularity can be found by scaling with rho of the given specularity,
    //   specular_interreflection = full_specular_interreflection / specular_rho.
    //   The light scattered from interreflection of the 2nd, 3rd and so on scattering event is assumed to be diffuse and its contribution is added to the diffuse tint.
    static DefaultShading create(float3 tint, float roughness, float specularity, float metallic, float abs_cos_theta) {
        DefaultShading shading;

        // No coat
        shading.m_coat_scale = 0;
        shading.m_coat_roughness = 0;
        shading.m_coat_transmission = 1;

        // Roughness
        shading.m_roughness = roughness;

        // Specularity
        float dielectric_specularity = specularity;
        float3 conductor_specularity = tint;
        shading.m_specularity = lerp(dielectric_specularity, conductor_specularity, metallic);

        // Specular directional-hemispherical reflectance function, rho.
        SpecularRho specular_rho = SpecularRho::fetch(abs_cos_theta, roughness);
        float dielectric_specular_rho = specular_rho.rho(dielectric_specularity);

        // Dielectric tint.
        float dielectric_lossless_specular_rho = dielectric_specular_rho / specular_rho.full;
        float dielectric_specular_transmission = 1.0f - dielectric_lossless_specular_rho;
        float3 dielectric_tint = tint * dielectric_specular_transmission;
        shading.m_diffuse_tint = dielectric_tint * (1.0f - metallic);

        // Compute microfacet specular interreflection. Assume it is diffuse and add its contribution to the diffuse tint.
        float3 single_bounce_specular_rho = specular_rho.rho(shading.m_specularity);
        float3 lossless_specular_rho = single_bounce_specular_rho / specular_rho.full;
        float3 specular_interreflection_rho = lossless_specular_rho - single_bounce_specular_rho;
        shading.m_diffuse_tint += specular_interreflection_rho;

        return shading;
    }

    // Sets up the specular microfacet and the diffuse reflection as described in setup_base_layer().
    // If the coat is enabled then a microfacet coat on top of the base layer is initialized as well.
    // * The base roughness is scaled by the coat roughness to simulate how light would scatter through the coat and 
    //   perceptually widen the highlight of the underlying material.
    // * The transmission through the coat is computed and baked into the diffuse tint.
    //   The contribution from coat interreflection is then added to the diffuse tint.
    // * The transmission cannot be baked into the specularity, as specularity only represents the contribution from the specular layer 
    //   when viewed head on and we need the transmission to affect the reflectance at grazing angles as well.
    //   Instead we store the transmission and scale the base layer's specular contribution by it when evaluating or sampling.
    static DefaultShading create(float3 tint, float roughness, float specularity, float metallic, float coat_scale, float coat_roughness, float abs_cos_theta) {
        float coat_modulated_roughness = modulate_roughness_under_coat(roughness, coat_roughness);
        roughness = lerp(roughness, coat_modulated_roughness, coat_scale);

        DefaultShading shading = create(tint, roughness, specularity, metallic, abs_cos_theta);

        if (coat_scale > 0) {
            // Clear coat with fixed index of refraction of 1.5 / specularity of 0.04, representative of polyurethane and glass.
            shading.m_coat_scale = coat_scale;
            shading.m_coat_roughness = coat_roughness;
            SpecularRho coat_rho = SpecularRho::fetch(abs_cos_theta, coat_roughness);
            float coat_single_bounce_rho = coat_scale * coat_rho.rho(COAT_SPECULARITY);
            float lossless_coat_rho = coat_single_bounce_rho / coat_rho.full;
            float coat_interreflection_rho = lossless_coat_rho - coat_single_bounce_rho;
            shading.m_coat_transmission = 1.0f - lossless_coat_rho;

            // Scale diffuse component by the coat transmission and add contribution from coat interreflection to the diffues tint. 
            shading.m_diffuse_tint *= shading.m_coat_transmission;
            shading.m_diffuse_tint += coat_interreflection_rho;
        }

        return shading;
    }

    // --------------------------------------------------------------------------------------------
    // Getters
    // --------------------------------------------------------------------------------------------
    float roughness() { return m_roughness; }
    float coat_scale() { return m_coat_scale; }
    float coat_roughness() { return m_coat_roughness; }

    // --------------------------------------------------------------------------------------------
    // Evaluations.
    // --------------------------------------------------------------------------------------------
    float3 evaluate(float3 wo, float3 wi) {
        bool is_same_hemisphere = wi.z * wo.z >= 0.0;
        if (!is_same_hemisphere)
            return float3(0.0, 0.0, 0.0);

        // Flip directions if on the backside of the material.
        if (wo.z < 0.0) {
            wi.z = -wi.z;
            wo.z = -wo.z;
        }

        float3 reflectance = m_diffuse_tint * BSDFs::Lambert::evaluate();
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        reflectance += BSDFs::GGX::evaluate(ggx_alpha, m_specularity, wo, wi) * m_coat_transmission;
        if (m_coat_scale > 0) {
            float coat_ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_coat_roughness);
            reflectance += m_coat_scale * BSDFs::GGX::evaluate(coat_ggx_alpha, COAT_SPECULARITY, wo, wi);
        }
        return reflectance;
    }

    // Evaluate the material lit by an area light.
    // Uses evaluation by most representative point internally.
    float3 evaluate_area_light(LightData light, float3 world_position, float3 wo, float3x3 world_to_shading_TBN, float ambient_visibility) {

        // Sphere light in local space
        float3 local_sphere_position = mul(world_to_shading_TBN, light.sphere_position() - world_position);
        float3 light_radiance = light.sphere_power() * rcp(4.0f * PI * dot(local_sphere_position, local_sphere_position));

        // Evaluate Lambert.
        Sphere local_sphere = Sphere::make(local_sphere_position, light.sphere_radius());
        Cone light_sphere_cap = sphere_to_sphere_cap(local_sphere.position, local_sphere.radius);
        float solidangle_of_light = solidangle(light_sphere_cap);
        CentroidAndSolidangle centroid_and_solidangle = centroid_and_solidangle_on_hemisphere(light_sphere_cap);
        float light_radiance_scale = centroid_and_solidangle.solidangle / solidangle_of_light;
        float3 radiance = m_diffuse_tint * BSDFs::Lambert::evaluate() * abs(centroid_and_solidangle.centroid_direction.z) * light_radiance * light_radiance_scale;

        // Scale ambient visibility by subtended solid angle.
        float solidangle_percentage = inverse_lerp(0, TWO_PI, solidangle_of_light);
        float scaled_ambient_visibility = lerp(1.0, ambient_visibility, solidangle_percentage);

        radiance *= scaled_ambient_visibility;

        radiance += evaluate_sphere_light_GGX(light, local_sphere_position, light_radiance, wo, m_specularity, m_roughness, scaled_ambient_visibility) * m_coat_transmission;

        if (m_coat_scale > 0)
            radiance += m_coat_scale * evaluate_sphere_light_GGX(light, local_sphere_position, light_radiance, wo, COAT_SPECULARITY, m_coat_roughness, scaled_ambient_visibility);

        return radiance;
    }

    // GGX sphere light evaluation by most representative point, heavily inspired by Real Shading in Unreal Engine 4.
    // For UE4 reference see the function AreaLightSpecular() in DeferredLightingCommon.usf. (15/1 -2018)
    float3 evaluate_sphere_light_GGX(LightData light, float3 local_light_position, float3 light_radiance, float3 wo, float3 specularity, float roughness, float ambient_visibility) {
        // Closest point on sphere to ray. Equation 11 in Real Shading in Unreal Engine 4, 2013.
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(roughness);
        float3 peak_reflection = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo);
        float3 closest_point_on_ray = dot(local_light_position, peak_reflection) * peak_reflection;
        float3 center_to_ray = closest_point_on_ray - local_light_position;
        float3 most_representative_point = local_light_position + center_to_ray * saturate(light.sphere_radius() * reciprocal_length(center_to_ray));
        float3 wi = normalize(most_representative_point);

        bool delta_GGX_distribution = ggx_alpha < 0.0005;
        if (delta_GGX_distribution) {
            // Check if peak reflection and the most representative point are aligned.
            float toggle = saturate(100000 * (dot(peak_reflection, wi) - 0.99999));
            float recip_divisor = rcp(PI * sphere_surface_area(light.sphere_radius()));
            float3 light_radiance = light.sphere_power() * recip_divisor;
            return specularity * light_radiance * toggle;
        } else {
            // Deprecated area light normalization term. Equation 10 and 14 in Real Shading in Unreal Engine 4, 2013. Included for completeness
            // float adjusted_ggx_alpha = saturate(ggx_alpha + light.sphere_radius() / (3 * length(local_light_position)));
            // float area_light_normalization_term = pow2(ggx_alpha / adjusted_ggx_alpha);

            // NOTE We could try fitting the constants and try cos_theta VS wo and local_sphere_position VS local_sphere_position.
            float cos_theta = max(wi.z, 0.0);
            float sin_theta_squared = pow2(light.sphere_radius()) / dot(most_representative_point, most_representative_point);
            float a2 = pow2(ggx_alpha);
            float area_light_normalization_term = a2 / (a2 + sin_theta_squared / (cos_theta * 3.6 + 0.4));
            float specular_ambient_visibility = lerp(1, ambient_visibility, a2);

            return BSDFs::GGX::evaluate(ggx_alpha, specularity, wo, wi) * cos_theta * light_radiance * area_light_normalization_term * specular_ambient_visibility;
        }
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

        float abs_cos_theta = abs(dot(wo, normal));

        radiance += evaluate_IBL_GGX(wo, normal, abs_cos_theta, m_roughness, m_specularity, mip_count) * m_coat_transmission;
        if (m_coat_scale > 0)
            radiance += m_coat_scale * evaluate_IBL_GGX(wo, normal, abs_cos_theta, m_coat_roughness, COAT_SPECULARITY, mip_count);

        return radiance;
    }

    // Evaluate GGX lit by an IBL.
    float3 evaluate_IBL_GGX(float3 wo, float3 normal, float abs_cos_theta, float roughness, float3 specularity, int mip_count) {
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(roughness);
        float3 wi = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo, normal);
        float3 rho = compute_specular_rho(specularity, abs_cos_theta, roughness);
        float2 ibl_tc = direction_to_latlong_texcoord(wi);
        return rho * environment_tex.SampleLevel(environment_sampler, ibl_tc, mip_count * roughness).rgb;
    }
};

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_DEFAULT_SHADING_H_