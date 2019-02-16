// Default shading.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_DEFAULT_SHADING_H_
#define _DX11_RENDERER_SHADERS_DEFAULT_SHADING_H_

#include "BSDFs/Diffuse.hlsl"
#include "BSDFs/GGX.hlsl"
#include "LightSources.hlsl"
#include "Utils.hlsl"

// ------------------------------------------------------------------------------------------------
// Textures.
// ------------------------------------------------------------------------------------------------

Texture2D environment_tex : register(t0);
SamplerState environment_sampler : register(s0);

Texture2D coverage_tex : register(t1);
SamplerState coverage_sampler : register(s1);

Texture2D tint_roughness_tex : register(t2);
SamplerState tint_roughness_sampler : register(s2);

Texture2D metallic_tex : register(t3);
SamplerState metallic_sampler : register(s3);

Texture2D<float2> ggx_with_fresnel_rho_tex : register(t15);

// ------------------------------------------------------------------------------------------------
// Specular rho helpers.
// ------------------------------------------------------------------------------------------------
struct PrecomputedSpecularRho {
    float base, full;
    float rho(float specularity) { return lerp(base, full, specularity); }
    float3 rho(float3 specularity) {
        return float3(rho(specularity.x), rho(specularity.y), rho(specularity.z));
    }
};

static PrecomputedSpecularRho fetch_specular_rho(float abs_cos_theta, float roughness) {
    float2 specular_rho = ggx_with_fresnel_rho_tex.Sample(bilinear_sampler, float2(abs_cos_theta, roughness));
    PrecomputedSpecularRho res;
    res.base = specular_rho.r;
    res.full = specular_rho.g;
    return res;
}

static float3 compute_specular_rho(float3 specularity, float abs_cos_theta, float roughness) {
    PrecomputedSpecularRho precomputed_specular_rho = fetch_specular_rho(abs_cos_theta, roughness);
    return precomputed_specular_rho.rho(specularity);
}

// ------------------------------------------------------------------------------------------------
// Default shading.
// ------------------------------------------------------------------------------------------------
struct DefaultShading {
    float3 m_diffuse_tint;
    float m_roughness;
    float3 m_specularity;
    float m_coverage;

    // --------------------------------------------------------------------------------------------
    // Factory function constructing a shading from a constant buffer.
    // --------------------------------------------------------------------------------------------
    static DefaultShading from_constants(uniform MaterialParams material_params, float abs_cos_theta, float2 texcoord) {
        DefaultShading shading;

        // Coverage
        shading.m_coverage = material_params.m_coverage;
        if (material_params.m_textures_bound & TextureBound::Coverage)
            shading.m_coverage *= coverage_tex.Sample(coverage_sampler, texcoord).a;

        // Tint and roughness
        float4 tint_roughness = { material_params.m_tint, material_params.m_roughness };
        if (material_params.m_textures_bound & TextureBound::Tint_Roughness) {
            float4 tex_sample = tint_roughness_tex.Sample(tint_roughness_sampler, texcoord);
            tex_sample.rgb = (material_params.m_textures_bound & TextureBound::Tint) == 0 ? float3(1, 1, 1) : tex_sample.rgb;
            tint_roughness *= tex_sample;
        }
        float3 tint = tint_roughness.rgb;
        shading.m_roughness = tint_roughness.a;

        // Metallic
        float metallic = material_params.m_metallic;
        if (material_params.m_textures_bound & TextureBound::Metallic)
            metallic *= metallic_tex.Sample(metallic_sampler, texcoord).a;

        // Specularity
        float dielectric_specularity = material_params.m_specularity * 0.08f; // See Physically-Based Shading at Disney bottom of page 8.
        float3 conductor_specularity = tint;
        shading.m_specularity = lerp(dielectric_specularity.rrr, conductor_specularity, metallic);

        // Specular directional-hemispherical reflectance function.
        PrecomputedSpecularRho precomputed_specular_rho = fetch_specular_rho(abs_cos_theta, shading.m_roughness);
        float dielectric_specular_rho = precomputed_specular_rho.rho(dielectric_specularity);
        float3 conductor_specular_rho = precomputed_specular_rho.rho(conductor_specularity);

        // Dielectric tint.
        float dielectric_lossless_specular_rho = dielectric_specular_rho / precomputed_specular_rho.full;
        float3 dielectric_tint = tint * (1.0f - dielectric_lossless_specular_rho);

        // Microfacet specular interreflection.
        float3 specular_rho = lerp(dielectric_specular_rho.rrr, conductor_specular_rho, metallic);
        float3 lossless_specular_rho = specular_rho / precomputed_specular_rho.full;
        float3 specular_secondary_scattering_rho = lossless_specular_rho - specular_rho;
        shading.m_diffuse_tint = dielectric_tint * (1.0f - metallic) + specular_secondary_scattering_rho; // Diffuse tint combines the tint of the diffuse scattering with the secondary scattering from the specular layer.

        return shading;
    }

    // --------------------------------------------------------------------------------------------
    // Getters
    // --------------------------------------------------------------------------------------------
    float coverage() { return m_coverage; }
    float roughness() { return m_roughness; }
    float3 diffuse_tint() { return m_diffuse_tint; }
    float3 specularity() { return m_specularity; }

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

        float3 diffuse = m_diffuse_tint * BSDFs::Lambert::evaluate();
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        float3 specular = BSDFs::GGX::evaluate(ggx_alpha, m_specularity, wo, wi);
        return diffuse + specular;
    }

    // Most representativepoint material evaluation, heavily inspired by Real Shading in Unreal Engine 4.
    // For UE4 reference see the function AreaLightSpecular() in DeferredLightingCommon.usf. (15/1 -2018)
    float3 evaluate_area_light(LightData light, float3 world_position, float3 wo, float3x3 world_to_shading_TBN, float ambient_visibility) {

        // Sphere light in local space
        float3 local_sphere_position = mul(world_to_shading_TBN, light.sphere_position() - world_position);
        float3 light_radiance = light.sphere_power() * rcp(4.0f * PI * dot(local_sphere_position, local_sphere_position));

        // Closest point on sphere to ray. Equation 11 in Real Shading in Unreal Engine 4, 2013.
        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(roughness());
        float3 peak_reflection = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo);
        float3 closest_point_on_ray = dot(local_sphere_position, peak_reflection) * peak_reflection;
        float3 center_to_ray = closest_point_on_ray - local_sphere_position;
        float3 most_representative_point = local_sphere_position + center_to_ray * saturate(light.sphere_radius() * reciprocal_length(center_to_ray));
        float3 wi = normalize(most_representative_point);

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

        { // Evaluate GGX.
            bool delta_GGX_distribution = ggx_alpha < 0.0005;
            if (delta_GGX_distribution) {
                // Check if peak reflection and the most representative point are aligned.
                float toggle = saturate(100000 * (dot(peak_reflection, wi) - 0.99999));
                float recip_divisor = rcp(PI * sphere_surface_area(light.sphere_radius()));
                float3 light_radiance = light.sphere_power() * recip_divisor;
                radiance += m_specularity * light_radiance * toggle;
            }
            else {
                // Deprecated area light normalization term. Equation 10 and 14 in Real Shading in Unreal Engine 4, 2013. Included for completeness
                // float adjusted_ggx_alpha = saturate(ggx_alpha + light.sphere_radius() / (3 * length(local_sphere_position)));
                // float area_light_normalization_term = pow2(ggx_alpha / adjusted_ggx_alpha);

                // NOTE We could try fitting the constants and try cos_theta VS wo and local_sphere_position VS local_sphere_position.
                float cos_theta = max(wi.z, 0.0);
                float sin_theta_squared = pow2(light.sphere_radius()) / dot(most_representative_point, most_representative_point);
                float a2 = pow2(ggx_alpha);
                float area_light_normalization_term = a2 / (a2 + sin_theta_squared / (cos_theta * 3.6 + 0.4));
                float specular_ambient_visibility = lerp(1, scaled_ambient_visibility, a2);

                radiance += BSDFs::GGX::evaluate(ggx_alpha, m_specularity, wo, wi) * cos_theta * light_radiance * area_light_normalization_term * specular_ambient_visibility;
            }
        }

        return radiance * scaled_ambient_visibility;
    }

    // Apply the shading model to the IBL.
    // TODO Take the current LOD and pixel density into account before choosing sample LOD.
    //      See http://casual-effects.blogspot.dk/2011/08/plausible-environment-lighting-in-two.html 
    //      for how to derive the LOD level for cubemaps.
    float3 evaluate_IBL(float3 wo, float3 normal) {
        float width, height, mip_count;
        environment_tex.GetDimensions(0, width, height, mip_count);

        float2 diffuse_tc = direction_to_latlong_texcoord(normal);
        float3 diffuse = m_diffuse_tint * environment_tex.SampleLevel(environment_sampler, diffuse_tc, mip_count).rgb;

        float ggx_alpha = BSDFs::GGX::alpha_from_roughness(m_roughness);
        float3 wi = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo, normal);
        float abs_cos_theta = abs(dot(wo, normal));
        float3 specular_rho = compute_specular_rho(m_specularity, abs_cos_theta, m_roughness);
        float2 specular_tc = direction_to_latlong_texcoord(wi);
        float3 specular = specular_rho * environment_tex.SampleLevel(environment_sampler, specular_tc, mip_count * m_roughness).rgb;

        return diffuse + specular;
    }

};

#endif // _DX11_RENDERER_SHADERS_DEFAULT_SHADING_H_