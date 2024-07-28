// Shading model utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_SHADING_MODELS_UTILS_H_
#define _DX11_RENDERER_SHADERS_SHADING_MODELS_UTILS_H_

#include <BSDFs/Diffuse.hlsl>
#include <BSDFs/GGX.hlsl>
#include <LightSources.hlsl>
#include <Utils.hlsl>

namespace ShadingModels {

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
struct SpecularRho {
    static const int angle_sample_count = 32;
    static const int roughness_sample_count = 32;

    float base, full;
    float rho(float specularity) { return lerp(base, full, specularity); }
    float3 rho(float3 specularity) {
        return float3(rho(specularity.x), rho(specularity.y), rho(specularity.z));
    }

    // Compensate for lost energy due to multiple scattering.
    // Multiple-scattering microfacet BSDFs with the smith model, Heitz et al., 2016 and 
    // Practical multiple scattering compensation for microfacet models, Emmanuel Turquin, 2018
    // showed that multiple-scattered reflectance has roughly the same distribution as single-scattering reflectance.
    // We can therefore account for energy lost to multi-scattering by computing the ratio of lost energy of a fully specular material,
    // and then scaling the specular reflectance by that ratio during evaluation, which increases reflectance to account for energy lost.
    float energy_loss_adjustment() { return 1.0f / full; }

    static SpecularRho fetch(float abs_cos_theta, float roughness) {
        // Adjust UV coordinates to start sampling half a pixel into the texture, as the pixel values correspond to the boundaries of the rho function.
        float cos_theta_uv = lerp(0.5 / angle_sample_count, (angle_sample_count - 0.5) / angle_sample_count, abs_cos_theta);
        float roughness_uv = lerp(0.5 / roughness_sample_count, (roughness_sample_count - 0.5) / roughness_sample_count, roughness);

        float2 specular_rho = ggx_with_fresnel_rho_tex.SampleLevel(bilinear_sampler, float2(cos_theta_uv, roughness_uv), 0);
        SpecularRho res;
        res.base = specular_rho.r;
        res.full = specular_rho.g;
        return res;
    }
};

static float3 compute_specular_rho(float3 specularity, float abs_cos_theta, float roughness) {
    return SpecularRho::fetch(abs_cos_theta, roughness).rho(specularity);
}

// ------------------------------------------------------------------------------------------------
// Parameters.
// ------------------------------------------------------------------------------------------------

struct TextureBound {
    static const unsigned int None = 0;
    static const unsigned int Tint = 1 << 0;
    static const unsigned int Roughness = 1 << 1;
    static const unsigned int Tint_Roughness = Tint | Roughness;
    static const unsigned int Coverage = 1 << 2;
    static const unsigned int Metallic = 1 << 3;
};

struct Parameters {
    float3 m_tint;
    unsigned int m_textures_bound;
    float m_roughness;
    float m_specularity;
    float m_metallic;
    float m_coverage;
    float m_coat;
    float m_coat_roughness;
    int shading_model; // Shading model pixel shaders are set on the host, but this can be used to validate that the correct pixel shaders are used.
    float __padding;

    float4 tint_roughness(float2 texcoord) {
        float4 tint_roughness = float4(m_tint, m_roughness);
        if (m_textures_bound & TextureBound::Tint_Roughness) {
            float4 tex_scale = tint_roughness_tex.Sample(tint_roughness_sampler, texcoord);
            tex_scale.rgb = (m_textures_bound & TextureBound::Tint) == 0 ? float3(1, 1, 1) : tex_scale.rgb;
            tint_roughness *= tex_scale;
        }
        return tint_roughness;
    }

    float metallic(float2 texcoord) {
        float metallic = m_metallic;
        if (m_textures_bound & TextureBound::Metallic)
            metallic *= metallic_tex.Sample(metallic_sampler, texcoord).a;
        return metallic;
    }

    float coverage(float2 texcoord) {
        float coverage = m_coverage;
        if (m_textures_bound & TextureBound::Coverage)
            coverage *= coverage_tex.Sample(coverage_sampler, texcoord).a;
        return coverage;
    }

    bool discard_from_cutout(float2 texcoord) {
        if (m_textures_bound & TextureBound::Coverage)
            return coverage_tex.Sample(coverage_sampler, texcoord).a < m_coverage;
        else
            return false;
    }

    float coat_scale() { return m_coat; }
    float coat_roughness() { return m_coat_roughness; }
};

//-----------------------------------------------------------------------------
// Sphere light BRDF approximations.
//-----------------------------------------------------------------------------

float3 evaluate_sphere_light_lambert(SphereLight light, float3 light_radiance, float3 wo, float3 tint, float ambient_visibility) {
    // Scale ambient visibility.
    Cone light_sphere_cap = light.get_sphere_cap();

    // Use the spherical cap of a sphere light to scale ambient visibility.
    // The rationale is that a large light needs AO to effectively occlude the light,
    // while a small light source can rely on shadowmaps or other means of direct sampling.
    float solidangle_of_light = solidangle(light_sphere_cap);
    float solidangle_percentage = inverse_lerp(0, TWO_PI, solidangle_of_light);
    float scaled_ambient_visibility = lerp(1.0, ambient_visibility, solidangle_percentage);

    // Evaluate Lambert lighting.
    CentroidAndSolidangle centroid_and_solidangle = centroid_and_solidangle_on_hemisphere(light_sphere_cap);
    float light_radiance_scale = centroid_and_solidangle.solidangle / solidangle(light_sphere_cap);
    float3 lambert_f = tint * BSDFs::Lambert::evaluate();
    float3 light_contribution = light_radiance * centroid_and_solidangle.centroid_direction.z * light_radiance_scale * scaled_ambient_visibility;
    return lambert_f * light_contribution;
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

    float specular_ambient_visibility = BSDFs::GGX::scaled_ambient_visibility(ggx_alpha, ambient_visibility);

    // Limit GGX alpha as nearly specular surfaces produce artifacts.
    ggx_alpha = max(0.0005, ggx_alpha);

    float cos_theta_i = wi.z;
    float sin_theta_squared = pow2(light.radius) / dot(most_representative_point, most_representative_point);
    float a2 = pow2(ggx_alpha);
    float area_light_normalization_term = a2 / (a2 + sin_theta_squared / (cos_theta_i * 3.6 + 0.4));

    return BSDFs::GGX::evaluate(ggx_alpha, specularity, wo, wi) * cos_theta_i * light_radiance * area_light_normalization_term * specular_ambient_visibility;
}

//-----------------------------------------------------------------------------
// IBL evaluation
//-----------------------------------------------------------------------------

float3 evaluate_IBL_lambert(float3 wo, float3 normal, float3 tint, float ambient_visibility) {
    float width, height, mip_count;
    environment_tex.GetDimensions(0, width, height, mip_count);

    float2 environment_tc = direction_to_latlong_texcoord(normal);
    return ambient_visibility * tint * environment_tex.SampleLevel(environment_sampler, environment_tc, mip_count).rgb;
}

// Evaluate GGX lit by an IBL.
// TODO Take the current LOD and pixel density into account before choosing sample LOD.
//      See http://casual-effects.blogspot.dk/2011/08/plausible-environment-lighting-in-two.html 
//      for how to derive the LOD level for cubemaps.
float3 evaluate_IBL_GGX(float3 wo, float3 normal, float ggx_alpha, float3 specularity, float ambient_visibility) {
    float width, height, mip_count;
    environment_tex.GetDimensions(0, width, height, mip_count);

    float specular_ambient_visibility = BSDFs::GGX::scaled_ambient_visibility(ggx_alpha, ambient_visibility);

    float roughness = BSDFs::GGX::roughness_from_alpha(ggx_alpha);
    float3 wi = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo, normal);
    float3 rho = compute_specular_rho(specularity, abs(dot(wo, normal)), roughness);
    float2 ibl_tc = direction_to_latlong_texcoord(wi);
    return rho * environment_tex.SampleLevel(environment_sampler, ibl_tc, mip_count * roughness).rgb;
}

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_UTILS_H_