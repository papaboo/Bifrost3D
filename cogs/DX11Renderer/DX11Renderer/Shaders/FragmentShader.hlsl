// Model fragment shaders.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "DefaultShading.hlsl"
#include "LightSources.hlsl"
#include "SPTD.hlsl"

#define SPTD_AREA_LIGHTS 0

cbuffer scene_variables : register(b0) {
    float4x4 view_projection_matrix;
    float4 camera_position;
    float4 environment_tint; // .w component is 0 if an environment tex is not bound, otherwise positive.
    float4x4 inverted_view_projection_matrix;
};

cbuffer lights : register(b1) {
    int4 light_count;
    LightData light_data[12];
}

cbuffer material : register(b3) {
    DefaultShading material;
}

Texture2D sptd_ggx_fit_tex : register(t14);

// Most representativepoint material evaluation, heavily inspired by Real Shading in Unreal Engine 4.
// For UE4 reference see the function AreaLightSpecular() in DeferredLightingCommon.usf. (15/1 -2018)
float3 evaluate_most_representative_point(LightData light, DefaultShading material, float2 texcoord,
                                          float3 world_position, float3x3 world_to_shading_TBN, float3 wo) {

    // TODO Precompute all light independent variables, such as the off specular peak and potentially others.

    // Sphere light in local space
    float3 local_sphere_position = mul(world_to_shading_TBN, light.sphere_position() - world_position);
    Sphere local_sphere = Sphere::make(local_sphere_position, light.sphere_radius());
    Cone light_sphere_cap = sphere_to_sphere_cap(local_sphere.position, local_sphere.radius);

    LightSample light_sample = sample_light(light, world_position); // TODO Only used for the radiance, so just replace by the radiance calculation.

    // Approximation of GGX off-specular peak direction.
    float ggx_alpha = BSDFs::GGX::alpha_from_roughness(material.roughness());
    float3 peak_reflection = BSDFs::GGX::approx_off_specular_peak(ggx_alpha, wo);

    // Closest point on sphere to ray. Equation 11 in Real Shading in Unreal Engine 4, 2013.
    // TODO Check at grazing angles. Should we switch to the centroid at those? Perhaps a weighted average with cos_theta as weight.
    float3 closest_point_on_ray = dot(local_sphere_position, peak_reflection) * peak_reflection;
    float3 center_to_ray = closest_point_on_ray - local_sphere_position;
    float3 most_representative_point = local_sphere_position + center_to_ray * saturate(local_sphere.radius / length(center_to_ray)); // TODO Use rsqrt
    float3 wi = normalize(most_representative_point);

    float3 diffuse_tint, specular_tint;
    material.evaluate_tints(wo, wi, texcoord, diffuse_tint, specular_tint);

    float3 radiance = float3(0, 0, 0);
    { // Evaluate Lambert.
        float solidangle_of_light = solidangle(light_sphere_cap);
        CentroidAndSolidangle centroid_and_solidangle = centroid_and_solidangle_on_hemisphere(light_sphere_cap);
        float light_radiance_scale = centroid_and_solidangle.solidangle / solidangle_of_light;
        radiance += diffuse_tint * BSDFs::Lambert::evaluate() * abs(centroid_and_solidangle.centroid_direction.z) * light_sample.radiance * light_radiance_scale;
    }

    { // Evaluate GGX/microfacet by finding the most representative point on the light source. 
        bool delta_GGX_distribution = ggx_alpha < 0.0005;
        if (delta_GGX_distribution) {
            // Check if perfect reflection and the most representative point are aligned.
            float toggle = dot(peak_reflection, wi) > 0.99999 ? 1 : 0;
            float inv_divisor = rcp(PI * sphere_surface_area(light.sphere_radius()));
            float light_radiance = light.sphere_power() * inv_divisor;
            radiance += specular_tint * light_radiance * toggle;
        } else {
            // Deprecated area light normalization term. Equation 10 and 14 in Real Shading in Unreal Engine 4, 2013.
            // float adjusted_ggx_alpha = saturate(ggx_alpha + local_sphere.radius / (3 * length(local_sphere_position)));
            // float area_light_normalization_term = pow2(ggx_alpha / adjusted_ggx_alpha);

            float sin_theta_squared = pow2(local_sphere.radius) / dot(local_sphere_position, local_sphere_position);
            float a2 = pow2(ggx_alpha);
            float area_light_normalization_term = a2 / (a2 + sin_theta_squared / (abs(wo.z) * 3.6 + 0.4));

            float3 halfway = normalize(wo + wi);
            radiance += specular_tint * BSDFs::GGX::evaluate(ggx_alpha, wo, wi, halfway) * abs(wi.z) * light_sample.radiance * area_light_normalization_term;
        }
    }

    return radiance;
}

struct PixelInput {
    float4 position : SV_POSITION;
    float4 world_position : WORLD_POSITION;
    float4 normal : NORMAL;
    float2 texcoord : TEXCOORD;
};

float3 integration(PixelInput input) {
    float3 normal = normalize(input.normal.xyz);

    // Apply IBL
    float3 wo = normalize(camera_position.xyz - input.world_position.xyz);
    float3 radiance = environment_tint.rgb * material.IBL(wo, normal, input.texcoord);

    float3x3 world_to_shading_TBN = create_TBN(normal);
    wo = mul(world_to_shading_TBN, wo);

    for (int l = 0; l < light_count.x; ++l) {
        LightData light = light_data[l];
        LightSample light_sample = sample_light(light_data[l], input.world_position.xyz);

        bool is_sphere_light = light.type() == LightType::Sphere && light.sphere_radius() > 0.0f;
        if (is_sphere_light) {
            // Apply SPTD area light approximation.
#if SPTD_AREA_LIGHTS
            float distance_to_camera = length(camera_position.xyz - input.world_position.xyz);
            radiance += SPTD::evaluate_sphere_light(light, material, input.texcoord, sptd_ggx_fit_tex,
                input.world_position.xyz, world_to_shading_TBN, wo, distance_to_camera);
#else
            radiance += evaluate_most_representative_point(light, material, input.texcoord, input.world_position.xyz, world_to_shading_TBN, wo);
#endif
        } else {
            // Apply regular delta lights.
            float3 wi = mul(world_to_shading_TBN, light_sample.direction_to_light);
            float3 f = material.evaluate(wo, wi, input.texcoord);
            radiance += f * light_sample.radiance * abs(wi.z);
        }
    }

    return radiance;
}

float4 opaque(PixelInput input) : SV_TARGET{
    // NOTE There may be a performance cost associated with having a potential discard, so we should probably have a separate pixel shader for cutouts.
    float coverage = material.coverage(input.texcoord);
    if (coverage < 0.33f)
        discard;

    return float4(integration(input), 1.0f);
}

float4 transparent(PixelInput input) : SV_TARGET{
    float coverage = material.coverage(input.texcoord);
    return float4(integration(input), coverage);
}