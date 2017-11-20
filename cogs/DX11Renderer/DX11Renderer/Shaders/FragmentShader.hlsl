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

cbuffer scene_variables : register(b0) {
    float4x4 view_projection_matrix;
    float4 camera_position;
    float4 environment_tint; // .w component is 0 if an environment tex is not bound, otherwise positive.
};

cbuffer lights : register(b1) {
    int4 light_count;
    LightData light_data[12];
}

cbuffer material : register(b3) {
    DefaultShading material;
}

// TODO Potentially move to default shading and use the other precomputation sampler.
Texture2D sptd_ggx_fit_tex : register(t14);
SamplerState sptd_ggx_fit_sampler : register(s14);

struct BRDFPivotTransform {
    float3 pivot;
    float brdf_scale;
};

BRDFPivotTransform sptd_ggx_pivot(float roughness, float3 wo) {
    float2 sptd_ggx_fit_uv = float2(roughness, 2.0f * acos(wo.z) / PI);
    float4 pivot_params = sptd_ggx_fit_tex.Sample(sptd_ggx_fit_sampler, sptd_ggx_fit_uv);

    BRDFPivotTransform res;
    res.brdf_scale = pivot_params.w;
    float pivot_norm = pivot_params.x;
    float pivot_theta = pivot_params.y;
    float3 pivot = pivot_norm * float3(sin(pivot_theta), 0, cos(pivot_theta));

    // Convert the pivot from local / wo space to tangent space. TODO Use TBN or perhaps inline. Isn't there some faster way to move the vector than create the matrix explicitly?
    float3x3 basis;
    basis[0] = wo.z < 0.999f ? normalize(wo - float3(0, 0, wo.z)) : float3(1, 0, 0);
    basis[1] = cross(float3(0, 0, 1), basis[0]);
    basis[2] = float3(0, 0, 1);
    res.pivot = mul(pivot, basis);
    return res;
}

BRDFPivotTransform sptd_lambert_pivot(float3 wo) {
    BRDFPivotTransform res;
    res.brdf_scale = 1.0f;
    float pivot_norm = 0.369589f;
    float pivot_theta = 0.0f;
    float3 pivot = pivot_norm * float3(sin(pivot_theta), 0, cos(pivot_theta));

    // Convert the pivot from local / wo space to tangent space. TODO Use TBN or perhaps inline. Isn't there some faster way to move the vector than create the matrix explicitly?
    float3x3 basis;
    basis[0] = wo.z < 0.999f ? normalize(wo - float3(0, 0, wo.z)) : float3(1, 0, 0);
    basis[1] = cross(float3(0, 0, 1), basis[0]);
    basis[2] = float3(0, 0, 1);
    res.pivot = mul(pivot, basis);
    return res;
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
    float3 wo = camera_position.xyz - input.world_position.xyz;
    float3 wi = -reflect(normalize(wo), normal);
    float3 radiance = environment_tint.rgb * material.IBL(normal, wi, input.texcoord);

    float3x3 world_to_shading_TBN = create_TBN(normal);
    wo = normalize(mul(world_to_shading_TBN, wo));

    // Compute GGX SPTD params
    BRDFPivotTransform sptd_specular = sptd_ggx_pivot(material.roughness(), wo);
    BRDFPivotTransform sptd_diffuse = sptd_lambert_pivot(wo);

    for (int l = 0; l < light_count.x; ++l) {
        LightData light = light_data[l];
        bool is_sphere_light = light.type() == LightType::Sphere && light.sphere_radius() > 0.0f;
        if (is_sphere_light) {
            // Sphere light in local space
            float3 sphere_position = mul(world_to_shading_TBN, light.sphere_position() - input.world_position.xyz);
            Sphere local_sphere = Sphere::make(sphere_position, light.sphere_radius());

            float3 l = light.sphere_power() / (PI * sphere_surface_area(light.sphere_radius()));

            float3 wi = normalize(local_sphere.position);
            float3 diffuse_tint, specular_tint;
            material.evaluate_tints(wo, wi, input.texcoord, diffuse_tint, specular_tint);

            // Evaluate BRDF. TODO Use brdf_scale?
            float specular_f = specular_tint * SPTD::evaluate_sphere_light(sptd_specular.pivot, local_sphere) / (4.0f * PI);
            float diffuse_f = diffuse_tint * SPTD::evaluate_sphere_light(sptd_diffuse.pivot, local_sphere) / (4.0f * PI);

            radiance += (diffuse_f + specular_f) * l;

        } else {
            // Apply regular delta lights
            LightSample light_sample = sample_light(light_data[l], input.world_position.xyz);
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