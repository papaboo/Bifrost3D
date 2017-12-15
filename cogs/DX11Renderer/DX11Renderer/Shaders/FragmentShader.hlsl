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
    // float2 sptd_ggx_fit_uv = float2(2.0f * acos(wo.z) / PI, roughness);
    float2 sptd_ggx_fit_uv = float2(abs(wo.z), roughness);
    float4 pivot_params = sptd_ggx_fit_tex.Sample(sptd_ggx_fit_sampler, sptd_ggx_fit_uv);

    BRDFPivotTransform res;
    res.brdf_scale = pivot_params.w;
    float pivot_norm = pivot_params.x;
    float pivot_cos_theta = pivot_params.y;
    float pivot_sin_theta = -sqrt(1.0f - pivot_cos_theta * pivot_cos_theta);
    float3 pivot = pivot_norm * float3(pivot_sin_theta, 0, pivot_cos_theta);

    // Convert the pivot from local / wo space to tangent space. TODO Isn't there some faster way to move the vector than create the matrix explicitly?
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
    BRDFPivotTransform ggx_sptd = sptd_ggx_pivot(material.roughness(), wo);
    Cone ggx_sptd_cap = SPTD::pivot_transform(Cone::make(float3(0.0f, 0.0f, 1.0f), 0.0f), ggx_sptd.pivot);

    for (int l = 0; l < light_count.x; ++l) {
        LightData light = light_data[l];
        LightSample light_sample = sample_light(light_data[l], input.world_position.xyz);

        bool is_sphere_light = light.type() == LightType::Sphere && light.sphere_radius() > 0.0f;
        if (is_sphere_light) {
            // Sphere light in local space
            float3 sphere_position = mul(world_to_shading_TBN, light.sphere_position() - input.world_position.xyz);
            Sphere local_sphere = Sphere::make(sphere_position, light.sphere_radius());
            Cone light_sphere_cap = SPTD::sphere_to_sphere_cap(local_sphere.position, local_sphere.radius);

            Cone hemisphere_sphere_cap = Cone::make(float3(0.0f, 0.0f, 1.0f), 0.0f);
            float3 centroid_of_union = SPTD::centroid_of_union(hemisphere_sphere_cap, light_sphere_cap);

            float3 diffuse_tint, specular_tint;
            material.evaluate_tints(wo, centroid_of_union, input.texcoord, diffuse_tint, specular_tint);

            { // Evaluate GGX/microfacet.
                { // Stretch highlight based on roughness and cos_theta.
                    // The stretching is performed by warping the lights sphere cap in local space along the view vector and its tangent.
                    float stretch_scale = 1.0 + 32.0 * material.roughness() * material.roughness() * (1.0f - sqrt(abs(wo.z)));
                    float3 perfect_reflection = float3(-wo.x, -wo.y, wo.z);
                    float2 delta_reflection = light_sphere_cap.direction.xy - perfect_reflection.xy;

                    // Project the delta reflection on to wo, which is itself projected down on the plane defined by the normal.
                    float2 wo_bitangent = wo.xy / length(wo.xy);
                    float2 wo_tangent = float2(-wo_bitangent.y, wo_bitangent.x);
                    float2 delta_x = wo_bitangent * dot(delta_reflection, wo_bitangent);
                    float2 delta_y = wo_tangent * dot(delta_reflection, wo_tangent);

                    // Warp the direction by compressing it horizontally and stretching it vertically.
                    light_sphere_cap.direction.xy = perfect_reflection + delta_x / stretch_scale + delta_y * stretch_scale;
                    light_sphere_cap.direction = normalize(light_sphere_cap.direction);
                }

                Cone ggx_light_sphere_cap = SPTD::pivot_transform(light_sphere_cap, ggx_sptd.pivot);
                float light_solidangle = SPTD::solidangle_of_union(ggx_light_sphere_cap, ggx_sptd_cap);
                float3 l = light.sphere_power() / (PI * sphere_surface_area(light.sphere_radius()));
                radiance += specular_tint * light_solidangle / (4.0f * PI) * l;
            }

            { // Evaluate Lambert. // TODO Optimize by perhaps approximating the union solidangle and centroid.
                // TODO Combine solidangle and centroid calculations.
                float solidangle_of_light = SPTD::solidangle(light_sphere_cap);
                float solidangle_of_union = SPTD::solidangle_of_union(hemisphere_sphere_cap, light_sphere_cap);
                float light_radiance_scale = solidangle_of_union / solidangle_of_light;
                radiance += diffuse_tint * BSDFs::Lambert::evaluate() * abs(centroid_of_union.z) * light_sample.radiance * light_radiance_scale;
            }

        } else {
            // Apply regular delta lights
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