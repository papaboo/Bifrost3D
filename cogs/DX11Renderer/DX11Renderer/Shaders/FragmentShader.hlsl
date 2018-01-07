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

Texture2D sptd_ggx_fit_tex : register(t14);

struct BRDFPivotTransform {
    float3 pivot;
    float brdf_scale;
};

BRDFPivotTransform sptd_ggx_pivot(float roughness, float3 wo) {
    float2 sptd_ggx_fit_uv = float2(abs(wo.z), roughness);
    float4 pivot_params = sptd_ggx_fit_tex.Sample(precomputation2D_sampler, sptd_ggx_fit_uv);

    BRDFPivotTransform res;
    res.brdf_scale = pivot_params.w;
    float pivot_norm = pivot_params.x;
    float pivot_cos_theta = pivot_params.y;
    float pivot_sin_theta = -sqrt(1.0f - pivot_cos_theta * pivot_cos_theta);
    float3 pivot = pivot_norm * float3(pivot_sin_theta, 0, pivot_cos_theta);

    // Convert the pivot from local / wo space to tangent space. TODO Inline. Row[1] seems to have no effect
    float3x3 basis;
    basis[0] = wo.z < 0.999f ? normalize(wo - float3(0, 0, wo.z)) : float3(1, 0, 0);
    basis[1] = cross(float3(0, 0, 1), basis[0]); // Has no effect. Looks like the whole thing can be inlined to use basis[0] and basis[2].z
    basis[2] = float3(0, 0, 1);
    res.pivot = mul(pivot, basis);
    return res;
}

float3 elongated_highlight_offset(float3 direction_to_camera, float3 direction_to_light, float elongation) {
    float2 camera_to_light = direction_to_light.xy - direction_to_camera.xy;
    float t = abs(direction_to_camera.z) / (abs(direction_to_light.z) + abs(direction_to_camera.z)); // NOTE Only needed as long as the direction to light and camera can be on opposite sides.
    float2 perfect_reflection_point_2D = direction_to_camera.xy + camera_to_light * t;

    float2 bitangent = normalize(camera_to_light);
    float2 tangent = float2(-bitangent.y, bitangent.x);

    float2 delta_x = tangent * dot(perfect_reflection_point_2D, tangent);
    float2 delta_y = bitangent * dot(perfect_reflection_point_2D, bitangent);
    float2 warped_reflection_point = perfect_reflection_point_2D - delta_x * elongation - delta_y / elongation;
    return float3(warped_reflection_point, 0.0);
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

    for (int l = 0; l < light_count.x; ++l) {
        LightData light = light_data[l];
        LightSample light_sample = sample_light(light_data[l], input.world_position.xyz);

        bool is_sphere_light = light.type() == LightType::Sphere && light.sphere_radius() > 0.0f;
        if (is_sphere_light) {
            // Sphere light in local space
            float3 sphere_position = mul(world_to_shading_TBN, light.sphere_position() - input.world_position.xyz);
            Sphere local_sphere = Sphere::make(sphere_position, light.sphere_radius());
            Cone light_sphere_cap = sphere_to_sphere_cap(local_sphere.position, local_sphere.radius);

            Cone hemisphere_sphere_cap = Cone::make(float3(0.0f, 0.0f, 1.0f), 0.0f);
            float3 centroid_of_cones = centroid_of_intersection(hemisphere_sphere_cap, light_sphere_cap);

            float3 diffuse_tint, specular_tint;
            material.evaluate_tints(wo, centroid_of_cones, input.texcoord, diffuse_tint, specular_tint);

            { // Evaluate Lambert.
                float solidangle_of_light = solidangle(light_sphere_cap);
                float visible_solidangle_of_light = solidangle_of_intersection(hemisphere_sphere_cap, light_sphere_cap);
                float light_radiance_scale = visible_solidangle_of_light / solidangle_of_light;
                radiance += diffuse_tint * BSDFs::Lambert::evaluate() * abs(centroid_of_cones.z) * light_sample.radiance * light_radiance_scale;
            }

            { // Evaluate GGX/microfacet.
                // Stretch highlight based on roughness and cos_theta.
                float3 direction_to_camera = wo * length(camera_position.xyz - input.world_position.xyz);
                float3 direction_to_light = centroid_of_cones * length(sphere_position); // light_sphere_cap.direction * length(sphere_position);
                float3 halfway = normalize(wo + centroid_of_cones);
                float elongation = 1.0; + 4.0 * material.roughness() * (1.0f - dot(wo, halfway));
                float3 intersection_offset = elongated_highlight_offset(direction_to_camera, direction_to_light, elongation);
                light_sphere_cap.direction = normalize(direction_to_light - intersection_offset); // Here there be side-effects outside of the scope.
                float3 adjusted_wo = normalize(direction_to_camera - intersection_offset);

                // NOTE If performance is a concern then the SPTD cap for the hemisphere could be precomputed and stored along with the pivot.
                BRDFPivotTransform adjusted_ggx_sptd = sptd_ggx_pivot(material.roughness(), adjusted_wo);
                Cone adjusted_ggx_sptd_cap = SPTD::pivot_transform(Cone::make(float3(0.0f, 0.0f, 1.0f), 0.0f), adjusted_ggx_sptd.pivot);

                Cone ggx_light_sphere_cap = SPTD::pivot_transform(light_sphere_cap, adjusted_ggx_sptd.pivot);
                float light_solidangle = solidangle_of_intersection(ggx_light_sphere_cap, adjusted_ggx_sptd_cap);
                float3 l = light.sphere_power() / (PI * sphere_surface_area(light.sphere_radius()));
                radiance += specular_tint * light_solidangle / (4.0f * PI) * l;
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