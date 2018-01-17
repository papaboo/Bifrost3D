// Spherical pivot transform distribution utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_SPTD_H_
#define _DX11_RENDERER_SHADERS_SPTD_H_

#include "Utils.hlsl"

namespace SPTD {

// ------------------------------------------------------------------------------------------------
// SPTD functions
// ------------------------------------------------------------------------------------------------
float2 pivot_transform(float2 r, float pivot) {
    float2 tmp1 = float2(r.x - pivot, r.y);
    float2 tmp2 = pivot * r - float2(1, 0);
    float x = dot(tmp1, tmp2);
    float y = tmp1.y * tmp2.x - tmp1.x * tmp2.y;
    float qf = dot(tmp2, tmp2);

    return float2(x, y) / qf;
}

// Equation 2 in SPDT, Dupuy et al. 17.
float3 pivot_transform(float3 r, float3 pivot) {
    float3 numerator = (dot(r, pivot) - 1.0f) * (r - pivot) - cross(r - pivot, cross(r, pivot));
    float denominator = pow2(dot(r, pivot) - 1.0f) + length_squared(cross(r, pivot));
    return numerator / denominator;
}

Cone pivot_transform(Cone cone, float3 pivot) {
    // Extract pivot length and direction.
    float pivot_mag = length(pivot);
    if (pivot_mag < 0.001f)
        // special case: the pivot is at the origin.
        return Cone::make(-cone.direction, cone.cos_theta);
    float3 pivot_dir = pivot / pivot_mag;

    // 2D cap direction.
    float cos_phi = dot(cone.direction, pivot_dir);
    float sin_phi = sqrt(1.0f - cos_phi * cos_phi);

    // 2D basis = (pivotDir, PivotOrthogonalDirection)
    float3 pivot_ortho_dir;
    if (abs(cos_phi) < 0.9999f)
        pivot_ortho_dir = (cone.direction - cos_phi * pivot_dir) / sin_phi;
    else
        pivot_ortho_dir = float3(0, 0, 0);

    // Compute cap 2D end points.
    float sin_theta_sqrd = sqrt(1.0f - cone.cos_theta * cone.cos_theta);
    float a1 = cos_phi * cone.cos_theta;
    float a2 = sin_phi * sin_theta_sqrd;
    float a3 = sin_phi * cone.cos_theta;
    float a4 = cos_phi * sin_theta_sqrd;
    float2 dir1 = float2(a1 + a2, a3 - a4);
    float2 dir2 = float2(a1 - a2, a3 + a4);

    // Project in 2D.
    float2 dir1_xf = pivot_transform(dir1, pivot_mag);
    float2 dir2_xf = pivot_transform(dir2, pivot_mag);

    // Compute the cap 2D direction.
    float area = dir1_xf.x * dir2_xf.y - dir1_xf.y * dir2_xf.x;
    float s = area > 0.0f ? 1.0f : -1.0f;
    float2 dir_xf = s * normalize(dir1_xf + dir2_xf);

    return Cone::make(dir_xf.x * pivot_dir + dir_xf.y * pivot_ortho_dir,
                        dot(dir_xf, dir1_xf));
}

// ------------------------------------------------------------------------------------------------
// Shading
// ------------------------------------------------------------------------------------------------

#include <DefaultShading.hlsl>

struct BRDFPivotTransform {
    float3 pivot;
    float brdf_scale;
};

BRDFPivotTransform sptd_BRDF_pivot(Texture2D sptd_BRDF_fit_tex, float roughness, float3 wo) {
    float2 sptd_ggx_fit_uv = float2(abs(wo.z), roughness);
    float4 pivot_params = sptd_BRDF_fit_tex.Sample(precomputation2D_sampler, sptd_ggx_fit_uv);

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

float3 evaluate_sphere_light(LightData light, DefaultShading material, Texture2D sptd_ggx_fit_tex, 
                             float3 world_position, float3x3 world_to_shading_TBN, float3 wo, float distance_to_camera) {
    // Sphere light in local space
    float3 sphere_position = mul(world_to_shading_TBN, light.sphere_position() - world_position);
    Sphere local_sphere = Sphere::make(sphere_position, light.sphere_radius());
    Cone light_sphere_cap = sphere_to_sphere_cap(local_sphere.position, local_sphere.radius);

    Cone hemisphere_sphere_cap = Cone::make(float3(0.0f, 0.0f, 1.0f), 0.0f);
    float3 centroid_of_cones = centroid_of_intersection(hemisphere_sphere_cap, light_sphere_cap);

    float3 diffuse_tint, specular_tint;
    material.evaluate_tints(wo, centroid_of_cones, diffuse_tint, specular_tint);

    float3 radiance = float3(0,0,0);
    { // Evaluate Lambert.
        float solidangle_of_light = solidangle(light_sphere_cap);
        float visible_solidangle_of_light = solidangle_of_intersection(hemisphere_sphere_cap, light_sphere_cap);
        float light_radiance_scale = visible_solidangle_of_light / solidangle_of_light;
        LightSample light_sample = sample_light(light, world_position);
        radiance += diffuse_tint * BSDFs::Lambert::evaluate() * abs(centroid_of_cones.z) * light_sample.radiance * light_radiance_scale;
    }

    { // Evaluate GGX/microfacet.
        // Stretch highlight based on roughness and cos_theta.
        float3 direction_to_camera = wo * distance_to_camera;
        float3 direction_to_light = centroid_of_cones * length(sphere_position); // light_sphere_cap.direction * length(sphere_position);
        float3 halfway = normalize(wo + centroid_of_cones);
        float elongation = 1.0; +4.0 * material.m_roughness * (1.0f - dot(wo, halfway));
        float3 intersection_offset = elongated_highlight_offset(direction_to_camera, direction_to_light, elongation);
        light_sphere_cap.direction = normalize(direction_to_light - intersection_offset); // Here there be side-effects outside of the scope.
        float3 adjusted_wo = normalize(direction_to_camera - intersection_offset);

        // NOTE If performance is a concern then the SPTD cap for the hemisphere could be precomputed and stored along with the pivot.
        BRDFPivotTransform adjusted_ggx_sptd = sptd_BRDF_pivot(sptd_ggx_fit_tex, material.m_roughness, adjusted_wo);
        Cone adjusted_ggx_sptd_cap = SPTD::pivot_transform(Cone::make(float3(0.0f, 0.0f, 1.0f), 0.0f), adjusted_ggx_sptd.pivot);

        Cone ggx_light_sphere_cap = SPTD::pivot_transform(light_sphere_cap, adjusted_ggx_sptd.pivot);
        float light_solidangle = solidangle_of_intersection(ggx_light_sphere_cap, adjusted_ggx_sptd_cap);
        float3 l = light.sphere_power() / (PI * sphere_surface_area(light.sphere_radius()));
        radiance += specular_tint * light_solidangle / (4.0f * PI) * l;
    }

    return radiance;
}

} // NS SPTD

#endif // _DX11_RENDERER_SHADERS_SPTD_H_