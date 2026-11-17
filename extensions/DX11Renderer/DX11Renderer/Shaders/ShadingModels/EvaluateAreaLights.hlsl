// Area light evaluation
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_SHADING_MODELS_AREA_LIGHT_EVALUATION_H_
#define _DX11_RENDERER_SHADERS_SHADING_MODELS_AREA_LIGHT_EVALUATION_H_

#include <BSDFs/Diffuse.hlsl>
#include <BSDFs/GGX.hlsl>
#include <ShadingModels/IShadingModel.hlsl>
#include <LightSources.hlsl>
#include <LtcAreaLight.hlsl>
#include <Utils.hlsl>

namespace ShadingModels {

//-----------------------------------------------------------------------------
// Sphere light BRDF approximations.
//-----------------------------------------------------------------------------

float3 evaluate_sphere_light_lambert(SphereLight light, float3 wo, float3 tint, float ambient_visibility) {
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
    float3 light_contribution = light.emitted_radiance() * centroid_and_solidangle.centroid_direction.z * light_radiance_scale * scaled_ambient_visibility;
    return lambert_f * light_contribution;
}

// GGX sphere light evaluation by most representative point, heavily inspired by Real Shading in Unreal Engine 4.
// For UE4 reference see the function AreaLightSpecular() in DeferredLightingCommon.usf. (15/1 -2018)
float3 evaluate_sphere_light_GGX(SphereLight light, float3 wo, float3 specularity, float ggx_alpha, float ambient_visibility) {
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
    ggx_alpha = max(0.002, ggx_alpha);

    float cos_theta_i = wi.z;
    float sin_theta_squared = pow2(light.radius) / dot(most_representative_point, most_representative_point);
    float a2 = pow2(ggx_alpha);
    float area_light_normalization_term = a2 / (a2 + sin_theta_squared / (cos_theta_i * 3.6 + 0.4));

    return BSDFs::GGX::evaluate(ggx_alpha, specularity, wo, wi) * cos_theta_i * light.emitted_radiance() * area_light_normalization_term * specular_ambient_visibility;
}

//-----------------------------------------------------------------------------
// Area light evaluation with linearly transformed cosine material approximations.
//-----------------------------------------------------------------------------

// Evaluates an area light and scales it by the BSDF.
float3 evaluate_area_light(LightData light, ShadingModels::BsdfLtcStack bsdf_ltcs, float3 world_wo, float3 world_position, float3 world_normal) {
#pragma warning (disable: 4000) // Disable warning about uninitialized variable. The compiler won't tell which variable it is and it's a false positive.
    if (light.is_spot_light() || light.is_sphere_light()) {
        // Adjust sphere light power to account for the two-sided disk having half the surface area of the sphere.
        float3 forward = light.is_spot_light() ? light.spot_direction() : normalize(light.sphere_position() - world_position);
        float3x3 light_coordinate_system = create_TBN(forward);

        float radius = light.is_spot_light() ? light.spot_radius() : light.sphere_radius();
        float3 disk_control_points[3] = {
            disk_control_points[0] = light.spot_position(),
            disk_control_points[1] = radius * light_coordinate_system[0], // light tangent
            disk_control_points[2] = radius * light_coordinate_system[1] }; // light bitangent

        float3 reflectance = float3(0, 0, 0);
#pragma warning (disable: 3557) // Disable warning about potential loop unrolling. For some materials the count will be a constant, for others it will be a variable.
        for (int bsdf_index = 0; bsdf_index < bsdf_ltcs.count; ++bsdf_index) {
            ShadingModels::BsdfLtc bsdf_ltc = bsdf_ltcs.bsdfs[bsdf_index];
            reflectance += bsdf_ltc.tint * LtcAreaLight::evaluate_disk_light(bsdf_ltc.shading_to_ltc, world_wo, world_position, world_normal, disk_control_points, true);
        }
#pragma warning (enable: 3557)
        float3 emitted_radiance = light.is_spot_light() ? light.spot_emitted_radiance() : light.sphere_emitted_radiance();
        return reflectance * emitted_radiance;
    }
#pragma warning (enable: 4000)

    // Red error radiance to indicate unsupported light source.
    return float3(1, 0, 0.5);
}

} // NS ShadingModels

#endif // _DX11_RENDERER_SHADERS_SHADING_MODELS_AREA_LIGHT_EVALUATION_H_