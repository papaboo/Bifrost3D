// Model fragment shaders.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "DefaultShading.hlsl"
#include "LightSources.hlsl"

cbuffer scene_variables : register(b0) {
    SceneVariables scene_vars;
};

cbuffer lights : register(b1) {
    int4 light_count;
    LightData light_data[12];
}

cbuffer material : register(b3) {
    MaterialParams material_params;
}

#if SPTD_AREA_LIGHTS
#include "SPTD.hlsl"
Texture2D sptd_ggx_fit_tex : register(t14);
#endif

struct PixelInput {
    float4 position : SV_POSITION;
    float4 world_position : WORLD_POSITION;
    float4 normal : NORMAL;
    float2 texcoord : TEXCOORD;
};

float3 integration(PixelInput input, bool is_front_face) {
    float3 world_wo = normalize(scene_vars.camera_position.xyz - input.world_position.xyz);
    float3 world_normal = normalize(input.normal.xyz) * (is_front_face ? 1.0 : -1.0);

    // Apply IBL
    float3x3 world_to_shading_TBN = create_TBN(world_normal);
    float3 wo = mul(world_to_shading_TBN, world_wo);

    const DefaultShading default_shading = DefaultShading::from_constants(material_params, wo, input.texcoord);
    float3 radiance = scene_vars.environment_tint.rgb * default_shading.evaluate_IBL(world_wo, world_normal);

    for (int l = 0; l < light_count.x; ++l) {
        LightData light = light_data[l];

        bool is_sphere_light = light.type() == LightType::Sphere && light.sphere_radius() > 0.0f;
        if (is_sphere_light) {
            // Apply SPTD area light approximation.
#if SPTD_AREA_LIGHTS
            float distance_to_camera = length(camera_position.xyz - input.world_position.xyz);
            radiance += SPTD::evaluate_sphere_light(light, default_shading, sptd_ggx_fit_tex,
                input.world_position.xyz, world_to_shading_TBN, wo, distance_to_camera);
#else
            radiance += default_shading.evaluate_area_light(light, input.world_position.xyz, wo, world_to_shading_TBN);
#endif
        } else {
            // Apply regular delta lights.
            LightSample light_sample = sample_light(light_data[l], input.world_position.xyz);
            float3 wi = mul(world_to_shading_TBN, light_sample.direction_to_light);
            float3 f = default_shading.evaluate(wo, wi);
            radiance += f * light_sample.radiance * abs(wi.z);
        }
    }

    return radiance;
}

float4 output_normals(PixelInput input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    float coverage = material_params.coverage(input.texcoord);
    if (coverage < 0.33f)
        discard;

    // TODO Move to screen space
    float3 world_normal = normalize(input.normal.xyz) * (is_front_face ? 1.0 : -1.0);

    return float4(world_normal * 0.5 + 0.5, 1.0f);
}

float4 opaque(PixelInput input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    // NOTE There may be a performance cost associated with having a potential discard, so we should probably have a separate pixel shader for cutouts.
    float coverage = material_params.coverage(input.texcoord);
    if (coverage < 0.33f)
        discard;

    return float4(integration(input, is_front_face), 1.0f);
}

float4 transparent(PixelInput input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    float coverage = material_params.coverage(input.texcoord);
    return float4(integration(input, is_front_face), coverage);
}