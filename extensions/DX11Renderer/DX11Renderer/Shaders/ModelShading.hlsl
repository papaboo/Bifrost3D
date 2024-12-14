// Model shading.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "ShadingModels/DefaultShading.hlsl"
#include "ShadingModels/DiffuseShading.hlsl"
#include "LightSources.hlsl"
#include "Utils.hlsl"

#pragma warning (disable: 3556) // Disable warning about dividing by integers is slow. It's a constant!!

// ------------------------------------------------------------------------------------------------
// Input buffers.
// ------------------------------------------------------------------------------------------------

cbuffer transform : register(b2) {
    float4x3 to_world_matrix;
};

cbuffer material : register(b3) {
    ShadingModels::Parameters material_params;
}

cbuffer lights : register(b12) {
    int4 light_count;
    LightData light_data[12];
}

cbuffer scene_variables : register(b13) {
    SceneVariables scene_vars;
};

Texture2D ssao_tex : register(t13);

struct Varyings {
    float4 position : SV_POSITION;
    float3 world_position : WORLD_POSITION;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD;
};

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

Varyings vs(float4 geometry : GEOMETRY, float2 texcoord : TEXCOORD) {
    Varyings output;
    output.world_position.xyz = mul(float4(geometry.xyz, 1.0f), to_world_matrix);
    output.position = mul(float4(output.world_position.xyz, 1.0f), scene_vars.view_projection_matrix);
    output.normal.xyz = normalize(mul(float4(decode_octahedral_normal(asint(geometry.w)), 0.0), to_world_matrix));
    output.texcoord = texcoord;
    return output;
}

// ------------------------------------------------------------------------------------------------
// Light integration.
// ------------------------------------------------------------------------------------------------

interface IShadingModelCreator {
    ShadingModels::IShadingModel create(float2 texcoord, float abs_cos_theta_o);
};

float3 integrate(IShadingModelCreator shading_model_creator, Varyings input, bool is_front_face, float ambient_visibility) {
    float3 world_wo = normalize(scene_vars.camera_world_position() - input.world_position.xyz);

    float3 world_normal = normalize(input.normal.xyz) * (is_front_face ? 1.0 : -1.0);
    world_normal = fix_backfacing_shading_normal(world_wo, world_normal, 0.002f);
    float3x3 world_to_shading_TBN = create_TBN(world_normal);

    float3 wo = mul(world_to_shading_TBN, world_wo);

    const ShadingModels::IShadingModel shading_model = shading_model_creator.create(input.texcoord, wo.z);

    // Apply IBL
    float3 radiance = scene_vars.environment_tint.rgb * shading_model.evaluate_IBL(world_wo, world_normal, ambient_visibility);

    for (int l = 0; l < light_count.x; ++l) {
        LightData light = light_data[l];

        bool is_sphere_light = light.type() == LightType::Sphere && light.sphere_radius() > 0.0f;
        if (is_sphere_light) {
            // Compute sphere light position in shading space
            SphereLight sphere_light = light.sphere_light();
            sphere_light.position = mul(world_to_shading_TBN, sphere_light.position - input.world_position.xyz);

            radiance += shading_model.evaluate_sphere_light(wo, sphere_light, ambient_visibility);
        } else {
            // Apply regular delta lights.
            LightSample light_sample = sample_light(light, input.world_position.xyz);
            float3 wi = mul(world_to_shading_TBN, light_sample.direction_to_light);
            float3 f = shading_model.evaluate(wo, wi);
            radiance += f * light_sample.radiance * abs(wi.z);
        }
    }

    return radiance;
}

float4 opaque(IShadingModelCreator shading_model_creator, Varyings input, bool is_front_face){
    // NOTE There may be a performance cost associated with having a potential discard, so we should probably have a separate pixel shader for cutouts.
    if (material_params.discard_from_cutout(input.texcoord))
        discard;

    float ambient_visibility = ssao_tex[input.position.xy + scene_vars.g_buffer_to_ao_index_offset].r;
    return float4(integrate(shading_model_creator, input, is_front_face, ambient_visibility), 1.0f);
}

float4 transparent(IShadingModelCreator shading_model_creator, Varyings input, bool is_front_face) {
    float coverage = material_params.coverage(input.texcoord);
    return float4(integrate(shading_model_creator, input, is_front_face, 1), coverage);
}

// ------------------------------------------------------------------------------------------------
// Default shading entry points
// ------------------------------------------------------------------------------------------------

struct DefaultShadingCreator : IShadingModelCreator {
    ShadingModels::IShadingModel create(float2 texcoord, float abs_cos_theta_o) {
        float4 tint_roughness = material_params.tint_roughness(texcoord);
        float3 tint = tint_roughness.rgb;
        float roughness = tint_roughness.w;

        float metallic = material_params.metallic(texcoord);
        float specularity = material_params.m_specularity;
        float coat_scale = material_params.coat_scale();
        float coat_roughness = material_params.coat_roughness();

        return ShadingModels::DefaultShading::create(tint, roughness, specularity, metallic, coat_scale, coat_roughness, abs_cos_theta_o);
    }
};

float4 default_opaque(Varyings input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    DefaultShadingCreator shading_creator;
    return opaque(shading_creator, input, is_front_face);
}

float4 default_transparent(Varyings input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    DefaultShadingCreator shading_creator;
    return transparent(shading_creator, input, is_front_face);
}

// ------------------------------------------------------------------------------------------------
// Diffuse shading entry points
// ------------------------------------------------------------------------------------------------

struct DiffuseShadingCreator : IShadingModelCreator {
    ShadingModels::IShadingModel create(float2 texcoord, float abs_cos_theta_o) {
        float3 tint = material_params.tint_roughness(texcoord).rgb;
        return ShadingModels::DiffuseShading::create(tint);
    }
};

float4 diffuse_opaque(Varyings input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    DiffuseShadingCreator shading_creator;
    return opaque(shading_creator, input, is_front_face);
}

float4 diffuse_transparent(Varyings input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    DiffuseShadingCreator shading_creator;
    return transparent(shading_creator, input, is_front_face);
}

// ------------------------------------------------------------------------------------------------
// Material property visualization.
// ------------------------------------------------------------------------------------------------

static const uint visualize_tint = 5;
static const uint visualize_roughness = 6;
static const uint visualize_metallic = 7;
static const uint visualize_coat = 8;
static const uint visualize_coat_roughness = 9;
static const uint visualize_coverage = 10;
static const uint visualize_UV = 11;

cbuffer visualization_mode : register(b4) {
    uint visualization_mode;
}

bool in_range(float x, float lower, float upper) {
    return lower <= x && x <= upper;
}

//          (0.6, 0.7)
//            |\
// (0.2, 0.6) |  \
// +----------+    \
// |                \ (0.8, 0.5)
// |                /
// +----------+    /
// (0.2, 0.4) |  /
//            |/
//          (0.6, 0.3)
//
bool inside_arrow(float2 uv) {
    bool inside_box = 0.2 <= uv.x && uv.x <= 0.6 &&
                      0.4 <= uv.y && uv.y <= 0.6;
    bool inside_head = uv.x >= 0.6
        && uv.y >= uv.x - 0.3
                       && uv.y <= -uv.x + 1.3;
    return inside_box || inside_head;
}

float4 visualize_material_params(Varyings input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    float3 world_wo = normalize(scene_vars.camera_world_position() - input.world_position.xyz);
    float3 world_normal = normalize(input.normal.xyz) * (is_front_face ? 1.0 : -1.0);

    float3x3 world_to_shading_TBN = create_TBN(world_normal);
    float3 wo = mul(world_to_shading_TBN, world_wo);

    if (visualization_mode == visualize_coverage) {
        float coverage = material_params.coverage(input.texcoord);
        return float4(coverage, coverage, coverage, 1);
    }

    if (visualization_mode == visualize_metallic) {
        float metallic = material_params.metallic(input.texcoord);
        return float4(metallic, metallic, metallic, 1);
    }

    if (visualization_mode == visualize_roughness) {
        float roughness = material_params.tint_roughness(input.texcoord).w;
        return float4(roughness, roughness, roughness, 1);
    }

    if (visualization_mode == visualize_coat) {
        float coat = material_params.coat_scale();
        return float4(coat, coat, coat, 1);
    }

    if (visualization_mode == visualize_coat_roughness) {
        float coat_roughness = material_params.coat_roughness();
        return float4(coat_roughness, coat_roughness, coat_roughness, 1);
    }

    if (visualization_mode == visualize_UV) {
        // Show UV as arrays colored by the texcoord along the direction of the arrow.
        // Every 25% we render a line across the texture. Black in the center and dark grey at 25%.
        const int block_count = 24;
        float2 uv_blocks = input.texcoord * block_count;
        int2 uv_blocks_indices = int2(uv_blocks);
        bool show_u = uv_blocks_indices.x % 2 == uv_blocks_indices.y % 2;
        float3 uv_tint = float3(0.5, 0.5, 0.5);
        if (show_u)
            uv_tint = inside_arrow(uv_blocks - uv_blocks_indices) ? float3(input.texcoord.x, 0, 0) : uv_tint;
        else
            uv_tint = inside_arrow(uv_blocks.yx - uv_blocks_indices.yx) ? float3(0, input.texcoord.y, 0) : uv_tint;

        // Lines at 25% and 75%.
        if (in_range(input.texcoord.x, 0.2475, 0.2525) || in_range(input.texcoord.x, 0.7475, 0.7525) ||
            in_range(input.texcoord.y, 0.2475, 0.2525) || in_range(input.texcoord.y, 0.7475, 0.7525))
            uv_tint = float3(0.25, 0.25, 0.25);

        // Lines at 50%.
        if (in_range(input.texcoord.x, 0.4975, 0.5025) || in_range(input.texcoord.y, 0.4975, 0.5025))
            uv_tint = float3(0, 0, 0);

        return float4(uv_tint, 1);
    }

    float3 tint = material_params.tint_roughness(input.texcoord).rgb;
    return float4(tint, 1);
}