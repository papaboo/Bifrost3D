// Model G-buffer output.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <ShadingModels/Utils.hlsl>
#include <Utils.hlsl>

// ------------------------------------------------------------------------------------------------
// Constants.
// ------------------------------------------------------------------------------------------------

cbuffer transform : register(b2) {
    float4x3 to_world_matrix;
};

cbuffer material : register(b3) {
    ShadingModels::Parameters material_params;
}

cbuffer scene_variables : register(b13) {
    SceneVariables scene_vars;
};

// ------------------------------------------------------------------------------------------------
// Opaque model.
// ------------------------------------------------------------------------------------------------

struct OpaqueVaryings {
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
};

OpaqueVaryings opaque_VS(float4 geometry : GEOMETRY) {
    OpaqueVaryings varyings;
    float3 world_position = mul(float4(geometry.xyz, 1.0f), to_world_matrix).xyz;
    varyings.position = mul(float4(world_position, 1.0f), scene_vars.view_projection_matrix);
    float3 world_normal = mul(float4(decode_octahedral_normal(asint(geometry.w)), 0.0), to_world_matrix);
    varyings.normal = mul(float4(world_normal, 0), scene_vars.world_to_view_matrix);
    return varyings;
}

float2 opaque_PS(OpaqueVaryings varyings) : SV_Target {
    float3 view_space_normal = normalize(varyings.normal);
    return encode_ss_octahedral_normal(view_space_normal);
}

// ------------------------------------------------------------------------------------------------
// Thin-walled model.
// ------------------------------------------------------------------------------------------------

struct ThinWalledVaryings {
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

ThinWalledVaryings thin_walled_VS(float4 geometry : GEOMETRY, float2 uv : TEXCOORD) {
    ThinWalledVaryings varyings;
    float3 world_position = mul(float4(geometry.xyz, 1.0f), to_world_matrix).xyz;
    varyings.position = mul(float4(world_position, 1.0f), scene_vars.view_projection_matrix);
    float3 world_normal = mul(float4(decode_octahedral_normal(asint(geometry.w)), 0.0), to_world_matrix);
    varyings.normal = mul(float4(world_normal, 0), scene_vars.world_to_view_matrix);
    varyings.uv = uv;
    return varyings;
}

float2 thin_walled_PS(ThinWalledVaryings varyings) : SV_Target {
    if (material_params.discard_from_cutout(varyings.uv))
        discard;

    float3 view_space_normal = normalize(varyings.normal);
    return encode_ss_octahedral_normal(view_space_normal);
}