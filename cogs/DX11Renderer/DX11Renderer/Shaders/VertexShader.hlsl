// Simple vertex shader.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

cbuffer scene_variables  : register(b0) {
    float4x4 mvp_matrix;
    float4x3 to_world_matrix;
    float4 camera_position;
};

struct Output {
    float4 position : SV_POSITION;
    float4 world_position : WORLD_POSITION;
    float4 normal : NORMAL;
    float2 texcoord : TEXCOORD;
};

Output main(float3 position : POSITION, float3 normal : NORMAL, float2 texcoord : TEXCOORD) {
    Output output;
    output.position = mul(float4(position, 1.0f), mvp_matrix);
    output.world_position.xyz = mul(float4(position, 1.0f), to_world_matrix);
    output.normal.xyz = mul(normal, to_world_matrix);
    output.texcoord = texcoord;
    return output;
}