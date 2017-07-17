// Model vertex shader.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// Utility functions.
// ------------------------------------------------------------------------------------------------
float sign(float v) { return v >= 0.0f ? +1.0f : -1.0f; }

float3 decode_octahedral_normal(unsigned int encoded_normal) {
    const int SHRT_MAX = 32767;
    const int SHRT_MIN = -32768;

    int encoding_x = int(encoded_normal & 0xFFFF) + SHRT_MIN;
    int encoding_y = int((encoded_normal >> 16) & 0xFFFF) + SHRT_MIN;

    float2 p2 = float2(encoding_x, encoding_y);
    float3 n = float3(encoding_x, encoding_y, SHRT_MAX - abs(p2.x) - abs(p2.y));
    if (n.z < 0.0f) {
        float tmp_x = (SHRT_MAX - abs(n.y)) * sign(n.x);
        n.y = (SHRT_MAX - abs(n.x)) * sign(n.y);
        n.x = tmp_x;
    }
    return n;
}

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

cbuffer scene_variables : register(b0) {
    float4x4 view_projection_matrix;
    float4 camera_position;
    float4 environment_tint; // .w component is 1 if an environment tex is bound, otherwise 0.
};

cbuffer transform : register(b2) {
    float4x3 to_world_matrix;
};

struct Output {
    float4 position : SV_POSITION;
    float4 world_position : WORLD_POSITION;
    float4 normal : NORMAL;
    float2 texcoord : TEXCOORD;
};

Output main(float4 geometry : GEOMETRY, float2 texcoord : TEXCOORD) {
    Output output;
    output.world_position.xyz = mul(float4(geometry.xyz, 1.0f), to_world_matrix);
    output.position = mul(float4(output.world_position.xyz, 1.0f), view_projection_matrix);
    output.normal.xyz = mul(decode_octahedral_normal(asuint(geometry.w)), to_world_matrix);
    output.texcoord = texcoord;
    return output;
}