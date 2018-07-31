// Debug shaders.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Utils.hlsl"

cbuffer scene_variables : register(b13) {
    SceneVariables scene_vars;
};

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

struct Varyings {
    float4 position : SV_POSITION;
};

Varyings main_vs(uint vertex_ID : SV_VertexID) {
    Varyings output;
    // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }
    output.position.x = vertex_ID == 2 ? 3 : -1;
    output.position.y = vertex_ID == 0 ? -3 : 1;
    output.position.zw = float2(0.0f, 1.0f);
    return output;
}

// ------------------------------------------------------------------------------------------------
// Display debug shader.
// ------------------------------------------------------------------------------------------------

Texture2D normal_tex : register(t0);
Texture2D depth_tex : register(t1);
Texture2D ao_tex : register(t2);

cbuffer debug_constants : register(b1) {
    int mode;
    int3 _padding;
};

float4 display_debug_ps(Varyings input) : SV_TARGET {
    float3 color;
    if (mode == 1) { // Normals
        float2 encoded_normal = normal_tex[input.position.xy].xy;
        float3 normal = decode_ss_octahedral_normal(encoded_normal);
        color = 0.5 * normal + 0.5;
    }
    else if (mode == 2) // Depth
        color = depth_tex[input.position.xy].rrr;
    else
        color = ao_tex[input.position.xy + scene_vars.g_buffer_to_ao_index_offset].rrr;
    return float4(color, 1);
}
