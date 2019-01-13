// Debug shaders.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Bifrost. See AUTHORS.txt for authors
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

float3 display_debug_ps(Varyings input) : SV_TARGET {
    if (mode == 1) { // Normals
        float2 encoded_normal = normal_tex[input.position.xy].xy;
        float3 normal = decode_ss_octahedral_normal(encoded_normal);
        return 0.5 * normal + 0.5;
    }
    else if (mode == 2) // Depth
        return depth_tex[input.position.xy].rrr;
    else if (mode == 3) { // Scene size visualized as 1x1x1 grid cells
        float g_width, g_height;
        depth_tex.GetDimensions(g_width, g_height);

        float z_over_w = depth_tex[input.position.xy].r;
        float2 viewport_uv = input.position.xy / float2(g_width, g_height);
        float3 position = position_from_depth(z_over_w, viewport_uv, scene_vars.inverted_view_projection_matrix);

        int3 trunc_position = floor(position);
        bool is_red = ((trunc_position.x & 1) == (trunc_position.y & 1)) ^ ((trunc_position.z & 1) == 1);
        float3 color = is_red ? float3(1, 0, 0) : float3(1, 1, 1);

        // Shade a bit by the normal
        float2 encoded_normal = normal_tex[input.position.xy].xy;
        float3 normal = decode_ss_octahedral_normal(encoded_normal);
        return color * (abs(normal.z * 0.75f) + 0.25f);
    }
    else
        return ao_tex[input.position.xy + scene_vars.g_buffer_to_ao_index_offset].rrr;
}
