// Simple vertex shader.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

cbuffer scene_variables  : register(b0) {
    matrix mvp_matrix;
    float4 color;
};

struct Output {
    float4 position : SV_POSITION;
    float4 normal : NORMAL;
};

Output main(float3 position : POSITION, float3 normal : NORMAL) {
    Output output;
    output.position = mul(float4(position, 1.0f), mvp_matrix);
    output.normal = float4(normal * 0.5f + 0.5f, 1.0);
    return output;
}