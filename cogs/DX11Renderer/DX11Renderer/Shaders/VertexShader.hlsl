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

float4 main(float3 in_pos : POSITION) : SV_POSITION {
    return mul(float4(in_pos, 1.0f), mvp_matrix);
}