// Simple fragment shader.
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

struct PixelInput {
    float4 position : SV_POSITION;
    float4 normal : NORMAL;
};

float4 main(PixelInput input) : SV_TARGET{
    return color * input.normal;
}