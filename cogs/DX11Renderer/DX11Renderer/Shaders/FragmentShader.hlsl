// Simple fragment shader.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

cbuffer scene_variables  : register(b0) {
    matrix mvp_matrix;
    float4 offset;
    float4 color;
};

float4 main() : SV_TARGET{
    return color;
}