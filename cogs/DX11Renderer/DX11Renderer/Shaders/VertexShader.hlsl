// Simple vertex shader.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

cbuffer scene_variables  : register(b0) {
    float4 offset;
    float4 color;
};

float4 main(float3 pos : POSITION) : SV_POSITION {
    return float4(pos + offset.xyz, 1.0f);
}