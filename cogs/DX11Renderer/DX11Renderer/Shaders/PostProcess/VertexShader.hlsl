// Post process vertex shader.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

struct Output {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

Output main(float2 position : POSITION) {
    Output output;
    output.position = float4(position, 0.99999, 1.0f); // Passthrough position.
    output.texcoord = position.xy * 0.5 + 0.5; // TODO Can I just get the screen position from some built-in instead?
    return output;
}