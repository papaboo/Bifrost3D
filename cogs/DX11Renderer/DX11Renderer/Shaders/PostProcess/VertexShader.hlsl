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

Output main(uint vertex_ID : SV_VertexID) {
    Output output;
    // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }
    output.position.x = vertex_ID == 2 ? 3 : -1;
    output.position.y = vertex_ID == 0 ? -3 : 1;
    output.position.zw = float2(0.99999, 1.0f);
    output.texcoord = output.position.xy * 0.5 + 0.5; // TODO Can I just get the screen position from some built-in instead?
    return output;
}