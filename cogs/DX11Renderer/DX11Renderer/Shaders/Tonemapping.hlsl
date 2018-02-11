// Tonemapping shaders.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Utils.hlsl"

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

struct Varyings {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

Varyings fullscreen_vs(uint vertex_ID : SV_VertexID) {
    Varyings output;
    // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }
    output.position.x = vertex_ID == 2 ? 3 : -1;
    output.position.y = vertex_ID == 0 ? -3 : 1;
    output.position.zw = float2(1.0, 1.0);
    output.texcoord = output.position.xy * 0.5 + 0.5;
    output.texcoord.y = 1.0f - output.texcoord.y;
    return output;
}

// ------------------------------------------------------------------------------------------------
// Shader for extracting log luminance.
// ------------------------------------------------------------------------------------------------

Texture2D pixels : register(t0);

float4 log_luminance_ps(Varyings input) : SV_TARGET {
    float3 pixel = pixels[int2(input.position.xy)].rgb;
    float log_luminance = log(luma(pixel));
    return float4(max(log_luminance, 0.0001f), 1, 1, 1);
}

// ------------------------------------------------------------------------------------------------
// Tonemapping utilities.
// ------------------------------------------------------------------------------------------------

Texture2D log_luminance_tex : register(t1);
SamplerState log_luminance_sampler : register(s1);

float get_average_luminance(float2 texcoord) {
    // TODO the log average texture should be sampled at the level where the smallest dimension has size 2, 
    // TODO Interpolate the average luminance across a couple of pixels and time.
    float width, height, mip_count;
    log_luminance_tex.GetDimensions(0, width, height, mip_count);

    float log_luminance = log_luminance_tex.SampleLevel(log_luminance_sampler, texcoord, mip_count - 3).r;
    return exp(log_luminance);
}

// Determines the color based on local exposure
float dynamic_exposure(float average_luminance, float threshold) {
    // Use geometric mean        
    average_luminance = max(average_luminance, 0.001f);

    float key_value = 1.03f - (2.0f / (2 + log10(average_luminance + 1)));

    float linear_exposure = (key_value / average_luminance);
    float log_exposure = log2(max(linear_exposure, 0.0001f));

    return exp2(log_exposure - threshold);
}

// ------------------------------------------------------------------------------------------------
// Tonemappers.
// ------------------------------------------------------------------------------------------------

// Advanced tonemapping operator
// http://perso.univ-lyon1.fr/jean-claude.iehl/Public/educ/GAMA/2007/gdc07/Post-Processing_Pipeline.pdf
float3 tonemap_reinhard(float3 color, float middlegrey, float average_luminance, float white_level_sqrd) {
    float luminance = luma(color);
    float scaled_luminance = luminance * middlegrey / average_luminance;
    float tonemapped_luminance = scaled_luminance * (1.0f + scaled_luminance / white_level_sqrd) / (1.0f + scaled_luminance);
    return color * (tonemapped_luminance / luminance);
}

// The filmic curve ALU only tonemapper from John Hable's presentation.
float3 tonemap_filmic_ALU(float3 color) {
    color = max(0, color - 0.004f);
    color = (color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f);

    // result has 1/2.2 baked in
    return pow(color, 2.2f);
}

// The filmic curve ALU only tonemapper from John Hable's presentation.
float3 tonemap_luminance_filmic_ALU(float3 color) {
    float luminance = luma(color);
    luminance = max(0, luminance - 0.004f);
    luminance = (luminance * (6.2f * luminance + 0.5f)) / (luminance * (6.2f * luminance + 1.7f) + 0.06f);
    float tonemapped_luminance = pow(luminance, 2.2f); // result has 1/2.2 baked in.
    return color * (tonemapped_luminance / luminance);
}

// ------------------------------------------------------------------------------------------------
// Tonemapping pixel shaders.
// ------------------------------------------------------------------------------------------------

float4 linear_tonemapping_ps(Varyings input) : SV_TARGET {
    return pixels[int2(input.position.xy)];
}

float4 reinhard_tonemapping_ps(Varyings input) : SV_TARGET {
    float average_luminance = get_average_luminance(input.texcoord);
    float3 color = pixels[int2(input.position.xy)].rgb;
    color = tonemap_reinhard(color, 1.0f, average_luminance, 9.0);
    return float4(color, 1);
}

float4 filmic_tonemapping_ps(Varyings input) : SV_TARGET {
    float average_luminance = get_average_luminance(input.texcoord);

    float exposure = dynamic_exposure(average_luminance, 0.0);

    float3 color = exposure * pixels[int2(input.position.xy)].rgb;
    return float4(tonemap_luminance_filmic_ALU(color), 1.0);
}
