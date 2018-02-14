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
    float log_luminance = log(max(luma(pixel), 0.0001f));
    return float4(log_luminance, 1, 1, 1);
}

// ------------------------------------------------------------------------------------------------
// Tonemapping utilities.
// ------------------------------------------------------------------------------------------------

Texture2D log_luminance_tex : register(t1);
SamplerState log_luminance_sampler : register(s1);

float get_average_luminance(float2 texcoord) {
    float width, height, mip_count;
    log_luminance_tex.GetDimensions(0, width, height, mip_count);

    float log_luminance = log_luminance_tex.SampleLevel(log_luminance_sampler, texcoord, mip_count).r;
    return exp(log_luminance);
}

float dynamic_linear_exposure(float average_luminance) {
    float key_value = 0.5f; // De-gammaed 0.18, see Reinhard et al., 2002
    return key_value / average_luminance;
}

// Compute linear exposure from the geometric mean. See MJP's tonemapping sample.
// https://mynameismjp.wordpress.com/2010/04/30/a-closer-look-at-tone-mapping/
float geometric_mean_linear_exposure(float average_luminance) {
    float key_value = 1.03f - (2.0f / (2 + log2(average_luminance + 1)));
    return key_value / average_luminance;
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

float3 uncharted2_tonemap_helper(float3 color, float shoulder_strength, float linear_strength, float linear_angle, float toe_strength, float toe_numerator, float toe_denominator) {
    float3 x = color;
    float A = shoulder_strength;
    float B = linear_strength;
    float C = linear_angle;
    float D = toe_strength;
    float E = toe_numerator;
    float F = toe_denominator;
    return ((x*(x*A + C*B) + D*E) / (x*(x*A + B) + D*F)) - E / F;
};

// Uncharted 2's filmic operator.
float3 uncharted2(float3 color, float shoulder_strength, float linear_strength, float linear_angle, float toe_strength, float toe_numerator, float toe_denominator, float linear_white) {
    return uncharted2_tonemap_helper(color, shoulder_strength, linear_strength, linear_angle, toe_strength, toe_numerator, toe_denominator) /
        uncharted2_tonemap_helper(float3(linear_white, linear_white, linear_white), shoulder_strength, linear_strength, linear_angle, toe_strength, toe_numerator, toe_denominator);
}


// ------------------------------------------------------------------------------------------------
// Tonemapping pixel shaders.
// ------------------------------------------------------------------------------------------------

float4 linear_tonemapping_ps(Varyings input) : SV_TARGET {
    return pixels[int2(input.position.xy)];
}

float4 reinhard_tonemapping_ps(Varyings input) : SV_TARGET {
    // TODO Apply exposure here instead of as part of the reinhard operator.
    float average_luminance = get_average_luminance(input.texcoord);
    float3 color = pixels[int2(input.position.xy)].rgb;
    color = tonemap_reinhard(color, 0.5f, average_luminance, 9.0);
    return float4(color, 1);
}

float4 filmic_tonemapping_ps(Varyings input) : SV_TARGET {
    // Exposure
    float average_luminance = get_average_luminance(input.texcoord);
    float exposure = dynamic_linear_exposure(average_luminance);
    float3 color = exposure * pixels[int2(input.position.xy)].rgb;

    // Tonemapping.
    float shoulder_strength = 0.22f;
    float linear_strength = 0.3f;
    float linear_angle = 0.1f;
    float toe_strength = 0.2f;
    float toe_numerator = 0.01f;
    float toe_denominator = 0.3f;
    float linear_white = 11.2f; 
    float3 tonemapped_color = uncharted2(color, shoulder_strength, linear_strength, linear_angle, toe_strength, toe_numerator, toe_denominator, linear_white);

    return float4(tonemapped_color, 1.0);
}
