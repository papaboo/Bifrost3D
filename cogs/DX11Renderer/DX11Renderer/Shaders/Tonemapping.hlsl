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

// Bradford chromatic adaptation transforms between ACES white point (D60) and sRGB white point (D65)
static const float3x3 D65_2_D60_CAT = {
    1.01303f,    0.00610531f, -0.014971f,
    0.00769823f, 0.998165f,   -0.00503203f,
    -0.00284131f, 0.00468516f,  0.924507f,
};

static const float3x3 sRGB_2_XYZ_MAT = {
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f,
};

static const float3x3 XYZ_2_AP1_MAT = {
    1.6410233797f, -0.3248032942f, -0.2364246952f,
    -0.6636628587f,  1.6153315917f,  0.0167563477f,
    0.0117218943f, -0.0082844420f,  0.9883948585f,
};

static const float3x3 AP1_2_XYZ_MAT = {
    0.6624541811f, 0.1340042065f, 0.1561876870f,
    0.2722287168f, 0.6740817658f, 0.0536895174f,
    -0.0055746495f, 0.0040607335f, 1.0103391003f,
};

static const float3 AP1_RGB2Y = {
    0.2722287168f, //AP1_2_XYZ_MAT[0][1],
    0.6740817658f, //AP1_2_XYZ_MAT[1][1],
    0.0536895174f, //AP1_2_XYZ_MAT[2][1]
};

inline float3 unreal4(float3 color, float slope = 0.91f, float toe = 0.53f, float shoulder = 0.23f, float black_clip = 0.0f, float white_clip = 0.035f) {

    static const float3x3 sRGB_to_AP1 = mul(XYZ_2_AP1_MAT, mul(D65_2_D60_CAT, sRGB_2_XYZ_MAT));

    // Use ACEScg primaries as working space
    float3 working_color = mul(sRGB_to_AP1, color);
    working_color = max(0.0, working_color);

    // Pre desaturate
    working_color = lerp(dot(working_color, AP1_RGB2Y), working_color, 0.96f);

    const float toe_scale = 1.0f + black_clip - toe;
    const float shoulder_scale = 1.0f + white_clip - shoulder;

    const float in_match = 0.18f, out_match = 0.18f;

    float toe_match;
    if (toe > 0.8f)
        // 0.18 will be on straight segment
        toe_match = (1.0f - toe - out_match) / slope + log10(in_match);
    else {
        // 0.18 will be on toe segment

        // Solve for toe_match such that input of InMatch gives output of OutMatch.
        const float bt = (out_match + black_clip) / toe_scale - 1.0f;
        toe_match = log10(in_match) - 0.5f * log((1.0f + bt) / (1.0f - bt)) * (toe_scale / slope);
    }

    float straight_match = (1.0f - toe) / slope - toe_match;
    float shoulder_match = shoulder / slope - straight_match;

    float3 log_color = log10(working_color);
    float3 straight_color = (log_color + straight_match) * slope;

    float3 toe_color = (-black_clip) + (2 * toe_scale) / (1 + exp((-2 * slope / toe_scale) * (log_color - toe_match)));
    float3 shoulder_color = (1 + white_clip) - (2 * shoulder_scale) / (1 + exp((2 * slope / shoulder_scale) * (log_color - shoulder_match)));

    float3 t = saturate((log_color - toe_match) / (shoulder_match - toe_match));
    t = shoulder_match < toe_match ? 1.0f - t : t;
    t = (3.0f - t * 2.0f) * t * t;
    float3 tone_color = lerp(toe_color, shoulder_color, t);

    // Post desaturate
    tone_color = lerp(dot(tone_color, AP1_RGB2Y), tone_color, 0.93f);

    // Returning positive AP1 values
    return max(0.0, tone_color);
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

float4 uncharted2_tonemapping_ps(Varyings input) : SV_TARGET {
    // Exposure
    float average_luminance = get_average_luminance(input.texcoord);
    float exposure = geometric_mean_linear_exposure(average_luminance);
    exposure = clamp(exposure, 0.25, 4.0);
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

float4 unreal4_tonemapping_ps(Varyings input) : SV_TARGET{
    // Exposure
    float average_luminance = get_average_luminance(input.texcoord);
    float exposure = geometric_mean_linear_exposure(average_luminance);
    exposure = clamp(exposure, 0.25, 4.0);
    float3 color = exposure * pixels[int2(input.position.xy)].rgb;

    // Tonemapping.
    return float4(unreal4(color), 1.0);
}
