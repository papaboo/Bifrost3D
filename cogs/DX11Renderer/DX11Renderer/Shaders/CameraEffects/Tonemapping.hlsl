// Tonemapping shaders.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Utils.hlsl"

namespace CameraEffects {

// ------------------------------------------------------------------------------------------------
// Fixed exposure constant copying
// ------------------------------------------------------------------------------------------------

RWStructuredBuffer<float> linear_exposure_write_buffer : register(u0);

[numthreads(1, 1, 1)]
void linear_exposure_from_constant_bias() {
    linear_exposure_write_buffer[0] = eye_adaptation(linear_exposure_write_buffer[0], exp2(log_lumiance_bias));
}

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
// Tonemappers.
// ------------------------------------------------------------------------------------------------

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
static const float3x3 D65_to_D60_cat = {
    1.01303, 0.00610531, -0.014971,
    0.00769823, 0.998165, -0.00503203,
    -0.00284131, 0.00468516, 0.924507 };

static const float3x3 sRGB_to_XYZ_mat = {
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041 };

static const float3x3 XYZ_to_AP1_mat = {
    1.6410233797, -0.3248032942, -0.2364246952,
    -0.6636628587, 1.6153315917, 0.0167563477,
    0.0117218943, -0.0082844420, 0.9883948585 };

static const float3x3 AP1_to_XYZ_mat = {
    0.6624541811, 0.1340042065, 0.1561876870,
    0.2722287168, 0.6740817658, 0.0536895174,
    -0.0055746495, 0.0040607335, 1.0103391003 };

static const float3 AP1_RGB2Y = {
    0.2722287168, // AP1_to_XYZ_mat[0][1],
    0.6740817658, // AP1_to_XYZ_mat[1][1],
    0.0536895174 // AP1_to_XYZ_mat[2][1]
};

inline float3 unreal4(float3 color, float slope = 0.91f, float toe = 0.53f, float shoulder = 0.23f, float black_clip = 0.0f, float white_clip = 0.035f) {

    static const float3x3 sRGB_to_AP1 = mul(XYZ_to_AP1_mat, mul(D65_to_D60_cat, sRGB_to_XYZ_mat));

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

SamplerState bilinear_sampler : register(s0);

Texture2D pixels : register(t0);
Buffer<float> linear_exposure_buffer : register(t1);
Texture2D bloom_texture : register(t2);

float3 get_pixel_color(int2 pixel_index, float2 uv) {
    float linear_exposure = linear_exposure_buffer[0];
    float3 low_intensity_color = min(pixels[pixel_index].rgb, bloom_threshold);
    float3 bloom_color = bloom_texture.SampleLevel(bilinear_sampler, uv, 0).rgb;
    return linear_exposure * (low_intensity_color + bloom_color);
}

float4 linear_tonemapping_ps(Varyings input) : SV_TARGET {
    return float4(get_pixel_color(int2(input.position.xy), input.texcoord), 1.0f);
}

float4 uncharted2_tonemapping_ps(Varyings input) : SV_TARGET {
    float3 color = get_pixel_color(int2(input.position.xy), input.texcoord);

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
    float3 color = get_pixel_color(int2(input.position.xy), input.texcoord);

    // Tonemapping.
    return float4(unreal4(color), 1.0);
}

} // NS CameraEffects