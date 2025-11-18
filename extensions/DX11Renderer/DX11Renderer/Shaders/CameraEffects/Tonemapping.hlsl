// Tonemapping shaders.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
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

float4 fullscreen_vs(uint vertex_ID : SV_VertexID) : SV_POSITION {
    float4 position;
    // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }
    position.x = vertex_ID == 2 ? 3 : -1;
    position.y = vertex_ID == 0 ? -3 : 1;
    position.zw = float2(1.0, 1.0);
    return position;
}

// ------------------------------------------------------------------------------------------------
// Tonemappers.
// ------------------------------------------------------------------------------------------------

// Bradford chromatic adaptation transforms between ACES white point (D60) and sRGB white point (D65)
static const float3x3 D65_to_D60 = {
    1.01303, 0.00610531, -0.014971,
    0.00769823, 0.998165, -0.00503203,
    -0.00284131, 0.00468516, 0.924507 };

static const float3x3 sRGB_to_XYZ = {
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041 };

static const float3x3 XYZ_to_AP1 = {
    1.6410233797, -0.3248032942, -0.2364246952,
    -0.6636628587, 1.6153315917, 0.0167563477,
    0.0117218943, -0.0082844420, 0.9883948585 };

static const float3x3 AP1_to_XYZ = {
    0.6624541811, 0.1340042065, 0.1561876870,
    0.2722287168, 0.6740817658, 0.0536895174,
    -0.0055746495, 0.0040607335, 1.0103391003 };

static const float3x3 sRGB_to_AP1 = mul(XYZ_to_AP1, mul(D65_to_D60, sRGB_to_XYZ));

static const float3x3 AP1_to_sRGB = {
    1.70479095, -0.621689737,-0.0832421705,
    -0.130263522, 1.14082849, -0.0105496496,
    -0.0240088310, -0.128999621, 1.15324795 };

static const float3 AP1_RGB2Y = {
    0.2722287168, // AP1_to_XYZ[0][1],
    0.6740817658, // AP1_to_XYZ[1][1],
    0.0536895174 // AP1_to_XYZ[2][1]
};

inline float3 unreal4(float3 color, float black_clip = 0.0f, float toe = 0.53f, float slope = 0.91f, float shoulder = 0.23f, float white_clip = 0.035f) {

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

    // Returning positive values
    return mul(AP1_to_sRGB, max(0.0, tone_color));
}

// Mean error^2: 3.6705141e-06
float3 agxDefaultContrastApprox(float3 c) {
    return -0.00232f + c * (0.1191f + c * (0.4298f + c * (-6.868f + c * (31.96f + c * (-40.14f + c * 15.5f)))));
}

// AgX tonemapping operator
// https://iolite-engine.com/blog_posts/minimal_agx_implementation
float3 agx(float3 linear_color) {
    // Transform to AgX color space
    const float3x3 linear_to_agx = { 0.842479062253094, 0.0784335999999992, 0.0792237451477643,
                                     0.0423282422610123, 0.878468636469772, 0.0791661274605434,
                                     0.0423756549057051, 0.0784336, 0.879142973793104 };
    float3 c = mul(linear_to_agx, linear_color);

    // Log2 space encoding
    const float min_exposure_value = -12.47393;
    const float max_exposure_value = 4.026069;
    c.r = log2(c.r);
    c.g = log2(c.g);
    c.b = log2(c.b);
    c = (c - min_exposure_value) / (max_exposure_value - min_exposure_value); // Inverse lerp between min and max by c

    // Apply sigmoid function approximation
    c = agxDefaultContrastApprox(saturate(c));

    // Transform back from AgX color space
    const float3x3 agx_to_tonemapped = { 1.19687900512017, -0.0980208811401368, -0.0990297440797205,
                                        -0.0528968517574562, 1.15190312990417, -0.0989611768448433,
                                        -0.0529716355144438, -0.0980434501171241, 1.15107367264116 };
    c = mul(agx_to_tonemapped, c);

    return pow(abs(c), 2.2);
}

// Khronos neutral tonemapping operator
// https://modelviewer.dev/examples/tone-mapping#commerce
float3 khronos_neutral_tone_mapping(float3 linear_color) {
    const float start_compression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(linear_color.r, min(linear_color.g, linear_color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    linear_color -= offset;

    float peak = max(linear_color.r, max(linear_color.g, linear_color.b));
    if (peak < start_compression) return linear_color;

    float d = 1.0 - start_compression;
    float new_peak = 1.0 - d * d / (peak + d - start_compression);
    linear_color *= new_peak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0f);
    return lerp(linear_color, new_peak, g);
}

// ------------------------------------------------------------------------------------------------
// Simple vignette using smoothstep.
// Adapted from https://github.com/mattdesl/lwjgl-basics/wiki/ShaderLesson3
// ------------------------------------------------------------------------------------------------

float simple_vignette_tint(float2 viewport_uv, float scale) {
    float outerRadius = 0.9f;
    float2 coord = viewport_uv - float2(0.5, 0.5);
    return 1.0 - smoothstep(0.1f, outerRadius, length(coord) * 1.5f * scale);
}

// ------------------------------------------------------------------------------------------------
// Film grain
// Adapted from https://gameidea.org/2023/12/01/film-grain-shader/
// ------------------------------------------------------------------------------------------------

float film_grain(float2 viewport_uv, float scale) {
    float2 grain_uv = viewport_uv + delta_time;
    float noise = frac(sin(dot(grain_uv, float2(12.9898, 78.233))) * 43758.5453) - 0.5;
    return scale * noise;
}

// ------------------------------------------------------------------------------------------------
// Tonemapping pixel shaders.
// ------------------------------------------------------------------------------------------------

Texture2D pixels : register(t0);
Buffer<float> linear_exposure_buffer : register(t1);
Texture2D bloom_texture : register(t2);

interface ITonemapper {
    float3 tonemap(float3 color);
};

class LinearTonemapper : ITonemapper {
    float3 tonemap(float3 color) { return color; }
};

class Unreal4Tonemapper : ITonemapper {
    float3 tonemap(float3 color) {
        return unreal4(color, tonemapping_black_clip, tonemapping_toe, tonemapping_slope, tonemapping_shoulder, tonemapping_white_clip);
    }
};

class KhronosNeutralTonemapper : ITonemapper {
    float3 tonemap(float3 color) { return khronos_neutral_tone_mapping(color); }
};

class AgXTonemapper : ITonemapper {
    float3 tonemap(float3 color) { return agx(color); }
};

float4 postprocess_pixel(int2 pixel_index, ITonemapper tonemapper) {
    // Bloom and exposure
    float linear_exposure = linear_exposure_buffer[0];
    float3 low_intensity_color = min(pixels[pixel_index + output_pixel_offset].rgb, bloom_threshold);
    float3 bloom_color = bloom_texture[pixel_index - output_viewport_offset].rgb;
    float3 color = linear_exposure * (low_intensity_color + bloom_color);

    // Vignette
    float2 viewport_size = input_viewport.zw;
    float2 viewport_uv = (pixel_index - output_viewport_offset) / viewport_size;
    color *= simple_vignette_tint(viewport_uv, vignette_strength);

    // Tonemap
    color = tonemapper.tonemap(color);

    // Film grain
    color += film_grain(viewport_uv, film_grain_strength);

    return float4(color, 1.0f);
}

float4 linear_tonemapping_ps(float4 position : SV_POSITION) : SV_TARGET {
    LinearTonemapper tonemapper;
    return postprocess_pixel(position.xy, tonemapper);
}

float4 unreal4_tonemapping_ps(float4 position : SV_POSITION) : SV_TARGET {
    Unreal4Tonemapper tonemapper;
    return postprocess_pixel(position.xy, tonemapper);
}

float4 khronos_neutral_tonemapping_ps(float4 position : SV_POSITION) : SV_TARGET {
    KhronosNeutralTonemapper tonemapper;
    return postprocess_pixel(position.xy, tonemapper);
}

float4 agx_tonemapping_ps(float4 position : SV_POSITION) : SV_TARGET {
    AgXTonemapper tonemapper;
    return postprocess_pixel(position.xy, tonemapper);
}

} // NS CameraEffects