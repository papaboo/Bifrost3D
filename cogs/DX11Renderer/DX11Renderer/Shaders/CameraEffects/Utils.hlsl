// Camera effects hlsl utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_CAMERA_EFFECTS_UTILS_H_
#define _DX11_RENDERER_SHADERS_CAMERA_EFFECTS_UTILS_H_

#include "../Utils.hlsl"

namespace CameraEffects {

cbuffer constants : register(b0) {
    float4 input_viewport; // offset in .xy, size in .zw.
    int2 output_viewport_offset;
    int2 output_pixel_offset; // input_viewport.xy - output_viewport.xy. Use for looking up individual pixels in the input at their output pixel position.

    float min_log_luminance;
    float max_log_luminance;
    float min_percentage;
    float max_percentage;
    float log_lumiance_bias;
    float eye_adaptation_brightness;
    float eye_adaptation_darkness;

    float bloom_threshold;
    int bloom_support;

    float delta_time;
    float2 padding;

    // float8 tonemapping params
    float4 tonemapping_params_1;
    float4 tonemapping_params_2;

    float filmic_black_clip() { return tonemapping_params_1.x; }
    float filmic_toe() { return tonemapping_params_1.y; }
    float filmic_slope() { return tonemapping_params_1.z; }
    float filmic_shoulder() { return tonemapping_params_1.w; }
    float filmic_white_clip() { return tonemapping_params_2.x; }

    float uncharted2_shoulder_strength() { return tonemapping_params_1.x; }
    float uncharted2_linear_strength() { return tonemapping_params_1.y; }
    float uncharted2_linear_angle() { return tonemapping_params_1.z; }
    float uncharted2_toe_strength() { return tonemapping_params_1.w; }
    float uncharted2_toe_numerator() { return tonemapping_params_2.x; }
    float uncharted2_toe_denominator() { return tonemapping_params_2.y; }
    float uncharted2_linear_white() { return tonemapping_params_2.z; }
}

float eye_adaptation(float current_exposure, float target_exposure) {
    float delta_exposure = target_exposure - current_exposure;
    float adaption_speed = (delta_exposure > 0.0) ? eye_adaptation_brightness : eye_adaptation_darkness;
    float factor = 1.0f - exp2(-delta_time * adaption_speed);
    return current_exposure + delta_exposure * factor;
}

} // NS CameraEffects

#endif // _DX11_RENDERER_SHADERS_CAMERA_EFFECTS_UTILS_H_