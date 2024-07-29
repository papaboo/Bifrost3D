// Camera effects hlsl utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
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
    float vignette_strength;
    float film_grain_strength;

    float tonemapping_black_clip;
    float tonemapping_toe;
    float tonemapping_slope;
    float tonemapping_shoulder;
    float tonemapping_white_clip;

    float3 _padding;
}

float eye_adaptation(float current_exposure, float target_exposure) {
    float delta_exposure = target_exposure - current_exposure;
    float adaption_speed = (delta_exposure > 0.0) ? eye_adaptation_brightness : eye_adaptation_darkness;
    float factor = 1.0f - exp2(-delta_time * adaption_speed);
    return current_exposure + delta_exposure * factor;
}

} // NS CameraEffects

#endif // _DX11_RENDERER_SHADERS_CAMERA_EFFECTS_UTILS_H_