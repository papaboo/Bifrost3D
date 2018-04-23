// Color grading hlsl utilities.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_COLOR_GRADING_UTILS_H_
#define _DX11_RENDERER_SHADERS_COLOR_GRADING_UTILS_H_

#include "../Utils.hlsl"

namespace ColorGrading {

cbuffer constants : register(b0) {
    float min_log_luminance;
    float max_log_luminance;
    float min_percentage;
    float max_percentage;
    float log_lumiance_bias;
    float eye_adaptation_brightness;
    float eye_adaptation_darkness;

    float bloom_threshold;

    float delta_time;
}

float eye_adaptation(float current_exposure, float target_exposure) {
    float delta_exposure = target_exposure - current_exposure;
    float adaption_speed = (delta_exposure > 0.0) ? eye_adaptation_brightness : eye_adaptation_darkness;
    float factor = 1.0f - exp2(-delta_time * adaption_speed);
    return current_exposure + delta_exposure * factor;
}

} // NS ColorGrading

#endif // _DX11_RENDERER_SHADERS_COLOR_GRADING_UTILS_H_