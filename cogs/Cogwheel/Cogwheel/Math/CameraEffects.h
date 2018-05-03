// Cogwheel camera effects settings and operators.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_CAMERA_EFFECTS_H_
#define _COGWHEEL_MATH_CAMERA_EFFECTS_H_

#include <Cogwheel/Math/Color.h>
#include <Cogwheel/Math/Matrix.h>
#include <Cogwheel/Math/Utils.h>

namespace Cogwheel {
namespace Math {
namespace CameraEffects {

enum class TonemappingMode { Linear, Filmic, Uncharted2, Count };
enum class ExposureMode { Fixed, LogAverage, Histogram, Count };

struct Settings final {
    struct {
        ExposureMode mode;
        float min_log_luminance;
        float max_log_luminance;
        float min_histogram_percentage;
        float max_histogram_percentage;
        float log_lumiance_bias;
        bool eye_adaptation_enabled;
        float eye_adaptation_brightness;
        float eye_adaptation_darkness;
    } exposure;

    struct {
        float receiver_threshold;
        float bandwidth; // Normalized bandwidth relative to the height of the image.
        float std_dev(int height) { return (bandwidth * height)* 0.25f; }
        float variance(int height) { return std_dev(height) * std_dev(height); }
    } bloom;

    struct {
        TonemappingMode mode;
    } tonemapping;

    static Settings default() {
        Settings res;
        res.exposure.mode = ExposureMode::Histogram;
        res.exposure.min_log_luminance = -4;
        res.exposure.max_log_luminance = 4;
        res.exposure.min_histogram_percentage = 0.7f;
        res.exposure.max_histogram_percentage = 0.95f;
        res.exposure.log_lumiance_bias = 0;
        res.exposure.eye_adaptation_enabled = true;
        res.exposure.eye_adaptation_brightness = 3.0f;
        res.exposure.eye_adaptation_darkness = 1.0f;

        res.bloom.receiver_threshold = 1.5f;
        res.bloom.bandwidth = 0.05f;

        res.tonemapping.mode = TonemappingMode::Uncharted2;

        return res;
    }
};


// ------------------------------------------------------------------------------------------------
// Free functions.
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// Advanced tonemapping operator
// ------------------------------------------------------------------------------------------------
// http://perso.univ-lyon1.fr/jean-claude.iehl/Public/educ/GAMA/2007/gdc07/Post-Processing_Pipeline.pdf
inline RGB reinhard(RGB color, float white_level_sqrd) {
    float color_luminance = luminance(color);
    float tonemapped_luminance = color_luminance * (1.0f + color_luminance / white_level_sqrd) / (1.0f + color_luminance);
    return color * (tonemapped_luminance / color_luminance);
}

// ------------------------------------------------------------------------------------------------
// Uncharted 2's filmic operator.
// ------------------------------------------------------------------------------------------------
inline RGB uncharted2(RGB color, float shoulder_strength, float linear_strength, float linear_angle, float toe_strength, float toe_numerator, float toe_denominator, float linear_white) {
    auto uncharted2_tonemap_helper = [](RGB color, float shoulder_strength, float linear_strength, float linear_angle, float toe_strength, float toe_numerator, float toe_denominator) -> RGB {
        RGB x = color;
        float A = shoulder_strength;
        float B = linear_strength;
        float C = linear_angle;
        float D = toe_strength;
        float E = toe_numerator;
        float F = toe_denominator;
        return ((x*(x*A + C*B) + D*E) / (x*(x*A + B) + D*F)) - E / F;
    };

    return uncharted2_tonemap_helper(color, shoulder_strength, linear_strength, linear_angle, toe_strength, toe_numerator, toe_denominator) /
        uncharted2_tonemap_helper(RGB(linear_white), shoulder_strength, linear_strength, linear_angle, toe_strength, toe_numerator, toe_denominator);
}

// ------------------------------------------------------------------------------------------------
// Unreal Engine 4 filmic tonemapping, see TonemapCommon.ush FilmToneMap function for the 
// tonemapping and ACES.ush for the conversion matrices.
// ------------------------------------------------------------------------------------------------

// Bradford chromatic adaptation transforms between ACES white point (D60) and sRGB white point (D65)
static const Matrix3x3f D65_to_D60_cat = {
    1.01303f,    0.00610531f, -0.014971f,
    0.00769823f, 0.998165f,   -0.00503203f,
    -0.00284131f, 0.00468516f,  0.924507f };

// https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
static const Matrix3x3f sRGB_to_XYZ_mat = {
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f };

static const Matrix3x3f XYZ_to_AP1_mat = {
    1.6410233797f, -0.3248032942f, -0.2364246952f,
    -0.6636628587f,  1.6153315917f,  0.0167563477f,
    0.0117218943f, -0.0082844420f,  0.9883948585f };

static const Matrix3x3f AP1_to_XYZ_mat = {
    0.6624541811f, 0.1340042065f, 0.1561876870f,
    0.2722287168f, 0.6740817658f, 0.0536895174f,
    -0.0055746495f, 0.0040607335f, 1.0103391003f };

static const Vector3f AP1_RGB2Y = AP1_to_XYZ_mat.get_row(1);

inline RGB unreal4(RGB color, float slope = 0.91f, float toe = 0.53f, float shoulder = 0.23f, float black_clip = 0.0f, float white_clip = 0.035f) {

    static const Matrix3x3f sRGB_to_AP1 = XYZ_to_AP1_mat * D65_to_D60_cat * sRGB_to_XYZ_mat;

    // Use ACEScg primaries as working space
    Vector3f working_color = sRGB_to_AP1 * Vector3f(color.r, color.g, color.b);
    working_color = max(Vector3f::zero(), working_color);

    // Pre desaturate
    working_color = lerp(Vector3f(dot(working_color, AP1_RGB2Y)), working_color, 0.96f);

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

    Vector3f log_color = Vector3f(log10(working_color.x), log10(working_color.y), log10(working_color.z));
    Vector3f straight_color = (log_color + straight_match) * slope;

    auto exp = [](Vector3f v) -> Vector3f { return Vector3f(expf(v.x), expf(v.y), expf(v.z)); };

    Vector3f toe_color = (-black_clip) + (2.0f * toe_scale) / (1.0f + exp((log_color - toe_match) * (-2 * slope / toe_scale)));
    toe_color = Vector3f(log_color.x < toe_match ? toe_color.x : straight_color.x,
                         log_color.y < toe_match ? toe_color.y : straight_color.y,
                         log_color.z < toe_match ? toe_color.z : straight_color.z);

    Vector3f shoulder_color = (1.0f + white_clip) - (2.0f * shoulder_scale) / (1.0f + exp((log_color - shoulder_match) * (2 * slope / shoulder_scale)));
    shoulder_color = Vector3f(log_color.x > shoulder_match ? shoulder_color.x : straight_color.x,
                              log_color.y > shoulder_match ? shoulder_color.y : straight_color.y,
                              log_color.z > shoulder_match ? shoulder_color.z : straight_color.z);

    Vector3f t = clamp((log_color - toe_match) / (shoulder_match - toe_match), Vector3f::zero(), Vector3f::one());
    t = shoulder_match < toe_match ? 1.0f - t : t;
    t = (3.0f - t * 2.0f) * t * t;
    Vector3f tone_color = lerp(toe_color, shoulder_color, t);

    // Post desaturate
    tone_color = lerp(Vector3f(dot(tone_color, AP1_RGB2Y)), tone_color, 0.93f);

    // Returning positive AP1 values
    return RGB(fmaxf(0.0f, tone_color.x), fmaxf(0.0f, tone_color.y), fmaxf(0.0f, tone_color.z));
}

} // NS CameraEffects
} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_CAMERA_EFFECTS_H_