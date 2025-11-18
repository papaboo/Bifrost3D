// Bifrost camera effects settings and operators.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_CAMERA_EFFECTS_H_
#define _BIFROST_MATH_CAMERA_EFFECTS_H_

#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Matrix.h>
#include <Bifrost/Math/Utils.h>

namespace Bifrost::Math::CameraEffects {

enum class TonemappingMode { Linear, Filmic, AgX, KhronosNeutral, Count };
enum class ExposureMode { Fixed, LogAverage, Histogram, Count };

struct TonemappingSettings {
    float black_clip;
    float toe;
    float slope;
    float shoulder;
    float white_clip;

    static TonemappingSettings ACES() { return { 0.0f, 0.53f, 0.91f, 0.23f, 0.035f }; }
    static TonemappingSettings uncharted2() { return { 0.0f, 0.55f, 0.63f, 0.47f, 0.01f }; }
    static TonemappingSettings HP() { return { 0.0f, 0.63f, 0.65f, 0.45f, 0.0f }; }
    static TonemappingSettings legacy() { return { 0.0f, 0.3f, 0.98f, 0.22f, 0.025f}; }
    static TonemappingSettings default() { return ACES(); }
};

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
        float threshold;
        float support; // Normalized support relative to the height of the image.
        float std_dev(int height) { return (support * height) * 0.25f; }
        float variance(int height) { return std_dev(height) * std_dev(height); }
    } bloom;

    float vignette;

    struct {
        TonemappingMode mode;
        TonemappingSettings settings;
    } tonemapping;

    float film_grain;

    static Settings default() {
        Settings res = {};
        res.exposure.mode = ExposureMode::Histogram;
        res.exposure.min_log_luminance = -4;
        res.exposure.max_log_luminance = 4;
        res.exposure.min_histogram_percentage = 0.7f;
        res.exposure.max_histogram_percentage = 0.95f;
        res.exposure.log_lumiance_bias = 0;
        res.exposure.eye_adaptation_enabled = true;
        res.exposure.eye_adaptation_brightness = 3.0f;
        res.exposure.eye_adaptation_darkness = 1.0f;

        res.bloom.threshold = INFINITY;
        res.bloom.support = 0.05f;

        res.vignette = 0.63f;

        res.tonemapping.mode = TonemappingMode::Filmic;
        res.tonemapping.settings = TonemappingSettings::default();

        res.film_grain = 1 / 255.0f;

        return res;
    }

    // Output linear colors without any exposure, vignetting or other screen space effects.
    static Settings linear() {
        Settings res = {};
        res.exposure.mode = ExposureMode::Fixed;
        res.exposure.min_log_luminance = -4;
        res.exposure.max_log_luminance = 4;
        res.exposure.min_histogram_percentage = 0.7f;
        res.exposure.max_histogram_percentage = 0.95f;
        res.exposure.log_lumiance_bias = 0;
        res.exposure.eye_adaptation_enabled = false;
        res.exposure.eye_adaptation_brightness = INFINITY;
        res.exposure.eye_adaptation_darkness = INFINITY;

        res.bloom.threshold = INFINITY;
        res.bloom.support = 0.00f;

        res.vignette = 0.0f;

        res.tonemapping.mode = TonemappingMode::Linear;
        res.tonemapping.settings = TonemappingSettings::default();

        res.film_grain = 0.0f;

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
// Unreal Engine 4 filmic tonemapping, see TonemapCommon.ush FilmToneMap function for the 
// tonemapping and ACES.ush for the conversion matrices.
// ------------------------------------------------------------------------------------------------

// Bradford chromatic adaptation transforms between ACES white point (D60) and sRGB white point (D65)
static const Matrix3x3f D65_to_D60 = {
    1.01303f,    0.00610531f, -0.014971f,
    0.00769823f, 0.998165f,   -0.00503203f,
    -0.00284131f, 0.00468516f,  0.924507f };

// https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
static const Matrix3x3f sRGB_to_XYZ = {
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f };

static const Matrix3x3f XYZ_to_AP1 = {
    1.6410233797f, -0.3248032942f, -0.2364246952f,
    -0.6636628587f,  1.6153315917f,  0.0167563477f,
    0.0117218943f, -0.0082844420f,  0.9883948585f };

static const Matrix3x3f AP1_to_XYZ = {
    0.6624541811f, 0.1340042065f, 0.1561876870f,
    0.2722287168f, 0.6740817658f, 0.0536895174f,
    -0.0055746495f, 0.0040607335f, 1.0103391003f };

static const Matrix3x3f sRGB_to_AP1 = XYZ_to_AP1 * D65_to_D60 * sRGB_to_XYZ;
static const Matrix3x3f AP1_to_sRGB = invert(sRGB_to_AP1);

static const Vector3f AP1_RGB2Y = AP1_to_XYZ.get_row(1);

inline RGB filmic(RGB color, float slope = 0.91f, float toe = 0.53f, float shoulder = 0.23f, float black_clip = 0.0f, float white_clip = 0.035f) {

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

    // Returning positive linear color values
    Vector3f ap1_color = Vector3f(fmaxf(0.0f, tone_color.x), fmaxf(0.0f, tone_color.y), fmaxf(0.0f, tone_color.z));
    Vector3f c = AP1_to_sRGB * ap1_color;
    return RGB(c.x, c.y, c.z);
}

inline RGB filmic(RGB color, TonemappingSettings s) {
    return filmic(color, s.slope, s.toe, s.shoulder, s.black_clip, s.white_clip);
}

// ------------------------------------------------------------------------------------------------
// AgX tonemapping operator
// https://iolite-engine.com/blog_posts/minimal_agx_implementation
// ------------------------------------------------------------------------------------------------

// Mean error^2: 3.6705141e-06
inline RGB agx_default_contrast_approx(RGB c) {
    return -0.00232f + c * (0.1191f + c * (0.4298f + c * (-6.868f + c * (31.96f + c * (-40.14f + c * 15.5f)))));
}

inline RGB agx(RGB linear_color) {
    // Transform to AgX color space
    const Matrix3x3f linear_to_agx = { 0.842479062253094f, 0.0784335999999992f, 0.0792237451477643f,
                                       0.0423282422610123f, 0.878468636469772f, 0.0791661274605434f,
                                       0.0423756549057051f, 0.0784336f, 0.879142973793104f };
    RGB c = linear_to_agx * linear_color;

    // Log2 space encoding
    const RGB min_exposure_value = RGB(-12.47393f);
    const RGB max_exposure_value = RGB(4.026069f);
    c.r = log2(c.r);
    c.g = log2(c.g);
    c.b = log2(c.b);
    c = inverse_lerp(min_exposure_value, max_exposure_value, c);

    // Apply sigmoid function approximation
    c = agx_default_contrast_approx(saturate(c));

    // Transform back from AgX color space
    const Matrix3x3f agx_to_tonemapped = { 1.19687900512017f, -0.0980208811401368f, -0.0990297440797205f,
                                          -0.0528968517574562f, 1.15190312990417f, -0.0989611768448433f,
                                          -0.0529716355144438f, -0.0980434501171241f, 1.15107367264116f };
    c = agx_to_tonemapped * c;

    return gammacorrect(c, 2.2f);
}

// ------------------------------------------------------------------------------------------------
// Khronos neutral tonemapping operator
// https://modelviewer.dev/examples/tone-mapping#commerce
// ------------------------------------------------------------------------------------------------

inline RGB khronos_neutral_tone_mapping(RGB linear_color) {
    const float start_compression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x = min(linear_color.r, min(linear_color.g, linear_color.b));
    float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
    linear_color -= offset;

    float peak = max(linear_color.r, max(linear_color.g, linear_color.b));
    if (peak < start_compression) return linear_color;

    float d = 1.0f - start_compression;
    float new_peak = 1.0f - d * d / (peak + d - start_compression);
    linear_color *= new_peak / peak;

    float g = 1.0f - 1.0f / (desaturation * (peak - new_peak) + 1.0f);
    return lerp(linear_color, RGB::white() * new_peak, g);
}

} // NS Bifrost::Math::CameraEffects

#endif // _BIFROST_MATH_CAMERA_EFFECTS_H_
