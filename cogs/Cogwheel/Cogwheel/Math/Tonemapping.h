// Cogwheel tonemapping operators and parameters.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_TONEMAPPING_H_
#define _COGWHEEL_MATH_TONEMAPPING_H_

#include <Cogwheel/Math/Color.h>

namespace Cogwheel {
namespace Math {
namespace Tonemapping {

enum class Operator { Linear, Reinhard, Filmic };

struct Parameters final {
    Operator mapping;
    float exposure;
    RGB white_point;

    static Parameters default() {
        Parameters res;
        res.mapping = Operator::Linear;
        res.exposure = 0;
        res.white_point = RGB::white();
        return res;
    }

    bool use_auto_exposure() { exposure = nanf(""); }
    bool using_auto_exposure() const { return isnan(exposure); }
};


// ------------------------------------------------------------------------------------------------
// Free functions.
// ------------------------------------------------------------------------------------------------

// Advanced tonemapping operator
// http://perso.univ-lyon1.fr/jean-claude.iehl/Public/educ/GAMA/2007/gdc07/Post-Processing_Pipeline.pdf
inline RGB reinhard(RGB color, float white_level_sqrd) {
    float luminance = luma(color);
    float tonemapped_luminance = luminance * (1.0f + luminance / white_level_sqrd) / (1.0f + luminance);
    return color * (tonemapped_luminance / luminance);
}

// Uncharted 2's filmic operator.
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

} // NS Tonemapping
} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_TONEMAPPING_H_