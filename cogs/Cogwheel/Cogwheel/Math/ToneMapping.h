// Cogwheel tonemapping operators and parameters.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_TONE_MAPPING_H_
#define _COGWHEEL_MATH_TONE_MAPPING_H_

#include <Cogwheel/Math/Color.h>

namespace Cogwheel {
namespace Math {
namespace ToneMapping {

enum class Operator { Linear, Simple, Reinhard, Filmic };

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

} // NS ToneMapping
} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_COLOR_H_