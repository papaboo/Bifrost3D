// OptiX monte carlo functions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_MONTE_CARLO_H_
#define _OPTIXRENDERER_MONTE_CARLO_H_

#include <OptiXRenderer/Defines.h>

namespace OptiXRenderer {
namespace MonteCarlo {

// Computes the balance heuristic of pdf1 and pdf2.
// It is assumed that pdf1 is always valid, i.e. not NaN.
// pdf2 is allowed to be NaN, but generally try to avoid it. :)
__inline_all__ float balance_heuristic(float pdf1, float pdf2) {
    float divisor = pdf1 + pdf2;
    float result = pdf1 / divisor;
    bool result_is_invalid = isinf(divisor) || isnan(result);
    return result_is_invalid ? (pdf1 <= pdf2 ? 0.0f : 1.0f) : result;
}

// Computes the power heuristic of pdf1 and pdf2.
// It is assumed that pdf1 is always valid, i.e. not NaN.
// pdf2 is allowed to be NaN, but generally try to avoid it. :)
__inline_all__ float power_heuristic(float pdf1, float pdf2) {
    pdf1 *= pdf1;
    pdf2 *= pdf2;
    return balance_heuristic(pdf1, pdf2);
}

__inline_all__ float MIS_weight(float pdf1, float pdf2) { return balance_heuristic(pdf1, pdf2); }

} // NS MonteCarlo
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_MONTE_CARLO_H_