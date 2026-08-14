// Bifrost linearly transformed cosine.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_MATH_LTC_H_
#define _BIFROST_MATH_LTC_H_

#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/Matrix.h>

namespace Bifrost::Math {

// ------------------------------------------------------------------------------------------------
// Represents an LTC fitted to isotropic lopes, as done in
// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines, Heitz et al., 2016.
// The implementation is a mix between what is found in the paper and the accompanying code sample.
// We have included 5 parameters here instead of the regular 4,
// as the EON Oren-Nayar fitting and the final GGX fitting differ on which matrix-element to scale to 1.
// ------------------------------------------------------------------------------------------------
struct IsotropicLTC {
public:
    // Elements in the inverse M matrix
    float e00, e11, e22, e02, e20;

    static _inline_all_archs_ IsotropicLTC from_inverse_M(float e00, float e11, float e22, float e02, float e20) {
        return { e00, e11, e22, e02, e20 };
    }
    static _inline_all_archs_ IsotropicLTC from_M(float e00, float e11, float e22, float e02, float e20) {
        Matrix3x3f inverse_M = invert_matrix(e00, e11, e22, e02, e20);
        return { inverse_M(0, 0), inverse_M(1, 1), inverse_M(2, 2), inverse_M(0, 2), inverse_M(2, 0) };
    }

    static _inline_all_archs_ IsotropicLTC identity() { return { 1, 1, 1, 0, 0 }; }

    _inline_all_archs_ Matrix3x3f get_inverse_M() const { return { e00, 0.0f, e02, 0.0f, e11, 0.0f, e20, 0.0f, e22 }; }
    _inline_all_archs_ Matrix3x3f get_M() const { return invert_matrix(e00, e11, e22, e02, e20); }
    _inline_all_archs_ float inverse_M_determinant() const { return e11 * (e00 * e22 - e02 * e20); }

    _inline_all_archs_ float PDF(Vector3f w) const {
        Vector3f w_original_scaled = get_inverse_M() * w;

        float l = 1.0f / magnitude(w_original_scaled); // magnitude(invert(inverse_M) * normalize(w_original)) in the paper source.
        float reciprocal_jacobian = (l * l * l) * inverse_M_determinant();

        float original_cos_theta = fmaxf(0.0f, w_original_scaled.z * l); // Multiplication with l amounts to normalize(w_original_scaled).z
        return Distributions::Cosine::PDF(original_cos_theta) * reciprocal_jacobian;
    }

    _inline_all_archs_ float evaluate(Vector3f w) const { return PDF(w); }

    _inline_all_archs_ Distributions::DirectionalSample sample(Vector2f random_sample) const {
        auto cosine_direction = Distributions::Cosine::sample(random_sample).direction;

        // Transform cosine sample to LTC sample.
        Vector3f ltc_direction = Math::normalize(get_M() * cosine_direction);

        return { ltc_direction, PDF(ltc_direction) };
    }

private:
    // Specialized 3x3 matrix inversion for diagonal cross matrices.
    _inline_all_archs_ static Matrix3x3f invert_matrix(float e00, float e11, float e22, float e02, float e20) {
        // As all non-zero elements are scaled by e11, we inline that into the precomputed determinant.
        // e11 is part of computing the determinant, so removing it produces e11 / determinant.
        // float determinant = e11 * (e00 * e22 - e02 * e20); // Included for reference
        float e11_over_determinant = 1.0f / (e00 * e22 - e02 * e20);

        Matrix3x3f inverse;

        inverse[0][0] = e22 * e11_over_determinant;
        inverse[0][1] = 0.0f;
        inverse[0][2] = - e02 * e11_over_determinant;

        inverse[1][0] = 0.0f;
        inverse[1][1] = 1.0f / e11;
        inverse[1][2] = 0.0f;

        inverse[2][0] = - e20 * e11_over_determinant;
        inverse[2][1] = 0.0f;
        inverse[2][2] = e00 * e11_over_determinant;

        return inverse;
    }
};

} // NS Bifrost::Math

#endif // _BIFROST_MATH_LTC_H_
