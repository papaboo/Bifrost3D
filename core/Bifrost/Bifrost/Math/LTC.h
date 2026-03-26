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
// as the EON Oren-Nayer fitting and the final GGX fitting differ on which matrix-element to scale to 1.
// ------------------------------------------------------------------------------------------------
struct IsotropicLTC {
public:
    float m00, m11, m22, m02, m20;

    static inline IsotropicLTC identity() { return { 1, 1, 1, 0, 0 }; }

    inline Matrix3x3f get_inverse_M() const { return { m00, 0, m02, 0, m11, 0, m20, 0, m22 }; }
    inline Matrix3x3f get_M() const { return invert(get_inverse_M()); }
    inline float inverse_M_determinant() const { return m11 * (m00 * m22 - m02 * m20); }

    inline float PDF(Vector3f w) const {
        Vector3f w_original_scaled = get_inverse_M() * w;

        float l = 1.0f / magnitude(w_original_scaled); // magnitude(invert(inverse_M) * normalize(w_original)) in the paper source.
        float reciprocal_jacobian = (l * l * l) * inverse_M_determinant();

        float original_cos_theta = fmaxf(0.0f, w_original_scaled.z * l); // Multiplication with l amounts to normalize(w_original_scaled).z
        return Distributions::Cosine::PDF(original_cos_theta) * reciprocal_jacobian;
    }

    inline float evaluate(Vector3f w) const { return PDF(w); }

    inline Vector3f sample(Vector2f random_sample) const {
        auto cosine_direction = Distributions::Cosine::sample(random_sample).direction;

        // Transform cosine sample to LTC sample.
        return Math::normalize(get_M() * cosine_direction);
    }
};

} // NS Bifrost::Math

#endif // _BIFROST_MATH_LTC_H_
