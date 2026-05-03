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

    inline Distributions::DirectionalSample sample(Vector2f random_sample) const {
        auto cosine_direction = Distributions::Cosine::sample(random_sample).direction;

        // Transform cosine sample to LTC sample.
        Vector3f ltc_direction = Math::normalize(get_M() * cosine_direction);

        return { ltc_direction, PDF(ltc_direction) };
    }
};

namespace LTCAreaLights {

// Edge integral using the fitted function to replace acos and gain increased precision.
// Real-Time Area Lighting: a Journey from Research to Production, Stephen Hill and Eric Heitz, Siggraph, 2017
inline Vector3f vector_edge_integral(Vector3f v1, Vector3f v2) {
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
    float b = 3.4175940f + (4.1616724f + y) * y;
    float v = a / b;

    float theta_over_sintheta = (x > 0.0f) ? v : 0.5f / sqrt(fmaxf(1.0f - x * x, 1e-7f)) - v;

    return cross(v1, v2) * theta_over_sintheta;
}

inline float edge_integral(Vector3f v1, Vector3f v2) { return vector_edge_integral(v1, v2).z; }

inline float evaluate_mesh_light_lambert(Vector3f normal, Vector3f wo, Vector3f position, Vector3f positions[3], bool two_sided) {
    // Construct orthonormal basis around normal with tangent pointing along the view direction.
    Vector3f tangent = normalize(wo - normal * dot(wo, normal));
    Vector3f bitangent = cross(normal, tangent);

    // Rotate area light in (T1, T2, N) basis
    Matrix3x3f M_inverse = Matrix3x3f({ tangent, bitangent, normal });
    // float3x3 M_inverse = transpose(float3x3(tangent, bitangent, normal));

    positions[0] = M_inverse * (positions[0] - position);
    positions[1] = M_inverse * (positions[1] - position);
    positions[2] = M_inverse * (positions[2] - position);

    // TODO triangle clipping

    // Project vertices onto sphere
    positions[0] = normalize(positions[0]);
    positions[1] = normalize(positions[1]);
    positions[2] = normalize(positions[2]);

    // Integrate triangle over cosine distribution.
    Vector3f F = { 0, 0, 0 };
    F += vector_edge_integral(positions[0], positions[1]);
    F += vector_edge_integral(positions[1], positions[2]);
    F += vector_edge_integral(positions[2], positions[0]);
    float integral = two_sided ? abs(F.z) : fmaxf(0.0, -F.z); // Negate integral due to winding order. TODO Fix later in MeshLightManager? Would be nice to keep this code in sync with other reference implementations

    return integral;
}

} // NS LTCAreaLights

} // NS Bifrost::Math

#endif // _BIFROST_MATH_LTC_H_
