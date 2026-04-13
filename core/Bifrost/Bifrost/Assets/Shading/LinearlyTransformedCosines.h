// Linearly transformed cosine fittings.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LTC_H
#define _BIFROST_ASSETS_SHADING_LTC_H

#include <Bifrost/Math/LTC.h>

namespace Bifrost::Assets::Shading::LTC {

// ------------------------------------------------------------------------------------------------
// Lambert reflection fit just returns the coefficients for the identity matrix.
// ------------------------------------------------------------------------------------------------
inline Math::IsotropicLTC lambert_LTC_coefficients() { return Math::IsotropicLTC::identity(); }

// ------------------------------------------------------------------------------------------------
// EON Oren-Nayar fit
// ------------------------------------------------------------------------------------------------

// Listing 2, EON: A Practical Energy-Preserving Rough Diffuse BRDF, Portsmouth et al, 2025
inline Math::IsotropicLTC oren_nayar_LTC_coefficients(float cos_theta_o, float roughness) {
    float mu = cos_theta_o;
    float m00 = 1.0f + roughness * (0.303392f + (-0.518982f + 0.111709f*mu)*mu + (-0.276266f + 0.335918f*mu) * roughness);
    float m02 = roughness * (-1.16407f + 1.15859f*mu + (0.150815f - 0.150105f*mu)*roughness) / (mu*mu*mu - 1.43545f);
    float m11 = 1.0f + roughness * (0.20013f + (-0.506373f + 0.261777f*mu)*mu);
    float m20 = roughness * (0.540852f + (-1.01625f + 0.475392f*mu)*mu) / (-1.0743f + (0.0725628f + mu)*mu);
    return { m00, m11, 1.0f, m02, m20 };
}

// ------------------------------------------------------------------------------------------------
// Isotropic GGX reflection fit
// ------------------------------------------------------------------------------------------------
extern const int GGX_reflection_angle_sample_count;
extern const int GGX_reflection_roughness_sample_count;
extern const Math::Vector4f GGX_reflection_LTC_params[];
Math::IsotropicLTC GGX_reflection_LTC_coefficients(float cos_theta_o, float roughness);

}

#endif // _BIFROST_ASSETS_SHADING_LTC_H