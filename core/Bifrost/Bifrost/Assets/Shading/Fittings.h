// Functions fitted to textures.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_FITTINGS_H
#define _BIFROST_ASSETS_SHADING_FITTINGS_H

#include <Bifrost/Math/Vector.h>

namespace Bifrost {
namespace Assets {
namespace Shading {
namespace Rho {

// ------------------------------------------------------------------------------------------------
// Burley rho fit
// ------------------------------------------------------------------------------------------------
extern const unsigned int burley_angle_sample_count;
extern const unsigned int burley_roughness_sample_count;
extern const float burley[];
float sample_burley(float wo_dot_normal, float roughness);

// ------------------------------------------------------------------------------------------------
// Default shading rho fit
// ------------------------------------------------------------------------------------------------
extern const unsigned int default_shading_angle_sample_count;
extern const unsigned int default_shading_roughness_sample_count;
extern const Bifrost::Math::Vector2f default_shading[];
Bifrost::Math::Vector2f sample_default_shading(float wo_dot_normal, float roughness);

// ------------------------------------------------------------------------------------------------
// GGX rho fit
// ------------------------------------------------------------------------------------------------
extern const unsigned int GGX_angle_sample_count;
extern const unsigned int GGX_roughness_sample_count;
extern const float GGX[];
float sample_GGX(float wo_dot_normal, float roughness);

// ------------------------------------------------------------------------------------------------
// GGX with fresnel rho fit
// ------------------------------------------------------------------------------------------------
extern const unsigned int GGX_with_fresnel_angle_sample_count;
extern const unsigned int GGX_with_fresnel_roughness_sample_count;
extern const float GGX_with_fresnel[];
float sample_GGX_with_fresnel(float wo_dot_normal, float roughness);

// ------------------------------------------------------------------------------------------------
// OrenNayar rho fit
// ------------------------------------------------------------------------------------------------
extern const unsigned int oren_nayar_angle_sample_count;
extern const unsigned int oren_nayar_roughness_sample_count;
extern const float oren_nayar[];
float sample_oren_nayar(float wo_dot_normal, float roughness);

} // NS Rho

namespace EstimateGGXAlpha {

    // ------------------------------------------------------------------------------------------------
// Estimate the alpha of the GGX distribution that with the given maximal PDF from a angle.
// ------------------------------------------------------------------------------------------------
extern const int alpha_sample_count;
extern const int cos_theta_sample_count;
extern const int max_PDF_sample_count;
extern const float alphas[];
float encode_PDF(float p);
float decode_PDF(float e);
float estimate_alpha(float cos_theta, float max_PDF);

} // NS EstimateGGXAlpha 

} // NS Shading
} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_SHADING_FITTINGS_H
