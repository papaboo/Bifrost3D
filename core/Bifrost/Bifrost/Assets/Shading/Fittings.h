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

namespace Bifrost::Assets::Shading {

namespace Rho {

// ------------------------------------------------------------------------------------------------
// Burley rho fit
// ------------------------------------------------------------------------------------------------
extern const int burley_angle_sample_count;
extern const int burley_roughness_sample_count;
extern const float burley[];
float sample_burley(float wo_dot_normal, float roughness);

// ------------------------------------------------------------------------------------------------
// Dielectric GGX rho fit
// ------------------------------------------------------------------------------------------------
extern const int dielectric_GGX_angle_sample_count;
extern const int dielectric_GGX_roughness_sample_count;
extern const int dielectric_GGX_specularity_sample_count;
extern const float dielectric_GGX_minimum_specularity;
extern const float dielectric_GGX_maximum_specularity;
extern const Math::Vector2f dielectric_GGX[];
Math::Vector2f sample_dielectric_GGX(float wo_dot_normal, float roughness, float specularity);

// ------------------------------------------------------------------------------------------------
// GGX rho fit
// ------------------------------------------------------------------------------------------------
extern const int GGX_angle_sample_count;
extern const int GGX_roughness_sample_count;
extern const float GGX[];
float sample_GGX(float wo_dot_normal, float roughness);

// ------------------------------------------------------------------------------------------------
// GGX with fresnel rho fit
// ------------------------------------------------------------------------------------------------
extern const int GGX_with_fresnel_angle_sample_count;
extern const int GGX_with_fresnel_roughness_sample_count;
extern const float GGX_with_fresnel[];
float sample_GGX_with_fresnel(float wo_dot_normal, float roughness);

} // NS Rho

namespace Estimate_GGX_bounded_VNDF_alpha {

// ------------------------------------------------------------------------------------------------
// Estimate the alpha of the GGX distribution that with the given maximal PDF from an angle.
// ------------------------------------------------------------------------------------------------
extern const int alpha_sample_count;
extern const int wo_dot_normal_sample_count;
extern const int max_PDF_sample_count;
extern const float alphas[];
float encode_PDF(float pdf);
float estimate_alpha(float wo_dot_normal, float max_PDF);

} // NS Estimate_GGX_bounded_VNDF_alpha 

} // NS Bifrost::Assets::Shading

#endif // _BIFROST_ASSETS_SHADING_FITTINGS_H
