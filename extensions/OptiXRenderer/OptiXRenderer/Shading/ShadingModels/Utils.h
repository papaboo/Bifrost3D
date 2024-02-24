// OptiX shading model utils.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODELS_UTILS_H_
#define _OPTIXRENDERER_SHADING_MODELS_UTILS_H_

#if GPU_DEVICE
rtTextureSampler<float, 2> estimate_GGX_alpha_texture;
rtTextureSampler<float2, 2> ggx_with_fresnel_rho_texture;
#else
#include <Bifrost/Assets/Shading/Fittings.h>
#endif

namespace OptiXRenderer {
namespace Shading {
namespace ShadingModels {

// ------------------------------------------------------------------------------------------------
// Specular rho helpers.
// ------------------------------------------------------------------------------------------------
struct SpecularRho {
    const static int angle_sample_count = 32;
    const static int roughness_sample_count = 32;

    float base, full;
    __inline_all__ float rho(float specularity) { return optix::lerp(base, full, specularity); }
    __inline_all__ optix::float3 rho(optix::float3 specularity) {
        return { rho(specularity.x), rho(specularity.y), rho(specularity.z) };
    }

    __inline_all__ static SpecularRho fetch(float abs_cos_theta, float roughness) {
#if GPU_DEVICE
        // Adjust UV coordinates to start sampling half a pixel into the texture, as the pixel values correspond to the boundaries of the rho function.
        float cos_theta_uv = optix::lerp(0.5 / angle_sample_count, (angle_sample_count - 0.5) / angle_sample_count, abs_cos_theta);
        float roughness_uv = optix::lerp(0.5 / roughness_sample_count, (roughness_sample_count - 0.5) / roughness_sample_count, roughness);

        float2 specular_rho = tex2D(ggx_with_fresnel_rho_texture, cos_theta_uv, roughness_uv);
        return { specular_rho.x, specular_rho.y };
#else
        return { Bifrost::Assets::Shading::Rho::sample_GGX_with_fresnel(abs_cos_theta, roughness),
                 Bifrost::Assets::Shading::Rho::sample_GGX(abs_cos_theta, roughness) };
#endif
    }
};

__inline_all__ static optix::float3 compute_specular_rho(optix::float3 specularity, float abs_cos_theta, float roughness) {
    return SpecularRho::fetch(abs_cos_theta, roughness).rho(specularity);
}

// ------------------------------------------------------------------------------------------------
// Helper to regularize the material by using a maximally allowed PDF and cos(theta) to estimate a minimally allowed GGX alpha / roughness.
// ------------------------------------------------------------------------------------------------
struct GGXMinimumRoughness {
    const static int angle_sample_count = 32;
    const static int max_PDF_sample_count = 32;

    __inline_all__ static float from_PDF(float abs_cos_theta, float max_PDF) {
#if GPU_DEVICE
        float encoded_PDF = encode_PDF(max_PDF);
        // Rescale uv to start and end at the center of the first and last pixel, as the center values represent the boundaries of the function.
        float2 uv = make_float2(optix::lerp(0.5f / max_PDF_sample_count, (max_PDF_sample_count - 0.5f) / max_PDF_sample_count, encoded_PDF),
                                optix::lerp(0.5f / angle_sample_count, (angle_sample_count - 0.5f) / angle_sample_count, abs_cos_theta));
        float min_alpha = tex2D(estimate_GGX_alpha_texture, uv.x, uv.y);
#else
        float min_alpha = Bifrost::Assets::Shading::Estimate_GGX_bounded_VNDF_alpha::estimate_alpha(abs_cos_theta, max_PDF);
#endif
        return BSDFs::GGX::roughness_from_alpha(min_alpha);
    }

    __inline_all__ static float encode_PDF(float pdf) {
        // Floats larger than 16777216 can't represent integers accurately, so we max at 16777216 - 1 to accurately represent the addition below.
        pdf = fminf(pdf, 16777215.0f);
        float non_linear_PDF = pdf / (1.0f + pdf);
        return (non_linear_PDF - 0.13f) / 0.87f;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODELS_UTILS_H_