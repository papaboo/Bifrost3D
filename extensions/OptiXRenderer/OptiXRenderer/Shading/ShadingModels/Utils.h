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

    // Compensate for lost energy due to multiple scattering.
    // Multiple-scattering microfacet BSDFs with the smith model, Heitz et al., 2016 and 
    // Practical multiple scattering compensation for microfacet models, Emmanuel Turquin, 2018
    // showed that multiple-scattered reflectance has roughly the same distribution as single-scattering reflectance.
    // We can therefore account for energy lost to multi-scattering by computing the ratio of lost energy of a fully specular material,
    // and then scaling the specular reflectance by that ratio during evaluation, which increases reflectance to account for energy lost.
    __inline_all__ float energy_loss_adjustment() const { return 1.0f / full; }

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

// ------------------------------------------------------------------------------------------------
// Helper to regularize the material by using a maximally allowed PDF and cos(theta) to estimate a minimally allowed GGX alpha / roughness.
// ------------------------------------------------------------------------------------------------
struct GGXMinimumRoughness {
    const static int angle_sample_count = 32;
    const static int max_PDF_sample_count = 32;

    __inline_all__ static float from_PDF(float abs_cos_theta, PDF max_PDF) {
        if (max_PDF.is_delta_dirac())
            return 0.0f;

#if GPU_DEVICE
        float encoded_PDF = encode_PDF(max_PDF.value());
        // Rescale uv to start and end at the center of the first and last pixel, as the center values represent the boundaries of the function.
        float2 uv = make_float2(optix::lerp(0.5f / max_PDF_sample_count, (max_PDF_sample_count - 0.5f) / max_PDF_sample_count, encoded_PDF),
                                optix::lerp(0.5f / angle_sample_count, (angle_sample_count - 0.5f) / angle_sample_count, abs_cos_theta));
        float min_alpha = tex2D(estimate_GGX_alpha_texture, uv.x, uv.y);
#else
        float min_alpha = Bifrost::Assets::Shading::Estimate_GGX_bounded_VNDF_alpha::estimate_alpha(abs_cos_theta, max_PDF.value());
#endif
        return BSDFs::GGX::roughness_from_alpha(min_alpha);
    }

    __inline_all__ static float encode_PDF(float pdf) {
        float non_linear_PDF = pdf / (1.0f + pdf);
        float encoded_PDF = (non_linear_PDF - 0.13f) / 0.87f;
        encoded_PDF = fminf(1.0f, encoded_PDF); // Clamps NaNs to 1
        return encoded_PDF;
    }
};

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODELS_UTILS_H_