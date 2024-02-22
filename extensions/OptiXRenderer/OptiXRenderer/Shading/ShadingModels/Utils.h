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
    const static int angle_sample_count = 64;
    const static int roughness_sample_count = 64;

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

} // NS ShadingModels
} // NS Shading
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODELS_UTILS_H_