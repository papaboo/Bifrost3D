// Fits for spherical pivot transformed distributions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SPTD_FIT_H_
#define _OPTIXRENDERER_SPTD_FIT_H_

#include <OptiXRenderer/Utils.h>
#include <OptiXRenderer/Distributions.h>

#include <optixu/optixpp_namespace.h>
#undef RGB

namespace OptiXRenderer {
namespace SPTD {

// ------------------------------------------------------------------------------------------------
// Fitted Spherical pivot.
// TODO Parameterize by distribution, fx uniform or cosine.
// ------------------------------------------------------------------------------------------------
struct Pivot {

    // lobe amplitude
    float amplitude;

    // parameterization 
    float distance;
    float theta;

    // pivot position
    inline optix::float3 position() const { return distance * optix::make_float3(sinf(theta), 0.0f, cosf(theta)); }

    float eval(const optix::float3& wi) const {
        optix::float3 xi = position();
        float num = 1.0f - dot(xi, xi);
        optix::float3 tmp = wi - xi;
        float den = dot(tmp, tmp);
        float p = num / den;
        float jacobian = p * p;
        float pdf = jacobian / (4.0f * PIf);
        return amplitude * pdf;
    }

    optix::float3 sample(const float U1, const float U2) const {
        const float sphere_theta = acosf(-1.0f + 2.0f * U1);
        const float sphere_phi = 2.0f * 3.14159f * U2;
        const optix::float3 sphere_sample = optix::make_float3(sinf(sphere_theta) * cosf(sphere_phi), sinf(sphere_theta) * sinf(sphere_phi), -cosf(sphere_theta));
        return OptiXRenderer::Distributions::SPTD::pivot_transform(sphere_sample, position());
    }

    /*
    float max_value() {
        float res = 0.0f;
        for (float U2 = 0.0f; U2 <= 1.0f; U2 += 0.05f)
            for (float U1 = 0.0f; U1 <= 1.0f; U1 += 0.05f) {
                float3 L = sample(U1, U2);
                float value = eval(L) / amplitude;
                res = std::max(value, res);
            }
        return res;
    }
    */
};

optix::float4 GGX_fit_lookup(float cos_theta, float ggx_alpha);

optix::TextureSampler GGX_fit_texture(optix::Context& context);

} // NS SPTD
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SPTD_FIT_H_