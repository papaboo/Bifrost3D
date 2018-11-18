// BRDFs for fitting.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _FIT_SPTD_BRDF_H_
#define _FIT_SPTD_BRDF_H_

#include <OptiXrenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Utils.h>

#include <cstdio>

using namespace optix;
using namespace OptiXRenderer;

namespace BRDF {

class Lambert {
public:

    BSDFResponse eval(const float3& wo, const float3& wi, float alpha) const {
        if (wo.z <= 0)
            return BSDFResponse::none();
        
        float pdf = wi.z * RECIP_PIf;
        return{ make_float3(pdf, pdf, pdf), pdf };
    }

    float3 sample_direction(const float3& wo, float alpha, float U1, float U2) const {
        float phi = TWO_PIf * U1;
        float r2 = U2;
        float r = sqrt(1.0f - r2);
        float z = sqrt(r2);
        return make_float3(cos(phi) * r, sin(phi) * r, z);
    }

    BSDFSample sample(const float3& wo, float alpha, float U1, float U2) const {
        BSDFSample result;
        result.direction = sample_direction(wo, alpha, U1, U2);
        BSDFResponse response = eval(wo, result.direction, alpha);
        result.weight = response.weight;
        result.PDF = response.PDF;
        return result;
    }
};

class GGX {
public:
    BSDFResponse eval(const float3& wo, const float3& wi, float alpha) const {
        if (wo.z <= 0)
            return BSDFResponse::none();

        float3 halfway = normalize(wo + wi);
        auto f = Shading::BSDFs::GGX::evaluate_with_PDF(make_float3(1,1,1), alpha, 1, wo, wi, halfway);
        f.weight *= wi.z; // eval scaled by cos theta
        return f;
    }

    BSDFSample sample(const float3& wo, float alpha, float U1, float U2) const {
        auto brdf_sample = Shading::BSDFs::GGX::sample(make_float3(1.0f, 1.0f, 1.0f), alpha, 1, wo, make_float2(U1, U2));
        brdf_sample.weight *= brdf_sample.direction.z; // eval scaled by cos theta
        return brdf_sample;
    }
};

class FastGGX {
public:

    BSDFResponse eval(const float3& wo, const float3& wi, float alpha) const {
        if (wo.z <= 0)
            return BSDFResponse::none();

        float G2 = wi.z <= 0.0f ? 0.0f : Shading::BSDFs::GGX::height_correlated_smith_G(alpha, wo, wi);

        // D
        const float3 H = normalize(wo + wi);
        const float slopex = H.x / H.z;
        const float slopey = H.y / H.z;
        float D = 1.0f / (1.0f + (slopex*slopex + slopey*slopey) / (alpha * alpha));
        D = D * D;
        D = D / (PIf * alpha * alpha * H.z*H.z*H.z*H.z);

        float pdf = fabsf(D * H.z / (4.0f * dot(wo, H)));
        float f = D * G2 / (4.0f * wo.z);
        return { make_float3(f, f, f), pdf };
    }

    float3 sample_direction(const float3& wo, float alpha, float U1, float U2) const {
        const float phi = TWO_PIf * U1;
        const float r = alpha * sqrtf(U2 / (1.0f - U2));
        const float3 halfway = normalize(make_float3(r * cosf(phi), r * sinf(phi), 1.0f));
        return reflect(-wo, halfway);
    }

    BSDFSample sample(const float3& wo, float alpha, float U1, float U2) const {
        BSDFSample result;
        result.direction = sample_direction(wo, alpha, U1, U2);
        BSDFResponse response = eval(wo, result.direction, alpha);
        result.weight = response.weight;
        result.PDF = response.PDF;
        return result;
    }
};

} // NS BRDF

#endif // _FIT_SPTD_BRDF_H_