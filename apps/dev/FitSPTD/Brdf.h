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

using namespace optix;
using namespace OptiXRenderer;

namespace BRDF {

class Lambert {
public:

    float eval(const float3& wo, const float3& wi, float alpha, float& pdf) const {
        if (wo.z <= 0) {
            pdf = 0;
            return 0;
        }

        pdf = wi.z * RECIP_PIf;
        return wi.z * RECIP_PIf; // eval scaled by cos theta
    }

    float3 sample(const float3& wo, float alpha, float U1, float U2) const {
        float phi = TWO_PIf * U1;
        float r2 = U2;
        float r = sqrt(1.0f - r2);
        float z = sqrt(r2);
        return make_float3(cos(phi) * r, sin(phi) * r, z);
    }
};

class GGX2 {
public:
    float eval(const float3& wo, const float3& wi, float alpha, float& pdf) const {
        if (wo.z <= 0) {
            pdf = 0;
            return 0;
        }

        float3 halfway = normalize(wo + wi);
        auto f = OptiXRenderer::Shading::BSDFs::GGXWithVNDF::evaluate_with_PDF(make_float3(1,1,1), alpha, wo, wi, halfway);
        pdf = f.PDF;
        return f.weight.x * wi.z; // eval scaled by cos theta
    }

    float3 sample(const float3& wo, float alpha, float U1, float U2) const {

        auto sample = OptiXRenderer::Shading::BSDFs::GGXWithVNDF::sample(make_float3(1.0f, 1.0f, 1.0f), alpha, wo, make_float2(U1, U2));
        return sample.direction;
    }
};

class GGX {
public:

    float eval(const float3& wo, const float3& wi, float alpha, float& pdf) const {
        if (wo.z <= 0) {
            pdf = 0;
            return 0;
        }

        // masking
        const float a_V = 1.0f / alpha / tanf(acosf(wo.z));
        const float LambdaV = (wo.z < 1.0f) ? 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a_V / a_V)) : 0.0f;

        // shadowing
        float G2;
        if (wi.z <= 0.0f)
            G2 = 0;
        else {
            const float a_L = 1.0f / alpha / tanf(acosf(wi.z));
            const float LambdaL = (wi.z < 1.0f) ? 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a_L / a_L)) : 0.0f;
            G2 = 1.0f / (1.0f + LambdaV + LambdaL);
        }

        // D
        const float3 H = normalize(wo + wi);
        const float slopex = H.x / H.z;
        const float slopey = H.y / H.z;
        float D = 1.0f / (1.0f + (slopex*slopex + slopey*slopey) / alpha / alpha);
        D = D * D;
        D = D / (PIf * alpha * alpha * H.z*H.z*H.z*H.z);

        pdf = fabsf(D * H.z / 4.0f / dot(wo, H));
        return D * G2 / 4.0f / wo.z;
    }

    float3 sample(const float3& wo, float alpha, float U1, float U2) const {
        const float phi = TWO_PIf * U1;
        const float r = alpha * sqrtf(U2 / (1.0f - U2));
        const float3 halfway = normalize(make_float3(r * cosf(phi), r * sinf(phi), 1.0f));
        return reflect(-wo, halfway);
    }

};

} // NS BRDF

#endif // _FIT_SPTD_BRDF_H_