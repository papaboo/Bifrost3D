// IBL convolution.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <BSDFs/GGX.hlsl>

namespace RNG {

    float van_der_corput(uint n, uint scramble) {
        n = reversebits(n) ^ scramble;
        return float((n >> 8) & 0xffffff) / float(1 << 24);
    }

    float sobol2(uint n, uint scramble) {
        for (uint v = 1u << 31u; n != 0; n >>= 1u, v ^= v >> 1u)
            if (n & 0x1)
                scramble ^= v;

        return float((scramble >> 8) & 0xffffff) / float(1 << 24);
    }

    float2 sample02(uint n, uint2 scramble = uint2(5569, 95597)) {
        return float2(van_der_corput(n, scramble.x), sobol2(n, scramble.y));
    }

    // Computes the power heuristic of pdf1 and pdf2.
    // It is assumed that pdf1 is always valid, i.e. not NaN.
    // pdf2 is allowed to be NaN, but generally try to avoid it. :)
    float power_heuristic(float pdf1, float pdf2) {
        pdf1 *= pdf1;
        pdf2 *= pdf2;
        float result = pdf1 / (pdf1 + pdf2);
        // This is where floating point math gets tricky!
        // If the mis weight is NaN then it can be caused by three things.
        // 1. pdf1 is so insanely high that pdf1 * pdf1 = infinity. In that case we end up with inf / (inf + pdf2^2) and return 1, unless pdf2 was larger than pdf1, i.e. 'more infinite :p', then we return 0.
        // 2. Conversely pdf2 can also be so insanely high that pdf2 * pdf2 = infinity. This is handled analogously to above.
        // 3. pdf2 can also be NaN. In this case the power heuristic is ill-defined and we return 0.
        return !isnan(result) ? result : (pdf1 > pdf2 ? 1.0f : 0.0f);
    }
}

float rougness_from_miplevel(uint base_width, uint mip_width, float rcp_mipmap_count) {
    uint size_factor = base_width / mip_width;
    uint mip_index = firstbithigh(size_factor);
    return mip_index * rcp_mipmap_count;
}

cbuffer constants : register(b0) {
    float c_rcp_mipmap_count;
    uint c_base_width;
    uint c_base_height;
    uint c_max_sample_count;
};

Texture2D environment_map : register(t0);
SamplerState env_sampler : register(s0);

RWTexture2D<float3> target_mip_level : register(u0);

// ------------------------------------------------------------------------------------------------
// GGX importance sampling IBL convolution.
// ------------------------------------------------------------------------------------------------
[numthreads(16, 16, 1)]
void GGX_convolute(uint3 thread_ID : SV_DispatchThreadID) {

    uint mip_width, mip_height;
    target_mip_level.GetDimensions(mip_width, mip_height);

    if (mip_width <= thread_ID.x || mip_height <= thread_ID.y)
        return;

    float2 up_uv = float2((thread_ID.x + 0.5f) / mip_width, 
                          (thread_ID.y + 0.5f) / mip_height);
    float3 up_vector = latlong_texcoord_to_direction(up_uv);
    float3x3 up_rotation = create_inverse_TBN(up_vector);

    // Compute roughness and alpha from current miplevel and max miplevel.
    float roughness = rougness_from_miplevel(c_base_width, mip_width, c_rcp_mipmap_count);
    float alpha = BSDFs::GGX::alpha_from_roughness(roughness);

    // Convolute.
    uint2 rng_scramble = reversebits(thread_ID.xy);
    float3 radiance = float3(0.0, 0.0, 0.0);
    for (unsigned int i = 0; i < c_max_sample_count; ++i) {
        float2 sample_uv = RNG::sample02(i, rng_scramble);
        float4 ggx_sample = BSDFs::GGX::sample(alpha, sample_uv);
        float2 bsdf_sample_uv = direction_to_latlong_texcoord(mul(up_rotation, ggx_sample.xyz));
        radiance += environment_map.SampleLevel(env_sampler, bsdf_sample_uv, 0).rgb;
    }

    target_mip_level[thread_ID.xy] = radiance / c_max_sample_count;
}

// ------------------------------------------------------------------------------------------------
// Multiple importance sampling IBL convolution using precompuated light samples.
// ------------------------------------------------------------------------------------------------
struct LightSample {
    float3 radiance;
    float PDF;
    float3 direction_to_light;
    float distance;
};

Texture2D per_pixel_PDF : register(t1);
StructuredBuffer<LightSample> light_samples : register(t2);

[numthreads(16, 16, 1)]
void MIS_convolute(uint3 thread_ID : SV_DispatchThreadID) {

    uint mip_width, mip_height;
    target_mip_level.GetDimensions(mip_width, mip_height);

    if (mip_width <= thread_ID.x || mip_height <= thread_ID.y)
        return;

    float2 up_uv = float2((thread_ID.x + 0.5f) / mip_width,
                          (thread_ID.y + 0.5f) / mip_height);
    float3 up_vector = latlong_texcoord_to_direction(up_uv);

    // Compute roughness and alpha from current miplevel and max miplevel.
    float roughness = rougness_from_miplevel(c_base_width, mip_width, c_rcp_mipmap_count);
    float alpha = BSDFs::GGX::alpha_from_roughness(roughness);

    // Convolute.
    unsigned int half_sample_count = c_max_sample_count / 2;
    float3 radiance = float3(0.0, 0.0, 0.0);

    for (unsigned int l = 0; l < half_sample_count; ++l) {
        LightSample light_sample = light_samples[l]; // TODO Offset by hash.

        float cos_theta = max(dot(light_sample.direction_to_light, up_vector), 0.0f);
        float ggx_f = BSDFs::GGX::D(alpha, cos_theta);
        float ggx_PDF = ggx_f * cos_theta; // Inlining GGX::PDF(alpha, cos_theta)
        if (!(ggx_PDF > 0.000000001f)) // NaN resilient check.
            continue;

        float mis_weight = RNG::power_heuristic(light_sample.PDF, ggx_PDF);
        radiance += light_sample.radiance * (mis_weight * ggx_f * cos_theta / light_sample.PDF);
    }

    uint2 rng_scramble = reversebits(thread_ID.xy);
    float3x3 up_TBN = create_inverse_TBN(up_vector);
    for (unsigned int s = 0; s < half_sample_count; ++s) {
        float4 bsdf_sample = BSDFs::GGX::sample(alpha, RNG::sample02(s, rng_scramble));
        if (!(bsdf_sample.w > 0.000000001f)) // NaN resilient check.
            continue;

        bsdf_sample.xyz = normalize(mul(up_TBN, bsdf_sample.xyz));
        float2 bsdf_sample_uv = direction_to_latlong_texcoord(bsdf_sample.xyz);
        float sin_theta = abs(sqrt(1.0f - bsdf_sample.y * bsdf_sample.y));
        float environment_PDF = per_pixel_PDF.SampleLevel(env_sampler, bsdf_sample_uv, 0).r * sin_theta;
        float mis_weight = RNG::power_heuristic(bsdf_sample.w, environment_PDF);
        radiance += environment_map.SampleLevel(env_sampler, bsdf_sample_uv, 0).rgb * mis_weight;
    }

    target_mip_level[thread_ID.xy] = radiance / half_sample_count;
}