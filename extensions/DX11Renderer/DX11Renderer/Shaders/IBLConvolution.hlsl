// IBL convolution.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <BSDFs/GGX.hlsl>
#include <RNG.hlsl>

float roughness_from_mip_width(uint mip_width, float rcp_smallest_width, float rcp_mipmap_count_minus_one) {
    return 1.0 - log2(mip_width * rcp_smallest_width) * rcp_mipmap_count_minus_one;
}

cbuffer constants : register(b0) {
    float c_rcp_mipmap_count_minus_one; // 1.0 / (mipmap_count - 1)
    float c_rcp_smallest_width; // 1.0 / width of the smallest mipmap
    uint c_max_sample_count;
    float __padding;
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
    float roughness = roughness_from_mip_width(mip_width, c_rcp_smallest_width, c_rcp_mipmap_count_minus_one);
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
// Future work:
// * Second-Order Approximation for Variance Reduction in Multiple Importance Sampling, Lu et al., 2013.
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
    float roughness = roughness_from_mip_width(mip_width, c_rcp_smallest_width, c_rcp_mipmap_count_minus_one);
    float alpha = BSDFs::GGX::alpha_from_roughness(roughness);

    // Convolute.
    unsigned int half_sample_count = c_max_sample_count / 2;
    float3 radiance = float3(0.0, 0.0, 0.0);

    uint light_sample_count, light_sample_stride;
    light_samples.GetDimensions(light_sample_count, light_sample_stride);
    uint light_index_offset = 0;
    for (unsigned int l = 0; l < half_sample_count; ++l) {
        LightSample light_sample = light_samples[(l + light_index_offset) % light_sample_count];

        float cos_theta = dot(light_sample.direction_to_light, up_vector);

        // NaN resilient check.
        // Unfortunately it is possible to draw light samples with a PDF of zero. Floating point rounding error is a pain.
        if (!(cos_theta > 0.00001f) || !(light_sample.PDF > 0.000000001f))
            continue;

        float ggx_f = BSDFs::GGX::D(alpha, cos_theta);
        float ggx_PDF = ggx_f * cos_theta; // Inlining GGX::PDF(alpha, cos_theta)

        float mis_weight = RNG::power_heuristic(light_sample.PDF, ggx_PDF);
        radiance += light_sample.radiance * (mis_weight * ggx_f * cos_theta / light_sample.PDF);
    }

    // If using the same seed pr pixel causes firesplotches, then try owen scrambling or Cranley patterson rotation.
    float3x3 up_TBN = create_inverse_TBN(up_vector);
    for (unsigned int s = 0; s < half_sample_count; ++s) {
        float4 bsdf_sample = BSDFs::GGX::sample(alpha, RNG::sample02(s));
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