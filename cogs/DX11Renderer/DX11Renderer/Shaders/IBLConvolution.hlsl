// IBL convolution.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <BSDFs/GGX.hlsl>

namespace RNG
{
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
}

cbuffer constants : register(b0) {
    uint mipmap_count;
    uint base_width;
    uint base_height;
    uint max_sample_count;
};

Texture2D env_tex : register(t0);
SamplerState env_sampler : register(s0);

RWTexture2D<float3> target_mip_level : register(u0);

// ------------------------------------------------------------------------------------------------
// Multiple importance sampling IBL convolution.
// ------------------------------------------------------------------------------------------------
[numthreads(16, 16, 1)]
void convolute(uint3 thread_ID : SV_DispatchThreadID) {

    uint mip_width, mip_height;
    target_mip_level.GetDimensions(mip_width, mip_height);

    if (mip_width <= thread_ID.x || mip_height <= thread_ID.y)
        return;

    float2 up_uv = float2((thread_ID.x + 0.5f) / mip_width, 
                          (thread_ID.y + 0.5f) / mip_height);
    float3 up_vector = latlong_texcoord_to_direction(up_uv);
    float3x3 inverse_up_rotation = transpose(create_tbn(up_vector));

    // Compute roughness and alpha from current miplevel and max miplevel.
    uint size_factor = base_width / mip_width;
    uint mip_index = firstbithigh(size_factor);
    float roughness = mip_index / (float)mipmap_count;
    float alpha = BSDFs::GGX::alpha_from_roughness(roughness);

    // Convolute.
    uint2 rng_scramble = reversebits(thread_ID.xy);
    float3 radiance = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < max_sample_count; ++i) {
        float2 sample_uv = RNG::sample02(i, rng_scramble);
        float4 ggx_sample = BSDFs::GGX::sample(alpha, sample_uv);
        float2 env_uv = direction_to_latlong_texcoord(mul(inverse_up_rotation, ggx_sample.xyz));
		radiance += env_tex.SampleLevel(env_sampler, env_uv, 0).rgb;
    }

    target_mip_level[thread_ID.xy] = radiance / max_sample_count;
}