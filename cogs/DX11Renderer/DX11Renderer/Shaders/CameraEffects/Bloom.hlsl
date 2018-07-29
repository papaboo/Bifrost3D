// Bloom shaders.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Utils.hlsl"

#pragma warning (disable: 3556) // Disable warning about dividing by integers is slow. It's a constant!!

namespace CameraEffects {

Texture2D image : register(t0);
RWTexture2D<float4> output_image : register(u0);

// ------------------------------------------------------------------------------------------------
// Gaussian bloom.
// ------------------------------------------------------------------------------------------------

Buffer<float2> bilinear_gaussian_samples : register (t1);

[numthreads(32, 32, 1)]
void sampled_gaussian_horizontal_filter(uint3 global_thread_ID : SV_DispatchThreadID) {
    float width, height;
    image.GetDimensions(width, height);
    float recip_width = 1.0 / width;

    float2 uv = (global_thread_ID.xy + input_viewport.xy + 0.5) / float2(width, height);

    int sample_count = bloom_bandwidth / 2;

    // NOTE Since the input buffer can be larger than the output buffer we should ideally clamp the uvs to the edge of the output buffer.
    // But, for now the input buffer pixels outside the viewport is either less than bloom_threshold or has proper pixel values, so we let the sampler 'go ham'.
    float3 sum = 0.0;
    for (int y = 0; y < sample_count; ++y) {
        float2 offset_weight = bilinear_gaussian_samples[y];
        offset_weight.x *= recip_width;
        float3 lower_sample = image.SampleLevel(bilinear_sampler, uv - float2(offset_weight.x, 0), 0).rgb;
        float3 upper_sample = image.SampleLevel(bilinear_sampler, uv + float2(offset_weight.x, 0), 0).rgb;
        sum += (max(0, lower_sample - bloom_threshold) + max(0, upper_sample - bloom_threshold)) * offset_weight.y;
    }

    output_image[global_thread_ID.xy] = float4(sum, 1.0);
}

[numthreads(32, 32, 1)]
void sampled_gaussian_vertical_filter(uint3 global_thread_ID : SV_DispatchThreadID) {
    float width, height;
    image.GetDimensions(width, height);
    float recip_height = 1.0 / height;

    float2 uv = (global_thread_ID.xy + 0.5) / float2(width, height);

    int sample_count = bloom_bandwidth / 2;

    float3 sum = 0.0;
    for (int y = 0; y < sample_count; ++y) {
        float2 offset_weight = bilinear_gaussian_samples[y];
        offset_weight.x *= recip_height;
        sum += (image.SampleLevel(bilinear_sampler, uv + float2(0, offset_weight.x), 0).rgb +
                image.SampleLevel(bilinear_sampler, uv + float2(0, -offset_weight.x), 0).rgb) * offset_weight.y;
    }

    output_image[global_thread_ID.xy] = float4(sum, 1.0);
}

// ------------------------------------------------------------------------------------------------
// Dual kawase filtering
// https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-26-50/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf
// ------------------------------------------------------------------------------------------------

[numthreads(32, 32, 1)]
void extract_high_intensity(uint3 global_thread_ID : SV_DispatchThreadID) {
    float4 pixel = image[global_thread_ID.xy + input_viewport.xy];
    output_image[global_thread_ID.xy] = float4(max(0.0, pixel.rgb - bloom_threshold), pixel.a);
}

[numthreads(32, 32, 1)]
void dual_kawase_downsample(uint3 global_thread_ID : SV_DispatchThreadID) {
    float width, height;
    output_image.GetDimensions(width, height);

    const float2 half_pixel_width = 0.5 * rcp(float2(width, height));
    float2 uv = global_thread_ID.xy * rcp(float2(width, height)) + half_pixel_width;

    float4 sum = image.SampleLevel(bilinear_sampler, uv, 0) * 4.0;
    sum += image.SampleLevel(bilinear_sampler, uv + float2( half_pixel_width.x,  half_pixel_width.y), 0);
    sum += image.SampleLevel(bilinear_sampler, uv + float2( half_pixel_width.x, -half_pixel_width.y), 0);
    sum += image.SampleLevel(bilinear_sampler, uv + float2(-half_pixel_width.x,  half_pixel_width.y), 0);
    sum += image.SampleLevel(bilinear_sampler, uv + float2(-half_pixel_width.x, -half_pixel_width.y), 0);

    output_image[global_thread_ID.xy] = sum / 8.0;
}

[numthreads(32, 32, 1)]
void dual_kawase_upsample(uint3 global_thread_ID : SV_DispatchThreadID) {
    float width, height;
    output_image.GetDimensions(width, height);

    const float2 half_pixel_width = 0.5 * rcp(float2(width, height));
    const float2 pixel_width = half_pixel_width * 2.0;;

    float2 uv = global_thread_ID.xy * rcp(float2(width, height)) + half_pixel_width;

    float4 sum = image.SampleLevel(bilinear_sampler, uv + float2(-half_pixel_width.x * 2.0, 0.0), 0);
    sum += image.SampleLevel(bilinear_sampler, uv + float2(-half_pixel_width.x, half_pixel_width.y), 0) * 2.0;
    sum += image.SampleLevel(bilinear_sampler, uv + float2(0.0, half_pixel_width.y * 2.0), 0);
    sum += image.SampleLevel(bilinear_sampler, uv + float2(half_pixel_width.x, half_pixel_width.y), 0) * 2.0;
    sum += image.SampleLevel(bilinear_sampler, uv + float2(half_pixel_width.x * 2.0, 0.0), 0);
    sum += image.SampleLevel(bilinear_sampler, uv + float2(half_pixel_width.x, -half_pixel_width.y), 0) * 2.0;
    sum += image.SampleLevel(bilinear_sampler, uv + float2(0.0, -half_pixel_width.y * 2.0), 0);
    sum += image.SampleLevel(bilinear_sampler, uv + float2(-half_pixel_width.x, -half_pixel_width.y), 0) * 2.0;

    output_image[global_thread_ID.xy] = sum / 12.0;
}

} // NS CameraEffects