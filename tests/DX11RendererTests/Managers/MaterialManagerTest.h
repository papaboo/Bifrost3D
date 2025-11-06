// DX11Renderer material manager test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_MATERIAL_MANAGER_TEST_H_
#define _DX11RENDERER_MATERIAL_MANAGER_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <Bifrost/Assets/Shading/Fittings.h>

#include <DX11Renderer/Managers/MaterialManager.h>
#include <DX11Renderer/Managers/ShaderManager.h>
#include <DX11Renderer/Managers/TextureManager.h>

namespace DX11Renderer::Managers {

// ------------------------------------------------------------------------------------------------
// Ensure that the hardcoded GGX rho texture dimensions used to adjust the sample coordinates 
// on the GPU match the ones on the CPU.
// ------------------------------------------------------------------------------------------------
GTEST_TEST(MaterialManager, GPU_GGX_rho_texture_dimensions_consistent_with_CPU) {
    using namespace Bifrost::Assets::Shading;

    // Setup GPU
    auto device = create_test_device();
    auto context = get_immidiate_context1(device);

    // Create compute shader to return the texture dimensions
    const char* rho_dimensions_cs =
        "#include <ShadingModels/Utils.hlsl>\n"
        "\n"
        "RWStructuredBuffer<int> rho_dimensions : register(u0);\n"
        "\n"
        "[numthreads(1, 1, 1)]\n"
        "void rho_dimensions_cs(uint2 threadIndex : SV_GroupThreadID) {\n"
        "   rho_dimensions[0] = ShadingModels::SpecularRho::angle_sample_count;\n"
        "   rho_dimensions[1] = ShadingModels::SpecularRho::roughness_sample_count;\n"
        "}\n";
    OComputeShader rho_dimensions_shader;
    OBlob rho_dimensions_blob = ShaderManager().compile_shader_source(rho_dimensions_cs, "cs_5_0", "rho_dimensions_cs");
    THROW_DX11_ERROR(device->CreateComputeShader(UNPACK_BLOB_ARGS(rho_dimensions_blob), nullptr, &rho_dimensions_shader));

    OUnorderedAccessView rho_dimensions_UAV;
    OBuffer rho_dimensions_buffer = create_default_buffer(device, DXGI_FORMAT_R32_UINT, nullptr, 2, nullptr, &rho_dimensions_UAV);
    context->CSSetUnorderedAccessViews(0, 1, &rho_dimensions_UAV, nullptr);

    context->CSSetShader(rho_dimensions_shader, nullptr, 0);
    context->Dispatch(1, 1, 1);

    int gpu_rho_dimensions[2];
    Readback::buffer(device, context, rho_dimensions_buffer, gpu_rho_dimensions, gpu_rho_dimensions + 2);

    EXPECT_EQ(gpu_rho_dimensions[0], Rho::GGX_angle_sample_count);
    EXPECT_EQ(gpu_rho_dimensions[1], Rho::GGX_roughness_sample_count);
}

// ------------------------------------------------------------------------------------------------
// Test that sampling the precomputed GGX rho texture results in the right values.
// Essentially this tests that the GPU UV offsets are consistent with the CPU UV offsets.
// ------------------------------------------------------------------------------------------------
GTEST_TEST(MaterialManager, sample_GPU_GGX_rho_texture_consistent_with_CPU) {
    using namespace Bifrost::Assets::Shading;

    // Setup GPU
    auto device = create_test_device();
    auto context = get_immidiate_context1(device);

    // Upload rho texture and bilinear sampler
    auto GGX_with_fresnel_rho_srv = MaterialManager::create_GGX_with_fresnel_rho_srv(device);
    context->CSSetShaderResources(15, 1, &GGX_with_fresnel_rho_srv);
    auto bilinear_sampler = TextureManager::create_clamped_linear_sampler(device);
    context->CSSetSamplers(15, 1, &bilinear_sampler);

    // Create compute shader to sample the rho texture
    const char* sample_rho_cs =
        "#include <ShadingModels/Utils.hlsl>\n"
        "\n"
        "RWStructuredBuffer<float2> rho_samples : register(u0);\n"
        "static const int samples_per_dimension = 4;\n"
        "\n"
        "[numthreads(samples_per_dimension, samples_per_dimension, 1)]\n"
        "void sample_rho_cs(uint2 threadIndex : SV_GroupThreadID) {\n"
        "   float cos_theta = threadIndex.x / (samples_per_dimension - 1.0f);\n"
        "   float roughness = threadIndex.y / (samples_per_dimension - 1.0f);\n"
        "   uint linear_index = threadIndex.x + threadIndex.y * samples_per_dimension;\n"
        "   ShadingModels::SpecularRho specular_rho = ShadingModels::SpecularRho::fetch(cos_theta, roughness);\n"
        "   rho_samples[linear_index] = float2(specular_rho.base, specular_rho.full);\n"
        "}\n";
    OComputeShader sample_rho_shader;
    OBlob sample_rho_blob = ShaderManager().compile_shader_source(sample_rho_cs, "cs_5_0", "sample_rho_cs");
    THROW_DX11_ERROR(device->CreateComputeShader(UNPACK_BLOB_ARGS(sample_rho_blob), nullptr, &sample_rho_shader));

    // Sample the rho texture on the GPU and readback the samples
    const int samples_per_dimension = 4;
    const int sample_count = samples_per_dimension * samples_per_dimension;
    OUnorderedAccessView buffer_UAV;
    OBuffer gpu_buffer = create_default_buffer(device, DXGI_FORMAT_R32G32_FLOAT, nullptr, sample_count, nullptr, &buffer_UAV);
    context->CSSetUnorderedAccessViews(0, 1, &buffer_UAV, nullptr);

    context->CSSetShader(sample_rho_shader, nullptr, 0);
    context->Dispatch(1, 1, 1);

    float2 gpu_rho_result[sample_count];
    Readback::buffer(device, context, gpu_buffer, gpu_rho_result, gpu_rho_result + sample_count);

    // Compare with the CPU version
    for (int y = 0; y < samples_per_dimension; y++) {
        float roughness = y / (samples_per_dimension - 1.0f);
        for (int x = 0; x < samples_per_dimension; x++) {
            float cos_theta = x / (samples_per_dimension - 1.0f);

            float actual_ggx_with_fresnel_rho = gpu_rho_result[x + y * samples_per_dimension].x;
            float actual_ggx_rho = gpu_rho_result[x + y * samples_per_dimension].y;

            float expected_ggx_rho = Rho::sample_GGX(cos_theta, roughness);
            float expected_ggx_with_fresnel_rho = Rho::sample_GGX_with_fresnel(cos_theta, roughness);

            EXPECT_FLOAT_EQ_EPS(actual_ggx_rho, expected_ggx_rho, 0.0001f) << "cos_theta: " << cos_theta << ", roughness: " << roughness;
            EXPECT_FLOAT_EQ_EPS(actual_ggx_with_fresnel_rho, expected_ggx_with_fresnel_rho, 0.0001f) << "cos_theta: " << cos_theta << ", roughness: " << roughness;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Ensure that the hardcoded dielectric GGX rho texture dimensions used to adjust the sample coordinates 
// on the GPU match the ones on the CPU.
// ------------------------------------------------------------------------------------------------
GTEST_TEST(MaterialManager, GPU_dielectric_GGX_rho_texture_dimensions_consistent_with_CPU) {
    using namespace Bifrost::Assets::Shading;

    // Setup GPU
    auto device = create_test_device();
    auto context = get_immidiate_context1(device);

    // Create compute shader to return the texture dimensions
    const char* rho_dimensions_cs =
        "#include <ShadingModels/Utils.hlsl>\n"
        "\n"
        "RWStructuredBuffer<int> rho_dimensions : register(u0);\n"
        "\n"
        "[numthreads(1, 1, 1)]\n"
        "void rho_dimensions_cs(uint2 threadIndex : SV_GroupThreadID) {\n"
        "   rho_dimensions[0] = ShadingModels::DielectricRho::angle_sample_count;\n"
        "   rho_dimensions[1] = ShadingModels::DielectricRho::roughness_sample_count;\n"
        "   rho_dimensions[2] = ShadingModels::DielectricRho::ior_i_over_o_sample_count;\n"
        "}\n";
    OComputeShader rho_dimensions_shader;
    OBlob rho_dimensions_blob = ShaderManager().compile_shader_source(rho_dimensions_cs, "cs_5_0", "rho_dimensions_cs");
    THROW_DX11_ERROR(device->CreateComputeShader(UNPACK_BLOB_ARGS(rho_dimensions_blob), nullptr, &rho_dimensions_shader));

    OUnorderedAccessView rho_dimensions_UAV;
    OBuffer rho_dimensions_buffer = create_default_buffer(device, DXGI_FORMAT_R32_UINT, nullptr, 3, nullptr, &rho_dimensions_UAV);
    context->CSSetUnorderedAccessViews(0, 1, &rho_dimensions_UAV, nullptr);

    context->CSSetShader(rho_dimensions_shader, nullptr, 0);
    context->Dispatch(1, 1, 1);

    int gpu_rho_dimensions[3];
    Readback::buffer(device, context, rho_dimensions_buffer, gpu_rho_dimensions, gpu_rho_dimensions + 3);

    EXPECT_EQ(gpu_rho_dimensions[0], Rho::dielectric_GGX_angle_sample_count);
    EXPECT_EQ(gpu_rho_dimensions[1], Rho::dielectric_GGX_roughness_sample_count);
    EXPECT_EQ(gpu_rho_dimensions[2], Rho::dielectric_GGX_ior_i_over_o_sample_count);
}

// ------------------------------------------------------------------------------------------------
// Test that sampling the precomputed dielectric GGX rho texture results in the right values.
// Essentially this tests that the GPU UV offsets are consistent with the CPU UV offsets.
// ------------------------------------------------------------------------------------------------
GTEST_TEST(MaterialManager, sample_GPU_dielectric_GGX_rho_texture_consistent_with_CPU) {
    using namespace Bifrost::Assets::Shading;

    // Setup GPU
    auto device = create_test_device();
    auto context = get_immidiate_context1(device);

    // Upload rho texture and bilinear sampler
    auto dielectric_GGX_rho_srv = MaterialManager::create_dielectric_GGX_srv(device);
    context->CSSetShaderResources(14, 1, &dielectric_GGX_rho_srv);
    auto bilinear_sampler = TextureManager::create_clamped_linear_sampler(device);
    context->CSSetSamplers(15, 1, &bilinear_sampler);

    // Create compute shader to sample the rho texture
    const char* sample_rho_cs =
        "#include <ShadingModels/Utils.hlsl>\n"
        "\n"
        "RWStructuredBuffer<float2> rho_samples : register(u0);\n"
        "static const int samples_per_dimension = 4;\n"
        "\n"
        "[numthreads(samples_per_dimension, samples_per_dimension, samples_per_dimension)]\n"
        "void sample_rho_cs(uint3 threadIndex : SV_GroupThreadID) {\n"
        "   float cos_theta = threadIndex.x / (samples_per_dimension - 1.0f);\n"
        "   float roughness = threadIndex.y / (samples_per_dimension - 1.0f);\n"
        "   float ior_i_over_o = 0.4 + 2.0 * threadIndex.z / (samples_per_dimension - 1.0f);\n" // ior_i_over_o samples in range [0.4, 2.4]
        "   uint linear_index = threadIndex.x + (threadIndex.y + threadIndex.z * samples_per_dimension) * samples_per_dimension;\n"
        "   ShadingModels::DielectricRho dielectric_rho = ShadingModels::DielectricRho::fetch(cos_theta, roughness, ior_i_over_o);\n"
        "   rho_samples[linear_index] = float2(dielectric_rho.total, dielectric_rho.reflected);\n"
        "}\n";
    OComputeShader sample_rho_shader;
    OBlob sample_rho_blob = ShaderManager().compile_shader_source(sample_rho_cs, "cs_5_0", "sample_rho_cs");
    THROW_DX11_ERROR(device->CreateComputeShader(UNPACK_BLOB_ARGS(sample_rho_blob), nullptr, &sample_rho_shader));

    // Sample the rho texture on the GPU and readback the samples
    const int samples_per_dimension = 4;
    const int sample_count = samples_per_dimension * samples_per_dimension * samples_per_dimension;
    OUnorderedAccessView buffer_UAV;
    OBuffer gpu_buffer = create_default_buffer(device, DXGI_FORMAT_R32G32_FLOAT, nullptr, sample_count, nullptr, &buffer_UAV);
    context->CSSetUnorderedAccessViews(0, 1, &buffer_UAV, nullptr);

    context->CSSetShader(sample_rho_shader, nullptr, 0);
    context->Dispatch(1, 1, 1);

    float2 gpu_rho_result[sample_count];
    Readback::buffer(device, context, gpu_buffer, gpu_rho_result, gpu_rho_result + sample_count);

    // Compare with the CPU version
    for (int z = 0; z < samples_per_dimension; z++) {
        float ior_i_over_o = 0.4f + 2.0f * z / (samples_per_dimension - 1.0f);
        for (int y = 0; y < samples_per_dimension; y++) {
            float roughness = y / (samples_per_dimension - 1.0f);
            for (int x = 0; x < samples_per_dimension; x++) {
                float cos_theta = x / (samples_per_dimension - 1.0f);

                auto cpu = Rho::sample_dielectric_GGX(cos_theta, roughness, ior_i_over_o);

                int linear_index = x + (y + z * samples_per_dimension) * samples_per_dimension;
                float gpu_total_rho = gpu_rho_result[linear_index].x;
                float gpu_reflection_rho = gpu_rho_result[linear_index].y;

                EXPECT_FLOAT_EQ_EPS(gpu_total_rho, cpu.total_rho, 0.0001f) << "cos_theta: " << cos_theta << ", roughness: " << roughness << ", ior_i_over_o: " << ior_i_over_o;
                EXPECT_FLOAT_EQ_EPS(gpu_reflection_rho, cpu.reflected_rho, 0.0001f) << "cos_theta: " << cos_theta << ", roughness: " << roughness << ", ior_i_over_o: " << ior_i_over_o;
            }
        }
    }
}

} // NS DX11Renderer::Managers

#endif // _DX11RENDERER_MATERIAL_MANAGER_TEST_H_