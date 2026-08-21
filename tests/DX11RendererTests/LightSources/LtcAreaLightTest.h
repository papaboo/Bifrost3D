// DX11Renderer LTC area light test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_LIGHT_SOURCES_LTC_AREA_LIGHT_TEST_H_
#define _DX11RENDERER_LIGHT_SOURCES_LTC_AREA_LIGHT_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <DX11Renderer/Managers/ShaderManager.h>

#include <Bifrost/Assets/Shading/LightSources/LtcAreaLight.h>
#include <Bifrost/Assets/Shading/LinearlyTransformedCosines.h>
#include <Bifrost/Math/LTC.h>
#include <Bifrost/Math/Triangle.h>

namespace DX11Renderer::LightSources {

// Sanity check that we don't accidentally transpose LTC matrices.
GTEST_TEST(LtcAreaLight, LTC_matrix_rows_are_equal_on_CPU_and_GPU) {
    using namespace Bifrost::Math;

    // Setup GPU
    auto device = create_test_device();
    auto context = get_immidiate_context1(device);
    
    // Create compute shader to return the LTC rows
    const char* extract_ltc_rows_cs =
        "#include <Utils.hlsl>\n"
        "\n"
        "RWStructuredBuffer<float4> ltc_rows : register(u0);\n"
        "\n"
        "[numthreads(1, 1, 1)]\n"
        "void extract_ltc_rows_cs() {\n"
        "    IsotropicLTC ltc = IsotropicLTC::from_M(1, 2, 3, 4, 5);\n"
        "    float3x3 M = ltc.get_M();\n"
        "    float3x3 inverse_M = ltc.get_inverse_M();\n"
        "    ltc_rows[0] = float4(M[0], 0);\n" // Indexing into an hlsl matrix returns the i'th row. https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math
        "    ltc_rows[1] = float4(M[1], 1);\n"
        "    ltc_rows[2] = float4(M[2], 2);\n"
        "    ltc_rows[3] = float4(inverse_M[0], 0);\n"
        "    ltc_rows[4] = float4(inverse_M[1], 1);\n"
        "    ltc_rows[5] = float4(inverse_M[2], 2);\n"
        "}\n";
    OComputeShader ltc_rows_shader;
    OBlob shader_blob = Managers::ShaderManager().compile_shader_source(extract_ltc_rows_cs, "cs_5_0", "extract_ltc_rows_cs");
    THROW_DX11_ERROR(device->CreateComputeShader(UNPACK_BLOB_ARGS(shader_blob), nullptr, &ltc_rows_shader));

    OUnorderedAccessView ltc_rows_UAV;
    OBuffer ltc_rows_buffer = create_default_buffer(device, DXGI_FORMAT_R32G32B32A32_FLOAT, nullptr, 6, nullptr, &ltc_rows_UAV);
    context->CSSetUnorderedAccessViews(0, 1, &ltc_rows_UAV, nullptr);

    context->CSSetShader(ltc_rows_shader, nullptr, 0);
    context->Dispatch(1, 1, 1);

    Vector4f gpu_ltc_rows[6];
    Readback::buffer(device, context, ltc_rows_buffer, gpu_ltc_rows, gpu_ltc_rows + 6);

    IsotropicLTC expected_ltc = IsotropicLTC::from_M(1, 2, 3, 4, 5);
    Matrix3x3f expected_M = expected_ltc.get_M();
    Matrix3x3f expected_inverse_M = expected_ltc.get_inverse_M();

    EXPECT_VECTOR4F_EQ(Vector4f(expected_M.get_row(0), 0), gpu_ltc_rows[0]);
    EXPECT_VECTOR4F_EQ(Vector4f(expected_M.get_row(1), 1), gpu_ltc_rows[1]);
    EXPECT_VECTOR4F_EQ(Vector4f(expected_M.get_row(2), 2), gpu_ltc_rows[2]);
    EXPECT_VECTOR4F_EQ(Vector4f(expected_inverse_M.get_row(0), 0), gpu_ltc_rows[3]);
    EXPECT_VECTOR4F_EQ(Vector4f(expected_inverse_M.get_row(1), 1), gpu_ltc_rows[4]);
    EXPECT_VECTOR4F_EQ(Vector4f(expected_inverse_M.get_row(2), 2), gpu_ltc_rows[5]);
}

GTEST_TEST(LtcAreaLight, Area_light_application_are_equal_on_CPU_and_GPU) {
    using namespace Bifrost::Assets::Shading;
    using namespace Bifrost::Math;

    // Setup scene
    Vector3f wo = normalize(Vector3f(1, -1, 1));
    float cos_theta = wo.z;

    Vector3f surface_point = Vector3f(0, 0, 0);
    Vector3f surface_normal = Vector3f(0, 0, 1);

    // Lights
    const int light_count = 2;
    Trianglef lights[light_count] = {
        Trianglef(Vector3f(-1, 1, 1), Vector3f(1, -1, 1), Vector3f(1, 1, 1)), // Triangle above surface
        Trianglef(Vector3f(1, 0, -2), Vector3f(1, -1, 1), Vector3f(1, 1, 1)), // Triangle at horizon
    };
    RGB emission = RGB(0.25f, 1, 4);
    bool two_sided = true;

    // LTCs
    float roughness = 0.5f;
    const int LTC_count = 3;
    IsotropicLTC LTCs[LTC_count] = {
        LTC::lambert_LTC_coefficients(),
        LTC::oren_nayar_LTC_coefficients(cos_theta, roughness),
        LTC::GGX_reflection_LTC_coefficients(cos_theta, roughness),
    };

    const int test_count = light_count * LTC_count;
    RGBA gpu_radiance[test_count];
    { // Apply on GPU
        // Setup GPU
        auto device = create_test_device();
        auto context = get_immidiate_context1(device);

        const std::string evaluate_ltc_area_light_cs =
            "#include <LtcAreaLight.hlsl>\n"
            "\n"
            "static const int light_count = " + std::to_string(light_count) + ";\n"
            "static const int LTC_count = " + std::to_string(LTC_count) + ";\n"
            "\n"
            "struct EmissiveTriangle { float3 positions[3]; };\n"
            "\n"
            "StructuredBuffer<EmissiveTriangle> lights : register(t0);\n"
            "StructuredBuffer<IsotropicLTC> LTCs : register(t1);\n"
            "RWStructuredBuffer<float4> radiance : register(u0);\n"
            "\n"
            "[numthreads(LTC_count, light_count, 1)]\n"
            "void evaluate_ltc_area_light_cs(uint2 thread_index : SV_GroupThreadID) {\n"
            "    IsotropicLTC bsdf = LTCs[thread_index.x];\n"
            "    EmissiveTriangle light = lights[thread_index.y];\n"
            "    float3 wo = normalize(float3(1, -1, 1));\n"
            "    float3 surface_point = float3(0, 0, 0);\n"
            "    float3 surface_normal = float3(0, 0, 1);\n"
            "    bool two_sided = true;\n"
            "    float3 emission = float3(0.25, 1, 4);\n"
            "    float3 emissions[3] = { emission, emission, emission };\n"
            "    float3 tinted_radiance = LtcAreaLight::evaluate_triangle_light(bsdf, wo, surface_point, surface_normal, light.positions, emissions, two_sided);\n"
            "    float base_radiance = LtcAreaLight::evaluate_triangle_light(bsdf, wo, surface_point, surface_normal, light.positions, two_sided);\n"
            "    radiance[thread_index.x + thread_index.y * LTC_count] = float4(tinted_radiance, base_radiance);\n"
            "}\n";

        OComputeShader ltc_evaluation_shader;
        OBlob shader_blob = Managers::ShaderManager().compile_shader_source(evaluate_ltc_area_light_cs.c_str(), "cs_5_0", "evaluate_ltc_area_light_cs");
        THROW_DX11_ERROR(device->CreateComputeShader(UNPACK_BLOB_ARGS(shader_blob), nullptr, &ltc_evaluation_shader));

        OShaderResourceView lights_SRV;
        create_structured_buffer(device, lights, light_count, &lights_SRV);
        context->CSSetShaderResources(0, 1, &lights_SRV);

        OShaderResourceView LTCs_SRV;
        create_structured_buffer(device, LTCs, LTC_count, &LTCs_SRV);
        context->CSSetShaderResources(1, 1, &LTCs_SRV);

        OUnorderedAccessView radiance_UAV;
        OBuffer radiance_buffer = create_default_buffer(device, DXGI_FORMAT_R32G32B32A32_FLOAT, nullptr, test_count, nullptr, &radiance_UAV);
        context->CSSetUnorderedAccessViews(0, 1, &radiance_UAV, nullptr);

        context->CSSetShader(ltc_evaluation_shader, nullptr, 0);
        context->Dispatch(1, 1, 1);

        Readback::buffer(device, context, radiance_buffer, gpu_radiance, gpu_radiance + test_count);
    }

    // Apply on CPU
    for (int y = 0; y < light_count; ++y)
        for (int x = 0; x < LTC_count; ++x) {
            // Compare equality of evaluating without emission
            float expected_radiance = Bifrost::Assets::Shading::LightSources::LtcAreaLight::evaluate_triangle_light(LTCs[x], wo, surface_point, surface_normal, &lights[y].v0, two_sided);
            float actual_radiance = gpu_radiance[x + y * LTC_count].a;
            EXPECT_FLOAT_EQ_PCT(expected_radiance, actual_radiance, 0.001f) << " combination [LTC: " << x << ", light: " << y << "]";

            // Compare equality of evaluating with white emission of intensity 1
            // Should give the same result as evaluating without emission
            RGB expected_radiance_with_emission = Bifrost::Assets::Shading::LightSources::LtcAreaLight::evaluate_triangle_light(LTCs[x], wo, surface_point, surface_normal, &lights[y].v0, emission, two_sided);
            RGB actual_radiance_with_emission = gpu_radiance[x + y * LTC_count].rgb();
            EXPECT_RGB_EQ_PCT(expected_radiance_with_emission, actual_radiance_with_emission, 0.001f) << " combination [LTC: " << x << ", light: " << y << "]";
        }
}

GTEST_TEST(LtcAreaLight, Surface_behind_light_not_lit) {
    auto device = create_test_device();
    auto context = get_immidiate_context1(device);

    const char* evaluate_ltc_area_light_cs =
        "#include <LtcAreaLight.hlsl>\n"
        "\n"
        "RWStructuredBuffer<float> radiance : register(u0);\n"
        "\n"
        "[numthreads(1, 1, 1)]\n"
        "void evaluate_ltc_area_light_cs(uint2 thread_index : SV_GroupThreadID) {\n"
        "    float3 wo = normalize(float3(1, -1, 1));\n"
        "    float3 surface_point = float3(0, 0, 0);\n"
        "    float3 surface_normal = float3(0, 0, 1);\n"
        "    float3 positions[3] = { float3(-1, 1, 1), float3(1, -1, 1), float3(1, 1, 1) };\n" // Light pointing away from surface
        "    bool two_sided = false;\n"
        "    radiance[0] = LtcAreaLight::evaluate_triangle_light_lambert(wo, surface_point, surface_normal, positions, two_sided);\n"
        "}\n";

    OComputeShader ltc_evaluation_shader;
    OBlob shader_blob = Managers::ShaderManager().compile_shader_source(evaluate_ltc_area_light_cs, "cs_5_0", "evaluate_ltc_area_light_cs");
    THROW_DX11_ERROR(device->CreateComputeShader(UNPACK_BLOB_ARGS(shader_blob), nullptr, &ltc_evaluation_shader));

    OUnorderedAccessView radiance_UAV;
    OBuffer radiance_buffer = create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, nullptr, 1, nullptr, &radiance_UAV);
    context->CSSetUnorderedAccessViews(0, 1, &radiance_UAV, nullptr);

    context->CSSetShader(ltc_evaluation_shader, nullptr, 0);
    context->Dispatch(1, 1, 1);

    float gpu_radiance[1];
    Readback::buffer(device, context, radiance_buffer, gpu_radiance, gpu_radiance + 1);
    float actual_light_contribution = gpu_radiance[0];

    float no_light_contribution = 0;
    EXPECT_FLOAT_EQ(no_light_contribution, actual_light_contribution);
}

GTEST_TEST(LtcAreaLight, Light_behind_surface_not_contributing) {
    auto device = create_test_device();
    auto context = get_immidiate_context1(device);

    const char* evaluate_ltc_area_light_cs =
        "#include <LtcAreaLight.hlsl>\n"
        "\n"
        "RWStructuredBuffer<float> radiance : register(u0);\n"
        "\n"
        "[numthreads(1, 1, 1)]\n"
        "void evaluate_ltc_area_light_cs(uint2 thread_index : SV_GroupThreadID) {\n"
        "    float3 wo = normalize(float3(1, -1, 1));\n"
        "    float3 surface_point = float3(0, 0, 0);\n"
        "    float3 surface_normal = float3(0, 0, 1);\n"
        "    float3 positions[3] = { float3(-1, 1, -1), float3(1, -1, -1), float3(1, 1, -1) };\n" // Light behind surface
        "    bool two_sided = true;\n"
        "    radiance[0] = LtcAreaLight::evaluate_triangle_light_lambert(wo, surface_point, surface_normal, positions, two_sided);\n"
        "}\n";

    OComputeShader ltc_evaluation_shader;
    OBlob shader_blob = Managers::ShaderManager().compile_shader_source(evaluate_ltc_area_light_cs, "cs_5_0", "evaluate_ltc_area_light_cs");
    THROW_DX11_ERROR(device->CreateComputeShader(UNPACK_BLOB_ARGS(shader_blob), nullptr, &ltc_evaluation_shader));

    OUnorderedAccessView radiance_UAV;
    OBuffer radiance_buffer = create_default_buffer(device, DXGI_FORMAT_R32_FLOAT, nullptr, 1, nullptr, &radiance_UAV);
    context->CSSetUnorderedAccessViews(0, 1, &radiance_UAV, nullptr);

    context->CSSetShader(ltc_evaluation_shader, nullptr, 0);
    context->Dispatch(1, 1, 1);

    float gpu_radiance[1];
    Readback::buffer(device, context, radiance_buffer, gpu_radiance, gpu_radiance + 1);
    float actual_light_contribution = gpu_radiance[0];

    float no_light_contribution = 0;
    EXPECT_FLOAT_EQ(no_light_contribution, actual_light_contribution);
}

} // NS DX11Renderer::LightSources

#endif // _DX11RENDERER_LIGHT_SOURCES_LTC_AREA_LIGHT_TEST_H_