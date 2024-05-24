// Test OptiXRenderer's GGX reflection layer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_LAYER_GGX_REFLECTION_H_
#define _OPTIXRENDERER_SHADING_MODEL_LAYER_GGX_REFLECTION_H_

#include <ShadingModels/ShadingModelTestUtils.h>
#include <Utils.h>

#include <OptiXRenderer/Shading/ShadingModels/DiffuseShading.h>
#include <OptiXRenderer/Shading/ShadingModels/Layers/GGXReflection.h>
#include <OptiXRenderer/Utils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

class GGXReflectionLayerWrapper {
public:
    float m_roughness;
    float m_specularity;
    float m_layer_opacity;
    float m_cos_theta;
    Shading::ShadingModels::Layers::GGXReflection<Shading::ShadingModels::DiffuseShading> m_shading_model;

    GGXReflectionLayerWrapper(float roughness, float specularity, float layer_opacity, float cos_theta_o)
        : m_roughness(roughness), m_specularity(specularity), m_layer_opacity(layer_opacity), m_cos_theta(cos_theta_o) {
        auto base_shading_model = Shading::ShadingModels::DiffuseShading(optix::make_float3(0));
        m_shading_model = Shading::ShadingModels::Layers::GGXReflection<Shading::ShadingModels::DiffuseShading>(
            roughness, optix::make_float3(specularity), cos_theta_o, base_shading_model, layer_opacity);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const { return m_shading_model.evaluate_with_PDF(wo, wi); }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const { return m_shading_model.sample(wo, random_sample); }

    optix::float3 rho(float abs_cos_theta_o) const { return m_shading_model.rho(abs_cos_theta_o); }

    std::string to_string() const {
        std::ostringstream out;
        out << "GGX Reflection layer:" << std::endl;
        out << "  Roughness:" << m_roughness << std::endl;
        out << "  Specularity:" << m_specularity << std::endl;
        out << "  Layer opacity:" << m_layer_opacity << std::endl;
        return out.str();
    }
};
GTEST_TEST(GGXReflectionLayer, white_hot_room) {
    using namespace Shading::ShadingModels;

    // A fully specular (white) material.
    float specularity = 1.0f;
    float layer_opacity = 1.0f;

    for (float roughness : { 0.2f, 0.7f })
        for (float cos_theta : { 0.1f, 0.4f, 0.7f, 1.0f }) {
            const optix::float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
            auto shading_model = GGXReflectionLayerWrapper(roughness, specularity, layer_opacity, wo.z);
            auto rho = ShadingModelTestUtils::directional_hemispherical_reflectance_function(shading_model, wo).reflectance;
            EXPECT_FLOAT3_LE(rho, 1.0002f) << " with roughness " << roughness << " and cos_theta " << cos_theta;
            EXPECT_FLOAT3_GE(rho, 0.9998f) << " with roughness " << roughness << " and cos_theta " << cos_theta;
        }
}

GTEST_TEST(GGXReflectionLayer, function_consistency) {
    using namespace Shading::ShadingModels;

    for (float cos_theta : { -0.7f, 0.1f, 0.4f, 0.7f, 1.0f }) {
        auto wo = BSDFTestUtils::wo_from_cos_theta(cos_theta);

        for (float roughness : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
            for (float specularity : { 0.05f, 0.7f }) {
                for (float layer_opacity : { 0.0f, 0.5f, 1.0f }) {
                    auto shading_model = GGXReflectionLayerWrapper(roughness, specularity, layer_opacity, wo.z);
                    ShadingModelTestUtils::consistency_test(shading_model, wo, 32);
                }
            }
        }
    }
}

/*
GTEST_TEST(GGXReflectionLayer, rho_correctness) {
    using namespace Shading::ShadingModels;

    for (float cos_theta : { -0.7f, 0.1f, 0.4f, 0.7f, 1.0f }) {
        auto wo = BSDFTestUtils::wo_from_cos_theta(cos_theta);

        for (float roughness : { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f }) {
            for (float specularity : { 0.05f, 0.7f }) {
                for (float layer_opacity : { 0.0f, 0.5f, 1.0f }) {
                    auto shading_model = GGXReflectionLayerWrapper(roughness, specularity, layer_opacity, wo.z);
                    auto expected_rho = BSDFTestUtils::directional_hemispherical_reflectance_function(shading_model, wo, 4096).reflectance;
                    auto actual_rho = shading_model.rho(abs(wo.z));
                    EXPECT_FLOAT3_EQ_EPS(expected_rho, actual_rho, 0.005f);
                }
            }
        }
    }
}
*/

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_LAYER_GGX_REFLECTION_H_