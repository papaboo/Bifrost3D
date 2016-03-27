// Test OptiXRe0nderer's default shading model.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_
#define _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_

#include <Utils.h>

#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {

GTEST_TEST(DefaultShadingModel, Fresnel) {
    using namespace OptiXRenderer::Shading::ShadingModels;
    using namespace optix;

    { // Test that specular reflections on non-metals are white and incident reflections are diffuse.
        Material material_params;
        material_params.base_color = make_float3(1.0f, 0.0f, 0.0f);
        material_params.base_roughness = 0.02f;
        material_params.metallic = 0.0f;
        material_params.specularity = 0.0f; // Testing specularity. Physically-based fubar value.
        DefaultShading material = DefaultShading(material_params);

        { // Test that indicent reflectivity is red.
            float3 wo = make_float3(0.0f, 0.0f, 1.0f);
            float3 weight = material.evaluate(wo, wo);
            EXPECT_GT(weight.x, 0.0f);
            EXPECT_FLOAT_EQ(weight.y, 0.0f);
            EXPECT_FLOAT_EQ(weight.z, 0.0f);
        }

        { // Test that specular reflectivity is white.
            float3 wo = normalize(make_float3(0.0f,  1.0f, 0.001f));
            float3 wi = normalize(make_float3(0.0f, -1.0f, 0.001f));
            float3 weight = material.evaluate(wo, wi);
            EXPECT_GT(weight.x, 0.0f);
            EXPECT_FLOAT_EQ(weight.x, weight.y);
            EXPECT_FLOAT_EQ(weight.x, weight.z);
        }
    }

    { // Test that specular reflections on metals are tinted.
        Material material_params;
        material_params.base_color = make_float3(1.0f, 0.766f, 0.336f);
        material_params.base_roughness = 0.02f;
        material_params.metallic = 1.0f;
        material_params.specularity = 0.02f; // Irrelevant when metallic is 1.
        DefaultShading material = DefaultShading(material_params);

        { // Test that indicent reflectivity is base color scaled.
            float3 wo = make_float3(0.0f, 0.0f, 1.0f);
            float3 weight = material.evaluate(wo, wo);
            float scale = material_params.base_color.x / weight.x;
            EXPECT_FLOAT_EQ(weight.x * scale, material_params.base_color.x);
            EXPECT_FLOAT_EQ(weight.y * scale, material_params.base_color.y);
            EXPECT_FLOAT_EQ(weight.z * scale, material_params.base_color.z);
        }

        { // Test that grazing angle reflectivity is base color scaled.
            float3 wo = normalize(make_float3(0.0f, 1.0f, 0.001f));
            float3 wi = normalize(make_float3(0.0f, -1.0f, 0.001f));
            float3 weight = material.evaluate(wo, wi);
            float scale = material_params.base_color.x / weight.x;
            EXPECT_FLOAT_EQ(weight.x * scale, material_params.base_color.x);
            EXPECT_FLOAT_EQ(weight.y * scale, material_params.base_color.y);
            EXPECT_FLOAT_EQ(weight.z * scale, material_params.base_color.z);
        }
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_DEFAULT_TEST_H_