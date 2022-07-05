// Test utils for OptiXRenderer's shading models.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_SHADING_MODEL_TEST_UTILS_H_
#define _OPTIXRENDERER_SHADING_MODEL_TEST_UTILS_H_

#include <BSDFTestUtils.h>

#include <gtest/gtest.h>

namespace OptiXRenderer {
namespace ShadingModelTestUtils {

using namespace optix;
    
Material gold_parameters() {
    Material gold_params = {};
    gold_params.tint = make_float3(1.0f, 0.766f, 0.336f);
    gold_params.roughness = 0.02f;
    gold_params.metallic = 1.0f;
    gold_params.specularity = 1.0f;
    return gold_params;
}

Material plastic_parameters() {
    Material plastic_params = {};
    plastic_params.tint = make_float3(0.02f, 0.27f, 0.33f);
    plastic_params.roughness = 0.7f;
    plastic_params.metallic = 0.0f;
    plastic_params.specularity = 0.02f;
    return plastic_params;
}

Material coated_plastic_parameters() {
    Material plastic_params = plastic_parameters();
    plastic_params.coat = 1.0f;
    plastic_params.coat_roughness = plastic_params.roughness;
    return plastic_params;
}

template <typename ShadingModel>
BSDFTestUtils::RhoResult directional_hemispherical_reflectance_function(ShadingModel shading_model, float3 wo) {
    unsigned int sample_count = 4096u;
    return BSDFTestUtils::directional_hemispherical_reflectance_function(shading_model, wo, sample_count);
}

template <typename ShadingModel>
void BSDF_consistency_test(ShadingModel shading_model, float3 wo, unsigned int sample_count) {
    BSDFTestUtils::BSDF_consistency_test(shading_model, wo, sample_count);
}

} // NS ShadingModelTestUtils
} // NS OptiXRenderer

#endif // _OPTIXRENDERER_SHADING_MODEL_TEST_UTILS_H_