// Test directional lights in Bifrost.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DIRECTIONAL_LIGHT_TEST_H_
#define _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DIRECTIONAL_LIGHT_TEST_H_

#include <Assets/Shading/LightSources/LightTestUtils.h>
#include <Expects.h>

#include <Bifrost/Assets/Shading/LightSources/DirectionalLight.h>

#include <gtest/gtest.h>

namespace Bifrost::Assets::Shading::LightSources {

GTEST_TEST(Assets_Shading_LightSources_DirectionalLight, light_function_consistency) {
    using namespace Bifrost::Math;

    Vector3f lit_position = Vector3f(0.0f, 0.0f, 2.0f);
    RGB light_radiance = RGB(1, 0.5f, 0.0f);

    for (Vector3f direction : { Vector3f(0, 0, 1), Vector3f(0, 0, -1) }) {
        DirectionalLight light = DirectionalLight(direction, light_radiance);
        LightTestUtils::light_consistency_test(light, lit_position, 1);
    }
}

} // NS Bifrost::Assets::Shading::LightSources

#endif // _BIFROST_ASSETS_SHADING_LIGHTSOURCES_DIRECTIONAL_LIGHT_TEST_H_