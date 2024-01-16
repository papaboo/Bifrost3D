// Test Bifrost BSDF precomputations.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_FITTINGS_TEST_H_
#define _BIFROST_ASSETS_SHADING_FITTINGS_TEST_H_

#include <Bifrost/Assets/Shading/Fittings.h>
#include <Expects.h>

namespace Bifrost::Assets {

GTEST_TEST(Assets_Shading_Estimate_GGX_bounded_VNDF_alpha, encode_decode_is_identity) {
    auto test_encode_decode_PDF = [=](float pdf, float accepted_error_percentage) {
        float encoded_PDF = Bifrost::Assets::Shading::Estimate_GGX_bounded_VNDF_alpha::encode_PDF(pdf);
        float decoded_PDF = Bifrost::Assets::Shading::Estimate_GGX_bounded_VNDF_alpha::decode_PDF(encoded_PDF);
        EXPECT_FLOAT_EQ_PCT(pdf, decoded_PDF, accepted_error_percentage);
    };

    // The error increases as the PDF increases. This is acceptable, however, as large PDF's all map to an encoed value near one,
    // the error is baked into the alpha estimation, and decode is only ever used for testing.
    test_encode_decode_PDF(0.1f, 0.000001f);
    test_encode_decode_PDF(1.0f, 0.000001f);
    test_encode_decode_PDF(10.0f, 0.000001f);
    test_encode_decode_PDF(1000.0f, 0.00004f);
    test_encode_decode_PDF(100000.0f, 0.002f);
}

} // NS Bifrost::Assets::Shading

#endif // _BIFROST_ASSETS_SHADING_FITTINGS_TEST_H_
