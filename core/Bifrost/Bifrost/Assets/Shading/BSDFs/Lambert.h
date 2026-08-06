// Bifrost functions for the Lambert BSDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_SHADING_BSDFS_LAMBERT_H_
#define _BIFROST_ASSETS_SHADING_BSDFS_LAMBERT_H_

#include <Bifrost/Assets/Shading/Utils.h>
#include <Bifrost/Math/Distributions.h>

namespace Bifrost::Assets::Shading::BSDFs {

namespace Lambert {

_inline_all_archs_ float evaluate() {
    return RECIP_PIf;
}

_inline_all_archs_ Math::RGB evaluate(Math::RGB albedo) {
    return albedo * RECIP_PIf;
}

_inline_all_archs_ PDF pdf(Math::Vector3f wo, Math::Vector3f wi) {
    return Math::Distributions::Cosine::PDF(wi.z);
}

_inline_all_archs_ BSDFResponse evaluate_with_PDF(Math::RGB albedo, Math::Vector3f wo, Math::Vector3f wi) {
    auto reflectance = evaluate(albedo);
    auto PDF = pdf(wo, wi);
    return { reflectance, PDF };
}

_inline_all_archs_ BSDFSample sample(Math::RGB albedo, Math::Vector2f random_sample) {
    auto cosine_sample = Math::Distributions::Cosine::sample(random_sample);
    BSDFSample bsdf_sample;
    bsdf_sample.direction = cosine_sample.direction;
    bsdf_sample.PDF = cosine_sample.PDF;
    bsdf_sample.reflectance = evaluate(albedo);
    return bsdf_sample;
}

} // NS Lambert
} // NS Bifrost::Assets::Shading::BSDFs

#endif // _BIFROST_ASSETS_SHADING_BSDFS_LAMBERT_H_