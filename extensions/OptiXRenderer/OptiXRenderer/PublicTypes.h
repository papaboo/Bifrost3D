// OptiX renderer public POD types.
// These types should not depent on any OptiX types so they can safely be exposed.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_PUBLIC_TYPES_H_
#define _OPTIXRENDERER_PUBLIC_TYPES_H_

#include <Bifrost/Core/Bitmask.h>

namespace OptiXRenderer {

// ------------------------------------------------------------------------------------------------
// Different rendering backends.
// ------------------------------------------------------------------------------------------------
enum class Backend {
    None,
    PathTracing,
    AIDenoisedPathTracing,
    DepthVisualization,
    AlbedoVisualization,
    TintVisualization,
    RoughnessVisualization,
    ShadingNormalVisualization,
};

// ------------------------------------------------------------------------------------------------
// Path regularization settings.
// The PDF scale scales how large a PDF we allow intersections to produce relative to the previous sample.
// Paths with large PDFs (glossy samples) at the previous bounce are allowed to continue with large PDF,
// paths with low PDFs (diffuse) will try to limit high frequency/specular samples.
// This effectively limits the amount of caustics the renderer can produce and allows images to converge faster.
// The scale decay will decay the effects of PDF scale as the accumulation count increases.
// ------------------------------------------------------------------------------------------------
struct PathRegularizationSettings {
    float PDF_scale;
    float scale_decay;

    float PDF_scale_at_accumulation(int accumulation) { return PDF_scale * (1.0f + scale_decay * accumulation); }
};

// ------------------------------------------------------------------------------------------------
// AI denoiser flags.
// ------------------------------------------------------------------------------------------------
enum class AIDenoiserFlag : unsigned char {
    None = 0,
    LogarithmicFeedback = 1 << 0,
    // Mutually exclusive debug flags
    VisualizeNoise = 1 << 1,
    VisualizeAlbedo = 1 << 2,
    Default = LogarithmicFeedback
};
typedef Bifrost::Core::Bitmask<AIDenoiserFlag> AIDenoiserFlags;

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_PUBLIC_TYPES_H_