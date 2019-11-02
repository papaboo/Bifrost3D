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
    AlbedoVisualization,
    DepthVisualization,
    RoughnessVisualization,
};

// ------------------------------------------------------------------------------------------------
// Path regularization settings.
// ------------------------------------------------------------------------------------------------
struct PathRegularizationSettings {
    float scale = 1.0f;
    float decay = 1.0f / 6.0f;
};

// ------------------------------------------------------------------------------------------------
// AI denoiser flags.
// ------------------------------------------------------------------------------------------------
enum class AIDenoiserFlag : unsigned char {
    None = 0,
    Albedo = 1 << 0,
    Normals = 1 << 1,
    LogarithmicFeedback = 1 << 2,
    // Mutually exclusive debug flags
    VisualizeNoise = 1 << 3,
    VisualizeAlbedo = 1 << 4,
    VisualizeNormals = 1 << 5,
    Default = LogarithmicFeedback
};
typedef Bifrost::Core::Bitmask<AIDenoiserFlag> AIDenoiserFlags;

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_PUBLIC_TYPES_H_