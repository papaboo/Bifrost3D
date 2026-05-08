// OptiX renderer POD types.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_TYPES_H_
#define _OPTIXRENDERER_TYPES_H_

#include <OptiXRenderer/Defines.h>

#include <Bifrost/Math/Matrix.h>

#include <cuda_fp16.h>

namespace OptiXRenderer {

using AccumulationElementType = Bifrost::Math::Vector4d;

struct RGBA16f { __half r, g, b, a; };
__inline_all__ RGBA16f make_rgba16f(Bifrost::Math::RGB rgb, float a = 1.0f) { return { rgb.r, rgb.g, rgb.b, a }; }

struct RayTypes {
    static const unsigned int Radiance = 0;
    // static const unsigned int Shadow = 1;
    static const unsigned int Count = 1;
};

//----------------------------------------------------------------------------
// Ray payloads.
//----------------------------------------------------------------------------

struct __align__(16) RadiancePayload {
    Bifrost::Math::RGB radiance;
    float __lala;
};

//----------------------------------------------------------------------------
// Pipeline and scene parameters
//----------------------------------------------------------------------------

struct PipelineParams {
    Bifrost::Math::Matrix3x3f view_to_world_rotation;
    Bifrost::Math::Matrix4x4f inverse_projection_matrix;
    Bifrost::Math::Matrix4x4f inverse_view_projection_matrix;

    unsigned int accumulations;
    unsigned int max_bounce_count;
    unsigned int frame_width;
    unsigned int frame_height;
    RGBA16f* output_buffer;
    AccumulationElementType* accumulation_buffer;

    float path_regularization_PDF_scale;

    struct {
        OptixTraversableHandle traversable;
        Bifrost::Math::RGB environment_tint;
    } scene;
};

struct RayGenData { };
struct MissShaderData { };
struct HitsShaderData {};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_