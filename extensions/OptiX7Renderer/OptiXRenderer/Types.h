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

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <optix_types.h>

namespace OptiXRenderer {

struct half4 { __half x, y, z, w; };
__inline_all__ half4 create_half4(float x, float y, float z, float w) { return { x, y, z, w }; }
__inline_all__ half4 create_half4(float3 v, float w) { return { v.x, v.y, v.z, w }; }

template <int DIM> struct VectorDim {};
template <> struct VectorDim<2> { typedef float2 VectorType; };
template <> struct VectorDim<3> { typedef float3 VectorType; };
template <> struct VectorDim<4> { typedef float4 VectorType; };

__inline_all__ float dot(float2 lhs, float2 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }
__inline_all__ float dot(float3 lhs, float3 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
__inline_all__ float dot(float4 lhs, float4 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w; }

template <int R, int C>
struct Matrix {
    static const int ROW_COUNT = R;
    static const int COLUMN_COUNT = C;
    static const int N = ROW_COUNT * COLUMN_COUNT;

    using RowType = typename VectorDim<C>::VectorType;
    using ColumnType = typename VectorDim<R>::VectorType;

    float m_elements[N];

    Matrix() = default;

    Matrix(float elements[N]) {
        for (int i = 0; i < N; ++i)
            m_elements[i] = elements[i];
    }

    static __inline_all__ Matrix<R, C> identity() {
        Matrix<R, C> res;
        for (int r = 0; r < ROW_COUNT; ++r)
            for (int c = 0; c < COLUMN_COUNT; ++c)
                res.m_elements[r * COLUMN_COUNT + c] = r == c ? 1.0f : 0.0f;
        return res;
    }

    __inline_all__ RowType get_row(int r) const {
        RowType res = {};
        float* res_ptr = reinterpret_cast<float*>(&res);
        const float* row_begin = m_elements + r * COLUMN_COUNT;
        for (int c = 0; c < COLUMN_COUNT; ++c)
            res_ptr[c] = row_begin[c];

        return res;
    }

    __inline_all__ ColumnType operator*(RowType rhs) const {
        ColumnType res = {};
        float* res_ptr = reinterpret_cast<float*>(&res);

        for (int r = 0; r < ROW_COUNT; ++r)
            res_ptr[r] = dot(get_row(r), rhs);

        return res;
    }
};
using Matrix3x3 = Matrix<3,3>;
using Matrix4x4 = Matrix<4,4>;

// Record for the shader binding table
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct PipelineParams {
    Matrix3x3 view_to_world_rotation;
    Matrix4x4 inverse_projection_matrix;
    Matrix4x4 inverse_view_projection_matrix;

    unsigned int accumulations;
    unsigned int max_bounce_count;
    unsigned int frame_width;
    unsigned int frame_height;
    half4* output_buffer;
#ifdef DOUBLE_PRECISION_ACCUMULATION_BUFFER
    double4* accumulation_buffer;
#else
    float4* accumulation_buffer;
#endif

    float path_regularization_PDF_scale;

    struct {
        float3 environment_tint;
    } scene;
};

struct RayGenData { };
struct MissShaderData { };

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_TYPES_H_