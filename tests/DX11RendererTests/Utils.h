// DX11Renderer testing utils.
// -------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// -------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERERTEST_UTILS_H_
#define _DX11RENDERERTEST_UTILS_H_

#include <DX11Renderer/Utils.h>

// -------------------------------------------------------------------------------------------------
// DX11 helpers.
// -------------------------------------------------------------------------------------------------

template <typename RandomAccessIterator>
inline void readback_texture2D(ID3D11Device1* device, ID3D11DeviceContext1* context, ID3D11Texture2D* gpu_texture,
                               RandomAccessIterator cpu_buffer_begin, RandomAccessIterator cpu_buffer_end) {
    unsigned int element_count = unsigned int(cpu_buffer_end - cpu_buffer_begin);

    D3D11_TEXTURE2D_DESC staging_desc;
    gpu_texture->GetDesc(&staging_desc);
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = D3D11_BIND_NONE;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = D3D11_RESOURCE_MISC_NONE;

    ID3D11Texture2D* staging_texture;
    THROW_ON_FAILURE(device->CreateTexture2D(&staging_desc, nullptr, &staging_texture));

    context->CopyResource(staging_texture, gpu_texture);
    // context->Flush();

    D3D11_MAPPED_SUBRESOURCE mapped_resource = {};
    THROW_ON_FAILURE(context->Map(staging_texture, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_NONE, &mapped_resource));
    memcpy(&(*cpu_buffer_begin), mapped_resource.pData, sizeof(std::iterator_traits<RandomAccessIterator>::value_type) * element_count);
    context->Unmap(staging_texture, 0);

    staging_texture->Release();
}

template <typename RandomAccessIterator>
inline void readback_buffer(ID3D11Device1* device, ID3D11DeviceContext1* context, ID3D11Buffer* gpu_buffer, 
                            RandomAccessIterator cpu_buffer_begin, RandomAccessIterator cpu_buffer_end) {
    unsigned int element_count = unsigned int(cpu_buffer_end - cpu_buffer_begin);

    D3D11_BUFFER_DESC staging_desc = {};
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.ByteWidth = sizeof(std::iterator_traits<RandomAccessIterator>::value_type) * element_count;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    ID3D11Buffer* staging_buffer;
    THROW_ON_FAILURE(device->CreateBuffer(&staging_desc, nullptr, &staging_buffer));

    context->CopyResource(staging_buffer, gpu_buffer);
    // context->Flush();

    D3D11_MAPPED_SUBRESOURCE mapped_resource = {};
    THROW_ON_FAILURE(context->Map(staging_buffer, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_NONE, &mapped_resource));
    memcpy(&(*cpu_buffer_begin), mapped_resource.pData, sizeof(std::iterator_traits<RandomAccessIterator>::value_type) * element_count);
    context->Unmap(staging_buffer, 0);

    staging_buffer->Release();
}

// -------------------------------------------------------------------------------------------------
// Comparison helpers.
// -------------------------------------------------------------------------------------------------

inline bool almost_equal_eps(float lhs, float rhs, float eps) {
    return lhs < rhs + eps && lhs + eps > rhs;
}

#define EXPECT_FLOAT_EQ_EPS(expected, actual, epsilon) EXPECT_PRED3(almost_equal_eps, expected, actual, epsilon)

inline bool almost_equal_percentage(float lhs, float rhs, float percentage) {
    float eps = lhs * percentage;
    return almost_equal_eps(lhs, rhs, eps);
}

#define EXPECT_FLOAT_EQ_PCT(expected, actual, percentage) EXPECT_PRED3(almost_equal_percentage, expected, actual, percentage)

#endif // _DX11RENDERERTEST_UTILS_H_