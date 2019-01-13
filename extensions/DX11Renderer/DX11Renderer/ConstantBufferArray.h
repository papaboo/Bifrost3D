// DirectX 11 constant buffer array.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_CONSTANT_BUFFER_ARRAY_H_
#define _DX11RENDERER_RENDERER_CONSTANT_BUFFER_ARRAY_H_

#include "Dx11Renderer/Types.h"
#include "Dx11Renderer/Utils.h"

#include <assert.h>
#include <memory>

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// Constant buffer array.
// A wrapper class around the constant buffer partial updates and slices 
// functionality introduced in DirectX 11.1.
// Future work:
// * Handle T's that are more than 256 byte.
// ------------------------------------------------------------------------------------------------
template<typename T>
class ConstantBufferArray {
public:

    // --------------------------------------------------------------------------------------------
    // Constructors and assignments
    // --------------------------------------------------------------------------------------------
    ConstantBufferArray() = default;

    ConstantBufferArray(ID3D11Device1& device, unsigned int element_count)
        : m_element_count(element_count) {
        static_assert(sizeof(T) <= CONSTANT_BUFFER_ALIGNMENT, "ConstantBufferArray only supports templated types T of size 256 bytes or less.");
        HRESULT hr = create_constant_buffer(device, element_count * get_element_stride(), &m_constant_buffer);
        THROW_DX11_ERROR(hr);
    }

    ConstantBufferArray(ID3D11Device1& device, T* elements, unsigned int element_count) 
        : m_element_count(element_count) {
        static_assert(sizeof(T) <= CONSTANT_BUFFER_ALIGNMENT, "ConstantBufferArray only supports templated types T of size 256 bytes or less.");

        D3D11_BUFFER_DESC desc = {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = element_count * get_element_stride();
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        auto data = std::make_unique<unsigned char[]>(desc.ByteWidth);
        for (unsigned int i = 0; i < element_count; ++i)
            memcpy(data.get() + i * get_element_stride(), elements + i, sizeof(T));

        D3D11_SUBRESOURCE_DATA resource_data = {};
        resource_data.pSysMem = data.get();

        THROW_DX11_ERROR(device.CreateBuffer(&desc, &resource_data, &m_constant_buffer));
    }

    ConstantBufferArray(ConstantBufferArray&& other) = default;
    ConstantBufferArray& operator=(ConstantBufferArray&& rhs) = default;

    // --------------------------------------------------------------------------------------------
    // Getters and setter.
    // --------------------------------------------------------------------------------------------
    
    inline ID3D11Buffer** get_buffer_addr() { return &m_constant_buffer; }
    inline unsigned int get_element_count() const { return m_element_count; }
    inline unsigned int get_element_stride() const { return CONSTANT_BUFFER_ALIGNMENT; }

    inline void set(ID3D11DeviceContext1* context, const T& element, unsigned int index, D3D11_COPY_FLAGS copy_flags = D3D11_COPY_NO_OVERWRITE) {
        assert(index < m_element_count);
        D3D11_BOX box;
        box.left = index * get_element_stride(); box.right = box.left + sizeof(T);
        box.front = box.top = 0;
        box.back = box.bottom = 1;
        context->UpdateSubresource1(m_constant_buffer, 0, &box, &element, sizeof(T), sizeof(T), copy_flags);
    }

    // --------------------------------------------------------------------------------------------
    // Shader constant buffer setters.
    // --------------------------------------------------------------------------------------------

    inline void VS_set(ID3D11DeviceContext1* context, unsigned int slot, unsigned int element_index) {
        unsigned int begin = element_index * get_element_stride() / 16;
        unsigned int size = get_element_stride() / 16;
        context->VSSetConstantBuffers1(slot, 1, &m_constant_buffer, &begin, &size);
    }

    inline void PS_set(ID3D11DeviceContext1* context, unsigned int slot, unsigned int element_index) {
        unsigned int begin = element_index * get_element_stride() / 16;
        unsigned int size = get_element_stride() / 16;
        context->PSSetConstantBuffers1(slot, 1, &m_constant_buffer, &begin, &size);
    }
private:
    ConstantBufferArray(ConstantBufferArray& other) = delete;
    ConstantBufferArray& operator=(ConstantBufferArray& rhs) = delete;

    OBuffer m_constant_buffer;
    unsigned int m_element_count;

};

} // DX11Renderer

#endif // _DX11RENDERER_RENDERER_CONSTANT_BUFFER_ARRAY_H_