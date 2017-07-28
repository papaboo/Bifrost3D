// DirectX 11 constant buffer array.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_CONSTANT_BUFFER_ARRAY_H_
#define _DX11RENDERER_RENDERER_CONSTANT_BUFFER_ARRAY_H_

#include "Dx11Renderer/Types.h"
#include "Dx11Renderer/Utils.h"

#include <assert.h>

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
public: // TODO Members should be private

    OID3D11Buffer m_constant_buffer;
    unsigned int m_element_count;

    ConstantBufferArray() : m_constant_buffer(nullptr), m_element_count(0u) { }

    ConstantBufferArray(ID3D11Device1& device, unsigned int element_count)
        : m_element_count(element_count) {
        // static_assert(sizeof(T) <= CONSTANT_BUFFER_ALIGNMENT);
        // TODO Check max constant buffer size.
        create_constant_buffer(device, CONSTANT_BUFFER_ALIGNMENT * element_count, &m_constant_buffer);
    }

    ConstantBufferArray(ConstantBufferArray&& other)
        : m_constant_buffer(other.m_constant_buffer)
        , m_element_count(rhs.m_element_count) { }

    ConstantBufferArray& operator=(ConstantBufferArray&& rhs) {
        m_constant_buffer = std::move(rhs.m_constant_buffer);
        m_element_count = rhs.m_element_count;
        return *this;
    }

    inline unsigned int resize(ID3D11Device1& device, ID3D11DeviceContext1& context, unsigned int element_count) {
        // TODO Check max constant buffer size.
        // TODO Can this be done without a context?
        OID3D11Buffer new_constant_buffer;
        create_constant_buffer(device, CONSTANT_BUFFER_ALIGNMENT * element_count, &new_constant_buffer);

        D3D11_BOX box;
        box.left = 0; box.right = m_element_count * sizeof(T);
        box.front = box.top = 0;
        box.back = box.bottom = 1;
        context.CopySubresourceRegion1(new_constant_buffer, 0, 0, 0, 0, m_constant_buffer, 0, &box, D3D11_COPY_NO_OVERWRITE);

        m_constant_buffer = new_constant_buffer;
        m_element_count = element_count;
        return element_count;
    }

    inline unsigned int get_element_count() const { return m_element_count; }

    inline void set(ID3D11DeviceContext1* context, const T& element, unsigned int index, D3D11_COPY_FLAGS copy_flags = D3D11_COPY_NO_OVERWRITE) {
        assert(index < m_element_count);
        D3D11_BOX box;
        box.left = index * CONSTANT_BUFFER_ALIGNMENT; box.right = box.left + sizeof(T);
        box.front = box.top = 0;
        box.back = box.bottom = 1;
        context->UpdateSubresource1(m_constant_buffer, 0, &box, &element, sizeof(T), sizeof(T), copy_flags);
    }

    inline void VS_set(ID3D11DeviceContext1* context, unsigned int slot, unsigned int element_index) {
        unsigned int begin = element_index * CONSTANT_BUFFER_ALIGNMENT / 16;
        unsigned int size = CONSTANT_BUFFER_ALIGNMENT / 16;
        context->VSSetConstantBuffers1(slot, 1, &m_constant_buffer, &begin, &size);
    }

    inline void PS_set(ID3D11DeviceContext1* context, unsigned int slot, unsigned int element_index) {
        unsigned int begin = element_index * CONSTANT_BUFFER_ALIGNMENT / 16;
        unsigned int size = CONSTANT_BUFFER_ALIGNMENT / 16;
        context->PSSetConstantBuffers1(slot, 1, &m_constant_buffer, &begin, &size);
    }
};

} // DX11Renderer

#endif // _DX11RENDERER_RENDERER_CONSTANT_BUFFER_ARRAY_H_