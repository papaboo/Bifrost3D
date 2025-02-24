// DirectX 11 renderer owned resource ptr.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_OWNED_RESOURCE_PTR_H_
#define _DX11RENDERER_RENDERER_OWNED_RESOURCE_PTR_H_

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// An owned pointer for DirectX 11 resources.
// The pointer is used when there will only ever be a single, unique owner of the resource.
// For shared ownership use Microsoft::WRL::ComPtr.
// Future work:
// * Resource interface casts.
// ------------------------------------------------------------------------------------------------
template <typename T>
class OwnedResourcePtr {
private:
    T* m_resource;

public:
    // --------------------------------------------------------------------------------------------
    // Constructors and destructor.
    // --------------------------------------------------------------------------------------------
    OwnedResourcePtr(T* res = nullptr)
        : m_resource(res) { }

    OwnedResourcePtr(OwnedResourcePtr&& other)
        : m_resource(other.detach()) { }

    OwnedResourcePtr& operator=(OwnedResourcePtr&& rhs) {
        if (m_resource) m_resource->Release();
        m_resource = rhs.detach();
        return *this;
    }

    ~OwnedResourcePtr() {
        if (m_resource != nullptr) {
            m_resource->Release();
            m_resource = nullptr;
        }
    }

    // --------------------------------------------------------------------------------------------
    // Data access.
    // --------------------------------------------------------------------------------------------

    inline operator bool() const { return m_resource != nullptr; }

    inline operator T*() { return m_resource; }
    inline operator T&() { return *m_resource; }
    inline T* get() { return m_resource; }
    inline T* operator->() { return m_resource; }
    inline const T* const operator->() const { return m_resource; }

    inline T** operator&() { return &m_resource; }
    inline T* const * operator&() const { return &m_resource; }

    inline bool operator==(T* rhs) const { return m_resource == rhs; }
    inline bool operator==(OwnedResourcePtr<T> rhs) const { return m_resource == rhs.m_resource; }
    inline bool operator!=(T* rhs) const { return m_resource != rhs; }
    inline bool operator!=(OwnedResourcePtr<T> rhs) const { return m_resource != rhs.m_resource; }

    // --------------------------------------------------------------------------------------------
    // Resource management.
    // --------------------------------------------------------------------------------------------

    inline unsigned int release() { 
        if (m_resource == nullptr) return 0u;
        unsigned int res = m_resource->Release(); m_resource = nullptr; return res;
    }
    inline void swap(OwnedResourcePtr<T>& other) {
        T* tmp = other.m_resource;
        other.m_resource = m_resource;
        m_resource = tmp;
    }

private:
    inline T* detach() { T* tmp = m_resource; m_resource = nullptr; return tmp; }

    OwnedResourcePtr(OwnedResourcePtr& other) = delete;
    OwnedResourcePtr& operator=(OwnedResourcePtr& rhs) = delete;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_OWNED_RESOURCE_PTR_H_