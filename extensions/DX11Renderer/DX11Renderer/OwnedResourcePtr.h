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
public:
    T* resource;

    // --------------------------------------------------------------------------------------------
    // Constructors and destructor.
    // --------------------------------------------------------------------------------------------
    OwnedResourcePtr(T* res = nullptr)
        : resource(res) { }

    OwnedResourcePtr(OwnedResourcePtr&& other)
        : resource(other.detach()) { }

    OwnedResourcePtr& operator=(OwnedResourcePtr&& rhs) {
        if (resource) resource->Release();
        resource = rhs.detach();
        return *this;
    }

    ~OwnedResourcePtr() {
        if (resource != nullptr) {
            resource->Release();
            resource = nullptr;
        }
    }

    // --------------------------------------------------------------------------------------------
    // Data access.
    // --------------------------------------------------------------------------------------------

    inline operator bool() const { return resource != nullptr; }

    inline operator T*() { return resource; }
    inline operator T&() { return *resource; }
    inline T* get() { return resource; }
    inline T* operator->() { return resource; }
    inline const T* const operator->() const { return resource; }

    inline T** operator&() { return &resource; }
    inline T* const * operator&() const { return &resource; }

    // --------------------------------------------------------------------------------------------
    // Resource management.
    // --------------------------------------------------------------------------------------------

    inline T* detach() { T* tmp = resource; resource = nullptr; return tmp; }
    inline unsigned int release() { 
        if (resource == nullptr) return 0u;
        unsigned int res = resource->Release(); resource = nullptr; return res;
    }
    inline void swap(OwnedResourcePtr<T>& other) {
        T* tmp = other.resource;
        other.resource = resource;
        resource = tmp;
    }

private:
    OwnedResourcePtr(OwnedResourcePtr& other) = delete;
    OwnedResourcePtr& operator=(OwnedResourcePtr& rhs) = delete;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_OWNED_RESOURCE_PTR_H_