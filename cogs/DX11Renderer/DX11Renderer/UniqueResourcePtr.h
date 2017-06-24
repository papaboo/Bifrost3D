// DirectX 11 renderer unique resource ptr.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_UNIQUE_RESOURCE_PTR_H_
#define _DX11RENDERER_RENDERER_UNIQUE_RESOURCE_PTR_H_

namespace DX11Renderer {

// ------------------------------------------------------------------------------------------------
// A unique pointer for DirectX 11 resources.
// The pointer is used when there will only ever be a single, unique owner of the resource.
// For shared ownership use Microsoft::WRL::ComPtr.
// Future work:
// * Resource interface casts.
// ------------------------------------------------------------------------------------------------
template <typename T>
class UniqueResourcePtr {
public:
    T* resource;

    // --------------------------------------------------------------------------------------------
    // Constructors and destructor.
    // --------------------------------------------------------------------------------------------
    UniqueResourcePtr(T* res = nullptr)
        : resource(res) { }

    UniqueResourcePtr(UniqueResourcePtr&& other)
        : resource(other.detach()) { }

    UniqueResourcePtr& operator=(UniqueResourcePtr&& rhs) {
        if (resource) resource->Release();
        resource = rhs.detach();
        return *this;
    }

    ~UniqueResourcePtr() {
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
    inline T* get() { return resource; }
    inline T* operator->() { return resource; }
    inline const T* const operator->() const { return resource; }

    inline T** operator&() { return &resource; }
    inline const T** const operator&() const { return &resource; }

    // --------------------------------------------------------------------------------------------
    // Resource management.
    // --------------------------------------------------------------------------------------------

    inline T* detach() { T* tmp = resource; resource = nullptr; return tmp; }
    inline unsigned int release() { resource->Release(); resource = nullptr; }
    inline void swap(UniqueResourcePtr<T>& other) {
        T* tmp = other.resource;
        other.resource = resource;
        resource = tmp;
    }

private:

    UniqueResourcePtr(UniqueResourcePtr& other) = delete;
    UniqueResourcePtr& operator=(UniqueResourcePtr& rhs) = delete;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_UNIQUE_RESOURCE_PTR_H_