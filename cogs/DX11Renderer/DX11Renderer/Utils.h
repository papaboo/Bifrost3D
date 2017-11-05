// DirectX 11 renderer utility functions.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_UTILS_H_
#define _DX11RENDERER_RENDERER_UTILS_H_

#include <DX11Renderer/Defines.h>
#include <DX11Renderer/Types.h>

#include <Cogwheel/Math/AABB.h>
#include <Cogwheel/Math/Ray.h>
#include <Cogwheel/Math/Transform.h>
#include <Cogwheel/Math/Vector.h>
#include <Cogwheel/Math/Utils.h>

#include <string>

#define NOMINMAX
#include <D3D11_1.h>
#include <D3DCompiler.h>
#undef RGB

namespace DX11Renderer {

#define UNPACK_BLOB_ARGS(blob) blob->GetBufferPointer(), blob->GetBufferSize()

template<typename ResourcePtr>
void safe_release(ResourcePtr* resource_ptr) {
    if (*resource_ptr) {
        (*resource_ptr)->Release();
        *resource_ptr = nullptr;
    }
}

inline void CHECK_HRESULT(HRESULT hr, const std::string& file, int line) {
    if (FAILED(hr)) {
        std::string error = "[file:" + file +
            " line:" + std::to_string(line) + "] DX 11";
        switch (hr) {
        case E_INVALIDARG:
            error += " invalid arg.";
            break;
        case DXGI_ERROR_INVALID_CALL:
            error += " DXGI error invalid call.";
            break;
        default:
            error += " unknown HRESULT code: " + std::to_string(hr);
            break;
        }
        printf("%s\n", error.c_str());
        throw std::exception(error.c_str());
    }
}

#define THROW_ON_FAILURE(hr) ::DX11Renderer::CHECK_HRESULT(hr, __FILE__,__LINE__)

// Copied from VS' assert.h to have an assert that is enabled in release builds as well.
#define always_assert(_Expression) (void)( (!!(_Expression)) || (_wassert(_CRT_WIDE(#_Expression), _CRT_WIDE(__FILE__), __LINE__), 0) )

inline int sizeof_dx_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_A8_UNORM:
        return 1;
    case DXGI_FORMAT_R16_UNORM:
        return 2;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return 4;
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        return 8;
    case DXGI_FORMAT_R32G32B32_FLOAT:
        return 12;
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return 16;
    default:
        throw std::exception("Unknown DXGI_FORMAT");
    }
};

inline OID3DBlob compile_shader(const std::wstring& filename, const char* target, const char* entry_point = "main") {
    OID3DBlob shader_bytecode;
    OID3DBlob error_messages = nullptr;
    HRESULT hr = D3DCompileFromFile(filename.c_str(),
        nullptr, // macroes
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry_point,
        target,
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
        0, // More flags. Unused.
        &shader_bytecode,
        &error_messages);
    if (FAILED(hr)) {
        if (hr == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND))
            printf("The system cannot find the file specified: '%ws'.\n", filename.c_str());
        else if (hr == HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND))
            printf("The system cannot find the path specified: '%ws'.\n", filename.c_str());
        else if (error_messages != nullptr)
            printf("Shader error: '%s'.\n", (char*)error_messages->GetBufferPointer());
        else 
            printf("Unknown error occured when trying to load: '%ws'.\n", filename.c_str());
        return nullptr;
    }

    return shader_bytecode;
}

inline HRESULT create_constant_buffer(ID3D11Device1& device, int byte_width, ID3D11Buffer** constant_buffer) {
    D3D11_BUFFER_DESC desc = {};
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.ByteWidth = byte_width;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    return device.CreateBuffer(&desc, nullptr, constant_buffer);
}

template <typename T>
inline HRESULT create_constant_buffer(ID3D11Device1& device, T& data, ID3D11Buffer** constant_buffer, D3D11_USAGE usage = D3D11_USAGE_IMMUTABLE) {
    D3D11_BUFFER_DESC desc = {};
    desc.Usage = usage;
    desc.ByteWidth = sizeof(T);
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA resource_data = {};
    resource_data.pSysMem = &data;
    return device.CreateBuffer(&desc, &resource_data, constant_buffer);
}

inline float3 make_float3(Cogwheel::Math::Vector3f v) {
    float3 r = { v.x, v.y, v.z};
    return r;
}

inline float alpha_sort_value(Cogwheel::Math::Vector3f camera_pos, Cogwheel::Math::Transform transform, Cogwheel::Math::AABB bounds) {
    using namespace Cogwheel::Math;

    Vector3f center = transform.inverse() * bounds.center();
    return magnitude_squared(camera_pos - center);
}

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_UTILS_H_