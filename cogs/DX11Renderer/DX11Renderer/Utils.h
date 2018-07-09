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
        fprintf(stderr, "%s\n", error.c_str());
        throw std::exception(error.c_str());
    }
}

#define THROW_DX11_ERROR(hr) ::DX11Renderer::CHECK_HRESULT(hr, __FILE__,__LINE__)

inline void _assert(const char* expression, const char* file, int line) {
    fprintf(stderr, "Assertion failed: '%s', file '%s' line '%d'.\n", expression, file, line);
    exit(1);
}
#define always_assert(EXPRESSION) ((EXPRESSION) ? (void)0 : _assert(#EXPRESSION, __FILE__, __LINE__))

struct PerformanceMarker {
    PerformanceMarker(ID3D11DeviceContext1& context, LPCWSTR event_name) {
        context.QueryInterface(IID_PPV_ARGS(&m_perf));
        m_perf->BeginEvent(event_name);
    }

    ~PerformanceMarker() {
        end();
    }

    void end() {
        if (m_perf != nullptr) {
            m_perf->EndEvent();
            m_perf->Release();
            m_perf = nullptr;
        }
    }

private:
    ID3DUserDefinedAnnotation* m_perf;
};

inline int sizeof_dx_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_A8_UNORM:
        return 1;
    case DXGI_FORMAT_R16_UNORM:
        return 2;
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_FLOAT:
        return 4;
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
        return 8;
    case DXGI_FORMAT_R32G32B32_FLOAT:
        return 12;
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return 16;
    default:
        throw std::exception("Unknown DXGI_FORMAT");
    }
};

inline ODevice1 get_device1(ID3D11DeviceContext1& context) {
    ID3D11Device* basic_device;
    context.GetDevice(&basic_device);
    ODevice1 device1;
    THROW_DX11_ERROR(basic_device->QueryInterface(IID_PPV_ARGS(&device1)));
    basic_device->Release();
    return device1;
}

inline OBlob compile_shader(const std::wstring& filename, const char* target, const char* entry_point,
                            const D3D_SHADER_MACRO* macros = nullptr) {
    OBlob shader_bytecode;
    OBlob error_messages = nullptr;
    HRESULT hr = D3DCompileFromFile(filename.c_str(),
        macros,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry_point,
        target,
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
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

inline float3 make_float3(Cogwheel::Math::Vector3f v) {
    float3 r = { v.x, v.y, v.z };
    return r;
}

inline float alpha_sort_value(Cogwheel::Math::Vector3f camera_pos, Cogwheel::Math::Transform transform, Cogwheel::Math::AABB bounds) {
    using namespace Cogwheel::Math;

    Vector3f center = transform.inverse() * bounds.center();
    return magnitude_squared(camera_pos - center);
}


// ------------------------------------------------------------------------------------------------
// Resource creation
// ------------------------------------------------------------------------------------------------
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

inline OBuffer create_default_buffer(ID3D11Device1& device, DXGI_FORMAT format, void* data, int element_count,
                                     ID3D11ShaderResourceView** buffer_SRV, ID3D11UnorderedAccessView** buffer_UAV = nullptr) {
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.StructureByteStride = sizeof_dx_format(format);
    buffer_desc.ByteWidth = sizeof_dx_format(format) * element_count;
    buffer_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    buffer_desc.MiscFlags = 0;
    buffer_desc.CPUAccessFlags = 0;

    OBuffer buffer;
    if (data != nullptr) {
        D3D11_SUBRESOURCE_DATA buffer_data = {};
        buffer_data.pSysMem = data;
        THROW_DX11_ERROR(device.CreateBuffer(&buffer_desc, &buffer_data, &buffer));
    } else
        THROW_DX11_ERROR(device.CreateBuffer(&buffer_desc, nullptr, &buffer));

    if (buffer_SRV) {
        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
        srv_desc.Format = format;
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srv_desc.Buffer.FirstElement = 0;
        srv_desc.Buffer.NumElements = element_count;
        THROW_DX11_ERROR(device.CreateShaderResourceView(buffer, &srv_desc, buffer_SRV));
    }

    if (buffer_UAV) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
        uav_desc.Format = format;
        uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = 0;
        uav_desc.Buffer.NumElements = element_count;
        uav_desc.Buffer.Flags = 0;
        THROW_DX11_ERROR(device.CreateUnorderedAccessView(buffer, &uav_desc, buffer_UAV));
    }

    return buffer;
};

inline OBuffer create_default_buffer(ID3D11Device1& device, DXGI_FORMAT format, int element_count,
                                     ID3D11ShaderResourceView** buffer_SRV, ID3D11UnorderedAccessView** buffer_UAV = nullptr) {
    return create_default_buffer(device, format, nullptr, element_count, buffer_SRV, buffer_UAV);
}

inline OTexture2D create_texture_2D(ID3D11Device1& device, DXGI_FORMAT format, void* pixels, unsigned int width, unsigned int height, D3D11_USAGE usage,
                                    ID3D11ShaderResourceView** texture_SRV, ID3D11UnorderedAccessView** texture_UAV = nullptr, ID3D11RenderTargetView** texture_RTV = nullptr) {
    D3D11_TEXTURE2D_DESC tex_desc;
    tex_desc.Width = width;
    tex_desc.Height = height;
    tex_desc.MipLevels = 1;
    tex_desc.ArraySize = 1;
    tex_desc.Format = format;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.SampleDesc.Quality = 0;
    tex_desc.Usage = usage;
    tex_desc.BindFlags = (texture_SRV == nullptr ? D3D11_BIND_NONE : D3D11_BIND_SHADER_RESOURCE) |
                         (texture_UAV == nullptr ? D3D11_BIND_NONE : D3D11_BIND_UNORDERED_ACCESS) |
                         (texture_RTV == nullptr ? D3D11_BIND_NONE : D3D11_BIND_RENDER_TARGET);
    tex_desc.CPUAccessFlags = 0;
    tex_desc.MiscFlags = 0;

    OTexture2D texture;
    if (pixels != nullptr) {
        D3D11_SUBRESOURCE_DATA resource_data = {};
        resource_data.SysMemPitch = sizeof_dx_format(tex_desc.Format) * width;
        resource_data.SysMemSlicePitch = resource_data.SysMemPitch * height;
        resource_data.pSysMem = pixels;

        THROW_DX11_ERROR(device.CreateTexture2D(&tex_desc, &resource_data, &texture));
    } else
        THROW_DX11_ERROR(device.CreateTexture2D(&tex_desc, nullptr, &texture));

    if (texture_SRV != nullptr)
        THROW_DX11_ERROR(device.CreateShaderResourceView(texture, nullptr, texture_SRV));
    if (texture_UAV != nullptr)
        THROW_DX11_ERROR(device.CreateUnorderedAccessView(texture, nullptr, texture_UAV));
    if (texture_RTV != nullptr)
        THROW_DX11_ERROR(device.CreateRenderTargetView(texture, nullptr, texture_RTV));

    return texture;
};

inline OTexture2D create_texture_2D(ID3D11Device1& device, DXGI_FORMAT format, void* pixels, unsigned int width, unsigned int height,
                                    ID3D11ShaderResourceView** texture_SRV, ID3D11UnorderedAccessView** texture_UAV = nullptr, ID3D11RenderTargetView** texture_RTV = nullptr) {
    return create_texture_2D(device, format, pixels, width, height, D3D11_USAGE_DEFAULT, texture_SRV, texture_UAV, texture_RTV);
}

inline OTexture2D create_texture_2D(ID3D11Device1& device, DXGI_FORMAT format, unsigned int width, unsigned int height,
                                    ID3D11ShaderResourceView** texture_SRV, ID3D11UnorderedAccessView** texture_UAV = nullptr, ID3D11RenderTargetView** texture_RTV = nullptr) {
    return create_texture_2D(device, format, nullptr, width, height, D3D11_USAGE_DEFAULT, texture_SRV, texture_UAV, texture_RTV);
}

// ------------------------------------------------------------------------------------------------
// Resource data readback utilities. 
// Highly inefficient and should only be used for debug or one-time purposes.
// ------------------------------------------------------------------------------------------------
namespace Readback {

template <typename RandomAccessIterator>
inline void texture2D(ID3D11Device1* device, ID3D11DeviceContext1* context, ID3D11Texture2D* gpu_texture,
                      RandomAccessIterator cpu_buffer_begin, RandomAccessIterator cpu_buffer_end) {
    unsigned int element_count = unsigned int(cpu_buffer_end - cpu_buffer_begin);

    D3D11_TEXTURE2D_DESC staging_desc;
    gpu_texture->GetDesc(&staging_desc);
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = D3D11_BIND_NONE;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = D3D11_RESOURCE_MISC_NONE;

    ID3D11Texture2D* staging_texture;
    THROW_DX11_ERROR(device->CreateTexture2D(&staging_desc, nullptr, &staging_texture));

    context->CopyResource(staging_texture, gpu_texture);
    // context->Flush();

    D3D11_MAPPED_SUBRESOURCE mapped_resource = {};
    THROW_DX11_ERROR(context->Map(staging_texture, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_NONE, &mapped_resource));
    memcpy(&(*cpu_buffer_begin), mapped_resource.pData, sizeof(std::iterator_traits<RandomAccessIterator>::value_type) * element_count);
    context->Unmap(staging_texture, 0);

    staging_texture->Release();
}

template <typename RandomAccessIterator>
inline void buffer(ID3D11Device1* device, ID3D11DeviceContext1* context, ID3D11Buffer* gpu_buffer,
                   RandomAccessIterator cpu_buffer_begin, RandomAccessIterator cpu_buffer_end) {
    unsigned int element_count = unsigned int(cpu_buffer_end - cpu_buffer_begin);

    D3D11_BUFFER_DESC staging_desc = {};
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.ByteWidth = sizeof(std::iterator_traits<RandomAccessIterator>::value_type) * element_count;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    ID3D11Buffer* staging_buffer;
    THROW_DX11_ERROR(device->CreateBuffer(&staging_desc, nullptr, &staging_buffer));

    context->CopyResource(staging_buffer, gpu_buffer);
    // context->Flush();

    D3D11_MAPPED_SUBRESOURCE mapped_resource = {};
    THROW_DX11_ERROR(context->Map(staging_buffer, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_NONE, &mapped_resource));
    memcpy(&(*cpu_buffer_begin), mapped_resource.pData, sizeof(std::iterator_traits<RandomAccessIterator>::value_type) * element_count);
    context->Unmap(staging_buffer, 0);

    staging_buffer->Release();
}

} // NS Readback
} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_UTILS_H_