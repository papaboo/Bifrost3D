// DirectX 11 renderer utility functions.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_UTILS_H_
#define _DX11RENDERER_RENDERER_UTILS_H_

#include <DX11Renderer/Types.h>

#define NOMINMAX
// #include <D3D11.h>
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

// TODO Handle cso files and errors related to files not found.
inline ID3DBlob* compile_shader(std::wstring filename, const char* target) {
    ID3DBlob* shader_bytecode;
    ID3DBlob* error_messages = nullptr;
    HRESULT hr = D3DCompileFromFile(filename.c_str(),
        nullptr, // macroes
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main",
        target,
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
        0, // More flags. Unused.
        &shader_bytecode,
        &error_messages);
    if (FAILED(hr)) { // TODO File not found not handled? Path not found unhandled as well.
        if (error_messages != nullptr)
            printf("Shader error: '%s'\n", (char*)error_messages->GetBufferPointer());
        return nullptr;
    }

    return shader_bytecode;
}

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_UTILS_H_