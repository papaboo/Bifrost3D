// DirectX 11 shader manager.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#include <DX11Renderer/ShaderManager.h>

#define NOMINMAX
#include <D3DCompiler.h>

#include <filesystem>

using namespace std::filesystem;

namespace DX11Renderer {

OBlob ShaderManager::compile_shader_source(const char* const shader_src, const char* const target, const char* const entry_point) {
    OBlob shader_bytecode;
    OBlob error_messages = nullptr;
    HRESULT hr = D3DCompile(shader_src, strlen(shader_src), nullptr,
        nullptr, // macroes
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry_point,
        target,
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
        0, // More flags. Unused.
        &shader_bytecode,
        &error_messages);
    if (FAILED(hr)) {
        if (error_messages != nullptr)
            printf("Shader error compiling '%s':\n%s\n", entry_point, (char*)error_messages->GetBufferPointer());
        else
            printf("Unknown error occured compiling '%s'.\n", entry_point);
        return nullptr;
    }
    return shader_bytecode;
}

OBlob ShaderManager::compile_shader_from_file(const path& shader_path, const char* const target, const char* const entry_point) {
    path resolved_path = shader_path.is_absolute() ? shader_path : "..\\Data\\DX11Renderer\\Shaders" / shader_path;

    OBlob shader_bytecode;
    OBlob error_messages = nullptr;
    HRESULT hr = D3DCompileFromFile(resolved_path.c_str(),
        nullptr, // macros
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry_point,
        target,
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
        0, // More flags. Unused.
        &shader_bytecode,
        &error_messages);
    if (FAILED(hr)) {
        if (hr == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND))
            printf("The system cannot find the file specified: '%ws'.\n", resolved_path.c_str());
        else if (hr == HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND))
            printf("The system cannot find the path specified: '%ws'.\n", resolved_path.c_str());
        else if (error_messages != nullptr)
            printf("Shader error compiling '%ws::%s':\n%s.\n", resolved_path.c_str(), entry_point, (char*)error_messages->GetBufferPointer());
        else
            printf("Unknown error occured compiling: '%ws::%s'.\n", resolved_path.c_str(), entry_point);
        return nullptr;
    }

    return shader_bytecode;
}

} // NS DX11Renderer