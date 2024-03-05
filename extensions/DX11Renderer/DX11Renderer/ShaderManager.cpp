// DirectX 11 shader manager.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#include <DX11Renderer/ShaderManager.h>
#include <DX11Renderer/Utils.h>

#define NOMINMAX
#include <D3DCompiler.h>

#include <fstream>

using namespace std::filesystem;

namespace DX11Renderer {

// For ID3DInclude examples see
// * https://www.asawicki.info/news_1515_implementing_id3d10include
// * https://github.com/TheRealMJP/BakingLab/blob/master/SampleFramework11/v1.02/Graphics/ShaderCompilation.cpp#L131
class ShaderLibInclude : public ID3DInclude {
public:
    path m_library_path;
    path m_local_path;

    ShaderLibInclude(path library_path, path shader_path = path()) {
        m_library_path = library_path;

        m_local_path = shader_path.parent_path();
        if (!m_local_path.is_absolute())
            m_local_path = m_library_path / m_local_path;
    }

    HRESULT Open(D3D_INCLUDE_TYPE include_type, LPCSTR shader_path, LPCVOID parent_shader_source, LPCVOID* shader_source_ref, UINT* shader_source_size) override {
        using namespace std::filesystem;

        path resolved_path = resolve_shader_path(include_type, path(shader_path));
        if (!exists(resolved_path))
            return E_FAIL;

        // Read shader source
        std::ifstream file_stream(resolved_path.c_str(), std::ios::in);
        std::string shader_content_str = read_stream(file_stream);

        // Copy source to output parameters
        *shader_source_size = UINT(shader_content_str.size());
        char* shader_source = new char[*shader_source_size];
        std::copy(shader_content_str.begin(), shader_content_str.end(), shader_source);
        *shader_source_ref = shader_source;

        return S_OK;
    }

    HRESULT Close(LPCVOID shader_source) override {
        delete[] (char*)shader_source;
        return S_OK;
    }

    path resolve_shader_path(D3D_INCLUDE_TYPE include_type, const path& shader_path) const {
        const path& path_prefix = include_type == D3D_INCLUDE_LOCAL ? m_local_path : m_library_path;
        return path_prefix / shader_path;
    }

    static std::string read_stream(std::istream &input) {
        // The characters in the stream are read one-by-one using a std::streambuf.
        // That is faster than reading them one-by-one using the std::istream.
        std::streambuf *buffer = input.rdbuf();
        
        // Copy the content untill the end of the file.
        std::string content = "";
        int c;
        while ((c = buffer->sbumpc()) != EOF)
            content += static_cast<char>(c);
        
        return content;
    }
};

ShaderManager::ShaderManager() {
    path data_directory;
    get_data_directory(data_directory);
    m_library_path = data_directory / "DX11Renderer" / "Shaders";
}

ShaderManager::ShaderManager(const path& data_directory)
    : m_library_path(data_directory / "DX11Renderer" / "Shaders") { }

OBlob ShaderManager::compile_shader_source(const char* const shader_src, const char* const target, const char* const entry_point) const {
    ShaderLibInclude shader_lib_include = ShaderLibInclude(m_library_path);
    OBlob shader_bytecode;
    OBlob error_messages = nullptr;
    HRESULT hr = D3DCompile(shader_src, strlen(shader_src), nullptr,
        nullptr, // macroes
        &shader_lib_include,
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

OBlob ShaderManager::compile_shader_from_file(const path& shader_path, const char* const target, const char* const entry_point) const {
    ShaderLibInclude shader_lib_include = ShaderLibInclude(m_library_path, shader_path);
    path resolved_path = shader_path.is_absolute() ? shader_path : shader_lib_include.resolve_shader_path(D3D_INCLUDE_SYSTEM, shader_path);

    OBlob shader_bytecode;
    OBlob error_messages = nullptr;
    HRESULT hr = D3DCompileFromFile(resolved_path.c_str(),
        nullptr, // macros
        &shader_lib_include,
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

OBlob ShaderManager::compile_shader_from_file(const char* const shader_path, const char* const target, const char* const entry_point) const {
    path converted_path = path(shader_path);
    return compile_shader_from_file(converted_path, target, entry_point);
}

} // NS DX11Renderer