// DirectX 11 shader manager.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_SHADER_MANAGER_H_
#define _DX11RENDERER_SHADER_MANAGER_H_

#include <DX11Renderer/Types.h>

#include <filesystem>

//-------------------------------------------------------------------------------------------------
// Forward declarations.
//-------------------------------------------------------------------------------------------------
namespace std::filesystem { class path; }

namespace DX11Renderer {

//-------------------------------------------------------------------------------------------------
// DirectX 11 shader manager.
// Used to compile shader sources.
//-------------------------------------------------------------------------------------------------
class ShaderManager final {
public:

    ShaderManager();
    ShaderManager(const std::filesystem::path& data_directory);
    ShaderManager(ShaderManager&& other) = default;
    ShaderManager(const ShaderManager& other) = default;

    ShaderManager& operator=(ShaderManager&& rhs) = default;
    ShaderManager& operator=(const ShaderManager& rhs) = default;

    OBlob ShaderManager::compile_shader_source(const char* const shader_src, const char* const target, const char* const entry_point) const;
    OBlob compile_shader_from_file(const std::filesystem::path& shader_path, const char* const target, const char* const entry_point) const;
    OBlob compile_shader_from_file(const char* const shader_path, const char* const target, const char* const entry_point) const;

private:
    std::filesystem::path m_library_path;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_SHADER_MANAGER_H_