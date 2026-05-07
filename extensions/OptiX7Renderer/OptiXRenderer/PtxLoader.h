// OptiX renderer ptx loader.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_PTX_LOADER_H_
#define _OPTIXRENDERER_PTX_LOADER_H_

#include <string>

namespace OptiXRenderer::PtxLoader {

std::string load_ptx_using_nvrtc(const std::string& relative_shader_path);

std::string load_precompiled_ptx(const std::string& relative_shader_path);

inline std::string load_ptx(const std::string& relative_shader_path) {
#ifdef CUDA_NVRTC_ENABLED
    return load_ptx_using_nvrtc(relative_shader_path);
#else
    return load_precompiled_ptx(relative_shader_path);
#endif
}

} // NS OptiXRenderer::PtxLoader

#endif // _OPTIXRENDERER_PTX_LOADER_H_