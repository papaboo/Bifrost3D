// DirectX 11 environment manager.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_ENVIRONMENT_MANAGER_H_
#define _DX11RENDERER_RENDERER_ENVIRONMENT_MANAGER_H_

#include "Dx11Renderer/Types.h"

#include "Bifrost/Math/Matrix.h"

#include <vector>

namespace std::filesystem { class path; }

namespace DX11Renderer {

class TextureManager;

//-------------------------------------------------------------------------------------------------
// Environment manager.
// Convolutes and uploads environments.
// Future work:
// * Fast Filtering of Reflection Probes https://www.ppsloan.org/publications/ggx_filtering.pdf
//-------------------------------------------------------------------------------------------------
class EnvironmentManager {
public:

    EnvironmentManager(ID3D11Device1& device, const std::filesystem::path& shader_directory, TextureManager& textures);

    // Render an environment to the active backbuffer.
    bool render(ID3D11DeviceContext1& render_context, int environment_ID);

    void handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& device_context);

private:
    EnvironmentManager(EnvironmentManager& other) = delete;
    EnvironmentManager& operator=(EnvironmentManager& rhs) = delete;

    struct Environment {
        float4 tint;
        int texture_ID;
        OTexture2D texture2D;
        OShaderResourceView srv;
    };

    TextureManager& m_textures;
    std::vector<Environment> m_envs = std::vector<Environment>(0);
    OSamplerState m_sampler;

    OVertexShader m_vertex_shader;
    OPixelShader m_pixel_shader;
    OComputeShader m_convolution_shader;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_ENVIRONMENT_MANAGER_H_