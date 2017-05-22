// DirectX 11 environment manager.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_ENVIRONMENT_MANAGER_H_
#define _DX11RENDERER_RENDERER_ENVIRONMENT_MANAGER_H_

#include "Dx11Renderer/Types.h"

#include "Cogwheel/Math/Matrix.h"

#include <vector>

namespace DX11Renderer {

class TextureManager;

//-------------------------------------------------------------------------------------------------
// Environment manager.
// Convolutes and uploads environments.
// Future work:
// * Convolute on the GPU.
//-------------------------------------------------------------------------------------------------
class EnvironmentManager {
public:

    EnvironmentManager(ID3D11Device1& device, const std::wstring& shader_folder_path, TextureManager& textures);
    ~EnvironmentManager();

    // Render an environment to the active backbuffer.
    bool render(ID3D11DeviceContext1& render_context, int environment_ID);

    void handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& device_context);

private:
    EnvironmentManager(EnvironmentManager& other) = delete;
    EnvironmentManager& operator=(EnvironmentManager& rhs) = delete;

    struct Environment {
        float4 tint;
        int texture_ID;
        ID3D11Texture2D* texture2D;
        ID3D11ShaderResourceView* srv;
    };

    TextureManager& m_textures;
    std::vector<Environment> m_envs = std::vector<Environment>(0);
    ID3D11SamplerState* m_sampler;

    ID3D11VertexShader* m_vertex_shader;
    ID3D11PixelShader* m_pixel_shader;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_ENVIRONMENT_MANAGER_H_