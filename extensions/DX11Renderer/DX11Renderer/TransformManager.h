// DirectX 11 transform manager.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_TRANSFORM_MANAGER_H_
#define _DX11RENDERER_RENDERER_TRANSFORM_MANAGER_H_

#include "Dx11Renderer/ConstantBufferArray.h"
#include "Dx11Renderer/Types.h"

#include <Bifrost/Math/Matrix.h>
#include <Bifrost/Math/Transform.h>

#include <vector>

namespace DX11Renderer {

//-------------------------------------------------------------------------------------------------
// Transform manager.
// Uploads and manages a buffer of transforms.
// Future work:
// * Upload to a single buffer and use that buffer for GPU frustum culling.
//   See NVIDIA's CAD/CAE pipeline presentations (Tavenrath)
//-------------------------------------------------------------------------------------------------
class TransformManager {
public:

    TransformManager() = default;
    TransformManager(ID3D11Device1& device, ID3D11DeviceContext1& context);
    TransformManager(TransformManager&& other) = default;
    TransformManager& operator=(TransformManager&& rhs) = default;
    
    inline Bifrost::Math::Transform& get_transform(unsigned int transform_index) { return m_transforms[transform_index]; }
    inline void bind_transform(ID3D11DeviceContext1& context, unsigned int slot, unsigned int transform_index) { 
        context.VSSetConstantBuffers(slot, 1, &m_GPU_transforms[transform_index]);
    }

    void handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& context);

private:
    TransformManager(TransformManager& other) = delete;
    TransformManager& operator=(TransformManager& rhs) = delete;

    std::vector<Bifrost::Math::Transform> m_transforms;
    std::vector<OBuffer> m_GPU_transforms;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_TRANSFORM_MANAGER_H_