// DirectX 11 transform manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_TRANSFORM_MANAGER_H_
#define _DX11RENDERER_RENDERER_TRANSFORM_MANAGER_H_

#include "Dx11Renderer/ConstantBufferArray.h"
#include "Dx11Renderer/Types.h"

#include <Cogwheel/Math/Matrix.h>
#include <Cogwheel/Math/Transform.h>

#include <vector>

namespace DX11Renderer {

//----------------------------------------------------------------------------
// Transform manager.
// Uploads and manages a buffer of transforms.
// Future work:
// * Support a variable number of transforms.
//----------------------------------------------------------------------------
class TransformManager {
private:

    std::vector<Cogwheel::Math::Transform> m_transforms;

    ConstantBufferArray<Cogwheel::Math::Matrix4x4f> m_constant_array;
    
public:

    TransformManager() = default;
    TransformManager(ID3D11Device1& device, ID3D11DeviceContext1& context);
    TransformManager(TransformManager&& other) = default;
    TransformManager& operator=(TransformManager&& rhs) = default;
    
    inline Cogwheel::Math::Transform& get_transform(unsigned int transform_index) { return m_transforms[transform_index]; }
    inline ID3D11Buffer** get_constant_buffer_addr() { return m_constant_array.get_buffer_addr(); }
    inline void bind_transform(ID3D11DeviceContext1& context, unsigned int slot, unsigned int transform_index) { m_constant_array.VS_set(&context, slot, transform_index); }

    void handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& context);

private:
    TransformManager(TransformManager& other) = delete;
    TransformManager& operator=(TransformManager& rhs) = delete;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_TRANSFORM_MANAGER_H_