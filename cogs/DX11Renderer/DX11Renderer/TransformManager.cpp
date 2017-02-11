// DirectX 11 transform manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Dx11Renderer/TransformManager.h"
#include "Dx11Renderer/Utils.h"

#include <Cogwheel/Math/Conversions.h>
#include <Cogwheel/Scene/SceneNode.h>

using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

namespace DX11Renderer {

TransformManager::TransformManager(ID3D11Device1& device, ID3D11DeviceContext1& context) {
    unsigned int transform_count = 128;

    m_transforms.resize(transform_count);
    m_transforms[0] = Transform::identity();

    m_constant_array = ConstantBufferArray<Matrix4x4f>(&device, transform_count);
    m_constant_array.set(&context, Matrix4x4f::identity(), 0, D3D11_COPY_DISCARD);
}
    
void TransformManager::handle_updates(ID3D11DeviceContext1& context) {
    if (SceneNodes::get_changed_nodes().is_empty())
        return;

    if (m_transforms.size() <= SceneNodes::capacity())
        m_transforms.resize(SceneNodes::capacity());

    for (SceneNodes::UID node_ID : SceneNodes::get_changed_nodes()) {
        if (SceneNodes::get_changes(node_ID).any_set(SceneNodes::Change::Created, SceneNodes::Change::Transform)) {
            m_transforms[node_ID] = SceneNodes::get_global_transform(node_ID);

            Matrix4x4f to_world = to_matrix4x4(m_transforms[node_ID]);
            m_constant_array.set(&context, to_world, node_ID);
        }
    }
}

} // NS DX11Renderer