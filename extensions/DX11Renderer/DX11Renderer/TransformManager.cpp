// DirectX 11 transform manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Dx11Renderer/TransformManager.h"
#include "Dx11Renderer/Utils.h"

#include <Bifrost/Math/Conversions.h>
#include <Bifrost/Scene/SceneNode.h>

using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace DX11Renderer {

TransformManager::TransformManager(ID3D11Device1& device, ID3D11DeviceContext1& context) {
    m_transforms.resize(1);
    m_transforms[0] = Transform::identity();

    m_GPU_transforms.resize(1);
    auto identity_matrix = Matrix4x4f::identity();
    create_constant_buffer(device, identity_matrix, &m_GPU_transforms[0]);
}

void TransformManager::handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& context) {
    if (SceneNodes::get_changed_nodes().is_empty())
        return;

    if (m_transforms.size() < SceneNodes::capacity()) {
        m_transforms.resize(SceneNodes::capacity());
        m_GPU_transforms.resize(SceneNodes::capacity());
    }

    for (SceneNodeID node_ID : SceneNodes::get_changed_nodes()) {
        auto node_changes = SceneNodes::get_changes(node_ID);
        if (node_changes.is_set(SceneNodes::Change::Destroyed))
            continue;

        if (node_changes.any_set(SceneNodes::Change::Created, SceneNodes::Change::Transform)) {
            m_transforms[node_ID] = SceneNodes::get_global_transform(node_ID);
            Matrix4x4f to_world = to_matrix4x4(m_transforms[node_ID]);
            if (m_GPU_transforms[node_ID] == nullptr)
                create_constant_buffer(device, to_world, &m_GPU_transforms[node_ID], D3D11_USAGE_DEFAULT);
            else
                context.UpdateSubresource(m_GPU_transforms[node_ID], 0u, nullptr, &to_world, 0u, 0u);
        }
    }
}

} // NS DX11Renderer