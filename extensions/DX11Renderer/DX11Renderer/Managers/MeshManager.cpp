// DirectX 11 mesh manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Dx11Renderer/Managers/MeshManager.h"
#include "Dx11Renderer/Utils.h"

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Math/OctahedralNormal.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

namespace DX11Renderer::Managers {

void release_mesh(Dx11Mesh& mesh) {
    mesh.index_count = mesh.vertex_count = mesh.vertex_buffer_count = 0;
    safe_release(&mesh.constant_buffer);
    safe_release(&mesh.indices);
    for (int b = 0; b < Dx11Mesh::MAX_BUFFER_COUNT; ++b)
        safe_release(mesh.vertex_buffers + b);
}

MeshManager::MeshManager(ID3D11Device1& device) {
    m_meshes = std::vector<Dx11Mesh>(Meshes::capacity(), Dx11Mesh::invalid());

    D3D11_BUFFER_DESC empty_desc = {};
    empty_desc.Usage = D3D11_USAGE_IMMUTABLE;
    empty_desc.ByteWidth = sizeof(Vector4f);
    empty_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    Vector4f lval = Vector4f::zero();
    D3D11_SUBRESOURCE_DATA empty_data = {};
    empty_data.pSysMem = &lval;
    THROW_DX11_ERROR(device.CreateBuffer(&empty_desc, &empty_data, &m_null_buffer));
}

MeshManager::~MeshManager() {
    for (Dx11Mesh& mesh : m_meshes)
        release_mesh(mesh);
}

template <typename T>
HRESULT upload_default_buffer(ID3D11Device1& device, T* data, int element_count, D3D11_BIND_FLAG flags, ID3D11Buffer** buffer) {
    D3D11_BUFFER_DESC desc = {};
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.ByteWidth = sizeof(T) * element_count;
    desc.BindFlags = flags;

    D3D11_SUBRESOURCE_DATA resource_data = {};
    resource_data.pSysMem = data;
    return device.CreateBuffer(&desc, &resource_data, buffer);
}

template <typename T>
HRESULT upload_index_buffer(ID3D11Device1& device, T* data, int element_count, ID3D11Buffer** buffer) {
    return upload_default_buffer(device, data, element_count, D3D11_BIND_INDEX_BUFFER, buffer);
}

template <typename T>
HRESULT upload_vertex_buffer(ID3D11Device1& device, T* data, int element_count, ID3D11Buffer** buffer) {
    return upload_default_buffer(device, data, element_count, D3D11_BIND_VERTEX_BUFFER, buffer);
}

void MeshManager::handle_updates(ID3D11Device1& device) {
    for (MeshID mesh_ID : Meshes::get_changed_meshes()) {
        Bifrost::Assets::Mesh mesh = mesh_ID;

        if (m_meshes.size() <= mesh_ID)
            m_meshes.resize(Meshes::capacity(), Dx11Mesh::invalid());

        auto mesh_changes = mesh.get_changes();
        if (mesh_changes.any_set(Meshes::Change::Created, Meshes::Change::Destroyed)) {
            Dx11Mesh& dx_mesh = m_meshes[mesh_ID];
            if (dx_mesh.vertex_count != 0) // Nothing to release
                release_mesh(dx_mesh);
        }

        // Destroyed meshes have been released above.
        if (mesh_changes.is_set(Meshes::Change::Destroyed))
            continue;
        assert(mesh.exists());

        if (mesh_changes.is_set(Meshes::Change::Created)) {
            Dx11Mesh dx_mesh = {};

            Bifrost::Math::AABB bounds = mesh.get_bounds();
            dx_mesh.bounds = { make_float3(bounds.minimum), make_float3(bounds.maximum) };

            // Expand the indexed buffers if an index buffer is used, but no normals are given.
            // In that case we need to compute hard normals per triangle and we can only store that for non-indexed buffers.
            // NOTE Alternatively look into storing the hard normals in a buffer and index into it based on the triangle ID?
            bool expand_indexed_buffers = mesh.get_primitive_count() != 0 && mesh.get_normals() == nullptr;

            if (!expand_indexed_buffers) { // Upload indices.
                dx_mesh.index_count = mesh.get_index_count();

                HRESULT hr = upload_index_buffer(device, mesh.get_primitives(), dx_mesh.index_count / 3, &dx_mesh.indices);
                if (FAILED(hr))
                    printf("Could not upload '%s' index buffer.\n", mesh.get_name().c_str());
            }

            dx_mesh.vertex_count = mesh.get_vertex_count();
            Vector3f* positions = mesh.get_positions();

            if (expand_indexed_buffers) {
                // Expand the positions.
                dx_mesh.vertex_count = mesh.get_index_count();
                positions = MeshUtils::expand_indexed_buffer(mesh.get_primitives(), mesh.get_primitive_count(), mesh.get_positions());
            }

            { // Upload geometry.
                auto create_vertex_geometry = [](Vector3f p, Vector3f n) -> Dx11VertexGeometry {
                    float3 dx_position = { p.x, p.y, p.z };
                    OctahedralNormal encoded_normal = OctahedralNormal::encode_precise(n);
                    int2 dx_normal = { encoded_normal.encoding.x, encoded_normal.encoding.y };
                    int packed_dx_normal = (dx_normal.x - SHRT_MIN) | (dx_normal.y << 16);
                    Dx11VertexGeometry geometry = { dx_position, packed_dx_normal };
                    return geometry;
                };

                Vector3f* normals = mesh.get_normals();

                Dx11VertexGeometry* geometry = new Dx11VertexGeometry[dx_mesh.vertex_count];
                if (normals == nullptr) {
                    // Compute hard normals. Positions have already been expanded if there is an index buffer.
#pragma omp parallel for
                    for (int i = 0; i < int(dx_mesh.vertex_count); i += 3) {
                        Vector3f p0 = positions[i], p1 = positions[i + 1], p2 = positions[i + 2];
                        Vector3f normal = normalize(cross(p1 - p0, p2 - p0));
                        geometry[i] = create_vertex_geometry(p0, normal);
                        geometry[i + 1] = create_vertex_geometry(p1, normal);
                        geometry[i + 2] = create_vertex_geometry(p2, normal);
                    }
                } else {
                    // Copy position and normal.
#pragma omp parallel for
                    for (int i = 0; i < int(dx_mesh.vertex_count); ++i)
                        geometry[i] = create_vertex_geometry(positions[i], normals[i]);
                }

                HRESULT hr = upload_vertex_buffer(device, geometry, dx_mesh.vertex_count, dx_mesh.geometry_address());
                if (FAILED(hr))
                    printf("Could not upload %s's geometry buffer.\n", mesh.get_name().c_str());

                delete[] geometry;
            }

            // Delete temporary expanded positions.
            if (positions != mesh.get_positions())
                delete[] positions;

            { // Upload texcoords if present, otherwise upload 'null buffer'.
                Vector2f* texcoords = mesh.get_texcoords();
                if (texcoords != nullptr) {

                    if (expand_indexed_buffers)
                        texcoords = MeshUtils::expand_indexed_buffer(mesh.get_primitives(), mesh.get_primitive_count(), texcoords);

                    HRESULT hr = upload_vertex_buffer(device, texcoords, dx_mesh.vertex_count, dx_mesh.texcoords_address());
                    if (FAILED(hr))
                        printf("Could not upload %s's texcoord buffer.\n", mesh.get_name().c_str());

                    if (texcoords != mesh.get_texcoords())
                        delete[] texcoords;
                } else {
                    m_null_buffer.get()->AddRef();
                    *dx_mesh.texcoords_address() = m_null_buffer;
                }
            }

            { // Upload tints and roughness if present, otherwise upload 'null buffer'.
                TintRoughness* tints = mesh.get_tint_and_roughness();
                if (tints != nullptr) {

                    if (expand_indexed_buffers)
                        tints = MeshUtils::expand_indexed_buffer(mesh.get_primitives(), mesh.get_primitive_count(), tints);

                    HRESULT hr = upload_vertex_buffer(device, tints, dx_mesh.vertex_count, dx_mesh.tint_and_roughness_address());
                    if (FAILED(hr))
                        printf("Could not upload %s's tint and roughness buffer.\n", mesh.get_name().c_str());

                    if (tints != mesh.get_tint_and_roughness())
                        delete[] tints;
                } else {
                    m_null_buffer.get()->AddRef();
                    *dx_mesh.tint_and_roughness_address() = m_null_buffer;
                }
            }

            // Constant buffer
            Dx11MeshConstans mesh_constants;
            mesh_constants.has_tint_and_roughness = mesh.get_tint_and_roughness() != nullptr;
            THROW_DX11_ERROR(create_constant_buffer(device, mesh_constants, &dx_mesh.constant_buffer));

            // Set the buffer count to the minimal number of buffers containing data.
            dx_mesh.vertex_buffer_count = Dx11Mesh::MAX_BUFFER_COUNT;
            while (dx_mesh.vertex_buffer_count > 1 && m_null_buffer == dx_mesh.vertex_buffers[dx_mesh.vertex_buffer_count - 1])
                dx_mesh.vertex_buffer_count--;

            m_meshes[mesh_ID] = dx_mesh;
        }
    }
}

} // NS DX11Renderer::Managers