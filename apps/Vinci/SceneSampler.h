// Vinci triangle scene sampler.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _VINCI_SCENE_SAMPLER_H_
#define _VINCI_SCENE_SAMPLER_H_

#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/SceneNode.h>
#include <Bifrost/Scene/SceneRoot.h>

#include <vector>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

// ------------------------------------------------------------------------------------------------
// Sample the triangles in a scene linearly. (independent of triangle area)
// The reason that we ignore area is that we are actually mostly interested in edges and corners,
// where the geometry is more interesting than just a flat plane.
// ------------------------------------------------------------------------------------------------
class SceneSampler final {
public:
    SceneSampler(SceneRoot scene_root, unsigned int random_seed)
        : m_rng(RNG::XorShift32(random_seed)), m_scene_root(scene_root) {

        // Collect valid mesh models.
        m_models = std::vector<MeshModels::UID>();
        m_models.reserve(MeshModels::capacity());
        unsigned long long total_triangle_count = 0;
        for (auto mesh_model_ID : MeshModels::get_iterable())
            if (MeshModels::has(mesh_model_ID)) {
                m_models.push_back(mesh_model_ID);
                Mesh mesh = MeshModels::get_mesh_ID(mesh_model_ID);
                total_triangle_count += mesh.get_primitive_count();
            }

        // Compute model CDF.
        m_model_CDF = std::vector<float>();
        m_model_CDF.reserve(m_models.size());
        float recip_total_triangle_count = 1.0f / total_triangle_count;
        total_triangle_count = 0;
        for (auto model_ID : m_models) {
            Mesh mesh = MeshModels::get_mesh_ID(model_ID);
            int triangle_count = mesh.get_primitive_count();
            m_model_CDF.push_back(triangle_count / recip_total_triangle_count);
            total_triangle_count += triangle_count;
        }
    }

    Transform sample_world_transform() {
        // Sample mesh model.
        float mesh_u = m_rng.sample1f();
        int model_index = 0;
        while (m_model_CDF[model_index] <= mesh_u)
            ++model_index;
        MeshModel model = m_models[model_index];

        // Sample triangle index. TODO Handle meshes without index buffer
        Mesh mesh = model.get_mesh();
        int primitive_index = int(mesh.get_primitive_count() * m_rng.sample1f());
        Vector3ui vertex_indices = mesh.get_primitives()[primitive_index];

        // Sample a position on a triangle.
        Vector3f barycentric_coord = sample_triangle_uniformly(m_rng.sample2f());
        Vector3f p0 = mesh.get_positions()[vertex_indices.x];
        Vector3f p1 = mesh.get_positions()[vertex_indices.y];
        Vector3f p2 = mesh.get_positions()[vertex_indices.z];

        Vector3f position = barycentric_coord.x * p0 + barycentric_coord.y * p1 + barycentric_coord.z * p2;
        Vector3f normal;
        if (mesh.get_normals() != nullptr) {
            Vector3f* normals = mesh.get_normals();
            normal = normalize(barycentric_coord.x * normals[vertex_indices.x] +
                               barycentric_coord.y * normals[vertex_indices.y] +
                               barycentric_coord.z * normals[vertex_indices.z]);
        } else
            normal = normalize(cross(p1 - p0, p2 - p0));

        Quaternionf rotation = Quaternionf::look_in(-normal);

        Transform global_transform = model.get_scene_node().get_global_transform() * Transform(position, rotation);
        global_transform.scale = 1.0f;
        return global_transform;
    }

private:
    // https://pharr.org/matt/blog/2019/02/27/triangle-sampling-1.html
    Vector3f sample_triangle_uniformly(Vector2f uv) {
        float su0 = sqrt(uv.x);
        float b0 = 1 - su0;
        float b1 = uv.y * su0;
        return { b0, b1, 1.f - b0 - b1 };
    }

    RNG::XorShift32 m_rng;

    SceneRoot m_scene_root;

    std::vector<float> m_model_CDF;
    std::vector<MeshModels::UID> m_models;
};

#endif _VINCI_SCENE_SAMPLER_H_