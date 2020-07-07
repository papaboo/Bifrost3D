// Vinci scene generator
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _VINCI_SCENE_GENERATOR_H_
#define _VINCI_SCENE_GENERATOR_H_

#include "TextureManager.h"

#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/Material.h>
#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/SceneNode.h>

#include <vector>

namespace SceneGenerator {

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

class RandomScene {
public:
    RandomScene(int seed, Cameras::UID camera_ID, std::string& texture_directory) 
    : m_seed(seed), m_camera_ID(camera_ID), m_textures(TextureManager(texture_directory)), m_root_node(SceneNodes::create("root node")) {
        new_scene();
    }

    inline SceneNode get_root_node() const { return m_root_node; }

    inline bool has_scene() const { return m_mesh_models.size() > 0; }

    void new_scene() {
        clear_scene();

        auto rng = RNG::LinearCongruential(m_seed);
        int node_count = 4 + int(rng.sample1f() * 44);

        typedef Mesh(*MeshGenerator)(RNG::LinearCongruential&);
        int mesh_generator_count = 4;
        MeshGenerator mesh_generators[] = { generate_cube, generate_cylinder, generate_sphere, generate_torus };

        m_root_node = SceneNodes::create("root node");
        for (int n = 0; n < node_count; ++n) {
            auto mesh_generator = mesh_generators[rng.sample1ui() % mesh_generator_count];
            Mesh mesh = mesh_generator(rng);
            m_meshes.push_back(mesh);
            SceneNode node = generate_scene_node(rng, mesh);
            node.set_parent(m_root_node);
        }

        m_seed = rng.get_seed();
    }

    ~RandomScene() {
        clear_scene();
        SceneNodes::destroy(m_root_node.get_ID());
    }

    void clear_scene() {
        for (auto mesh : m_meshes)
            Meshes::destroy(mesh.get_ID());
        m_meshes.clear();
        for (auto material : m_materials)
            Materials::destroy(material.get_ID());
        m_materials.clear();
        for (auto node : m_nodes)
            SceneNodes::destroy(node.get_ID());
        m_nodes.clear();
        for (auto model : m_mesh_models)
            MeshModels::destroy(model.get_ID());
        m_mesh_models.clear();
    }

private:

    unsigned int m_seed;

    Cameras::UID m_camera_ID;

    // Collections of objects in the scene.
    TextureManager m_textures;
    SceneNode m_root_node;
    std::vector<Mesh> m_meshes;
    std::vector<Material> m_materials;
    std::vector<SceneNode> m_nodes;
    std::vector<MeshModel> m_mesh_models;

    SceneNode generate_scene_node(RNG::LinearCongruential& rng, Mesh mesh) {

        // Generate random position in front of the camera
        auto bounds = mesh.get_bounds();
        float radius = magnitude(bounds.center() - bounds.minimum);
        Ray ray = CameraUtils::ray_from_viewport_point(m_camera_ID, rng.sample2f());
        float t = radius + 3 * rng.sample1f() * radius;
        Vector3f translation = ray.position_at(t);

        // Generate random transform
        Vector3f axis = normalize(rng.sample3f() * 2 - 1);
        Quaternionf rotation = Quaternionf::from_angle_axis(rng.sample1f() * 2 * PI<float>(), axis);
        Transform transform = Transform(translation, rotation);

        // Generate random material
        auto material_ID = m_textures.generate_random_material(rng);
        m_materials.push_back(material_ID);

        // Assemble in scene node.
        SceneNodes::UID node_ID = SceneNodes::create("Node", transform);
        m_nodes.push_back(node_ID);
        auto mesh_model = MeshModels::create(node_ID, mesh.get_ID(), material_ID);
        m_mesh_models.push_back(mesh_model);

        return node_ID;
    }

    static Mesh generate_cube(RNG::LinearCongruential& rng) {
        Vector3f scaling = rng.sample3f() * 1.5f + 0.5f;
        return MeshCreation::cube(1, scaling);
    }

    static Mesh generate_cylinder(RNG::LinearCongruential& rng) {
        Mesh cylinder = MeshCreation::cylinder(1, 128);
        return scale_mesh(cylinder, rng);
    }

    static Mesh generate_sphere(RNG::LinearCongruential& rng) {
        Mesh sphere = MeshCreation::revolved_sphere(128, 128);
        return scale_mesh(sphere, rng);
    }

    static Mesh generate_torus(RNG::LinearCongruential& rng) {
        Mesh torus = MeshCreation::torus(128, 128, 0.01f + 2.0f * rng.sample1f());
        return scale_mesh(torus, rng);
    }

    // --------------------------------------------------------------------------------------------
    // Mesh modifications
    // --------------------------------------------------------------------------------------------

    static Mesh scale_mesh(Mesh mesh, RNG::LinearCongruential& rng) {
        Vector3f scaling = rng.sample3f() * 1.5f + 0.5f;
        Vector3f* positions = mesh.get_positions();
        for (unsigned int i = 0; i < mesh.get_vertex_count(); ++i)
            positions[i] *= scaling;
        MeshUtils::compute_normals(mesh.get_ID());

        AABB bounds = mesh.get_bounds();
        bounds.minimum *= scaling;
        bounds.maximum *= scaling;
        mesh.set_bounds(bounds);

        return mesh;
    }
};

} // NS SceneGenerator

#endif // _VINCI_SCENE_GENERATOR_H_