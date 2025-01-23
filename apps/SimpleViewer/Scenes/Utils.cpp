// SimpleViewer scene utilities.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scenes/Utils.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Math/Vector.h>
#include <Bifrost/Math/Transform.h>
#include <Bifrost/Math/Quaternion.h>

#include <glTFLoader/glTFLoader.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Scene;
using namespace Bifrost::Math;

namespace Scenes {

SceneNode create_checkered_floor(float floor_size, float checker_size) {
    // 2x2 checkered texture to be repeated across the floor.
    unsigned int size = 2;
    ImageID tint_roughness_image_ID = Images::create2D("Floor color", PixelFormat::RGBA32, 2.2f, Vector2ui(size, size));
    Images::set_mipmapable(tint_roughness_image_ID, true);
    unsigned char* tint_roughness_pixels = (unsigned char*)Images::get_pixels(tint_roughness_image_ID);
    for (unsigned int y = 0; y < size; ++y) {
        for (unsigned int x = 0; x < size; ++x) {
            bool is_black = (x & 1) != (y & 1);
            unsigned char* pixel = tint_roughness_pixels + (x + y * size) * 4u;
            unsigned char intensity = is_black ? 1 : 255;
            pixel[0] = pixel[1] = pixel[2] = intensity;
            pixel[3] = is_black ? 15 : 255;
        }
    }

    // Material
    Materials::Data material_data = Materials::Data::create_dielectric(RGB::white(), 0.4f, 0.04f);
    material_data.tint_roughness_texture_ID = Textures::create2D(tint_roughness_image_ID, MagnificationFilter::None, MinificationFilter::Trilinear);
    material_data.flags = MaterialFlag::ThinWalled;
    MaterialID material_ID = Materials::create("Floor", material_data);

    // Mesh scaled to be floor size and with texture coordinates to match the checker size.
    Mesh plane_mesh = MeshCreation::plane(2, { MeshFlag::Position, MeshFlag::Texcoord });
    for (Vector3f& position_itr : plane_mesh.get_position_iterable())
        position_itr *= floor_size;
    float uv_scale = floor_size / (2 * checker_size); // Multiply by 2, as a texture is 2x2 checkers big.
    for (Vector2f& texcoord_itr : plane_mesh.get_texcoord_iterable())
        texcoord_itr = (texcoord_itr - 0.5f) * uv_scale; // Subtract 0.5 to keep high precision of texcoords near the center of the floor.

    SceneNode plane_node = SceneNodes::create("Floor");
    MeshModels::create(plane_node.get_ID(), plane_mesh.get_ID(), material_ID);
    
    return plane_node;
}

void replace_material(Material material, SceneNode parent_node, const std::string& child_scene_node_name) {
    parent_node.apply_to_children_recursively([=](SceneNode node) {
        if (node.get_name() == child_scene_node_name) {
            MeshModel mesh_model = MeshModels::get_attached_mesh_model(node.get_ID());
            if (mesh_model.exists())
                mesh_model.set_material(material);
        }
        });
}

Bifrost::Scene::SceneNode load_shader_ball(const std::filesystem::path& data_directory, Material material) {
    printf("Shader ball curtesy of https://github.com/derkreature/ShaderBall\n");
    auto shader_ball_path = data_directory / "SimpleViewer" / "Shaderball.glb";
    SceneNode shader_ball_node = glTFLoader::load(shader_ball_path.generic_string());

    // Delete original materials as they get replaced
    shader_ball_node.apply_to_children_recursively([=](SceneNode node) {
        MeshModel mesh_model = MeshModels::get_attached_mesh_model(node.get_ID());
        Material material = mesh_model.get_material();
        Materials::destroy(material.get_ID());
    });

    // Set base materials to rubber
    Materials::Data rubber_material_data = Materials::Data::create_dielectric(RGB(0.05f), 1, 0.04f);
    Material rubber_material = Materials::create("Rubber", rubber_material_data);
    for (std::string node_name : { "Node1", "Node4", "Node5" })
        replace_material(rubber_material, shader_ball_node, node_name);

    // Set surrounding surface to tested material
    for (std::string node_name : { "Node2", "Node3", "Node6" })
        replace_material(material, shader_ball_node, node_name);

    return shader_ball_node;
}

} // NS Scenes
