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
    Image tint_roughness_image = Image::create2D("Floor color", PixelFormat::RGBA32, true, Vector2ui(size, size));
    tint_roughness_image.set_mipmapable(true);
    unsigned char* tint_roughness_pixels = (unsigned char*)tint_roughness_image.get_pixels();
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
    material_data.tint_roughness_texture_ID = Texture::create2D(tint_roughness_image, MagnificationFilter::None, MinificationFilter::Trilinear).get_ID();
    material_data.flags = MaterialFlag::ThinWalled;
    Material material = Material("Floor", material_data);

    // Mesh scaled to be floor size and with texture coordinates to match the checker size.
    Mesh plane_mesh = MeshCreation::plane(2, { MeshFlag::Position, MeshFlag::Texcoord });
    for (Vector3f& position_itr : plane_mesh.get_position_iterable())
        position_itr *= floor_size;
    float uv_scale = floor_size / (2 * checker_size); // Multiply by 2, as a texture is 2x2 checkers big.
    for (Vector2f& texcoord_itr : plane_mesh.get_texcoord_iterable())
        texcoord_itr = (texcoord_itr - 0.5f) * uv_scale; // Subtract 0.5 to keep high precision of texcoords near the center of the floor.

    SceneNode plane_node = SceneNode("Floor");
    MeshModel(plane_node, plane_mesh, material);
    
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

Bifrost::Scene::SceneNode load_shader_ball(const std::filesystem::path& resource_directory, Material material) {
    printf("Mori knob curtesy of Yasutoshi Mori\n");
    auto shader_ball_path = resource_directory / "Shaderball.gltf";
    SceneNode shader_ball_node = glTFLoader::load(shader_ball_path.generic_string());

    // Rubber material for the inside
    Material rubber_material = Material::create_dielectric("Rubber", RGB(0.05f), 1);

    const std::string inside_node_name = "Node2";
    const std::string outside_node_name = "Node5";

    shader_ball_node.apply_to_children_recursively([&](SceneNode node) {
        std::string& node_name = node.get_name();
        MeshModel mesh_model = MeshModels::get_attached_mesh_model(node.get_ID());
        if (mesh_model.exists()) {
            if (node_name == outside_node_name)
                mesh_model.set_material(material);
            else if (node_name == inside_node_name)
                mesh_model.set_material(rubber_material);
            else {
                // Delete anything but the shaderball.
                mesh_model.get_mesh().destroy();
                mesh_model.get_material().destroy();
                mesh_model.destroy();
            }
        }
    });

    return shader_ball_node;
}

} // NS Scenes
