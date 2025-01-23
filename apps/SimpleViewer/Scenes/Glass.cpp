// SimpleViewer sphere scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scenes/Sphere.h>
#include <Scenes/Utils.h>

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Math/Transform.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>

#include <glTFLoader/glTFLoader.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace Scenes {

void create_glass_scene(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::SceneNode root_node, const std::filesystem::path& data_directory) {

    MeshFlags positions_and_normals = { MeshFlag::Position, MeshFlag::Normal };

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 3.0f, -10.0f);
        cam_transform.look_at(Vector3f(0, 1.0f, 0.0f));
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Add a directional light.
        Transform light_transform = Transform(Vector3f(20.0f, 20.0f, -20.0f));
        light_transform.look_at(Vector3f::zero());
        SceneNode light_node = SceneNodes::create("light", light_transform);
        light_node.set_parent(root_node);
        LightSources::create_directional_light(light_node.get_ID(), RGB(3.0f, 2.9f, 2.5f));
    }

    { // Create floor.
        SceneNode floor_node = create_checkered_floor(400, 1);
        floor_node.set_global_transform(Transform(Vector3f(0, -1.0f, 0)));
        floor_node.set_parent(root_node);
    }

    { // Add glass shaderball
        Materials::Data glass_material_data = Materials::Data::create_glass(RGB(0.95f), 0.25f, glass_specularity);
        Material glass_material = Materials::create("Glass shader ball", glass_material_data);

        SceneNode shader_ball_node = load_shader_ball(data_directory, glass_material);
        Transform transform = Transform(Vector3f::zero(), Quaternionf::from_angle_axis(1.1f * PI<float>(), Vector3f::up()), 0.01f);
        shader_ball_node.set_global_transform(transform);
        shader_ball_node.apply_delta_transform(Transform(Vector3f(0, -102, 0)));
        shader_ball_node.set_parent(root_node);
    }

    { // Magnifying glass
        SceneNode magnifying_glass_node = SceneNodes::create("Magnifying glass");

        { // Glass
            Materials::Data glass_material_data = Materials::Data::create_glass(RGB(0.975f), 0.0f, glass_specularity);
            Material glass_material = Materials::create("Magnifying glass", glass_material_data);

            // Create sphere and flatten it to use as the glass in the magnifying glass.
            Mesh glass_mesh = MeshCreation::revolved_sphere(64, 32, positions_and_normals);
            Vector3f* positions = glass_mesh.get_positions();
            Vector3f* normals = glass_mesh.get_normals();
            for (unsigned int v = 0; v < glass_mesh.get_vertex_count(); ++v) {
                positions[v].z *= 0.1f;
                normals[v].x *= 0.1f;
                normals[v].y *= 0.1f;
                normals[v] = normalize(normals[v]);
            }

            SceneNode glass_node = SceneNodes::create("Glass");
            MeshModels::create(glass_node.get_ID(), glass_mesh.get_ID(), glass_material.get_ID());
            glass_node.set_parent(magnifying_glass_node);
        }

        { // Frame
            Materials::Data frame_material_data = Materials::Data::create_metal(gold_tint, 0.5f);
            Material frame_material = Materials::create("Magnifying glass frame", frame_material_data);

            Mesh frame_mesh = MeshCreation::torus(64, 64, 0.05f, positions_and_normals);

            SceneNode frame_node = SceneNodes::create("Magnifying glass frame");
            Quaternionf rotation = Quaternionf::from_angle_axis(0.5f * PI<float>(), Vector3f::right());
            frame_node.set_global_transform(Transform(Vector3f::zero(), rotation));
            MeshModels::create(frame_node.get_ID(), frame_mesh.get_ID(), frame_material.get_ID());
            frame_node.set_parent(magnifying_glass_node);

            Mesh handle_mesh = MeshCreation::cylinder(1, 64, positions_and_normals);
            for (Vector3f& position : handle_mesh.get_position_iterable()) {
                position.x *= 0.08f;
                position.z *= 0.08f;
                position.y *= 0.6f;
            }

            SceneNode handle_node = SceneNodes::create("Magnifying glass handle", Transform(Vector3f(0, -0.8f, 0)));
            MeshModels::create(handle_node.get_ID(), handle_mesh.get_ID(), frame_material.get_ID());
            handle_node.set_parent(magnifying_glass_node);
        }

        // Move magnifying glass
        magnifying_glass_node.set_local_transform(Transform(Vector3f(3, 0.1f, 0)));
        magnifying_glass_node.set_parent(root_node);
    }

    { // Diamond
        auto diamond_path = data_directory / "SimpleViewer" / "Diamond.glb";
        SceneNode diamond_node = glTFLoader::load(diamond_path.generic_string());
        Quaternionf rotation = Quaternionf::from_angle_axis(-0.5f * PI<float>(), Vector3f::right());
        diamond_node.set_global_transform(Transform(Vector3f(-3, 0, 0), rotation));
        diamond_node.set_parent(root_node);

        Materials::Data diamond_material_data = Materials::Data::create_glass(RGB(1.0f), 0.0f, diamond_specularity);
        Material diamond_material = Materials::create("Diamond", diamond_material_data);
        replace_material(diamond_material, diamond_node, "pCone1_DiamondOutside_0");
    }

    { // Pool of water. The pool is 2x0.5x2, with two levels of tiles.
        SceneNode pool_node = SceneNodes::create("Pool");

        { // Pool sides grout
            Materials::Data grout_material_data = Materials::Data::create_dielectric(RGB(0.2f, 0.2f, 0.2f), 0.9f, 0.04f);
            Material grout_material = Materials::create("Pool tile", grout_material_data);

            Mesh grout_side1 = MeshCreation::cube(1, Vector3f(2.0f, 0.5f, 0.25f), positions_and_normals);
            Mesh grout_side2 = MeshCreation::cube(1, Vector3f(0.25f, 0.5f, 1.5f), positions_and_normals);
            Mesh grout_floor = MeshCreation::cube(1, Vector3f(1.5f, 0.02f, 1.5f), positions_and_normals);
            MeshUtils::TransformedMesh meshes[5] = {
                { grout_side1.get_ID(), Transform(Vector3f(0, 0, 0.875f)) },
                { grout_side1.get_ID(), Transform(Vector3f(0, 0, -0.875f)) },
                { grout_side2.get_ID(), Transform(Vector3f(0.875f, 0, 0)) },
                { grout_side2.get_ID(), Transform(Vector3f(-0.875f, 0, 0)) },
                { grout_floor.get_ID(), Transform(Vector3f(0, -0.24f, 0)) },
            };
            Mesh grout_mesh = MeshUtils::combine("Pool grout", meshes, meshes + 5);
            Meshes::destroy(grout_side1.get_ID());
            Meshes::destroy(grout_side2.get_ID());
            Meshes::destroy(grout_floor.get_ID());

            SceneNode grout_node = SceneNodes::create("Pool grout");
            MeshModels::create(grout_node.get_ID(), grout_mesh.get_ID(), grout_material.get_ID());
            grout_node.set_parent(pool_node);
        }

        { // Tiles
            Materials::Data tile_material_data = Materials::Data::create_coated_dielectric(RGB(0.001f, 0.56f, 0.81f), 0.8f, 0.04f, 0.05f);
            Material tile_material = Materials::create("Pool tile", tile_material_data);

            Mesh tmp_tile_mesh = MeshCreation::beveled_cube(5, 0.002f, Vector3f(0.235f, 0.005f, 0.235f), positions_and_normals);
            Mesh tile_mesh = MeshUtils::merge_duplicate_vertices(tmp_tile_mesh, { MeshFlag::Position, MeshFlag::Normal });
            Meshes::destroy(tmp_tile_mesh.get_ID());

            { // Bump upwards facing flat vertices slightly
                RNG::XorShift32 rng = RNG::XorShift32(19349669);

                for (unsigned int v = 0; v < tile_mesh.get_vertex_count(); ++v) {
                    Vector3f normal = tile_mesh.get_normals()[v];
                    bool upwards_facing = abs(normal.y) == 1;
                    if (upwards_facing) {
                        float y_offset = (rng.sample1f() - 0.5f) * 0.002f;
                        Vector3f& position = tile_mesh.get_positions()[v];
                        position.y += y_offset;
                    }
                }
                MeshUtils::compute_normals(tile_mesh.get_ID());
            }

            { // Places tiles
                RNG::XorShift32 rng = RNG::XorShift32(73856093);

                auto create_tile_at = [&](Transform transform) {
                    SceneNode tile_node = SceneNodes::create("Tile", transform);
                    MeshModels::create(tile_node.get_ID(), tile_mesh.get_ID(), tile_material.get_ID());
                    tile_node.set_parent(pool_node);
                };

                // Place upwards facing
                for (float x = -0.875f; x < 1; x += 0.25)
                    for (float z = -0.875f; z < 1; z += 0.25) {
                        bool inside_pool = -0.75f < x && x < 0.75f && -0.75f < z && z < 0.75f;
                        float y = inside_pool ? -0.23f : 0.25f;

                        create_tile_at(Transform(Vector3f(x, y, z)));
                    }

                // Place tiles on sides
                Quaternionf right_rotation = Quaternionf::from_angle_axis(0.5f * PI<float>(), Vector3f::right());
                Quaternionf forward_rotation = Quaternionf::from_angle_axis(0.5f * PI<float>(), Vector3f::forward());
                for (float x = -0.875f; x < 1; x += 0.25)
                    for (float y = -0.125f; y < 0.25; y += 0.25) {
                        create_tile_at(Transform(Vector3f(-1, y, x), forward_rotation));
                        create_tile_at(Transform(Vector3f(1, y, x), forward_rotation));
                        create_tile_at(Transform(Vector3f(x, y, -1), right_rotation));
                        create_tile_at(Transform(Vector3f(x, y, 1), right_rotation));

                        bool inside_pool = -0.75f < x && x < 0.75f;
                        if (inside_pool) {
                            create_tile_at(Transform(Vector3f(-0.75f, y, x), forward_rotation));
                            create_tile_at(Transform(Vector3f(0.75f, y, x), forward_rotation));
                            create_tile_at(Transform(Vector3f(x, y, -0.75f), right_rotation));
                            create_tile_at(Transform(Vector3f(x, y, 0.75f), right_rotation));
                        }
                    }
            }
        }

        { // Water
            Materials::Data water_material_data = Materials::Data::create_glass(RGB(0.95f), 0.0f, water_specularity);
            Material water_material = Materials::create("Water", water_material_data);

            // Add waves to the water surface.
            Mesh water_mesh = MeshCreation::cube(128, Vector3f(1.5f, 0.38f, 1.5f), positions_and_normals);
            for (Vector3f& position : water_mesh.get_position_iterable())
                if (position.y > 0)
                    position.y += 0.02f + 0.02f * (sin(10 * position.x) + sin(7 * position.z));
            water_mesh.compute_bounds();
            MeshUtils::compute_normals(water_mesh.get_ID());

            SceneNode water_node = SceneNodes::create("Water", Transform(Vector3f(0, -0.051f, 0)));
            MeshModels::create(water_node.get_ID(), water_mesh.get_ID(), water_material.get_ID());
            water_node.set_parent(pool_node);
        }

        // Move pool
        pool_node.set_local_transform(Transform(Vector3f(0, -0.749f, -4)));
        pool_node.set_parent(root_node);
    }

    // Image of gradient from 0 to 1
    Image roughness_gradient = Images::create2D("Gradient", PixelFormat::Roughness8, 1.0f, Vector2ui(8, 1));
    unsigned char* roughness_gradient_pixels = roughness_gradient.get_pixels<unsigned char>();
    for (unsigned int x = 0; x < 8; ++x)
        roughness_gradient_pixels[x] = x * 31;
    Texture roughness_gradient_texture = Textures::create2D(roughness_gradient.get_ID(), MagnificationFilter::None, MinificationFilter::Linear, WrapMode::Clamp, WrapMode::Clamp);

    { // Thin-walled material VS a thin box
        // Glass material with varying roughness.
        Materials::Data box_material_data = Materials::Data::create_glass(RGB(0.8f), 1.0f, glass_specularity);
        box_material_data.tint_roughness_texture_ID = roughness_gradient_texture.get_ID();

        { // Glass box
            Material box_material = Materials::create("Glass sheet", box_material_data);
            Mesh box_mesh = MeshCreation::cube(1, Vector3f(1, 1, 0.001f));

            // Fix UVs so they match the UVs of the plane below by aligning them with the xy plane.
            // The box positions are in range [-0.5f, 0.5f].
            Vector3f* positions = box_mesh.get_positions();
            Vector2f* uvs = box_mesh.get_texcoords();
            for (unsigned int v = 0; v < box_mesh.get_vertex_count(); ++v)
                uvs[v] = Vector2f(positions[v].x, positions[v].y) + Vector2f(0.5f);

            SceneNode box_node = SceneNodes::create("Glass sheet", Transform(Vector3f(6, -0.5f, 0)));
            MeshModels::create(box_node.get_ID(), box_mesh.get_ID(), box_material.get_ID());
            box_node.set_parent(root_node);
        }

        { // Thin-walled plane
            box_material_data.flags |= MaterialFlag::ThinWalled;
            Material thinwalled_material = Materials::create("Thin-walled glass sheet", box_material_data);
            Mesh plane_mesh = MeshCreation::plane(1);
            Quaternionf rotation = Quaternionf::from_angle_axis(0.5f * PI<float>(), Vector3f::right());
            SceneNode plane_node = SceneNodes::create("Thin-walled glass sheet", Transform(Vector3f(6, 0.5f, 0), rotation));
            MeshModels::create(plane_node.get_ID(), plane_mesh.get_ID(), thinwalled_material.get_ID());
            plane_node.set_parent(root_node);
        }
    }

    { // Coated default material VS default material inside a thin glass box
        // Default material with varying roughness.
        Materials::Data nickel_material_data = Materials::Data::create_metal(nickel_tint, 1.0f);
        nickel_material_data.tint_roughness_texture_ID = roughness_gradient_texture.get_ID();
        nickel_material_data.flags |= MaterialFlag::ThinWalled;
        float coat_roughness = 0.1f;

        Mesh plane_mesh = MeshCreation::plane(1);
        Quaternionf plane_rotation = Quaternionf::from_angle_axis(0.5f * PI<float>(), Vector3f::right());

        { // Default material in coat box
            Material nickel_material = Materials::create("Nickel", nickel_material_data);
            SceneNode plane_node = SceneNodes::create("Nickel", Transform(Vector3f(-6, -0.5f, 0), plane_rotation));
            MeshModels::create(plane_node.get_ID(), plane_mesh.get_ID(), nickel_material.get_ID());
            plane_node.set_parent(root_node);

            // Coat outside plane
            Materials::Data coat_material_data = Materials::Data::create_glass(RGB::white(), coat_roughness, coat_specularity);
            Material coat_material = Materials::create("Coat", coat_material_data);

            Mesh box_mesh = MeshCreation::cube(1, Vector3f(1, 1, 0.002f));
            SceneNode box_node = SceneNodes::create("Nickel coat", Transform(Vector3f(-6, -0.5f, 0)));
            MeshModels::create(box_node.get_ID(), box_mesh.get_ID(), coat_material.get_ID());
            box_node.set_parent(root_node);
        }

        { // Coated default material
            nickel_material_data.coat = 1.0f;
            nickel_material_data.coat_roughness = coat_roughness;
            Material coated_nickel_material = Materials::create("Coated nickel", nickel_material_data);
            SceneNode plane_node = SceneNodes::create("Coated nickel", Transform(Vector3f(-6, 0.5f, 0), plane_rotation));
            MeshModels::create(plane_node.get_ID(), plane_mesh.get_ID(), coated_nickel_material.get_ID());
            plane_node.set_parent(root_node);
        }
    }
}

} // NS Scenes