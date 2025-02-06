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

void create_glass_scene(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::SceneNode root_node, const std::filesystem::path& resource_directory) {

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
        SceneNode light_node = SceneNode("light", light_transform);
        light_node.set_parent(root_node);
        DirectionalLight(light_node, RGB(3.0f, 2.9f, 2.5f));
    }

    { // Create floor.
        SceneNode floor_node = create_checkered_floor(400, 1);
        floor_node.set_global_transform(Transform(Vector3f(0, -1.0f, 0)));
        floor_node.set_parent(root_node);
    }

    { // Add glass shaderball
        Material glass_material = Material::create_transmissive("Glass shader ball", RGB(0.95f), 0.25f, glass_specularity);

        SceneNode shader_ball_node = load_shader_ball(resource_directory, glass_material);
        Transform transform = Transform(Vector3f(0,-0.25f,0), Quaternionf::from_angle_axis(0.1f * PI<float>(), Vector3f::up()), 1.5f);
        shader_ball_node.set_global_transform(transform);
        shader_ball_node.set_parent(root_node);
    }

    { // Magnifying glass
        SceneNode magnifying_glass_node = SceneNode("Magnifying glass");

        { // Glass
            Material glass_material = Material::create_transmissive("Magnifying glass", RGB(0.975f), 0.0f, glass_specularity);

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

            SceneNode glass_node = SceneNode("Glass");
            MeshModel(glass_node, glass_mesh, glass_material);
            glass_node.set_parent(magnifying_glass_node);
        }

        { // Frame
            Material frame_material = Material::create_metal("Magnifying glass frame", gold_tint, 0.5f);

            Mesh frame_mesh = MeshCreation::torus(64, 64, 0.05f, positions_and_normals);

            SceneNode frame_node = SceneNode("Magnifying glass frame");
            Quaternionf rotation = Quaternionf::from_angle_axis(0.5f * PI<float>(), Vector3f::right());
            frame_node.set_global_transform(Transform(Vector3f::zero(), rotation));
            MeshModel(frame_node, frame_mesh, frame_material);
            frame_node.set_parent(magnifying_glass_node);

            Mesh handle_mesh = MeshCreation::cylinder(1, 64, positions_and_normals);
            for (Vector3f& position : handle_mesh.get_position_iterable()) {
                position.x *= 0.08f;
                position.z *= 0.08f;
                position.y *= 0.6f;
            }

            SceneNode handle_node = SceneNode("Magnifying glass handle", Transform(Vector3f(0, -0.8f, 0)));
            MeshModel(handle_node, handle_mesh, frame_material);
            handle_node.set_parent(magnifying_glass_node);
        }

        // Move magnifying glass
        magnifying_glass_node.set_local_transform(Transform(Vector3f(3, 0.1f, 0)));
        magnifying_glass_node.set_parent(root_node);
    }

    { // Diamond
        auto diamond_path = resource_directory / "Diamond.glb";
        SceneNode diamond_node = glTFLoader::load(diamond_path.generic_string());
        Quaternionf rotation = Quaternionf::from_angle_axis(-0.5f * PI<float>(), Vector3f::right());
        diamond_node.set_global_transform(Transform(Vector3f(-3, 0, 0), rotation));
        diamond_node.set_parent(root_node);

        Material diamond_material = Material::create_transmissive("Diamond", RGB(1.0f), 0.0f, diamond_specularity);
        replace_material(diamond_material, diamond_node, "pCone1_DiamondOutside_0");
    }

    { // Pool of water. The pool is 2x0.5x2, with two levels of tiles.
        SceneNode pool_node = SceneNode("Pool");

        { // Pool sides grout
            Material grout_material = Material::create_dielectric("Pool tile", RGB(0.2f), 0.9f);

            Mesh grout_side1 = MeshCreation::box(1, Vector3f(2.0f, 0.5f, 0.25f), positions_and_normals);
            Mesh grout_side2 = MeshCreation::box(1, Vector3f(0.25f, 0.5f, 1.5f), positions_and_normals);
            Mesh grout_floor = MeshCreation::box(1, Vector3f(1.5f, 0.02f, 1.5f), positions_and_normals);
            MeshUtils::TransformedMesh meshes[5] = {
                { grout_side1, Transform(Vector3f(0, 0, 0.875f)) },
                { grout_side1, Transform(Vector3f(0, 0, -0.875f)) },
                { grout_side2, Transform(Vector3f(0.875f, 0, 0)) },
                { grout_side2, Transform(Vector3f(-0.875f, 0, 0)) },
                { grout_floor, Transform(Vector3f(0, -0.24f, 0)) },
            };
            Mesh grout_mesh = MeshUtils::combine("Pool grout", meshes, meshes + 5);
            grout_side1.destroy();
            grout_side2.destroy();
            grout_floor.destroy();

            SceneNode grout_node = SceneNode("Pool grout");
            MeshModel(grout_node, grout_mesh, grout_material);
            grout_node.set_parent(pool_node);
        }

        { // Tiles
            Material tile_material = Material::create_coated_dielectric("Pool tile", RGB(0.001f, 0.56f, 0.81f), 0.8f, 0.04f, 0.05f);

            Mesh tmp_tile_mesh = MeshCreation::beveled_box(5, 0.002f, Vector3f(0.235f, 0.005f, 0.235f), positions_and_normals);
            Mesh tile_mesh = MeshUtils::merge_duplicate_vertices(tmp_tile_mesh, { MeshFlag::Position, MeshFlag::Normal });
            tmp_tile_mesh.destroy();

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
                MeshUtils::compute_normals(tile_mesh);
            }

            { // Places tiles
                RNG::XorShift32 rng = RNG::XorShift32(73856093);

                auto create_tile_at = [&](Transform transform) {
                    SceneNode tile_node = SceneNode("Tile", transform);
                    MeshModel(tile_node, tile_mesh, tile_material);
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
            Material water_material = Material::create_transmissive("Water", RGB(0.95f), 0.0f, water_specularity);

            Mesh water_surface = MeshCreation::plane(128, positions_and_normals);
            for (Vector3f& position : water_surface.get_position_iterable())
                position.y += 0.02f * (sin(10 * position.x) + sin(7 * position.z));
            water_surface.compute_bounds();
            MeshUtils::compute_normals(water_surface);

            SceneNode water_node = SceneNode("Water", Transform(Vector3f(0, 0.125f, 0), Quaternionf::identity(), 1.5f));
            MeshModel(water_node, water_surface, water_material);
            water_node.set_parent(pool_node);
        }

        // Move pool
        pool_node.set_local_transform(Transform(Vector3f(0, -0.749f, -4)));
        pool_node.set_parent(root_node);
    }

    // Image of gradient from 0 to 1
    Image roughness_gradient = Image::create2D("Gradient", PixelFormat::Roughness8, 1.0f, Vector2ui(8, 1));
    unsigned char* roughness_gradient_pixels = roughness_gradient.get_pixels<unsigned char>();
    for (unsigned int x = 0; x < 8; ++x)
        roughness_gradient_pixels[x] = x * 31;
    Texture roughness_gradient_texture = Texture::create2D(roughness_gradient, MagnificationFilter::None, MinificationFilter::Linear, WrapMode::Clamp, WrapMode::Clamp);

    { // Thin-walled material VS a thin box
        // Glass material with varying roughness.
        Materials::Data box_material_data = Materials::Data::create_transmissive(RGB(0.8f), 1.0f, glass_specularity);
        box_material_data.tint_roughness_texture_ID = roughness_gradient_texture.get_ID();

        { // Glass box
            Material box_material = Material("Glass sheet", box_material_data);
            Mesh box_mesh = MeshCreation::box(1, Vector3f(1, 1, 0.001f));

            // Fix UVs so they match the UVs of the plane below by aligning them with the xy plane.
            // The box positions are in range [-0.5f, 0.5f].
            Vector3f* positions = box_mesh.get_positions();
            Vector2f* uvs = box_mesh.get_texcoords();
            for (unsigned int v = 0; v < box_mesh.get_vertex_count(); ++v)
                uvs[v] = Vector2f(positions[v].x, positions[v].y) + Vector2f(0.5f);

            SceneNode box_node = SceneNode("Glass sheet", Transform(Vector3f(6, -0.5f, 0)));
            MeshModel(box_node, box_mesh, box_material);
            box_node.set_parent(root_node);
        }

        { // Thin-walled plane
            box_material_data.flags |= MaterialFlag::ThinWalled;
            Material thinwalled_material = Material("Thin-walled glass sheet", box_material_data);
            Mesh plane_mesh = MeshCreation::plane(1);
            Quaternionf rotation = Quaternionf::from_angle_axis(0.5f * PI<float>(), Vector3f::right());
            SceneNode plane_node = SceneNode("Thin-walled glass sheet", Transform(Vector3f(6, 0.5f, 0), rotation));
            MeshModel(plane_node, plane_mesh, thinwalled_material);
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
            Material nickel_material = Material("Nickel", nickel_material_data);
            SceneNode plane_node = SceneNode("Nickel", Transform(Vector3f(-6, -0.5f, 0), plane_rotation));
            MeshModel(plane_node, plane_mesh, nickel_material);
            plane_node.set_parent(root_node);

            // Coat outside plane
            Material coat_material = Material::create_transmissive("Coat", RGB::white(), coat_roughness, coat_specularity);

            Mesh box_mesh = MeshCreation::box(1, Vector3f(1, 1, 0.002f));
            SceneNode box_node = SceneNode("Nickel coat", Transform(Vector3f(-6, -0.5f, 0)));
            MeshModel(box_node, box_mesh, coat_material);
            box_node.set_parent(root_node);
        }

        { // Coated default material
            nickel_material_data.coat = 1.0f;
            nickel_material_data.coat_roughness = coat_roughness;
            Material coated_nickel_material = Material("Coated nickel", nickel_material_data);
            SceneNode plane_node = SceneNode("Coated nickel", Transform(Vector3f(-6, 0.5f, 0), plane_rotation));
            MeshModel(plane_node, plane_mesh, coated_nickel_material);
            plane_node.set_parent(root_node);
        }
    }
}

} // NS Scenes