// Cogwheel obj model loader.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#pragma warning(disable : 4996)

#include <ObjLoader/ObjLoader.h>

#include <Cogwheel/Assets/Material.h>
#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Core/Array.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <ObjLoader/tiny_obj_loader.h>

using namespace Cogwheel;
using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Scene;

namespace ObjLoader {

void split_path(std::string& directory, std::string& filename, const std::string& path) {
    std::string::const_iterator slash_itr = path.end();
    std::string::const_iterator itr = path.end()-1;
    while (itr > path.begin()) {
        --itr;
        if (*itr == '/' || *itr == '\\') {
            slash_itr = itr;
            break;
        }
    }

    if (slash_itr == path.end()) {
        directory = "";
        filename = path;
    } else {
        directory = std::string(path.begin(), slash_itr + 1);
        filename = std::string(slash_itr + 1, path.end());
    }
}

SceneNodes::UID load(const std::string& path, ImageLoader image_loader) {
    std::string directory, filename;
    split_path(directory, filename, path);

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> tiny_materials;
    std::string error;
    bool obj_loaded = tinyobj::LoadObj(shapes, tiny_materials, error, path.c_str(), directory.c_str());

    if (!error.empty()) {
        printf("ObjLoader::load error: '%s'.\n", error.c_str());
        if (!obj_loaded)
            return SceneNodes::UID::invalid_UID();
    }

    SceneNodes::UID root_ID = shapes.size() > 1u ? SceneNodes::create(std::string(filename.begin(), filename.end()-4)) : SceneNodes::UID::invalid_UID();

    Core::Array<Materials::UID> materials = Core::Array<Materials::UID>(unsigned int(tiny_materials.size()));
    for (size_t i = 0; i < tiny_materials.size(); ++i) {
        tinyobj::material_t tiny_mat = tiny_materials[i];

        Materials::Data material_data;
        material_data.tint = Math::RGB(tiny_mat.diffuse[0], tiny_mat.diffuse[1], tiny_mat.diffuse[2]);
        material_data.roughness = sqrt(sqrt(2.0f / (tiny_mat.shininess + 2.0f))); // Map from blinn shininess to material roughness.
        bool is_metallic = tiny_mat.illum == 3 || tiny_mat.illum == 5;
        material_data.metallic = is_metallic ? 1.0f : 0.0f;
        material_data.specularity = (tiny_mat.specular[0] + tiny_mat.specular[1] + tiny_mat.specular[2]) / 3.0f;
        material_data.coverage = tiny_mat.dissolve;
        material_data.transmission = 0.0f; // (tiny_mat.transmittance[0] + tiny_mat.transmittance[1] + tiny_mat.transmittance[2]) / 3.0f;

        if (!tiny_mat.alpha_texname.empty()) {
            Images::UID image_ID = image_loader(directory + tiny_mat.alpha_texname);
            if (Images::get_pixel_format(image_ID) != PixelFormat::I8) {
                Images::UID new_image_ID = ImageUtils::change_format(image_ID, PixelFormat::I8); // TODO Own change format. Intensity should come from the alpha channel if there is one, otherwise from red.
                Images::destroy(image_ID);
                image_ID = new_image_ID;
            }
            material_data.coverage_texture_ID = Textures::create2D(image_ID);
        } else
            material_data.coverage_texture_ID = Textures::UID::invalid_UID();

        if (!tiny_mat.diffuse_texname.empty()) {
            Images::UID image_ID = image_loader(directory + tiny_mat.diffuse_texname);

            // Use diffuse alpha for coverage, if no explicit coverage texture has been set.
            if (channel_count(Images::get_pixel_format(image_ID)) == 4 && material_data.coverage_texture_ID == Textures::UID::invalid_UID()) {
                Image image = image_ID;
                unsigned int mipmap_count = image.get_mipmap_count();
                Math::Vector2ui size = Math::Vector2ui(image.get_width(), image.get_height());
                Images::UID coverage_image_ID = Images::create(image.get_name(), PixelFormat::I8, image.get_gamma(), size, mipmap_count);

                float min_coverage = 1.0f;
                for (unsigned int m = 0; m < mipmap_count; ++m)
                    for (unsigned int y = 0; y < image.get_height(m); ++y)
                        for (int x = 0; x < int(image.get_width(m)); ++x) {
                            Math::Vector2ui index = Math::Vector2ui(x, y);
                            float coverage = image.get_pixel(index, m).a;
                            min_coverage = fminf(min_coverage, coverage);
                            Math::RGBA pixel = Math::RGBA(coverage, coverage, coverage, coverage);
                            Images::set_pixel(coverage_image_ID, pixel, index, m);
                        }

                if (min_coverage < 1.0f)
                    material_data.coverage_texture_ID = Textures::create2D(coverage_image_ID);
            }

            if (channel_count(Images::get_pixel_format(image_ID)) != 4) {
                Images::UID new_image_ID = ImageUtils::change_format(image_ID, PixelFormat::RGBA32);
                Images::destroy(image_ID);
                image_ID = new_image_ID;
            }
            material_data.tint_texture_ID = Textures::create2D(image_ID);
        } else
            material_data.tint_texture_ID = Textures::UID::invalid_UID();

        materials[unsigned int(i)] = Materials::create(tiny_mat.name, material_data);
    }

    for (size_t i = 0; i < shapes.size(); ++i) {

        tinyobj::mesh_t tiny_mesh = shapes[i].mesh;

        Meshes::UID mesh_ID = Meshes::UID::invalid_UID();
        { // Create mesh 
            assert(tiny_mesh.indices.size() != 0); // Assert that there are indices, because currently we don't support non-indexed meshes.
            assert((tiny_mesh.indices.size() % 3) == 0); // Assert that indices are for triangles.
            assert((tiny_mesh.positions.size() % 3) == 0); // Assert that positions are three dimensional.
            assert((tiny_mesh.normals.size() % 3) == 0); // Assert that normals are three dimensional.
            assert((tiny_mesh.texcoords.size() % 2) == 0); // Assert that texcoords are two dimensional.

            unsigned char mesh_flags = MeshFlags::Position;
            if (tiny_mesh.normals.size() > 0)
                mesh_flags |= MeshFlags::Normal;
            if (tiny_mesh.texcoords.size() > 0)
                mesh_flags |= MeshFlags::Texcoord;

            unsigned int triangle_count = unsigned int(tiny_mesh.indices.size() / 3u);
            unsigned int vertex_count = unsigned int(tiny_mesh.positions.size() / 3u);
            mesh_ID = Meshes::create(shapes[i].name, triangle_count, vertex_count, mesh_flags);

            Mesh cogwheel_mesh = mesh_ID;
            memcpy(cogwheel_mesh.get_primitives(), tiny_mesh.indices.data(), tiny_mesh.indices.size() * sizeof(unsigned int));
            memcpy(cogwheel_mesh.get_positions(), tiny_mesh.positions.data(), tiny_mesh.positions.size() * sizeof(float));
            if (mesh_flags & MeshFlags::Normal)
                memcpy(cogwheel_mesh.get_normals(), tiny_mesh.normals.data(), tiny_mesh.normals.size() * sizeof(float));
            if (mesh_flags & MeshFlags::Texcoord)
                memcpy(cogwheel_mesh.get_texcoords(), tiny_mesh.texcoords.data(), tiny_mesh.texcoords.size() * sizeof(float));

            cogwheel_mesh.compute_bounds();
        }

        SceneNodes::UID node_ID = SceneNodes::create(shapes[i].name);
        if (root_ID != SceneNodes::UID::invalid_UID())
            SceneNodes::set_parent(node_ID, root_ID);
        else
            root_ID = node_ID;

        int material_index = tiny_mesh.material_ids.size() == 0 ? -1 : tiny_mesh.material_ids[0];
        Materials::UID material_ID = material_index >= 0 ? materials[material_index] : Materials::UID::invalid_UID();
        MeshModels::UID model_ID = MeshModels::create(node_ID, mesh_ID, material_ID);
    }

    return root_ID;
}

} // NS ObjLoader
