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

#include <map>

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

    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> tiny_materials;
    std::string error;
    bool obj_loaded = tinyobj::LoadObj(&attributes, &shapes, &tiny_materials, &error, path.c_str(), directory.c_str());

    if (!error.empty()) {
        printf("ObjLoader::load error: '%s'.\n", error.c_str());
        if (!obj_loaded)
            return SceneNodes::UID::invalid_UID();
    }

    SceneNodes::UID root_ID = shapes.size() > 1u ? SceneNodes::create(std::string(filename.begin(), filename.end()-4)) : SceneNodes::UID::invalid_UID();

    Core::Array<Materials::UID> materials = Core::Array<Materials::UID>(unsigned int(tiny_materials.size()));
    for (size_t i = 0; i < tiny_materials.size(); ++i) {
        tinyobj::material_t tiny_mat = tiny_materials[i];

        Materials::Data material_data = {};
        material_data.flags = MaterialFlag::None;
        material_data.tint = Math::RGB(tiny_mat.diffuse[0], tiny_mat.diffuse[1], tiny_mat.diffuse[2]);
        material_data.tint_texture_ID = Textures::UID::invalid_UID();
        material_data.roughness = sqrt(sqrt(2.0f / (tiny_mat.shininess + 2.0f))); // Map from blinn shininess to material roughness.
        bool is_metallic = tiny_mat.illum == 3 || tiny_mat.illum == 5;
        material_data.metallic = is_metallic ? 1.0f : 0.0f;
        material_data.specularity = (tiny_mat.specular[0] + tiny_mat.specular[1] + tiny_mat.specular[2]) / 3.0f;
        material_data.coverage = tiny_mat.dissolve;
        material_data.coverage_texture_ID = Textures::UID::invalid_UID();
        material_data.transmission = 0.0f; // (tiny_mat.transmittance[0] + tiny_mat.transmittance[1] + tiny_mat.transmittance[2]) / 3.0f;

        // Warn about completely transparent object. Happens from time to time and it's hell to debug a missing model.
        if (material_data.coverage <= 0.0f)
            printf("ObjLoader::load warning: Coverage set to %.3f. Material %s is completely transparent.\n", material_data.coverage, tiny_mat.name.c_str());

        if (!tiny_mat.alpha_texname.empty()) {
            Images::UID image_ID = image_loader(directory + tiny_mat.alpha_texname);
            if (image_ID == Images::UID::invalid_UID())
                printf("ObjLoader::load error: Could not load image at '%s'.\n", (directory + tiny_mat.alpha_texname).c_str());
            else {
                if (Images::get_pixel_format(image_ID) != PixelFormat::I8) {
                    Images::UID new_image_ID = ImageUtils::change_format(image_ID, PixelFormat::I8); // TODO Own change format. Intensity should come from the alpha channel if there is one, otherwise from red.
                    Images::destroy(image_ID);
                    image_ID = new_image_ID;
                }
                material_data.coverage_texture_ID = Textures::create2D(image_ID);
            }
        }

        if (!tiny_mat.diffuse_texname.empty()) {
            Images::UID image_ID = image_loader(directory + tiny_mat.diffuse_texname);
            if (image_ID == Images::UID::invalid_UID())
                printf("ObjLoader::load error: Could not load image at '%s'.\n", (directory + tiny_mat.alpha_texname).c_str());
            else {
                // Use diffuse alpha for coverage, if no explicit coverage texture has been set.
                if (channel_count(Images::get_pixel_format(image_ID)) == 4 && material_data.coverage_texture_ID == Textures::UID::invalid_UID()) {
                    Image image = image_ID;
                    unsigned int mipmap_count = image.get_mipmap_count();
                    Math::Vector2ui size = Math::Vector2ui(image.get_width(), image.get_height());
                    Images::UID coverage_image_ID = Images::create2D(image.get_name(), PixelFormat::I8, image.get_gamma(), size, mipmap_count);

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
            }
        }

        materials[unsigned int(i)] = Materials::create(tiny_mat.name, material_data);
    }

    for (size_t s = 0; s < int(shapes.size()); ++s) {
        tinyobj::shape_t shape = shapes[s];

        // Base normal and texcoords on the first vertex.
        tinyobj::index_t first_vertex_index = shape.mesh.indices[0];
        MeshFlags mesh_flags = MeshFlag::Position;
        if (first_vertex_index.normal_index != -1)
            mesh_flags |= MeshFlag::Normal;
        if (first_vertex_index.texcoord_index != -1)
            mesh_flags |= MeshFlag::Texcoord;

        // Tiny index comparer for the map.
        struct IndexComparer {
            inline bool operator()(tinyobj::index_t lhs, tinyobj::index_t rhs) const {
                if (lhs.vertex_index != rhs.vertex_index)
                    return lhs.vertex_index < rhs.vertex_index;
                else if (lhs.normal_index != rhs.normal_index)
                    return lhs.normal_index < rhs.normal_index;
                else if (lhs.texcoord_index != rhs.texcoord_index)
                    return lhs.texcoord_index < rhs.texcoord_index;
                else
                    return false;
            }
        };

        std::map<tinyobj::index_t, unsigned int, IndexComparer> vertex_index_map; // TODO Sort and reduce to make a faster structure?
        unsigned int vertex_count = 0;
        for (auto tiny_vertex_index : shape.mesh.indices) {
            auto res = vertex_index_map.emplace(tiny_vertex_index, vertex_count);
            if (res.second)
                ++vertex_count;
        }

        unsigned int triangle_count = unsigned int(shape.mesh.num_face_vertices.size());

        Mesh cogwheel_mesh = Meshes::create(shape.name, triangle_count, vertex_count, mesh_flags);

        // Copy indices.
        auto* mesh_primitives = cogwheel_mesh.get_primitives();
        for (unsigned int p = 0; p < cogwheel_mesh.get_primitive_count(); ++p) {
            unsigned int i0 = vertex_index_map[shape.mesh.indices[3 * p + 0]];
            unsigned int i1 = vertex_index_map[shape.mesh.indices[3 * p + 1]];
            unsigned int i2 = vertex_index_map[shape.mesh.indices[3 * p + 2]];
            mesh_primitives[p] = { i0, i1, i2 };
        }

        // Copy positions
        auto* mesh_positions = cogwheel_mesh.get_positions();
        for (auto index_pair : vertex_index_map) {
            float vx = attributes.vertices[3 * index_pair.first.vertex_index + 0];
            float vy = attributes.vertices[3 * index_pair.first.vertex_index + 1];
            float vz = attributes.vertices[3 * index_pair.first.vertex_index + 2];
            mesh_positions[index_pair.second] = { vx, vy, vz };
        }

        // Copy normals
        auto* mesh_normals = cogwheel_mesh.get_normals();
        if (mesh_normals != nullptr) {
            for (auto index_pair : vertex_index_map) {
                float nx = attributes.normals[3 * index_pair.first.normal_index + 0];
                float ny = attributes.normals[3 * index_pair.first.normal_index + 1];
                float nz = attributes.normals[3 * index_pair.first.normal_index + 2];
                mesh_normals[index_pair.second] = { nx, ny, nz };
            }
        }

        // Copy texcoords
        auto* mesh_texcoord = cogwheel_mesh.get_texcoords();
        if (mesh_texcoord != nullptr) {
            for (auto index_pair : vertex_index_map) {
                float u = attributes.texcoords[2 * index_pair.first.texcoord_index + 0];
                float v = attributes.texcoords[2 * index_pair.first.texcoord_index + 1];
                mesh_texcoord[index_pair.second] = { u, v };
            }
        }

        cogwheel_mesh.compute_bounds();

        SceneNodes::UID node_ID = SceneNodes::create(shape.name);
        if (root_ID != SceneNodes::UID::invalid_UID())
            SceneNodes::set_parent(node_ID, root_ID);
        else
            root_ID = node_ID;

        int material_index = shape.mesh.material_ids[0]; // No per facet material support. TODO Add it in the future by splitting up the shape.
        Materials::UID material_ID = material_index >= 0 ? materials[material_index] : Materials::UID::invalid_UID();
        MeshModels::UID model_ID = MeshModels::create(node_ID, cogwheel_mesh.get_ID(), material_ID);
    }

    return root_ID;
}

inline bool string_ends_with(const std::string& s, const std::string& end) {
    if (s.length() < end.length())
        return false;

    return s.compare(s.length() - end.length(), end.length(), end) == 0;
}

bool file_supported(const std::string& filename) {
    return string_ends_with(filename, ".obj");
}

} // NS ObjLoader
