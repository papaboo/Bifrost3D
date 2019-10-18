// Bifrost obj model loader.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#pragma warning(disable : 4996)

#include <ObjLoader/ObjLoader.h>

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Core/Array.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <ObjLoader/tiny_obj_loader.h>

#include <map>

using namespace Bifrost;
using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace ObjLoader {

struct RGBA32 { unsigned char r, g, b, a; };

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
    std::string warning;
    std::string error;

    bool obj_loaded = tinyobj::LoadObj(&attributes, &shapes, &tiny_materials, &warning, &error, path.c_str(), directory.c_str());

    if (!warning.empty())
        printf("ObjLoader::load warning: '%s'.\n", warning.c_str());
    if (!error.empty())
        printf("ObjLoader::load error: '%s'.\n", error.c_str());

    if (!obj_loaded)
        return SceneNodes::UID::invalid_UID();

    SceneNodes::UID root_ID = shapes.size() > 1u ? SceneNodes::create(std::string(filename.begin(), filename.end()-4)) : SceneNodes::UID::invalid_UID();

    Core::Array<Materials::UID> materials = Core::Array<Materials::UID>(unsigned int(tiny_materials.size()));
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < int(tiny_materials.size()); ++i) {
        tinyobj::material_t tiny_mat = tiny_materials[i];

        Materials::Data material_data = {};
        material_data.flags = MaterialFlag::None;
        material_data.tint = RGB(tiny_mat.diffuse[0], tiny_mat.diffuse[1], tiny_mat.diffuse[2]);
        material_data.tint_roughness_texture_ID = Textures::UID::invalid_UID();
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
                if (Images::get_pixel_format(image_ID) != PixelFormat::A8) {
                    Images::UID new_image_ID = ImageUtils::change_format(image_ID, PixelFormat::A8); // TODO Own change format. Alpha should come from the alpha channel if there is one, otherwise from red.
                    Images::destroy(image_ID);
                    image_ID = new_image_ID;
                }
                material_data.coverage_texture_ID = Textures::create2D(image_ID);
            }
        }

        if (!tiny_mat.diffuse_texname.empty()) {
            Image image = image_loader(directory + tiny_mat.diffuse_texname);
            if (!image.exists())
                printf("ObjLoader::load error: Could not load image at '%s'.\n", (directory + tiny_mat.diffuse_texname).c_str());
            else {
                // Use diffuse alpha for coverage, if no explicit coverage texture has been set.
                if (channel_count(image.get_pixel_format()) == 4 && material_data.coverage_texture_ID == Textures::UID::invalid_UID()) {
                    unsigned int mipmap_count = image.get_mipmap_count();
                    Vector2ui size = Vector2ui(image.get_width(), image.get_height());
                    Image coverage_image = Images::create2D(image.get_name(), PixelFormat::A8, image.get_gamma(), size, mipmap_count);

                    float min_coverage = 1.0f;
                    for (unsigned int m = 0; m < mipmap_count; ++m)
                        for (unsigned int y = 0; y < image.get_height(m); ++y)
                            for (int x = 0; x < int(image.get_width(m)); ++x) {
                                Vector2ui index = Vector2ui(x, y);
                                RGBA tint_pixel = image.get_pixel(index, m);
                                min_coverage = fminf(min_coverage, tint_pixel.a);
                                RGBA coverage_pixel = RGBA(RGB(1.0f), tint_pixel.a);
                                coverage_image.set_pixel(coverage_pixel, index, m);

                                // Set roughness to 1 in the original tint/roughness image.
                                tint_pixel.a = 1.0f;
                                image.set_pixel(tint_pixel, index, m);
                            }

                    if (min_coverage < 1.0f)
                        material_data.coverage_texture_ID = Textures::create2D(coverage_image.get_ID());
                }

                material_data.tint_roughness_texture_ID = Textures::create2D(image.get_ID());
            }
        }

        if (!tiny_mat.roughness_texname.empty()) {
            Image roughness_map = image_loader(directory + tiny_mat.roughness_texname);
            if (!roughness_map.exists())
                printf("ObjLoader::load error: Could not load image at '%s'.\n", (directory + tiny_mat.roughness_texname).c_str());
            else {
                // If no diffuse map exists, then save a single channel texture as roughness texture.
                if (!Textures::has(material_data.tint_roughness_texture_ID)) {
                    if (roughness_map.get_pixel_format() != PixelFormat::Roughness8) {
                        Image tmp_image = roughness_map;
                        roughness_map = ImageUtils::change_format(roughness_map.get_ID(), PixelFormat::Roughness8, 1.0f, [](RGBA c) -> RGBA { return RGBA(c.r, c.r, c.r, c.r); });
                        Images::destroy(tmp_image.get_ID());
                    }
                    material_data.tint_roughness_texture_ID = Textures::create2D(roughness_map.get_ID());;
                } else {
                    // Diffuse texture exists.
                    Image tint_roughness_map = Textures::get_image_ID(material_data.tint_roughness_texture_ID);
                    
                    // Image should have a fourth channel to store roughness in.
                    if (channel_count(tint_roughness_map.get_pixel_format()) != 4) {
                        Image tmp_image = tint_roughness_map;
                        tint_roughness_map = ImageUtils::change_format(tint_roughness_map.get_ID(), PixelFormat::RGBA32, 2.2f);
                        Images::destroy(tmp_image.get_ID());
                    }

                    // Blit roughness texture if possible.
                    if (tint_roughness_map.get_width() == roughness_map.get_width() && tint_roughness_map.get_height() == roughness_map.get_height() &&
                        roughness_map.get_pixel_format() == PixelFormat::Roughness8) {
                        RGBA32* tint_roughness_pixels = tint_roughness_map.get_pixels<RGBA32>();
                        unsigned char* roughness_pixels = roughness_map.get_pixels<unsigned char>();
                        for (unsigned int p = 0; p < roughness_map.get_pixel_count(); ++p)
                            tint_roughness_pixels[p].a = roughness_pixels[p];
                    } else {
                        // Otherwise sample roughness texture.
                        Textures::UID tmp_roughness_tex_ID = Textures::create2D(roughness_map.get_ID());
                        #pragma omp parallel for schedule(dynamic, 16)
                        for (int y = 0; y < int(tint_roughness_map.get_height()); ++y)
                            for (unsigned int x = 0; x < tint_roughness_map.get_width(); ++x) {
                                Vector2f uv = { (x + 0.5f) / tint_roughness_map.get_width(), (y + 0.5f) / tint_roughness_map.get_height() };
                                RGBA pixel = tint_roughness_map.get_pixel(Vector2ui(x, y));
                                pixel.a = sample2D(tmp_roughness_tex_ID, uv).r;
                                tint_roughness_map.set_pixel(pixel, Vector2ui(x, y));
                            }
                        Textures::destroy(tmp_roughness_tex_ID);

                        Textures::destroy(material_data.tint_roughness_texture_ID);
                        material_data.tint_roughness_texture_ID = Textures::create2D(tint_roughness_map.get_ID());;
                    }
                }
            }
        }

        materials[unsigned int(i)] = Materials::create(tiny_mat.name, material_data);
    }

    #pragma omp parallel for schedule(dynamic, 16)
    for (int s = 0; s < int(shapes.size()); ++s) {
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

        Mesh bifrost_mesh = Meshes::create(shape.name, triangle_count, vertex_count, mesh_flags);

        // Copy indices.
        auto* mesh_primitives = bifrost_mesh.get_primitives();
        for (unsigned int p = 0; p < bifrost_mesh.get_primitive_count(); ++p) {
            unsigned int i0 = vertex_index_map[shape.mesh.indices[3 * p + 0]];
            unsigned int i1 = vertex_index_map[shape.mesh.indices[3 * p + 1]];
            unsigned int i2 = vertex_index_map[shape.mesh.indices[3 * p + 2]];
            mesh_primitives[p] = { i0, i1, i2 };
        }

        // Copy positions
        auto* mesh_positions = bifrost_mesh.get_positions();
        for (auto index_pair : vertex_index_map) {
            float vx = attributes.vertices[3 * index_pair.first.vertex_index + 0];
            float vy = attributes.vertices[3 * index_pair.first.vertex_index + 1];
            float vz = attributes.vertices[3 * index_pair.first.vertex_index + 2];
            mesh_positions[index_pair.second] = { vx, vy, vz };
        }

        // Copy normals
        auto* mesh_normals = bifrost_mesh.get_normals();
        if (mesh_normals != nullptr) {
            for (auto index_pair : vertex_index_map) {
                float nx = attributes.normals[3 * index_pair.first.normal_index + 0];
                float ny = attributes.normals[3 * index_pair.first.normal_index + 1];
                float nz = attributes.normals[3 * index_pair.first.normal_index + 2];
                mesh_normals[index_pair.second] = { nx, ny, nz };
            }
        }

        // Copy texcoords
        auto* mesh_texcoord = bifrost_mesh.get_texcoords();
        if (mesh_texcoord != nullptr) {
            for (auto index_pair : vertex_index_map) {
                float u = attributes.texcoords[2 * index_pair.first.texcoord_index + 0];
                float v = attributes.texcoords[2 * index_pair.first.texcoord_index + 1];
                mesh_texcoord[index_pair.second] = { u, v };
            }
        }

        bifrost_mesh.compute_bounds();

        SceneNodes::UID node_ID = SceneNodes::create(shape.name);
        if (root_ID != SceneNodes::UID::invalid_UID())
            SceneNodes::set_parent(node_ID, root_ID);
        else
            root_ID = node_ID;

        int material_index = shape.mesh.material_ids[0]; // No per facet material support. TODO Add it in the future by splitting up the shape.
        Materials::UID material_ID = material_index >= 0 ? materials[material_index] : Materials::UID::invalid_UID();
        MeshModels::UID model_ID = MeshModels::create(node_ID, bifrost_mesh.get_ID(), material_ID);
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
