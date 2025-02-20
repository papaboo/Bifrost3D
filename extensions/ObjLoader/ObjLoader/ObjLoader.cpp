// Bifrost obj model loader.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#pragma warning(disable : 4996)

#include <ObjLoader/ObjLoader.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Core/Array.h>
#include <Bifrost/Scene/SceneNode.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <ObjLoader/tiny_obj_loader.h>

#include <map>
#include <unordered_map>

using namespace Bifrost;
using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace ObjLoader {

using ImageCache = std::unordered_map<std::string, Image>;

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

// Loads images in materials.
// Images that fail to load are are also added to the cache to prevent attempting to load them multiple times and to make all image lookups valid later.
void load_material_images(const std::vector<tinyobj::material_t>& materials, const std::string& directory, ImageLoader image_loader,
    ImageCache& tint_cache, ImageCache& coverage_cache) {
    for (auto mat : materials) {

        std::string& coverage_name = mat.alpha_texname;
        if (!coverage_name.empty() && coverage_cache.find(coverage_name) == coverage_cache.end()) {
            std::string coverage_image_path = directory + coverage_name;
            Image coverage_image = image_loader(coverage_image_path);
            coverage_cache.insert({ coverage_name, coverage_image });

            if (coverage_image.exists())
                if (coverage_image.get_pixel_format() != PixelFormat::Alpha8)
                    coverage_image.change_format(PixelFormat::Alpha8, 1.0f);
            else
                printf("ObjLoader::load error: Could not load image at '%s'.\n", coverage_image_path.c_str());
        }

        std::string& tint_name = mat.diffuse_texname;
        if (!tint_name.empty() && tint_cache.find(tint_name) == tint_cache.end()) {
            std::string tint_image_path = directory + tint_name;
            Image tint_image = image_loader(tint_image_path);
            tint_cache.insert({ tint_name, tint_image });

            if (tint_image.exists()) {
                // Extract alpha as coverage and set the roughness to 1 if the image has an alpha channel
                if (channel_count(tint_image.get_pixel_format()) == 4) {
                    unsigned int mipmap_count = tint_image.get_mipmap_count();
                    Vector2ui size = tint_image.get_size_2D();
                    Image coverage_image = Image::create2D(tint_image_path, PixelFormat::Alpha8, 1.0f, size);
                    UNorm8* coverage_pixels = coverage_image.get_pixels<UNorm8>();

                    float min_coverage = 1.0f;
                    if (tint_image.get_pixel_format() == PixelFormat::RGBA32) {
                        min_coverage = 255.0f;
                        RGBA32* tint_pixels = tint_image.get_pixels<RGBA32>();
                        for (unsigned int p = 0; p < tint_image.get_pixel_count(); ++p) {
                            coverage_pixels[p] = tint_pixels[p].a;
                            min_coverage = fminf(min_coverage, coverage_pixels[p]);
                            tint_pixels[p].a = byte(255);
                        }
                        min_coverage /= 255;

                    } else {

                        for (unsigned int p = 0; p < tint_image.get_pixel_count(); ++p) {
                            RGBA tint_pixel = tint_image.get_pixel(p);

                            min_coverage = fminf(min_coverage, tint_pixel.a);
                            coverage_pixels[p] = unsigned char(tint_pixel.a * 255 + 0.5f);

                            // Set roughness to 1 in the original tint/roughness image.
                            tint_pixel.a = 1.0f;
                            tint_image.set_pixel(tint_pixel, p);
                        }
                    }

                    if (min_coverage < 1.0f)
                        coverage_cache.insert({ tint_name, coverage_image });
                }
            } else
                printf("ObjLoader::load error: Could not load image at '%s'.\n", tint_image_path.c_str());
        }

        if (!mat.roughness_texname.empty())
            printf("ObjLoader::load error: Roughness texture not supported.\n");
        if (!mat.metallic_texname.empty())
            printf("ObjLoader::load error: Metallic texture not supported.\n");
        if (!mat.sheen_texname.empty())
            printf("ObjLoader::load error: Sheen texture not supported.\n");
        if (!mat.emissive_texname.empty())
            printf("ObjLoader::load error: Emissive texture not supported.\n");
        if (!mat.normal_texname.empty())
            printf("ObjLoader::load error: Normal map not supported.\n");
    }
}

SceneNode load(const std::string& path, ImageLoader image_loader) {
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
        return SceneNode::invalid();

    SceneNode root = shapes.size() > 1u ? SceneNode(std::string(filename.begin(), filename.end()-4)) : SceneNodeID::invalid_UID();

    ImageCache tint_images;
    ImageCache coverage_images;
    load_material_images(tiny_materials, directory, image_loader, tint_images, coverage_images);

    Core::Array<Material> materials = Core::Array<Material>(unsigned int(tiny_materials.size()));
    for (int i = 0; i < int(tiny_materials.size()); ++i) {
        tinyobj::material_t tiny_mat = tiny_materials[i];

        Materials::Data material_data = {};
        material_data.flags = MaterialFlag::None;
        material_data.tint = RGB(tiny_mat.diffuse[0], tiny_mat.diffuse[1], tiny_mat.diffuse[2]);
        material_data.roughness = sqrt(2.0f / (tiny_mat.shininess + 2.0f)); // Map from blinn shininess to material roughness. See D_Blinn in https://github.com/EpicGames/UnrealEngine/blob/d94b38ae3446da52224bedd2568c078f828b4039/Engine/Shaders/Private/BRDF.ush
        bool is_metallic = tiny_mat.illum == 3 || tiny_mat.illum == 5;
        material_data.metallic = is_metallic ? 1.0f : 0.0f;
        material_data.specularity = (tiny_mat.specular[0] + tiny_mat.specular[1] + tiny_mat.specular[2]) / 3.0f;
        material_data.coverage = tiny_mat.dissolve;
        material_data.transmission = 0.0f; // (tiny_mat.transmittance[0] + tiny_mat.transmittance[1] + tiny_mat.transmittance[2]) / 3.0f;

        // Warn about completely transparent object. Happens from time to time and it's hell to debug a missing model.
        if (material_data.coverage <= 0.0f)
            printf("ObjLoader::load warning: Coverage set to %.3f. Material %s is completely transparent.\n", material_data.coverage, tiny_mat.name.c_str());

        // Coverage image
        if (!tiny_mat.alpha_texname.empty() || !tiny_mat.diffuse_texname.empty()) {
            Image alpha_image = Image();

            // First lookup in coverage images
            auto& alpha_image_itr = coverage_images.find(tiny_mat.alpha_texname);
            if (alpha_image_itr != coverage_images.end())
                alpha_image = alpha_image_itr->second;

            // Then attempt diffuse alpha channel images if coverage images didn't provide an image
            if (!alpha_image.exists()) {
                auto& alpha_image_itr = coverage_images.find(tiny_mat.diffuse_texname);
                if (alpha_image_itr != coverage_images.end())
                    alpha_image = alpha_image_itr->second;
            }

            if (alpha_image.exists())
                material_data.coverage_texture_ID = Texture::create2D(alpha_image).get_ID();
        }

        // Diffuse image
        if (!tiny_mat.diffuse_texname.empty()) {
            auto& tint_image_itr = tint_images.find(tiny_mat.diffuse_texname);
            if (tint_image_itr != tint_images.end()) {
                Image tint_image = tint_image_itr->second;
                if (tint_image.exists())
                    material_data.tint_roughness_texture_ID = Texture::create2D(tint_image).get_ID();
            }
        }

        materials[i] = Material(tiny_mat.name, material_data);
    }

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

        Mesh bifrost_mesh = Mesh(shape.name, triangle_count, vertex_count, mesh_flags);

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

        SceneNode node = SceneNode(shape.name);
        if (root != SceneNode::invalid())
            node.set_parent(root);
        else
            root = node;

        int material_index = shape.mesh.material_ids[0]; // No per facet material support. TODO Add it in the future by splitting up the shape.
        Material material = material_index >= 0 ? materials[material_index] : Material::invalid();
        MeshModel(node, bifrost_mesh, material);
    }

    return root;
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
