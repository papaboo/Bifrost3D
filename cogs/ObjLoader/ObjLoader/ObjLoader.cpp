// Cogwheel obj model laoder.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#pragma warning(disable : 4996)

#include <ObjLoader/ObjLoader.h>

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <ObjLoader/tiny_obj_loader.h>

using namespace Cogwheel::Assets;
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

SceneNodes::UID load(const std::string& path) {
    std::string directory, filename;
    split_path(directory, filename, path);

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string error;
    bool obj_loaded = tinyobj::LoadObj(shapes, materials, error, path.c_str(), directory.c_str());

    if (!error.empty()) {
        printf("ObjLoader::load error: %s.\n", error.c_str());
        if (!obj_loaded)
            return SceneNodes::UID::invalid_UID();
    }

    SceneNodes::UID root_ID = shapes.size() > 1u ? SceneNodes::create(std::string(filename.begin(), filename.end()-4)) : SceneNodes::UID::invalid_UID();

    for (size_t i = 0; i < shapes.size(); i++) {

        Meshes::UID mesh_ID = Meshes::UID::invalid_UID();
        { // Create mesh 
            assert((shapes[i].mesh.indices.size() % 3) == 0); // Assert that indices are for triangles.
            assert((shapes[i].mesh.positions.size() % 3) == 0); // Assert that positions are three dimensional.
            assert((shapes[i].mesh.normals.size() % 3) == 0); // Assert that normals are three dimensional.
            assert((shapes[i].mesh.texcoords.size() % 2) == 0); // Assert that texcoords are two dimensional.

            tinyobj::mesh_t tiny_mesh = shapes[i].mesh;
            mesh_ID = Meshes::create(shapes[i].name, unsigned int(tiny_mesh.indices.size() / 3u), unsigned int(tiny_mesh.positions.size() / 3u));

            // TODO Verify if any of these arrays are null.
            Mesh& cogwheel_mesh = Meshes::get_mesh(mesh_ID);
            memcpy(cogwheel_mesh.m_indices, tiny_mesh.indices.data(), tiny_mesh.indices.size() * sizeof(unsigned int));
            memcpy(cogwheel_mesh.m_positions, tiny_mesh.positions.data(), tiny_mesh.positions.size() * sizeof(float));
            memcpy(cogwheel_mesh.m_normals, tiny_mesh.normals.data(), tiny_mesh.normals.size() * sizeof(float));
            memcpy(cogwheel_mesh.m_texcoords, tiny_mesh.texcoords.data(), tiny_mesh.texcoords.size() * sizeof(float));

            Meshes::compute_bounds(mesh_ID);
        }

        SceneNodes::UID node_ID = SceneNodes::create(shapes[i].name);
        if (root_ID != SceneNodes::UID::invalid_UID())
            SceneNodes::set_parent(node_ID, root_ID);
        else
            root_ID = node_ID;

        MeshModels::UID model_ID = MeshModels::create(node_ID, mesh_ID);
    }

    return root_ID;
}

} // NS ObjLoader
