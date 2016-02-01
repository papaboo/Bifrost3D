// Cogwheel model for meshes.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MESH_MODEL_H_
#define _COGWHEEL_ASSETS_MESH_MODEL_H_

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Scene/SceneNode.h>

namespace Cogwheel {
namespace Assets {

//----------------------------------------------------------------------------
// A mesh model contains the mesh and material IDs and combines them with 
// the scene node ID.
// Future work:
// * Add material IDs.
// * Add model properties: Static, visible, etc.
//----------------------------------------------------------------------------
struct MeshModel final {
    Scene::SceneNodes::UID m_scene_node_ID;
    Assets::Meshes::UID m_mesh_ID;
};

class MeshModels final {
public:
    typedef Core::TypedUIDGenerator<Meshes> UIDGenerator;
    typedef UIDGenerator::UID UID;

    static bool is_allocated() { return m_models != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(MeshModels::UID model_ID) { return m_UID_generator.has(model_ID); }

    static MeshModels::UID create(Scene::SceneNodes::UID scene_node_ID, Assets::Meshes::UID mesh_ID);

    static inline MeshModel get_model(MeshModels::UID model_ID) { return m_models[model_ID]; }
    static inline void set_model(MeshModels::UID model_ID, MeshModel model) { m_models[model_ID] = model; }

    static inline Scene::SceneNodes::UID get_scene_node_ID(MeshModels::UID model_ID) { return m_models[model_ID].m_scene_node_ID; }
    static inline void set_scene_node_ID(MeshModels::UID model_ID, Scene::SceneNodes::UID node_ID) { m_models[model_ID].m_scene_node_ID = node_ID; }

    static inline Assets::Meshes::UID get_mesh_ID(MeshModels::UID model_ID) { return m_models[model_ID].m_mesh_ID; }
    static inline void set_mesh_ID(MeshModels::UID model_ID, Assets::Meshes::UID mesh_ID) { m_models[model_ID].m_mesh_ID = mesh_ID; }

    static UIDGenerator::ConstIterator begin() { return m_UID_generator.begin(); }
    static UIDGenerator::ConstIterator end() { return m_UID_generator.end(); }

private:
    static void reserve_node_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;
    static MeshModel* m_models;
};

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_MODEL_H_