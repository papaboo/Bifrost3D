// Bifrost model for meshes.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_MESH_MODEL_H_
#define _BIFROST_ASSETS_MESH_MODEL_H_

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/Material.h>
#include <Bifrost/Core/Bitmask.h>
#include <Bifrost/Core/ChangeSet.h>
#include <Bifrost/Core/Iterable.h>
#include <Bifrost/Core/UniqueIDGenerator.h>
#include <Bifrost/Scene/SceneNode.h>

namespace Bifrost {
namespace Assets {

//-------------------------------------------------------------------------------------------------
// A mesh model contains the mesh and material IDs and combines them with the scene node ID.
//-------------------------------------------------------------------------------------------------
class MeshModels final {
public:
    typedef Core::TypedUIDGenerator<MeshModels> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static bool is_allocated() { return m_models != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(MeshModels::UID model_ID) { return m_UID_generator.has(model_ID); }

    static MeshModels::UID create(Scene::SceneNodes::UID scene_node_ID, Meshes::UID mesh_ID, Materials::UID material_ID);
    static void destroy(MeshModels::UID model_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline Scene::SceneNodes::UID get_scene_node_ID(MeshModels::UID model_ID) { return m_models[model_ID].scene_node_ID; }
    static inline Meshes::UID get_mesh_ID(MeshModels::UID model_ID) { return m_models[model_ID].mesh_ID; }
    static inline Materials::UID get_material_ID(MeshModels::UID model_ID) { return m_models[model_ID].material_ID; }
    static void set_material_ID(MeshModels::UID model_ID, Materials::UID material_ID);

    //---------------------------------------------------------------------------------------------
    // Changes since last game loop tick.
    //---------------------------------------------------------------------------------------------
    enum class Change : unsigned char {
        None = 0u,
        Created = 1u << 0u,
        Destroyed = 1u << 1u,
        Material = 1u << 4u,
        All = Created | Destroyed | Material
    }; 
    typedef Core::Bitmask<Change> Changes;

    static inline Changes get_changes(MeshModels::UID model_ID) { return m_changes.get_changes(model_ID); }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_models() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { m_changes.reset_change_notifications(); }

private:
    static void reserve_model_data(unsigned int new_capacity, unsigned int old_capacity);

    struct Model final {
        Scene::SceneNodes::UID scene_node_ID;
        Meshes::UID mesh_ID;
        Materials::UID material_ID;
    };

    static UIDGenerator m_UID_generator;
    static Model* m_models;
    static Core::ChangeSet<Changes, UID> m_changes;
};

//-------------------------------------------------------------------------------------------------
// MeshModel UID wrapper.
//-------------------------------------------------------------------------------------------------
class MeshModel final {
public:
    //---------------------------------------------------------------------------------------------
    // Class management.
    //---------------------------------------------------------------------------------------------
    MeshModel() : m_ID(MeshModels::UID::invalid_UID()) {}
    MeshModel(MeshModels::UID id) : m_ID(id) {}

    inline const MeshModels::UID get_ID() const { return m_ID; }
    inline bool exists() const { return MeshModels::has(m_ID); }

    inline bool operator==(MeshModel rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(MeshModel rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters.
    // -----------------------------------------------------------------------
    inline Scene::SceneNode get_scene_node() const { return MeshModels::get_scene_node_ID(m_ID); }
    inline Mesh get_mesh() const { return MeshModels::get_mesh_ID(m_ID); }
    inline Material get_material() const { return MeshModels::get_material_ID(m_ID); }
    inline void set_material_ID(Materials::UID material_ID) { MeshModels::set_material_ID(m_ID, material_ID); }

    inline MeshModels::Changes get_changes() const { return MeshModels::get_changes(m_ID); }

private:
    MeshModels::UID m_ID;
};

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_MESH_MODEL_H_
