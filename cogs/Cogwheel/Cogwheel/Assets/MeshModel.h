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
#include <Cogwheel/Assets/Material.h>
#include <Cogwheel/Core/Bitmask.h>
#include <Cogwheel/Core/ChangeSet.h>
#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Scene/SceneNode.h>

namespace Cogwheel {
namespace Assets {

//----------------------------------------------------------------------------
// A mesh model contains the mesh and material IDs and combines them with 
// the scene node ID.
// When a MeshModel ID is destroyed, it is still possible to access it's values 
// until clear_change_notifications has been called, usually at the 
// end of the game loop.
//----------------------------------------------------------------------------
class MeshModels final {
public:
    typedef Core::TypedUIDGenerator<Meshes> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static bool is_allocated() { return m_models != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(MeshModels::UID model_ID) { return m_UID_generator.has(model_ID); }

    static MeshModels::UID create(Scene::SceneNodes::UID scene_node_ID, Assets::Meshes::UID mesh_ID, Assets::Materials::UID material_ID);
    static void destroy(MeshModels::UID model_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline Scene::SceneNodes::UID get_scene_node_ID(MeshModels::UID model_ID) { return m_models[model_ID].scene_node_ID; }
    static inline Assets::Meshes::UID get_mesh_ID(MeshModels::UID model_ID) { return m_models[model_ID].mesh_ID; }
    static inline Assets::Materials::UID get_material_ID(MeshModels::UID model_ID) { return m_models[model_ID].material_ID; }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    enum class Change : unsigned char {
        None = 0u,
        Created = 1u << 0u,
        Destroyed = 1u << 1u,
        All = Created | Destroyed
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
        Assets::Meshes::UID mesh_ID;
        Assets::Materials::UID material_ID;
    };

    static UIDGenerator m_UID_generator;
    static Model* m_models;
    static Core::ChangeSet<Changes, UID> m_changes;
};

// ---------------------------------------------------------------------------
// MeshModel UID wrapper.
// ---------------------------------------------------------------------------
class MeshModel final {
public:
    // -----------------------------------------------------------------------
    // Class management.
    // -----------------------------------------------------------------------
    MeshModel() : m_ID(MeshModels::UID::invalid_UID()) {}
    MeshModel(MeshModels::UID id) : m_ID(id) {}

    inline const MeshModels::UID get_ID() const { return m_ID; }
    inline bool exists() const { return MeshModels::has(m_ID); }

    inline bool operator==(MeshModel rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(MeshModel rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters.
    // -----------------------------------------------------------------------
    inline Scene::SceneNode get_scene_node() { return MeshModels::get_scene_node_ID(m_ID); }
    inline Assets::Mesh get_mesh() { return MeshModels::get_mesh_ID(m_ID); }
    inline Assets::Material get_material() { return MeshModels::get_material_ID(m_ID); }

    inline MeshModels::Changes get_changes() { return MeshModels::get_changes(m_ID); }

private:
    MeshModels::UID m_ID;
};

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_MODEL_H_