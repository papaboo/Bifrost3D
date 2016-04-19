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
#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Scene/SceneNode.h>

#include <vector>

namespace Cogwheel {
namespace Assets {

struct MeshModel final {
    Scene::SceneNodes::UID scene_node_ID;
    Assets::Meshes::UID mesh_ID;
    Assets::Materials::UID material_ID;
};

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
    static bool has(MeshModels::UID model_ID);

    static MeshModels::UID create(Scene::SceneNodes::UID scene_node_ID, Assets::Meshes::UID mesh_ID, Assets::Materials::UID material_ID);
    static void destroy(MeshModels::UID model_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline MeshModel get_model(MeshModels::UID model_ID) { return m_models[model_ID]; }
    static inline void set_model(MeshModels::UID model_ID, MeshModel model) { m_models[model_ID] = model; }

    static inline Scene::SceneNodes::UID get_scene_node_ID(MeshModels::UID model_ID) { return m_models[model_ID].scene_node_ID; }
    static inline Assets::Meshes::UID get_mesh_ID(MeshModels::UID model_ID) { return m_models[model_ID].mesh_ID; }
    static inline Assets::Materials::UID get_material_ID(MeshModels::UID model_ID) { return m_models[model_ID].material_ID; }

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    struct Changes {
        static const unsigned char None = 0u;
        static const unsigned char Created = 1u << 0u;
        static const unsigned char Destroyed = 1u << 1u;
        static const unsigned char All = Created | Destroyed;
    }; 
    
    static inline unsigned char get_changes(MeshModels::UID model_ID) { return m_changes[model_ID]; }
    static inline bool has_changes(MeshModels::UID model_ID, unsigned char change_bitmask = Changes::All) {
        return (m_changes[model_ID] & change_bitmask) != Changes::None;
    }

    typedef std::vector<UID>::iterator model_created_iterator;
    static Core::Iterable<model_created_iterator> get_created_models() {
        return Core::Iterable<model_created_iterator>(m_models_created.begin(), m_models_created.end());
    }

    typedef std::vector<UID>::iterator model_destroyed_iterator;
    static Core::Iterable<model_destroyed_iterator> get_destroyed_models() {
        return Core::Iterable<model_destroyed_iterator>(m_models_destroyed.begin(), m_models_destroyed.end());
    }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_models() {
        return Core::Iterable<ChangedIterator>(m_models_changed.begin(), m_models_changed.end());
    }

    static void reset_change_notifications();

private:
    static void reserve_model_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;
    static MeshModel* m_models;
    static unsigned char* m_changes; // Bitmask of changes. Could be reduced to 2 bits pr model.
    static std::vector<UID> m_models_changed;

    // Change notifications.
    static std::vector<UID> m_models_created;
    static std::vector<UID> m_models_destroyed;
};

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_MODEL_H_