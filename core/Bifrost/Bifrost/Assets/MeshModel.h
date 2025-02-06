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

//----------------------------------------------------------------------------
// Mesh ID
//----------------------------------------------------------------------------
class MeshModels;
typedef Core::TypedUIDGenerator<MeshModels> MeshModelIDGenerator;
typedef MeshModelIDGenerator::UID MeshModelID;

//-------------------------------------------------------------------------------------------------
// A mesh model contains the mesh and material IDs and combines them with the scene node ID.
//-------------------------------------------------------------------------------------------------
class MeshModels final {
public:
    using Iterator = MeshModelIDGenerator::ConstIterator;

    static bool is_allocated() { return m_models != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(MeshModelID model_ID) { return m_UID_generator.has(model_ID); }

    static MeshModelID create(Scene::SceneNodeID scene_node_ID, MeshID mesh_ID, MaterialID material_ID);
    static void destroy(MeshModelID model_ID);

    static Iterator begin() { return m_UID_generator.begin(); }
    static Iterator end() { return m_UID_generator.end(); }
    static Core::Iterable<Iterator> get_iterable() { return Core::Iterable<Iterator>(begin(), end()); }

    static MeshModelID get_attached_mesh_model(Scene::SceneNodeID scene_node_ID);

    static inline Scene::SceneNodeID get_scene_node_ID(MeshModelID model_ID) { return m_models[model_ID].scene_node_ID; }
    static inline MeshID get_mesh_ID(MeshModelID model_ID) { return m_models[model_ID].mesh_ID; }
    static inline MaterialID get_material_ID(MeshModelID model_ID) { return m_models[model_ID].material_ID; }
    static void set_material_ID(MeshModelID model_ID, MaterialID material_ID);

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

    static inline Changes get_changes(MeshModelID model_ID) { return m_changes.get_changes(model_ID); }

    typedef std::vector<MeshModelID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_models() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { m_changes.reset_change_notifications(); }

private:
    static void reserve_model_data(unsigned int new_capacity, unsigned int old_capacity);

    struct Model final {
        Scene::SceneNodeID scene_node_ID;
        MeshID mesh_ID;
        MaterialID material_ID;
    };

    static MeshModelIDGenerator m_UID_generator;
    static Model* m_models;
    static Core::ChangeSet<Changes, MeshModelID> m_changes;
};

//-------------------------------------------------------------------------------------------------
// MeshModel ID wrapper.
//-------------------------------------------------------------------------------------------------
class MeshModel final {
public:
    //---------------------------------------------------------------------------------------------
    // Class management.
    //---------------------------------------------------------------------------------------------
    MeshModel() : m_ID(MeshModelID::invalid_UID()) {}
    MeshModel(MeshModelID id) : m_ID(id) {}
    MeshModel(Scene::SceneNode scene_node, Mesh mesh, Material material) :
        m_ID(MeshModels::create(scene_node.get_ID(), mesh.get_ID(), material.get_ID())) {}

    static MeshModel invalid() { return MeshModelID::invalid_UID(); }

    inline void destroy() { MeshModels::destroy(m_ID); }
    inline bool exists() const { return MeshModels::has(m_ID); }
    inline const MeshModelID get_ID() const { return m_ID; }

    inline bool operator==(MeshModel rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(MeshModel rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters.
    // -----------------------------------------------------------------------
    inline Scene::SceneNode get_scene_node() const { return MeshModels::get_scene_node_ID(m_ID); }
    inline Mesh get_mesh() const { return MeshModels::get_mesh_ID(m_ID); }
    inline Material get_material() const { return MeshModels::get_material_ID(m_ID); }
    inline void set_material(Material material) { MeshModels::set_material_ID(m_ID, material.get_ID()); }

    inline MeshModels::Changes get_changes() const { return MeshModels::get_changes(m_ID); }

private:
    MeshModelID m_ID;
};

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_MESH_MODEL_H_
