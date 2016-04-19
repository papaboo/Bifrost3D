// Cogwheel mesh.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MESH_H_
#define _COGWHEEL_ASSETS_MESH_H_

#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/AABB.h>
#include <Cogwheel/Math/Vector.h>

#include <vector>

namespace Cogwheel {
namespace Assets {

namespace MeshFlags {
static const unsigned char Position   = 1u << 0u;
static const unsigned char Normal     = 1u << 1u;
static const unsigned char Texcoords  = 1u << 2u;
static const unsigned char AllBuffers = Position | Normal | Texcoords;
}

//----------------------------------------------------------------------------
// Container for the buffers that make up a mesh, such as positions and normals.
// Future work:
// * Verify that creating and destroying meshes don't leak!
//----------------------------------------------------------------------------
struct Mesh final {
    unsigned int indices_count;
    unsigned int vertex_count;
    
    Math::Vector3ui* indices;
    Math::Vector3f* positions;
    Math::Vector3f* normals;
    Math::Vector2f* texcoords;

    Mesh()
        : indices_count(0u)
        , vertex_count(0u)
        , indices(nullptr)
        , positions(nullptr)
        , normals(nullptr)
        , texcoords(nullptr) { }

    Mesh(unsigned int indices_count, unsigned int vertex_count, unsigned char buffer_bitmask = MeshFlags::AllBuffers)
        : indices_count(indices_count)
        , vertex_count(vertex_count)
        , indices(new Math::Vector3ui[indices_count])
        , positions((buffer_bitmask & MeshFlags::Position) ? new Math::Vector3f[vertex_count] : nullptr)
        , normals((buffer_bitmask & MeshFlags::Normal) ? new Math::Vector3f[vertex_count] : nullptr)
        , texcoords((buffer_bitmask & MeshFlags::Texcoords) ? new Math::Vector2f[vertex_count] : nullptr) {
    }
};

class Meshes final {
public:
    typedef Core::TypedUIDGenerator<Meshes> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static bool is_allocated() { return m_meshes != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(Meshes::UID mesh_ID) { return m_UID_generator.has(mesh_ID); }

    static Meshes::UID create(const std::string& name, unsigned int indices_count, unsigned int vertex_count, unsigned char buffer_bitmask = MeshFlags::AllBuffers);
    static void destroy(Meshes::UID mesh_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(Meshes::UID mesh_ID) { return m_names[mesh_ID]; }
    static inline void set_name(Meshes::UID mesh_ID, const std::string& name) { m_names[mesh_ID] = name; }

    static inline Mesh& get_mesh(Meshes::UID mesh_ID) { return m_meshes[mesh_ID]; }
    static inline unsigned int get_indices_count(Meshes::UID mesh_ID) { return m_meshes[mesh_ID].indices_count; }
    static inline Math::Vector3ui* get_indices(Meshes::UID mesh_ID) { return m_meshes[mesh_ID].indices; }
    static inline unsigned int get_vertex_count(Meshes::UID mesh_ID) { return m_meshes[mesh_ID].vertex_count; }
    static inline Math::Vector3f* get_positions(Meshes::UID mesh_ID) { return m_meshes[mesh_ID].positions; }
    static inline Math::Vector3f* get_normals(Meshes::UID mesh_ID) { return m_meshes[mesh_ID].normals; }
    static inline Math::Vector2f* get_texcoords(Meshes::UID mesh_ID) { return m_meshes[mesh_ID].texcoords; }
    static inline Math::AABB get_bounds(Meshes::UID mesh_ID) { return m_bounds[mesh_ID]; }
    static inline void set_bounds(Meshes::UID mesh_ID, Math::AABB bounds) { m_bounds[mesh_ID] = bounds; }
    static Math::AABB compute_bounds(Meshes::UID mesh_ID);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    struct Changes {
        static const unsigned char None = 0u;
        static const unsigned char Created = 1u << 0u;
        static const unsigned char Destroyed = 1u << 1u;
        static const unsigned char All = Created | Destroyed;
    };

    static inline unsigned char get_changes(Meshes::UID mesh_ID) { return m_changes[mesh_ID]; }
    static inline bool has_changes(Meshes::UID mesh_ID, unsigned char change_bitmask = Changes::All) {
        return (m_changes[mesh_ID] & change_bitmask) != Changes::None;
    }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_meshes() {
        return Core::Iterable<ChangedIterator>(m_meshes_changed.begin(), m_meshes_changed.end());
    }

    static void reset_change_notifications();

private:
    static void reserve_mesh_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;
    static std::string* m_names;

    static Mesh* m_meshes;
    static Math::AABB* m_bounds;

    static unsigned char* m_changes; // Bitmask of changes.
    static std::vector<UID> m_meshes_changed;
};

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_H_