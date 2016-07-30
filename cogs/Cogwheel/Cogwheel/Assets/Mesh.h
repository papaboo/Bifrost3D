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
#include <Cogwheel/Math/Transform.h>
#include <Cogwheel/Math/Vector.h>

#include <vector>

namespace Cogwheel {
namespace Assets {

namespace MeshFlags {
static const unsigned char None       = 0u;
static const unsigned char Position   = 1u << 0u;
static const unsigned char Normal     = 1u << 1u;
static const unsigned char Texcoord  = 1u << 2u;
static const unsigned char AllBuffers = Position | Normal | Texcoord;
}

//----------------------------------------------------------------------------
// Container for mesh properties and their bufers.
// Future work:
// * Verify that creating and destroying meshes don't leak!
// * Array access functions should probably map the data as read- or writable
//   and set appropiate change flags.
//----------------------------------------------------------------------------
class Meshes final {
public:
    typedef Core::TypedUIDGenerator<Meshes> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static inline bool is_allocated() { return m_buffers != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static inline bool has(Meshes::UID mesh_ID) { return m_UID_generator.has(mesh_ID); }

    static Meshes::UID create(const std::string& name, unsigned int index_count, unsigned int vertex_count, unsigned char buffer_bitmask = MeshFlags::AllBuffers);
    static void destroy(Meshes::UID mesh_ID);

    static inline ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static inline ConstUIDIterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(Meshes::UID mesh_ID) { return m_names[mesh_ID]; }
    static inline void set_name(Meshes::UID mesh_ID, const std::string& name) { m_names[mesh_ID] = name; }

    static inline unsigned int get_index_count(Meshes::UID mesh_ID) { return m_buffers[mesh_ID].index_count; }
    static inline Math::Vector3ui* get_indices(Meshes::UID mesh_ID) { return m_buffers[mesh_ID].indices; }
    static inline unsigned int get_vertex_count(Meshes::UID mesh_ID) { return m_buffers[mesh_ID].vertex_count; }
    static inline Math::Vector3f* get_positions(Meshes::UID mesh_ID) { return m_buffers[mesh_ID].positions; }
    static inline Math::Vector3f* get_normals(Meshes::UID mesh_ID) { return m_buffers[mesh_ID].normals; }
    static inline Math::Vector2f* get_texcoords(Meshes::UID mesh_ID) { return m_buffers[mesh_ID].texcoords; }
    static inline Math::AABB get_bounds(Meshes::UID mesh_ID) { return m_bounds[mesh_ID]; }
    static inline void set_bounds(Meshes::UID mesh_ID, Math::AABB bounds) { m_bounds[mesh_ID] = bounds; }
    static Math::AABB compute_bounds(Meshes::UID mesh_ID);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    static struct Changes {
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
    static inline Core::Iterable<ChangedIterator> get_changed_meshes() {
        return Core::Iterable<ChangedIterator>(m_meshes_changed.begin(), m_meshes_changed.end());
    }

    static void reset_change_notifications();

private:
    static void reserve_mesh_data(unsigned int new_capacity, unsigned int old_capacity);

    struct Buffers {
        unsigned int index_count;
        unsigned int vertex_count;

        Math::Vector3ui* indices;
        Math::Vector3f* positions;
        Math::Vector3f* normals;
        Math::Vector2f* texcoords;
    };

    static UIDGenerator m_UID_generator;
    static std::string* m_names;

    static Buffers* m_buffers;
    static Math::AABB* m_bounds;

    static unsigned char* m_changes; // Bitmask of changes.
    static std::vector<UID> m_meshes_changed;
};

// ---------------------------------------------------------------------------
// Mesh UID wrapper.
// ---------------------------------------------------------------------------
class Mesh final {
public:
    // -----------------------------------------------------------------------
    // Class management.
    // -----------------------------------------------------------------------
    Mesh() : m_ID(Meshes::UID::invalid_UID()) {}
    Mesh(Meshes::UID id) : m_ID(id) {}

    inline const Meshes::UID get_ID() const { return m_ID; }
    inline bool exists() const { return Meshes::has(m_ID); }

    inline bool operator==(Mesh rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(Mesh rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline std::string get_name() const { return Meshes::get_name(m_ID); }
    inline void set_name(const std::string& name) { Meshes::set_name(m_ID, name); }
    inline unsigned int get_index_count() { return Meshes::get_index_count(m_ID); }
    inline Math::Vector3ui* get_indices() { return Meshes::get_indices(m_ID); }
    inline unsigned int get_vertex_count() { return Meshes::get_vertex_count(m_ID); }
    inline Math::Vector3f* get_positions() { return Meshes::get_positions(m_ID); }
    inline Math::Vector3f* get_normals() { return Meshes::get_normals(m_ID); }
    inline Math::Vector2f* get_texcoords() { return Meshes::get_texcoords(m_ID); }
    inline Math::AABB get_bounds() { return Meshes::get_bounds(m_ID); }
    inline void set_bounds(Math::AABB bounds) { Meshes::set_bounds(m_ID, bounds); }

    inline Math::AABB compute_bounds() { return Meshes::compute_bounds(m_ID); }

    inline unsigned char get_changes() { return Meshes::get_changes(m_ID); }
    inline bool has_changes(unsigned char change_bitmask) { return Meshes::has_changes(m_ID, change_bitmask); }

private:
    const Meshes::UID m_ID;
};

//----------------------------------------------------------------------------
// Mesh utilities.
// Future work:
// * Forsyth index sorting. https://code.google.com/archive/p/vcacne/
// * What about vertex sorting ? Base it on a morton curve or order of appearance in the index array.
//----------------------------------------------------------------------------
namespace MeshUtils {

    // Future work
    // * Take N meshes and transforms as arguments.
    Meshes::UID combine(Meshes::UID mesh0_ID, Math::Transform transform0,
                        Meshes::UID mesh1_ID, Math::Transform transform1);

    inline Meshes::UID combine_and_destroy(Meshes::UID mesh0_ID, Math::Transform transform0,
                                           Meshes::UID mesh1_ID, Math::Transform transform1) {
        Meshes::UID combined_ID = combine(mesh0_ID, transform0, mesh1_ID, transform1);
        if (combined_ID != Meshes::UID::invalid_UID()) {
            Meshes::destroy(mesh0_ID);
            Meshes::destroy(mesh1_ID);
        }
        return combined_ID;
    }

} // NS MeshUtils

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_H_