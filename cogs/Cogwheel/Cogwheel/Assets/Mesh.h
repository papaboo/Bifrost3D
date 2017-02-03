// Cogwheel mesh.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MESH_H_
#define _COGWHEEL_ASSETS_MESH_H_

#include <Cogwheel/Core/Bitmask.h>
#include <Cogwheel/Core/ChangeSet.h>
#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/AABB.h>
#include <Cogwheel/Math/Transform.h>
#include <Cogwheel/Math/Vector.h>

namespace Cogwheel {
namespace Assets {

enum class MeshFlag : unsigned char {
    None       = 0u,
    Position   = 1u << 0u,
    Normal     = 1u << 1u,
    Texcoord   = 1u << 2u,
    AllBuffers = Position | Normal | Texcoord
};
typedef Core::Bitmask<MeshFlag> MeshFlags;

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

    static Meshes::UID create(const std::string& name, unsigned int primitive_count, unsigned int vertex_count, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);
    static void destroy(Meshes::UID mesh_ID);

    static inline ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static inline ConstUIDIterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(Meshes::UID mesh_ID) { return m_names[mesh_ID]; }
    static inline void set_name(Meshes::UID mesh_ID, const std::string& name) { m_names[mesh_ID] = name; }

    static inline unsigned int get_primitive_count(Meshes::UID mesh_ID) { return m_buffers[mesh_ID].primitive_count; }
    static inline Math::Vector3ui* get_primitives(Meshes::UID mesh_ID) { return m_buffers[mesh_ID].primitives; }
    static inline unsigned int get_index_count(Meshes::UID mesh_ID) { return get_primitive_count(mesh_ID) * 3; }
    static inline unsigned int* get_indices(Meshes::UID mesh_ID) { return (unsigned int*)(void*)get_primitives(mesh_ID); }

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
    enum class Change : unsigned char {
        None = 0u,
        Created = 1u << 0u,
        Destroyed = 1u << 1u,
        All = Created | Destroyed,
    };
    typedef Core::Bitmask<Change> Changes;

    static inline Changes get_changes(Meshes::UID mesh_ID) { return m_changes.get_changes(mesh_ID); }

    typedef std::vector<UID>::iterator ChangedIterator;
    static inline Core::Iterable<ChangedIterator> get_changed_meshes() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { m_changes.reset_change_notifications(); }

private:
    static void reserve_mesh_data(unsigned int new_capacity, unsigned int old_capacity);

    struct Buffers {
        unsigned int primitive_count;
        unsigned int vertex_count;

        Math::Vector3ui* primitives;
        Math::Vector3f* positions;
        Math::Vector3f* normals;
        Math::Vector2f* texcoords;
    };

    static UIDGenerator m_UID_generator;
    static std::string* m_names;

    static Buffers* m_buffers;
    static Math::AABB* m_bounds;

    static Core::ChangeSet<Changes, UID> m_changes;
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
    inline unsigned int get_primitive_count() { return Meshes::get_primitive_count(m_ID); }
    inline Math::Vector3ui* get_primitives() { return Meshes::get_primitives(m_ID); }
    inline unsigned int get_index_count() { return Meshes::get_index_count(m_ID); }
    inline unsigned int* get_indices() { return Meshes::get_indices(m_ID); }
    inline Core::Iterable<Math::Vector3ui*> get_primitive_iterable() { return Core::Iterable<Math::Vector3ui*>(get_primitives(), get_primitive_count()); }
    inline unsigned int get_vertex_count() { return Meshes::get_vertex_count(m_ID); }
    inline Math::Vector3f* get_positions() { return Meshes::get_positions(m_ID); }
    inline Core::Iterable<Math::Vector3f*> get_position_iterable() { return Core::Iterable<Math::Vector3f*>(get_positions(), get_vertex_count()); }
    inline Math::Vector3f* get_normals() { return Meshes::get_normals(m_ID); }
    inline Core::Iterable<Math::Vector3f*> get_normal_iterable() { return Core::Iterable<Math::Vector3f*>(get_normals(), get_vertex_count()); }
    inline Math::Vector2f* get_texcoords() { return Meshes::get_texcoords(m_ID); }
    inline Core::Iterable<Math::Vector2f*> get_texcoord_iterable() { return Core::Iterable<Math::Vector2f*>(get_texcoords(), get_vertex_count()); }
    inline Math::AABB get_bounds() { return Meshes::get_bounds(m_ID); }
    inline void set_bounds(Math::AABB bounds) { Meshes::set_bounds(m_ID, bounds); }

    inline Math::AABB compute_bounds() { return Meshes::compute_bounds(m_ID); }

    inline MeshFlags get_flags() {
        MeshFlags mesh_flags = get_positions() ? MeshFlag::Position : MeshFlag::None;
        mesh_flags |= get_normals() ? MeshFlag::Normal : MeshFlag::None;
        mesh_flags |= get_texcoords() ? MeshFlag::Texcoord : MeshFlag::None;
        return mesh_flags;
    }

    inline Meshes::Changes get_changes() { return Meshes::get_changes(m_ID); }

private:
    const Meshes::UID m_ID;
};

//----------------------------------------------------------------------------
// Mesh utilities.
// Future work:
// * Forsyth index sorting. https://code.google.com/archive/p/vcacne/
// * What about vertex sorting? Base it on a morton curve or order of appearance in the index array.
// * Utility function for computing tangents and normals on bump mapped surfaces. Possibly splitting the mesh.
//----------------------------------------------------------------------------
namespace MeshUtils {

//-------------------------------------------------------------------------
// Mesh combine utilities.
//-------------------------------------------------------------------------
struct TransformedMesh {
    Meshes::UID mesh_ID;
    Math::Transform transform;
};

Meshes::UID combine(const std::string& name, 
                    const TransformedMesh* const meshes_begin, const TransformedMesh* const meshes_end, 
                    MeshFlags flags = MeshFlag::AllBuffers);

inline Meshes::UID combine(const std::string& name, 
                           Meshes::UID mesh0_ID, Math::Transform transform0,
                           Meshes::UID mesh1_ID, Math::Transform transform1, 
                           MeshFlags flags = MeshFlag::AllBuffers) {
    TransformedMesh mesh0 = { mesh0_ID, transform0 };
    TransformedMesh mesh1 = { mesh1_ID, transform1 };
    TransformedMesh meshes[2] = { mesh0, mesh1 };
    return combine(name, meshes, meshes + 2, flags);
}

// Computes a list of hard normals from a list of triangle positions.
// This function assumes that the positions are used to describe triangles.
void compute_hard_normals(Math::Vector3f* positions_begin, Math::Vector3f* positions_end, Math::Vector3f* normals_begin);

// Computes a list of hard normals from a list of triangle positions.
// This function assumes that the positions are used to describe triangles.
void compute_normals(Math::Vector3ui* primitives_begin, Math::Vector3ui* primitives_end,
                     Math::Vector3f* normals_begin, Math::Vector3f* normals_end, Math::Vector3f* positions_begin);
void compute_normals(Meshes::UID mesh_ID);

// Expands a buffer and a list of triangle vertex indices into a non-indexed buffer.
// Useful for expanding meshes that uses indexing into a mesh that does not.
template <typename RandomAccessIterator>
void expand_indexed_buffer(Math::Vector3ui* primitives, int primitive_count, RandomAccessIterator buffer_itr, 
                           RandomAccessIterator expanded_buffer_itr) {
    for (Math::Vector3ui primitive : Core::Iterable<Math::Vector3ui*>(primitives, primitive_count)) {
        *expanded_buffer_itr++ = buffer_itr[primitive.x];
        *expanded_buffer_itr++ = buffer_itr[primitive.y];
        *expanded_buffer_itr++ = buffer_itr[primitive.z];
    }
};

// Expands a buffer and a list of triangle vertex indices into a non-indexed buffer.
// Useful for expanding meshes that uses indexing into a mesh that does not.
template <typename RandomAccessIterator>
typename std::iterator_traits<RandomAccessIterator>::value_type* 
    expand_indexed_buffer(Math::Vector3ui* primitives, int primitive_count, RandomAccessIterator buffer) {

    auto expanded_buffer = new std::iterator_traits<RandomAccessIterator>::value_type[primitive_count * 3];
    expand_indexed_buffer(primitives, primitive_count, buffer, expanded_buffer);
    return expanded_buffer;
};

} // NS MeshUtils

//----------------------------------------------------------------------------
// Utility tests for verifying the validity of a mesh.
//----------------------------------------------------------------------------
namespace MeshTests {

// Tests that the normals corrospond to the winding order,
// i.e. that the front face defined by the winding order 
// is facing the same general direction as the normals.
// TODO optional function that all invalid primitives are passed to.
// Returns the number of primitives whose winding order did not corrospond to their vertex normals.
unsigned int normals_correspond_to_winding_order(Meshes::UID mesh_ID);

// TODO callback function that receives found degenerate primitives.
unsigned int count_degenerate_primitives(Meshes::UID mesh_ID, float epsilon_squared = 0.000001f);

} // NS MeshTests

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_H_