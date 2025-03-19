// Bifrost mesh.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_MESH_H_
#define _BIFROST_ASSETS_MESH_H_

#include <Bifrost/Core/Bitmask.h>
#include <Bifrost/Core/ChangeSet.h>
#include <Bifrost/Core/Iterable.h>
#include <Bifrost/Core/UniqueIDGenerator.h>
#include <Bifrost/Math/AABB.h>
#include <Bifrost/Math/Color.h>
#include <Bifrost/Math/Matrix.h>
#include <Bifrost/Math/Transform.h>
#include <Bifrost/Math/Vector.h>

namespace Bifrost::Assets {

enum class MeshFlag : unsigned char {
    None             = 0u,
    Position         = 1u << 0u,
    Normal           = 1u << 1u,
    Texcoord         = 1u << 2u,
    TintAndRoughness = 1u << 3u,
    AllBuffers = Position | Normal | Texcoord | TintAndRoughness
};
typedef Core::Bitmask<MeshFlag> MeshFlags;

//----------------------------------------------------------------------------
// Mesh ID
//----------------------------------------------------------------------------
class Meshes;
typedef Core::TypedUIDGenerator<Meshes> MeshIDGenerator;
typedef MeshIDGenerator::UID MeshID;

struct TintRoughness {
    Math::RGB24 tint;
    Math::UNorm8 roughness;

    __always_inline__ bool operator==(TintRoughness rhs) const { return memcmp(this, &rhs, sizeof(rhs)) == 0; }
};

//----------------------------------------------------------------------------
// Container for mesh properties and their bufers.
// Future work:
// * Verify that creating and destroying meshes don't leak!
// * Array access functions should probably map the data as read- or writable
//   and set appropiate change flags.
//----------------------------------------------------------------------------
class Meshes final {
public:
    using Iterator = MeshIDGenerator::ConstIterator;

    static inline bool is_allocated() { return m_buffers != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static inline bool has(MeshID mesh_ID) { return m_UID_generator.has(mesh_ID) && get_changes(mesh_ID) != Change::Destroyed; }

    static MeshID create(const std::string& name, unsigned int primitive_count, unsigned int vertex_count, MeshFlags buffer_bitmask = MeshFlag::AllBuffers);
    static void destroy(MeshID mesh_ID);

    static inline Iterator begin() { return m_UID_generator.begin(); }
    static inline Iterator end() { return m_UID_generator.end(); }
    static inline Core::Iterable<Iterator> get_iterable() { return Core::Iterable(begin(), end()); }

    static inline const std::string& get_name(MeshID mesh_ID) { return m_names[mesh_ID]; }
    static inline void set_name(MeshID mesh_ID, const std::string& name) { m_names[mesh_ID] = name; }

    static inline unsigned int get_primitive_count(MeshID mesh_ID) { return m_buffers[mesh_ID].primitive_count; }
    static inline Math::Vector3ui* get_primitives(MeshID mesh_ID) { return m_buffers[mesh_ID].primitives; }
    static inline unsigned int get_index_count(MeshID mesh_ID) { return get_primitive_count(mesh_ID) * 3; }
    static inline unsigned int* get_indices(MeshID mesh_ID) { return (unsigned int*)(void*)get_primitives(mesh_ID); }

    static inline unsigned int get_vertex_count(MeshID mesh_ID) { return m_buffers[mesh_ID].vertex_count; }
    static inline Math::Vector3f* get_positions(MeshID mesh_ID) { return m_buffers[mesh_ID].positions; }
    static inline Math::Vector3f* get_normals(MeshID mesh_ID) { return m_buffers[mesh_ID].normals; }
    static inline Math::Vector2f* get_texcoords(MeshID mesh_ID) { return m_buffers[mesh_ID].texcoords; }
    static inline TintRoughness* get_tint_and_roughness(MeshID mesh_ID) { return m_buffers[mesh_ID].tint_and_roughness; }
    static inline Math::AABB get_bounds(MeshID mesh_ID) { return m_bounds[mesh_ID]; }
    static inline void set_bounds(MeshID mesh_ID, Math::AABB bounds) { m_bounds[mesh_ID] = bounds; }
    static Math::AABB compute_bounds(MeshID mesh_ID);

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

    static inline Changes get_changes(MeshID mesh_ID) { return m_changes.get_changes(mesh_ID); }

    typedef std::vector<MeshID>::iterator ChangedIterator;
    static inline Core::Iterable<ChangedIterator> get_changed_meshes() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications();

private:
    static void reserve_mesh_data(unsigned int new_capacity, unsigned int old_capacity);

    struct Buffers {
        unsigned int primitive_count;
        unsigned int vertex_count;

        Math::Vector3ui* primitives;
        Math::Vector3f* positions;
        Math::Vector3f* normals;
        Math::Vector2f* texcoords;
        TintRoughness* tint_and_roughness;
    };
    static void delete_buffers(Buffers& buffers);

    static MeshIDGenerator m_UID_generator;
    static std::string* m_names;

    static Buffers* m_buffers;
    static Math::AABB* m_bounds;

    static Core::ChangeSet<Changes, MeshID> m_changes;
};

// ---------------------------------------------------------------------------
// Mesh UID wrapper.
// ---------------------------------------------------------------------------
class Mesh final {
public:
    // -----------------------------------------------------------------------
    // Class management.
    // -----------------------------------------------------------------------
    Mesh() : m_ID(MeshID::invalid_UID()) {}
    Mesh(MeshID id) : m_ID(id) {}
    Mesh(const std::string& name, unsigned int primitive_count, unsigned int vertex_count, MeshFlags buffer_bitmask = MeshFlag::AllBuffers)
        : m_ID(Meshes::create(name, primitive_count, vertex_count, buffer_bitmask)) { }

    static Mesh invalid() { return MeshID::invalid_UID(); }

    inline void destroy() { Meshes::destroy(m_ID); }
    inline bool exists() const { return Meshes::has(m_ID); }
    inline const MeshID get_ID() const { return m_ID; }

    inline bool operator==(Mesh rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(Mesh rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline const std::string& get_name() const { return Meshes::get_name(m_ID); }
    inline void set_name(const std::string& name) { Meshes::set_name(m_ID, name); }

    inline unsigned int get_index_count() { return Meshes::get_index_count(m_ID); }
    inline unsigned int* get_indices() { return Meshes::get_indices(m_ID); }
    inline Core::Iterable<unsigned int*> get_indices_iterable() { return Core::Iterable(get_indices(), get_index_count()); }
    inline unsigned int get_primitive_count() { return Meshes::get_primitive_count(m_ID); }
    inline Math::Vector3ui* get_primitives() { return Meshes::get_primitives(m_ID); }
    inline Core::Iterable<Math::Vector3ui*> get_primitive_iterable() { return Core::Iterable(get_primitives(), get_primitive_count()); }

    inline unsigned int get_vertex_count() { return Meshes::get_vertex_count(m_ID); }
    inline Math::Vector3f* get_positions() { return Meshes::get_positions(m_ID); }
    inline Core::Iterable<Math::Vector3f*> get_position_iterable() { return Core::Iterable(get_positions(), get_vertex_count()); }
    inline Math::Vector3f* get_normals() { return Meshes::get_normals(m_ID); }
    inline Core::Iterable<Math::Vector3f*> get_normal_iterable() { return Core::Iterable(get_normals(), get_vertex_count()); }
    inline Math::Vector2f* get_texcoords() { return Meshes::get_texcoords(m_ID); }
    inline Core::Iterable<Math::Vector2f*> get_texcoord_iterable() { return Core::Iterable(get_texcoords(), get_vertex_count()); }
    inline TintRoughness* get_tint_and_roughness() { return Meshes::get_tint_and_roughness(m_ID); }
    inline Core::Iterable<TintRoughness*> get_tint_and_roughness_iterable() { return Core::Iterable(get_tint_and_roughness(), get_vertex_count()); }

    inline Math::AABB get_bounds() { return Meshes::get_bounds(m_ID); }
    inline void set_bounds(Math::AABB bounds) { Meshes::set_bounds(m_ID, bounds); }
    inline Math::AABB compute_bounds() { return Meshes::compute_bounds(m_ID); }

    inline MeshFlags get_flags() {
        MeshFlags mesh_flags = get_positions() ? MeshFlag::Position : MeshFlag::None;
        mesh_flags |= get_normals() ? MeshFlag::Normal : MeshFlag::None;
        mesh_flags |= get_texcoords() ? MeshFlag::Texcoord : MeshFlag::None;
        mesh_flags |= get_tint_and_roughness() ? MeshFlag::TintAndRoughness : MeshFlag::None;
        return mesh_flags;
    }

    inline Meshes::Changes get_changes() const { return Meshes::get_changes(m_ID); }

private:
    MeshID m_ID;
};

//----------------------------------------------------------------------------
// Mesh utilities.
// Future work:
// * Forsyth index sorting. https://code.google.com/archive/p/vcacne/
// * What about vertex sorting? Base it on a morton curve or order of appearance in the index array.
// * Utility function for computing tangents and normals on bump mapped surfaces. Possibly splitting the mesh.
//----------------------------------------------------------------------------
namespace MeshUtils {

Mesh deep_clone(Mesh mesh_ID);
void transform_mesh(Mesh mesh_ID, Math::Matrix3x4f affine_transform);
void transform_mesh(Mesh mesh_ID, Math::Transform transform);

//-------------------------------------------------------------------------
// Mesh combine utilities.
//-------------------------------------------------------------------------
struct TransformedMesh {
    Mesh mesh;
    Math::Transform transform;
};

Mesh combine(const std::string& name, 
             const TransformedMesh* const meshes_begin, const TransformedMesh* const meshes_end, 
             MeshFlags flags = MeshFlag::AllBuffers);

inline Mesh combine(const std::string& name, 
                    Mesh mesh0_ID, Math::Transform transform0,
                    Mesh mesh1_ID, Math::Transform transform1, 
                    MeshFlags flags = MeshFlag::AllBuffers) {
    TransformedMesh mesh0 = { mesh0_ID, transform0 };
    TransformedMesh mesh1 = { mesh1_ID, transform1 };
    TransformedMesh meshes[2] = { mesh0, mesh1 };
    return combine(name, meshes, meshes + 2, flags);
}

// Computes a list of hard normals from a list of triangle positions.
// This function assumes that the positions are used to describe triangles.
void compute_hard_normals(Math::Vector3f* positions_begin, Math::Vector3f* positions_end, Math::Vector3f* normals_begin);

// Computes per vertex normals.
void compute_normals(Math::Vector3ui* primitives_begin, Math::Vector3ui* primitives_end,
                     Math::Vector3f* normals_begin, Math::Vector3f* normals_end, Math::Vector3f* positions_begin);
void compute_normals(Mesh mesh);

// Expands a buffer and a list of triangle vertex indices into a non-indexed buffer.
// Useful for expanding meshes that uses indexing into a mesh that does not.
template <typename RandomAccessIterator>
void expand_indexed_buffer(Math::Vector3ui* primitives, int primitive_count, RandomAccessIterator buffer_itr, 
                           RandomAccessIterator expanded_buffer_itr) {
    for (Math::Vector3ui primitive : Core::Iterable(primitives, primitive_count)) {
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

// Creates a new mesh where vertices with the same vertex attributes have been merged.
Mesh merge_duplicate_vertices(Mesh mesh, MeshFlags attribute_types = MeshFlag::AllBuffers);

} // NS MeshUtils

//----------------------------------------------------------------------------
// Utility tests for verifying the validity of a mesh.
//----------------------------------------------------------------------------
namespace MeshTests {

// Tests that the normals corrospond to the winding order,
// i.e. that the front face defined by the winding order 
// is facing the same general direction as the normals.
// TODO optional function that all invalid primitives are passed to.
// Returns the number of primitives whose winding order did not correspond to their vertex normals.
unsigned int normals_correspond_to_winding_order(Mesh mesh);

// TODO callback function that receives found degenerate primitives.
unsigned int count_degenerate_primitives(Mesh mesh, float epsilon_squared = 0.000001f);

// Tests that no indices index out of bounds.
inline bool has_invalid_indices(Mesh mesh) {
    unsigned int max_index = 0;
    for (unsigned int i = 0; i < mesh.get_index_count(); ++i)
        max_index = Math::max(max_index, mesh.get_indices()[i]);

    return !(max_index < mesh.get_vertex_count());
}

} // NS MeshTests

} // NS Bifrost::Assets

#endif // _BIFROST_ASSETS_MESH_H_
