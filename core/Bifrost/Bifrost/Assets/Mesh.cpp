// Bifrost mesh.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Assets/Mesh.h>

#include <Bifrost/Core/Array.h>
#include <Bifrost/Math/Conversions.h>

#include <assert.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Assets {

MeshIDGenerator Meshes::m_UID_generator = MeshIDGenerator(0u);
std::string* Meshes::m_names = nullptr;
Meshes::Buffers* Meshes::m_buffers = nullptr;
AABB* Meshes::m_bounds = nullptr;

Core::ChangeSet<Meshes::Changes, MeshID> Meshes::m_changes;

void Meshes::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = MeshIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_buffers = new Buffers[capacity];
    m_bounds = new AABB[capacity];
    m_changes = Core::ChangeSet<Changes, MeshID>(capacity);

    // Allocate dummy element at 0.
    m_names[0] = "Dummy Node";
    m_buffers[0] = {};
    m_bounds[0] = AABB(Vector3f(nanf("")), Vector3f(nanf("")));
}

void Meshes::delete_buffers(Meshes::Buffers& buffers) {
    delete[] buffers.primitives; buffers.primitives = nullptr;
    delete[] buffers.positions; buffers.positions = nullptr;
    delete[] buffers.normals; buffers.normals = nullptr;
    delete[] buffers.texcoords; buffers.texcoords = nullptr;
    delete[] buffers.tint_and_roughness; buffers.tint_and_roughness = nullptr;
}

void Meshes::deallocate() {
    if (!is_allocated())
        return;

    for (MeshID id : m_UID_generator)
        delete_buffers(m_buffers[id]);
    delete[] m_names; m_names = nullptr;
    delete[] m_buffers; m_buffers = nullptr;
    delete[] m_bounds; m_bounds = nullptr;
    
    m_changes.resize(0);

    m_UID_generator = MeshIDGenerator(0u);
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void Meshes::reserve_mesh_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_names != nullptr);
    assert(m_buffers != nullptr);
    assert(m_bounds != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;
    m_names = resize_and_copy_array(m_names, new_capacity, copyable_elements);
    m_buffers = resize_and_copy_array(m_buffers, new_capacity, copyable_elements);
    m_bounds = resize_and_copy_array(m_bounds, new_capacity, copyable_elements);
    m_changes.resize(new_capacity);
}

void Meshes::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_mesh_data(m_UID_generator.capacity(), old_capacity);
}

MeshID Meshes::create(const std::string& name, unsigned int primitive_count, unsigned int vertex_count, MeshFlags buffer_bitmask) {
    assert(m_buffers != nullptr);
    assert(m_names != nullptr);
    assert(m_bounds != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    MeshID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_mesh_data(m_UID_generator.capacity(), old_capacity);

    m_names[id] = name;
    m_buffers[id].primitive_count = primitive_count;
    m_buffers[id].primitives = new Vector3ui[primitive_count];
    m_buffers[id].vertex_count = vertex_count;
    m_buffers[id].positions = (buffer_bitmask.contains(MeshFlag::Position)) ? new Vector3f[vertex_count] : nullptr;
    m_buffers[id].normals = (buffer_bitmask.contains(MeshFlag::Normal)) ? new Vector3f[vertex_count] : nullptr;
    m_buffers[id].texcoords = (buffer_bitmask.contains(MeshFlag::Texcoord)) ? new Vector2f[vertex_count] : nullptr;
    m_buffers[id].tint_and_roughness = (buffer_bitmask.contains(MeshFlag::TintAndRoughness)) ? new TintRoughness[vertex_count] : nullptr;
    m_bounds[id] = AABB::invalid();
    m_changes.set_change(id, Change::Created);

    return id;
}

void Meshes::destroy(MeshID mesh_ID) {
    if (has(mesh_ID)) {
        delete_buffers(m_buffers[mesh_ID]);
        m_changes.add_change(mesh_ID, Change::Destroyed);
    }
}

AABB Meshes::compute_bounds(MeshID mesh_ID) {
    Buffers& buffers = m_buffers[mesh_ID];

    AABB bounds = AABB(buffers.positions[0], buffers.positions[0]);
    for (Vector3f* position_itr = buffers.positions + 1; position_itr < (buffers.positions + buffers.vertex_count); ++position_itr)
        bounds.grow_to_contain(*position_itr);

    m_bounds[mesh_ID] = bounds;
    return bounds;
}

void Meshes::reset_change_notifications() {
    for (MeshID mesh_ID : get_changed_meshes())
        if (get_changes(mesh_ID).is_set(Change::Destroyed))
            m_UID_generator.erase(mesh_ID);
    m_changes.reset_change_notifications();
}

//-----------------------------------------------------------------------------
// Mesh utils.
//-----------------------------------------------------------------------------

namespace MeshUtils {

Mesh deep_clone(Mesh mesh) {
    Mesh new_mesh = Meshes::create(mesh.get_name() + "_clone", mesh.get_primitive_count(), mesh.get_vertex_count(), mesh.get_flags());

    Vector3f* positions_begin = mesh.get_positions();
    if (positions_begin != nullptr)
        std::copy_n(positions_begin, mesh.get_vertex_count(), new_mesh.get_positions());

    Vector3f* normals_begin = mesh.get_normals();
    if (normals_begin != nullptr)
        std::copy_n(normals_begin, mesh.get_vertex_count(), new_mesh.get_normals());

    Vector2f* texcoords_begin = mesh.get_texcoords();
    if (texcoords_begin != nullptr)
        std::copy_n(texcoords_begin, mesh.get_vertex_count(), new_mesh.get_texcoords());

    TintRoughness* tint_begin = mesh.get_tint_and_roughness();
    if (tint_begin != nullptr)
        std::copy_n(tint_begin, mesh.get_vertex_count(), new_mesh.get_tint_and_roughness());

    Vector3ui* primitives_begin = mesh.get_primitives();
    if (primitives_begin != nullptr)
        std::copy_n(primitives_begin, mesh.get_primitive_count(), new_mesh.get_primitives());

    new_mesh.set_bounds(mesh.get_bounds());

    return new_mesh;
}

void transform_mesh(Mesh mesh, Matrix3x4f affine_transform) {
    Matrix3x3f rotation;
    rotation.set_column(0, affine_transform.get_column(0));
    rotation.set_column(1, affine_transform.get_column(1));
    rotation.set_column(2, affine_transform.get_column(2));
    Vector3f translation = affine_transform.get_column(3);

    // Transform positions.
    Vector3f* positions_itr = mesh.get_positions();
    if (positions_itr != nullptr) {
        AABB bounding_box = AABB(Vector3f(std::numeric_limits<float>::infinity()), Vector3f(-std::numeric_limits<float>::infinity()));
        Vector3f* positions_end = positions_itr + mesh.get_vertex_count();
        for (; positions_itr != positions_end; ++positions_itr) {
            *positions_itr = rotation * *positions_itr + translation;
            bounding_box.grow_to_contain(*positions_itr);
        }
        mesh.set_bounds(bounding_box);
    }

    // Transform normals.
    Vector3f* normals_itr = mesh.get_normals();
    if (normals_itr != nullptr) {
        Matrix3x3f normal_rotation = transpose(invert(rotation));
        Vector3f* normals_end = normals_itr + mesh.get_vertex_count();
        for (; normals_itr != normals_end; ++normals_itr)
            *normals_itr = normal_rotation * *normals_itr;
    }
}

void transform_mesh(Mesh mesh, Transform transform) {
    transform_mesh(mesh, to_matrix3x4(transform));
}

Mesh combine(const std::string& name,
             const TransformedMesh* const meshes_begin,
             const TransformedMesh* const meshes_end,
             MeshFlags flags) {

    auto meshes = Core::Iterable<const TransformedMesh* const>(meshes_begin, meshes_end);

    unsigned int primitive_count = 0u;
    unsigned int vertex_count = 0u;
    for (TransformedMesh transformed_mesh : meshes) {
        Mesh mesh = transformed_mesh.mesh;
        primitive_count += mesh.get_primitive_count();
        vertex_count += mesh.get_vertex_count();
    }

    // Determine shared buffers.
    for (TransformedMesh transformed_mesh : meshes)
        flags &= transformed_mesh.mesh.get_flags();

    Mesh merged_mesh = Mesh(Meshes::create(name, primitive_count, vertex_count, flags));

    { // Always combine primitives.
        Vector3ui* primitives = merged_mesh.get_primitives();
        unsigned int primitive_offset = 0u;
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh;
            for (Vector3ui primitive : mesh.get_primitive_iterable())
                *(primitives++) = primitive + primitive_offset;
            primitive_offset += mesh.get_vertex_count();
        }
    }

    if (flags.contains(MeshFlag::Position)) {
        Vector3f* positions = merged_mesh.get_positions();
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh;
            for (Vector3f position : mesh.get_position_iterable())
                *(positions++) = transformed_mesh.transform * position;
        }
    }

    if (flags.contains(MeshFlag::Normal)) {
        Vector3f* normals = merged_mesh.get_normals();
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh;
            for (Vector3f normal : mesh.get_normal_iterable())
                *(normals++) = transformed_mesh.transform.rotation * normal;
        }
    }

    if (flags.contains(MeshFlag::Texcoord)) {
        Vector2f* texcoords = merged_mesh.get_texcoords();
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh;
            memcpy(texcoords, mesh.get_texcoords(), sizeof(Vector2f) * mesh.get_vertex_count());
            texcoords += mesh.get_vertex_count();
        }
    }

    if (flags.contains(MeshFlag::TintAndRoughness)) {
        TintRoughness* colors = merged_mesh.get_tint_and_roughness();
        for (TransformedMesh transformed_mesh : meshes) {
            Mesh mesh = transformed_mesh.mesh;
            memcpy(colors, mesh.get_tint_and_roughness(), sizeof(TintRoughness) * mesh.get_vertex_count());
            colors += mesh.get_vertex_count();
        }
    }

    merged_mesh.compute_bounds();

    return merged_mesh.get_ID();
}

void compute_hard_normals(Vector3f* positions_begin, Vector3f* positions_end, Vector3f* normals_begin) {
    while (positions_begin < positions_end) {
        Vector3f p0 = *positions_begin++;
        Vector3f p1 = *positions_begin++;
        Vector3f p2 = *positions_begin++;

        Vector3f normal = normalize(cross(p1 - p0, p2 - p0));
        *normals_begin++ = normal;
        *normals_begin++ = normal;
        *normals_begin++ = normal;
    }
}

void compute_normals(Vector3ui* primitives_begin, Vector3ui* primitives_end,
                     Vector3f* normals_begin, Vector3f* normals_end, Vector3f* positions_begin) {
    std::fill(normals_begin, normals_end, Vector3f::zero());

    while (primitives_begin < primitives_end) {
        Vector3ui primitive = *primitives_begin++;

        Vector3f p0 = positions_begin[primitive.x];
        Vector3f p1 = positions_begin[primitive.y];
        Vector3f p2 = positions_begin[primitive.z];

        Vector3f normal = cross(p1 - p0, p2 - p0);
        normals_begin[primitive.x] += normal;
        normals_begin[primitive.y] += normal;
        normals_begin[primitive.z] += normal;
    }

    std::for_each(normals_begin, normals_end, [](Vector3f& n) { n = normalize(n); });
}

void compute_normals(Mesh mesh) {
    compute_normals(mesh.get_primitives(), mesh.get_primitives() + mesh.get_primitive_count(),
                    mesh.get_normals(), mesh.get_normals() + mesh.get_vertex_count(),
                    mesh.get_positions());
}

Mesh merge_duplicate_vertices(Mesh mesh, MeshFlags attribute_types) {
    // Limit the duplicate checks to avaliable buffers.
    attribute_types &= mesh.get_flags() & MeshFlag::AllBuffers;
    if (attribute_types.none_set())
        return Mesh();

    unsigned int vertex_count = mesh.get_vertex_count();
    Vector3f* positions = attribute_types.is_set(MeshFlag::Position) ? mesh.get_positions() : nullptr;
    Vector3f* normals = attribute_types.is_set(MeshFlag::Normal) ? mesh.get_normals() : nullptr;
    Vector2f* uvs = attribute_types.is_set(MeshFlag::Texcoord) ? mesh.get_texcoords() : nullptr;
    TintRoughness* colors = attribute_types.is_set(MeshFlag::TintAndRoughness) ? mesh.get_tint_and_roughness() : nullptr;

    // Detect duplicate vertices and fill array of new vertex indices.
    unsigned int duplicate_count = 0;
    Core::Array<unsigned int> new_vertex_indices(vertex_count);
    new_vertex_indices[0] = 0;
    int next_vertex_index = 1;
    for (unsigned int v = 1; v < vertex_count; ++v) {
        bool duplicate_found = false;

        // Search backwards to see if there are duplicates.
        for (unsigned int j = v - 1; j != UINT_MAX && !duplicate_found; --j) {
            bool is_equal = true;
            if (positions != nullptr)
                is_equal &= positions[v] == positions[j];
            if (normals != nullptr)
                is_equal &= normals[v] == normals[j];
            if (uvs != nullptr)
                is_equal &= uvs[v] == uvs[j];
            if (colors != nullptr)
                is_equal &= colors[v] == colors[j];

            if (is_equal) {
                duplicate_found = true;
                ++duplicate_count;

                // v is a duplicate of j and should point to the same new vertex index as j.
                new_vertex_indices[v] = new_vertex_indices[j];
            }
        }

        if (!duplicate_found)
            new_vertex_indices[v] = next_vertex_index++;
    }

    Mesh new_mesh = Meshes::create(mesh.get_name() + "_without_duplicates", mesh.get_primitive_count(), vertex_count - duplicate_count, mesh.get_flags());

    // Copy vertices
    positions = mesh.get_positions();
    normals = mesh.get_normals();
    uvs = mesh.get_texcoords();
    Vector3f* new_positions = new_mesh.get_positions();
    Vector3f* new_normals = new_mesh.get_normals();
    Vector2f* new_uvs = new_mesh.get_texcoords();
    TintRoughness* new_colors = new_mesh.get_tint_and_roughness();
    for (unsigned int v = 0; v < vertex_count; ++v) {
        unsigned int new_v = new_vertex_indices[v];
        if (positions != nullptr)
            new_positions[new_v] = positions[v];
        if (normals != nullptr)
            new_normals[new_v] = normals[v];
        if (uvs != nullptr)
            new_uvs[new_v] = uvs[v];
        if (colors != nullptr)
            new_colors[new_v] = colors[v];
    }

    // Copy primitives one index at a time to support primitives of arbitrary arity.
    unsigned int index_count = mesh.get_index_count();
    unsigned int* old_indices = mesh.get_indices();
    unsigned int* new_indices = new_mesh.get_indices();
    for (unsigned int i = 0; i < index_count; ++i) {
        unsigned int old_vertex_index = old_indices[i];
        unsigned int new_vertex_index = new_vertex_indices[old_vertex_index];
        new_indices[i] = new_vertex_index;
    }

    return new_mesh;
}

} // NS MeshUtils

namespace MeshTests {

unsigned int normals_correspond_to_winding_order(Mesh mesh) {
    unsigned int failed_primitives = 0;

    for (Vector3ui primitive : mesh.get_primitive_iterable()) {
        Vector3f v0 = mesh.get_positions()[primitive.x];
        Vector3f v1 = mesh.get_positions()[primitive.y];
        Vector3f v2 = mesh.get_positions()[primitive.z];
        Vector3f primitive_normal = cross(v1 - v0, v2 - v0); // Not normalized, as we only care about the sign of the dot product below.

        bool primitive_failed = false;
        Vector3f n0 = mesh.get_normals()[primitive.x];
        if (dot(primitive_normal, n0) <= 0.0f) {
            // error_callback(mesh_ID, primitive_index, primitive.x);
            primitive_failed = true;
        }

        Vector3f n1 = mesh.get_normals()[primitive.y];
        if (dot(primitive_normal, n1) <= 0.0f) {
            // error_callback(mesh_ID, primitive_index, primitive.x);
            primitive_failed = true;
        }

        Vector3f n2 = mesh.get_normals()[primitive.z];
        if (dot(primitive_normal, n2) <= 0.0f) {
            // error_callback(mesh_ID, primitive_index, primitive.x);
            primitive_failed = true;
        }

        if (primitive_failed)
            ++failed_primitives;
    }

    return failed_primitives;
}

unsigned int count_degenerate_primitives(Mesh mesh, float epsilon_squared) {
    unsigned int degenerate_primitives = 0;

    for (Vector3ui primitive : mesh.get_primitive_iterable()) {
        bool degenerate_indices = primitive.x == primitive.y ||
                                  primitive.x == primitive.z ||
                                  primitive.y == primitive.z;
        Vector3f p0 = mesh.get_positions()[primitive.x];
        Vector3f p1 = mesh.get_positions()[primitive.y];
        Vector3f p2 = mesh.get_positions()[primitive.z];
        bool degenerate_positions = magnitude_squared(p0 - p1) < epsilon_squared ||
                                    magnitude_squared(p0 - p2) < epsilon_squared ||
                                    magnitude_squared(p1 - p2) < epsilon_squared;

        if (degenerate_indices || degenerate_positions)
            ++degenerate_primitives;
    }

    return degenerate_primitives;
}

} // NS MeshUtils

} // NS Assets
} // NS Bifrost
