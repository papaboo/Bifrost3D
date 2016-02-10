// Cogwheel mesh.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MESH_H_
#define _COGWHEEL_ASSETS_MESH_H_

#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/AABB.h>
#include <Cogwheel/Math/Vector.h>

namespace Cogwheel {
namespace Assets {

//----------------------------------------------------------------------------
// Container for the buffers that make up a mesh, such as positions and normals.
// Future work:
// * Pass bitmask of vertex attributes to be created to constructor.
// * Simplify to POD struct and get rid of the RAII destructor. 
//   Then let Meshes handle desctruction.
//----------------------------------------------------------------------------
struct Mesh final {
    unsigned int m_indices_count;
    unsigned int m_vertex_count;
    
    Math::Vector3ui* m_indices;
    Math::Vector3f* m_positions;
    Math::Vector3f* m_normals;
    Math::Vector2f* m_texcoords;

    Mesh()
        : m_indices_count(0u)
        , m_indices(nullptr)
        , m_vertex_count(0u)
        , m_positions(nullptr)
        , m_normals(nullptr)
        , m_texcoords(nullptr) {
    }

    Mesh(unsigned int indices_count, unsigned int vertex_count)
        : m_indices_count(indices_count)
        , m_vertex_count(vertex_count)
        , m_indices(new Math::Vector3ui[m_indices_count])
        , m_positions(new Math::Vector3f[m_vertex_count])
        , m_normals(new Math::Vector3f[m_vertex_count])
        , m_texcoords(new Math::Vector2f[m_vertex_count]) {
    }

    Mesh(Mesh&& other) {
        m_indices_count = other.m_indices_count; other.m_indices_count = 0u;
        m_vertex_count = other.m_vertex_count; other.m_vertex_count = 0u;
        m_indices = other.m_indices; other.m_indices = nullptr;
        m_positions = other.m_positions; other.m_positions = nullptr;
        m_normals = other.m_normals; other.m_normals = nullptr;
        m_texcoords = other.m_texcoords; other.m_texcoords = nullptr;
    }

    Mesh& operator=(Mesh&& other) {
        m_indices_count = other.m_indices_count; other.m_indices_count = 0u;
        m_vertex_count = other.m_vertex_count; other.m_vertex_count = 0u;
        m_indices = other.m_indices; other.m_indices = nullptr;
        m_positions = other.m_positions; other.m_positions = nullptr;
        m_normals = other.m_normals; other.m_normals = nullptr;
        m_texcoords = other.m_texcoords; other.m_texcoords = nullptr;
        return *this;
    }

    ~Mesh() {
        delete[] m_indices;
        delete[] m_positions;
        delete[] m_normals;
        delete[] m_texcoords;
    }

private:
    // Delete copy constructors to avoid issues with shared ownership.
    Mesh(Mesh& other) = delete;
    Mesh& operator=(const Mesh& other) = delete;
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

    static Meshes::UID create(const std::string& name, unsigned int indices_count, unsigned int vertex_count);

    static inline std::string get_name(Meshes::UID mesh_ID) { return m_names[mesh_ID]; }
    static inline void set_name(Meshes::UID mesh_ID, const std::string& name) { m_names[mesh_ID] = name; }

    static inline Mesh& get_mesh(Meshes::UID mesh_ID) { return m_meshes[mesh_ID]; }
    static inline Math::AABB get_bounds(Meshes::UID mesh_ID) { return m_bounds[mesh_ID]; }
    static inline void set_bounds(Meshes::UID mesh_ID, Math::AABB bounds) { m_bounds[mesh_ID] = bounds; }
    static Math::AABB compute_bounds(Meshes::UID mesh_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }

private:
    static void reserve_node_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;
    static std::string* m_names;

    static Mesh* m_meshes;
    static Math::AABB* m_bounds;
};

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MESH_H_