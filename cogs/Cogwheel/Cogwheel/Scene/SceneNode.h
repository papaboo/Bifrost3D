// Cogwheel scene node.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_SCENE_NODE_H_
#define _COGWHEEL_SCENE_SCENE_NODE_H_

#include <Cogwheel/Core/UniqueIDGenerator.h>
#include <Cogwheel/Math/Transform.h>

#include <vector>

namespace Cogwheel {
namespace Scene {

// ---------------------------------------------------------------------------
// Container class for the cogwheel scene node.
// Future work
// * Allocate all (or most) internal arrays in one big chunk.
// * Change the sibling/children layout, so sibling IDs or perhaps siblings are always allocated next too each other?
//  * Requires an extra indirection though, since node ID's won't match the node positions anymore.
//  * Then I could use Core::Array to quickly construct a list of children.
//  * Could be done (incrementally?) when all mutating modules have been run.
// ---------------------------------------------------------------------------
class SceneNodes final {
public:
    typedef Core::TypedUIDGenerator<SceneNodes> UIDGenerator;
    typedef UIDGenerator::UID UID;

    static bool is_allocated() { return m_global_transforms != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(SceneNodes::UID node_ID) { return m_UID_generator.has(node_ID); }

    static SceneNodes::UID create(const std::string& name);

    static inline std::string get_name(SceneNodes::UID node_ID) { return m_names[node_ID]; }
    static inline void set_name(SceneNodes::UID node_ID, const std::string& name) { m_names[node_ID] = name; }

    static inline SceneNodes::UID get_parent_ID(SceneNodes::UID node_ID) { return m_parent_IDs[node_ID]; }
    static void set_parent(SceneNodes::UID node_ID, const SceneNodes::UID parent_ID);
    static bool has_child(SceneNodes::UID node_ID, SceneNodes::UID tested_child_ID);
    static std::vector<SceneNodes::UID> get_sibling_IDs(SceneNodes::UID node_ID);
    static std::vector<SceneNodes::UID> get_children_IDs(SceneNodes::UID node_ID);

    static Math::Transform get_local_transform(SceneNodes::UID node_ID);
    static void set_local_transform(SceneNodes::UID node_ID, Math::Transform transform);
    static Math::Transform get_global_transform(SceneNodes::UID node_ID) { return m_global_transforms[node_ID];}
    static void set_global_transform(SceneNodes::UID node_ID, Math::Transform transform);

    template<typename F>
    static void traverser_recursively(SceneNodes::UID node_ID, F& function);
    template<typename F>
    static void traverser_children_recursively(SceneNodes::UID node_ID, F& function);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    // static Range<SceneNodes::UID*> get_changed_transforms();

private:
    static void reserve_node_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;
    static std::string* m_names;

    static SceneNodes::UID* m_parent_IDs;
    static SceneNodes::UID* m_sibling_IDs;
    static SceneNodes::UID* m_first_child_IDs;

    static Math::Transform* m_global_transforms;
};

// ---------------------------------------------------------------------------
// The cogwheel scene node.
// The scene node implements the Entity Component System, 
// https://en.wikipedia.org/wiki/Entity_component_system, with entities being 
// managed by their own managers, e.g. the Foo component is managed by the FooManager.
// ---------------------------------------------------------------------------
class SceneNode final {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors
    // -----------------------------------------------------------------------
    SceneNode() : m_ID(SceneNodes::UID::invalid_UID()) { }
    SceneNode(SceneNodes::UID id) : m_ID(id) { }
    ~SceneNode() { }

    inline const SceneNodes::UID get_ID() const { return m_ID; }

    inline std::string get_name() const { return SceneNodes::get_name(m_ID); }
    inline void set_name(std::string name) { SceneNodes::set_name(m_ID, name); }

    inline SceneNode get_parent() const { return SceneNode(SceneNodes::get_parent_ID(m_ID)); }
    inline void set_parent(SceneNode parent) { SceneNodes::set_parent(m_ID, parent.get_ID()); }
    inline bool has_child(SceneNode tested_child) { return SceneNodes::has_child(m_ID, tested_child.get_ID()); }
    std::vector<SceneNode> get_children() const;

    inline bool exists() const { return SceneNodes::has(m_ID); }

    inline Math::Transform get_local_transform() const { return SceneNodes::get_local_transform(m_ID); }
    inline void set_local_transform(Math::Transform transform) { SceneNodes::set_local_transform(m_ID, transform); }
    inline Math::Transform get_global_transform() const { return SceneNodes::get_global_transform(m_ID); }
    inline void set_global_transform(Math::Transform transform) { SceneNodes::set_global_transform(m_ID, transform); }

    template<typename F>
    inline void traverser_recursively(F& function) { SceneNodes::traverser_recursively<F>(m_ID, function); }
    template<typename F>
    inline void traverser_children_recursively(F& function) { SceneNodes::traverser_children_recursively<F>(m_ID, function); }

    inline bool operator==(SceneNode rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(SceneNode rhs) const { return m_ID != rhs.m_ID; }

private:
    const SceneNodes::UID m_ID;
};

// ---------------------------------------------------------------------------
// Temdplate implementations.
// ---------------------------------------------------------------------------

template<typename F>
void SceneNodes::traverser_children_recursively(SceneNodes::UID node_ID, F& function) {
    UID node = m_first_child_IDs[node_ID];
    if (node == UID::invalid_UID())
        return;

    do {
        function(node);

        if (m_first_child_IDs[node] != UID::invalid_UID())
            // Visit the next child.
            node = m_first_child_IDs[node];
        else if (m_sibling_IDs[node] != UID::invalid_UID())
            // Visit the next sibling.
            node = m_sibling_IDs[node];
        else
        {
            // search upwards for next node not visited.
            node = m_parent_IDs[node];
            while (node != node_ID) {
                UID parent_sibling = m_sibling_IDs[node];
                if (parent_sibling == UID::invalid_UID())
                    node = m_parent_IDs[node];
                else {
                    node = parent_sibling;
                    break;
                }
            }
        }
    } while (node != node_ID);
}

template<typename F>
void SceneNodes::traverser_recursively(SceneNodes::UID node_ID, F& function) {
    function(node_ID);
    traverser_children_recursively(node_ID, function);
}

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_SCENE_NODE_H_