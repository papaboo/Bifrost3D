// Bifrost scene node.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_SCENE_SCENE_NODE_H_
#define _BIFROST_SCENE_SCENE_NODE_H_

#include <Bifrost/Core/Bitmask.h>
#include <Bifrost/Core/ChangeSet.h>
#include <Bifrost/Core/Iterable.h>
#include <Bifrost/Core/UniqueIDGenerator.h>
#include <Bifrost/Math/Transform.h>

namespace Bifrost {
namespace Scene {

// ---------------------------------------------------------------------------
// Container class for the bifrost scene node.
// Future work
// * A parent changed event: (node_id, old_parent_id). Is this actually needed by anything when transforms are global?
// * Change the sibling/children layout, so sibling IDs or perhaps siblings are always allocated next too each other?
//   * Requires an extra indirection though, since node ID's won't match the node positions anymore.
//   * Could be done (incrementally?) when all mutations in a tick are done.
// * The change notification count is going to explode when setting up or tearing down a scene. 
//   We should implement a better solution for these cases.
//   Perhaps a great big 'a lot has changed, rebuild everything and ignore the notifications' flag?
// ---------------------------------------------------------------------------
class SceneNodes final {
public:
    typedef Core::TypedUIDGenerator<SceneNodes> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static bool is_allocated() { return m_global_transforms != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(SceneNodes::UID node_ID) { return m_UID_generator.has(node_ID); }

    static SceneNodes::UID create(const std::string& name, Math::Transform transform = Math::Transform::identity());
    static void destroy(SceneNodes::UID node_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

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
    static void apply_delta_transform(SceneNodes::UID node_ID, Math::Transform delta_transform);

    template<typename F>
    static void apply_recursively(SceneNodes::UID node_ID, F& function);
    template<typename F>
    static void apply_to_children_recursively(SceneNodes::UID node_ID, F& function);

    //-------------------------------------------------------------------------
    // Changes since last game loop tick.
    //-------------------------------------------------------------------------
    enum class Change : unsigned char {
        None      = 0u,
        Created   = 1u << 0u,
        Destroyed = 1u << 1u,
        Transform = 1u << 2u,
        All = Created | Destroyed | Transform
    };
    typedef Core::Bitmask<Change> Changes;

    static inline Changes get_changes(SceneNodes::UID node_ID) { return m_changes.get_changes(node_ID); }

    typedef std::vector<UID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_nodes() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications() { return m_changes.reset_change_notifications(); }

private:
    static void reserve_node_data(unsigned int new_capacity, unsigned int old_capacity);
    static void SceneNodes::unsafe_set_global_transform(SceneNodes::UID node_ID, Math::Transform transform);


    static UIDGenerator m_UID_generator;
    static std::string* m_names;

    static SceneNodes::UID* m_parent_IDs;
    static SceneNodes::UID* m_sibling_IDs;
    static SceneNodes::UID* m_first_child_IDs;

    static Math::Transform* m_global_transforms;

    static Core::ChangeSet<Changes, UID> m_changes;
};

// ---------------------------------------------------------------------------
// The bifrost scene node.
// The scene node implements the Entity Component System, 
// https://en.wikipedia.org/wiki/Entity_component_system, with entities being 
// managed by their own managers, e.g. the Foo component is managed by the FooManager.
// ---------------------------------------------------------------------------
class SceneNode final {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors.
    // -----------------------------------------------------------------------
    SceneNode() : m_ID(SceneNodes::UID::invalid_UID()) { }
    SceneNode(SceneNodes::UID id) : m_ID(id) { }

    inline const SceneNodes::UID get_ID() const { return m_ID; }
    inline bool exists() const { return SceneNodes::has(m_ID); }

    inline bool operator==(SceneNode rhs) const { return m_ID == rhs.m_ID; }
    inline bool operator!=(SceneNode rhs) const { return m_ID != rhs.m_ID; }

    // -----------------------------------------------------------------------
    // Getters and setters.
    // -----------------------------------------------------------------------
    inline std::string get_name() const { return SceneNodes::get_name(m_ID); }
    inline void set_name(std::string name) { SceneNodes::set_name(m_ID, name); }

    inline SceneNode get_parent() const { return SceneNode(SceneNodes::get_parent_ID(m_ID)); }
    inline void set_parent(SceneNode parent) { SceneNodes::set_parent(m_ID, parent.get_ID()); }
    inline bool has_child(SceneNode tested_child) { return SceneNodes::has_child(m_ID, tested_child.get_ID()); }
    std::vector<SceneNode> get_children() const;

    inline Math::Transform get_local_transform() const { return SceneNodes::get_local_transform(m_ID); }
    inline void set_local_transform(Math::Transform transform) { SceneNodes::set_local_transform(m_ID, transform); }
    inline Math::Transform get_global_transform() const { return SceneNodes::get_global_transform(m_ID); }
    inline void set_global_transform(Math::Transform transform) { SceneNodes::set_global_transform(m_ID, transform); }
    inline void apply_delta_transform(Math::Transform transform) { SceneNodes::apply_delta_transform(m_ID, transform); }

    inline SceneNodes::Changes get_changes() const { return SceneNodes::get_changes(m_ID); }

    // -----------------------------------------------------------------------
    // Applies a function recursively.
    // -----------------------------------------------------------------------
    template<typename F>
    inline void apply_recursively(F& function) { SceneNodes::apply_recursively<F>(m_ID, function); }
    template<typename F>
    inline void apply_to_children_recursively(F& function) { SceneNodes::apply_to_children_recursively<F>(m_ID, function); }

private:
    SceneNodes::UID m_ID;
};

// ---------------------------------------------------------------------------
// Template implementations.
// ---------------------------------------------------------------------------

template<typename F>
void SceneNodes::apply_to_children_recursively(SceneNodes::UID node_ID, F& function) {
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
void SceneNodes::apply_recursively(SceneNodes::UID node_ID, F& function) {
    function(node_ID);
    apply_to_children_recursively(node_ID, function);
}

} // NS Scene
} // NS Bifrost

#endif // _BIFROST_SCENE_SCENE_NODE_H_
