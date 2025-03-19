// Bifrost scene node.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Scene/SceneNode.h>

#include <assert.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Scene {

SceneNodeIDGenerator SceneNodes::m_UID_generator = SceneNodeIDGenerator(0u);
std::string* SceneNodes::m_names = nullptr;

SceneNodeID* SceneNodes::m_parent_IDs = nullptr;
SceneNodeID* SceneNodes::m_sibling_IDs = nullptr;
SceneNodeID* SceneNodes::m_first_child_IDs = nullptr;

Transform* SceneNodes::m_global_transforms = nullptr;

Core::ChangeSet<SceneNodes::Changes, SceneNodeID> SceneNodes::m_changes;

void SceneNodes::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = SceneNodeIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];

    m_parent_IDs = new SceneNodeID[capacity];
    m_sibling_IDs = new SceneNodeID[capacity];
    m_first_child_IDs = new SceneNodeID[capacity];

    m_global_transforms = new Transform[capacity];

    m_changes = Core::ChangeSet<Changes, SceneNodeID>(capacity);

    // Allocate dummy element at 0.
    m_names[0] = "Dummy Node";
    m_parent_IDs[0] = m_first_child_IDs[0] = m_sibling_IDs[0] = SceneNodeID::invalid_UID();
    m_global_transforms[0] = Transform::identity();
}

void SceneNodes::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = SceneNodeIDGenerator(0u);
    delete[] m_names; m_names = nullptr;

    delete[] m_parent_IDs; m_parent_IDs = nullptr;
    delete[] m_sibling_IDs; m_sibling_IDs = nullptr;
    delete[] m_first_child_IDs; m_first_child_IDs = nullptr;

    delete[] m_global_transforms; m_global_transforms = nullptr;

    m_changes.resize(0);
}

void SceneNodes::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_node_data(m_UID_generator.capacity(), old_capacity);
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void SceneNodes::reserve_node_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_first_child_IDs != nullptr);
    assert(m_global_transforms != nullptr);
    assert(m_names != nullptr);
    assert(m_parent_IDs != nullptr);
    assert(m_sibling_IDs != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;

    m_names = resize_and_copy_array(m_names, new_capacity, copyable_elements);

    m_parent_IDs = resize_and_copy_array(m_parent_IDs, new_capacity, copyable_elements);
    m_sibling_IDs = resize_and_copy_array(m_sibling_IDs, new_capacity, copyable_elements);
    m_first_child_IDs = resize_and_copy_array(m_first_child_IDs, new_capacity, copyable_elements);

    m_global_transforms = resize_and_copy_array(m_global_transforms, new_capacity, copyable_elements);

    m_changes.resize(new_capacity);
}

SceneNodeID SceneNodes::create(const std::string& name, Transform transform) {
    assert(m_first_child_IDs != nullptr);
    assert(m_global_transforms != nullptr);
    assert(m_names != nullptr);
    assert(m_parent_IDs != nullptr);
    assert(m_sibling_IDs != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    SceneNodeID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_node_data(m_UID_generator.capacity(), old_capacity);

    m_names[id] = name;
    m_parent_IDs[id] = m_first_child_IDs[id] = m_sibling_IDs[id] = SceneNodeID::invalid_UID();
    m_global_transforms[id] = transform;
    m_changes.set_change(id, Change::Created);

    return id;
}

void SceneNodes::destroy(SceneNodeID node_ID) {
    // We don't actually destroy anything when destroying a node. The properties will get overwritten later when a node is created in same the spot.
    if (has(node_ID))
        m_changes.add_change(node_ID, Change::Destroyed);
}

void SceneNodes::set_parent(SceneNodeID node_ID, const SceneNodeID parent_ID) {
    assert(m_parent_IDs != nullptr);
    assert(m_sibling_IDs != nullptr);
    assert(m_first_child_IDs != nullptr);

    SceneNodeID old_parent_ID = m_parent_IDs[node_ID];
    if (node_ID != parent_ID && node_ID != SceneNodeID::invalid_UID()) {
        // Detach current node from it's place in the hierarchy
        if (old_parent_ID != SceneNodeID::invalid_UID()) {
            // Find old previous sibling.
            SceneNodeID sibling = m_first_child_IDs[old_parent_ID];
            if (sibling == node_ID) {
                m_first_child_IDs[old_parent_ID] = m_sibling_IDs[sibling];
            } else {
                while (node_ID != m_sibling_IDs[sibling])
                    sibling = m_sibling_IDs[sibling];
                m_sibling_IDs[sibling] = m_sibling_IDs[node_ID];
            }
        }

        // Attach it to the new parent as the first child and link it to the other siblings.
        m_parent_IDs[node_ID] = parent_ID;
        m_sibling_IDs[node_ID] = m_first_child_IDs[parent_ID];
        m_first_child_IDs[parent_ID] = node_ID;
    }
}

std::vector<SceneNodeID> SceneNodes::get_sibling_IDs(SceneNodeID node_ID) {
    assert(m_parent_IDs != nullptr);

    SceneNodeID parent_ID = m_parent_IDs[node_ID];

    return get_children_IDs(parent_ID);
}

std::vector<SceneNodeID> SceneNodes::get_children_IDs(SceneNodeID node_ID) {
    assert(m_first_child_IDs != nullptr);
    assert(m_sibling_IDs != nullptr);

    std::vector<SceneNodeID> res(0);
    SceneNodeID child = m_first_child_IDs[node_ID];
    while (child != SceneNodeID::invalid_UID()) {
        res.push_back(child);
        child = m_sibling_IDs[child];
    }
    return res;
}

bool SceneNodes::has_child(SceneNodeID node_ID, SceneNodeID tested_Child_ID) {
    assert(m_first_child_IDs != nullptr);
    assert(m_sibling_IDs != nullptr);

    SceneNodeID child = m_first_child_IDs[node_ID];
    while (child != SceneNodeID::invalid_UID()) {
        if (child == tested_Child_ID)
            return true;
        child = m_sibling_IDs[child];
    }
    return false;
}

std::vector<SceneNode> SceneNode::get_children() const {
    std::vector<SceneNodeID> children_IDs = SceneNodes::get_children_IDs(m_ID);
    std::vector<SceneNode> children;
    children.reserve(children_IDs.size());
    for (SceneNodeID id : children_IDs)
        children.push_back(SceneNode(id));
    return children;
}

Transform SceneNodes::get_local_transform(SceneNodeID node_ID) {
    SceneNodeID parent_ID = m_parent_IDs[node_ID];
    const Transform parent_transform = m_global_transforms[parent_ID];
    const Transform transform = m_global_transforms[node_ID];
    return Transform::delta(parent_transform, transform);
}

void SceneNodes::set_local_transform(SceneNodeID node_ID, Transform transform) {
    assert(m_global_transforms != nullptr);
    assert(m_parent_IDs != nullptr);

    if (node_ID == SceneNodeID::invalid_UID()) return;

    // Update global transform.
    SceneNodeID parent_ID = m_parent_IDs[node_ID];
    Transform parent_transform = m_global_transforms[parent_ID];
    Transform new_global_transform = parent_transform * transform;
    unsafe_set_global_transform(node_ID, new_global_transform);
}

void SceneNodes::set_global_transform(SceneNodeID node_ID, Transform transform) {
    assert(m_global_transforms != nullptr);

    if (node_ID == SceneNodeID::invalid_UID()) return;

    unsafe_set_global_transform(node_ID, transform);
}

void SceneNodes::apply_delta_transform(SceneNodeID node_ID, Transform delta_transform) {
    assert(m_global_transforms != nullptr);

    if (node_ID == SceneNodeID::invalid_UID()) return;

    Transform new_global_transform = m_global_transforms[node_ID] * delta_transform;
    unsafe_set_global_transform(node_ID, new_global_transform);
}

void SceneNodes::unsafe_set_global_transform(SceneNodeID node_ID, Transform new_transform) {
    Transform old_transform = m_global_transforms[node_ID];
    Transform delta_transform = Transform::delta(new_transform, old_transform);
    m_global_transforms[node_ID] = new_transform;
    m_changes.add_change(node_ID, Change::Transform);

    // Update global transforms of all children.
    Transform inverse_old_transform = old_transform.inverse();
    apply_to_children_recursively(node_ID, [=](SceneNodeID child_ID) {
        Transform delta_transform = inverse_old_transform * m_global_transforms[child_ID]; // Inlined Transform::delta(); to ensure that old_transform isn't inverted on every application.
        m_global_transforms[child_ID] = new_transform * delta_transform;
        m_changes.add_change(child_ID, Change::Transform);
    });
}

void SceneNodes::reset_change_notifications() {
    for (SceneNodeID node_ID : get_changed_nodes())
        if (get_changes(node_ID).is_set(Change::Destroyed))
            m_UID_generator.erase(node_ID);
    m_changes.reset_change_notifications();
}

} // NS Scene
} // NS Bifrost
