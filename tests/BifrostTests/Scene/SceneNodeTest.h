// Test Bifrost Scene Nodes.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_SCENE_SCENE_NODE_TEST_H_
#define _BIFROST_SCENE_SCENE_NODE_TEST_H_

#include <Bifrost/Scene/SceneNode.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Scene {

GTEST_TEST(Scene_SceneNode, resizing) {
    SceneNodes::allocate(8u);
    EXPECT_GE(SceneNodes::capacity(), 8u);

    // Test that capacity can be increased.
    unsigned int larger_capacity = SceneNodes::capacity() + 4u;
    SceneNodes::reserve(larger_capacity);
    EXPECT_GE(SceneNodes::capacity(), larger_capacity);

    // Test that capacity won't be decreased.
    SceneNodes::reserve(5u);
    EXPECT_GE(SceneNodes::capacity(), larger_capacity);

    SceneNodes::deallocate();
    EXPECT_LT(SceneNodes::capacity(), larger_capacity);
}

GTEST_TEST(Scene_SceneNode, sentinel_node) {
    SceneNodes::allocate(1u);

    SceneNode sentinel = SceneNodeID::invalid_UID();
    SceneNode node = SceneNodes::create("Foo");

    // Test that sentinel node cannot have it's parent set.
    sentinel.set_parent(node);
    SceneNode parent_set = sentinel.get_parent();
    EXPECT_NE(parent_set, node);
    EXPECT_EQ(parent_set, sentinel);

    EXPECT_EQ(sentinel.get_children().size(), 0u);

    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneNode, create) {
    SceneNodes::allocate(2u);
    SceneNodeID node_ID = SceneNodes::create("Foo");
    EXPECT_TRUE(SceneNodes::has(node_ID));

    EXPECT_EQ("Foo", SceneNodes::get_name(node_ID));

    // Test scene node created notification.
    Core::Iterable<SceneNodes::ChangedIterator> changed_nodes = SceneNodes::get_changed_nodes();
    EXPECT_EQ(1, changed_nodes.end() - changed_nodes.begin());
    EXPECT_EQ(node_ID, *changed_nodes.begin());
    EXPECT_EQ(SceneNodes::Change::Created, SceneNodes::get_changes(node_ID));

    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneNode, destroy) {
    SceneNodes::allocate(2u);
    SceneNodeID node_ID = SceneNodes::create("Foo");
    EXPECT_TRUE(SceneNodes::has(node_ID));

    SceneNodes::reset_change_notifications();

    SceneNodes::destroy(node_ID);
    EXPECT_FALSE(SceneNodes::has(node_ID));

    // Test scene node destroyed notification.
    Core::Iterable<SceneNodes::ChangedIterator> changed_nodes = SceneNodes::get_changed_nodes();
    EXPECT_EQ(1, changed_nodes.end() - changed_nodes.begin());
    EXPECT_EQ(node_ID, *changed_nodes.begin());
    EXPECT_EQ(SceneNodes::Change::Destroyed, SceneNodes::get_changes(node_ID));

    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneNode, create_and_destroy_notifications) {
    SceneNodes::allocate(8u);

    SceneNodeID node_ID0 = SceneNodes::create("Foo");
    SceneNodeID node_ID1 = SceneNodes::create("Bar");
    EXPECT_TRUE(SceneNodes::has(node_ID0));
    EXPECT_TRUE(SceneNodes::has(node_ID1));

    { // Test scene node create notifications.
        Core::Iterable<SceneNodes::ChangedIterator> changed_nodes = SceneNodes::get_changed_nodes();
        EXPECT_EQ(2, changed_nodes.end() - changed_nodes.begin());

        bool node0_created = false;
        bool node1_created = false;
        bool other_changes = false;
        for (const SceneNodeID node_ID : changed_nodes) {
            bool node_created = SceneNodes::get_changes(node_ID) == SceneNodes::Change::Created;
            if (node_ID == node_ID0 && node_created)
                node0_created = true;
            else if (node_ID == node_ID1 && node_created)
                node1_created = true;
            else
                other_changes = true;
        }

        EXPECT_TRUE(node0_created);
        EXPECT_TRUE(node1_created);
        EXPECT_FALSE(other_changes);
    }

    SceneNodes::reset_change_notifications();

    { // Test destroy.
        SceneNodes::destroy(node_ID0);
        EXPECT_FALSE(SceneNodes::has(node_ID0));

        Core::Iterable<SceneNodes::ChangedIterator> changed_nodes = SceneNodes::get_changed_nodes();
        EXPECT_EQ(1, changed_nodes.end() - changed_nodes.begin());

        SceneNode node = *changed_nodes.begin();
        bool node0_destroyed = node.get_ID() == node_ID0 && node.get_changes() == SceneNodes::Change::Destroyed;
        EXPECT_TRUE(node0_destroyed);
    }

    SceneNodes::reset_change_notifications();

    { // Test that destroyed node cannot be destroyed again.
        EXPECT_FALSE(SceneNodes::has(node_ID0));

        SceneNodes::destroy(node_ID0);
        EXPECT_FALSE(SceneNodes::has(node_ID0));
        EXPECT_TRUE(SceneNodes::get_changed_nodes().is_empty());
    }

    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneNode, parenting) {
    SceneNodes::allocate(4u);
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");
    SceneNode n2 = SceneNodes::create("n2");

    EXPECT_FALSE(n1.get_parent().exists());
    
    // Set n0 to the parent and check the parent-child relationship.
    n1.set_parent(n0);
    EXPECT_TRUE(n1.get_parent().exists());
    EXPECT_EQ(n0, n1.get_parent());
    EXPECT_EQ(1u, n0.get_children().size());
    EXPECT_TRUE(n0.has_child(n1));

    // Set n2 to the parent and check the parent-child relationship.
    n1.set_parent(n2);
    EXPECT_TRUE(n1.get_parent().exists());
    EXPECT_EQ(n2, n1.get_parent());
    EXPECT_EQ(1u, n2.get_children().size());
    EXPECT_TRUE(n2.has_child(n1));
    // ... also check that n0 no longer has any children.
    EXPECT_EQ(0u, n0.get_children().size());
    
    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneNode, creating_hierarchy) {
    // Tests the following hierachy
    //      id3
    //    /  |  \
    // id0  id4  id6
    //      / \    \
    //    id2 id5  id1
    SceneNodes::allocate(1u);
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");
    SceneNode n2 = SceneNodes::create("n2");
    SceneNode n3 = SceneNodes::create("n3");
    SceneNode n4 = SceneNodes::create("n4");
    SceneNode n5 = SceneNodes::create("n5");
    SceneNode n6 = SceneNodes::create("n6");
    
    EXPECT_TRUE(n0.exists());
    EXPECT_TRUE(n1.exists());
    EXPECT_TRUE(n2.exists());
    EXPECT_TRUE(n3.exists());
    EXPECT_TRUE(n4.exists());
    EXPECT_TRUE(n5.exists());
    EXPECT_TRUE(n6.exists());

    n0.set_parent(n3);
    n4.set_parent(n3);
    n6.set_parent(n3);
    n2.set_parent(n4);
    n5.set_parent(n4);
    n1.set_parent(n6);

    EXPECT_EQ(n3, n0.get_parent());
    EXPECT_TRUE(n3.has_child(n0));
    EXPECT_EQ(n3, n4.get_parent());
    EXPECT_TRUE(n3.has_child(n4));
    EXPECT_EQ(n3, n6.get_parent());
    EXPECT_TRUE(n3.has_child(n6));
    EXPECT_EQ(n4, n2.get_parent());
    EXPECT_TRUE(n4.has_child(n2));
    EXPECT_EQ(n4, n5.get_parent());
    EXPECT_TRUE(n4.has_child(n5));
    EXPECT_EQ(n6, n1.get_parent());
    EXPECT_TRUE(n6.has_child(n1));

    // Now parent id4 below id0, just for fun and profit.
    //     n3
    //    /  \
    //   n0  n6
    //   |    |
    //   n4  n1
    //  / \
    // n2 n5
    n4.set_parent(n0);

    EXPECT_FALSE(n3.get_parent().exists());
    EXPECT_EQ(n3, n0.get_parent());
    EXPECT_TRUE(n3.has_child(n0));
    EXPECT_EQ(n3, n6.get_parent());
    EXPECT_TRUE(n3.has_child(n6));
    EXPECT_EQ(n0, n4.get_parent());
    EXPECT_TRUE(n0.has_child(n4));
    EXPECT_EQ(n6, n1.get_parent());
    EXPECT_TRUE(n6.has_child(n1));
    EXPECT_EQ(n4, n2.get_parent());
    EXPECT_TRUE(n4.has_child(n2));
    EXPECT_EQ(n4, n5.get_parent());
    EXPECT_TRUE(n4.has_child(n5));

    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneNode, graph_traversal) {
    // Tests the following hierachy
    //      id3
    //    /  |  \
    // id0  id4  id6
    //      / \    \
    //    id2 id5  id1
    SceneNodes::allocate(1u);
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");
    SceneNode n2 = SceneNodes::create("n2");
    SceneNode n3 = SceneNodes::create("n3");
    SceneNode n4 = SceneNodes::create("n4");
    SceneNode n5 = SceneNodes::create("n5");
    SceneNode n6 = SceneNodes::create("n6");

    n0.set_parent(n3);
    n4.set_parent(n3);
    n6.set_parent(n3);
    n2.set_parent(n4);
    n5.set_parent(n4);
    n1.set_parent(n6);

    unsigned int* visits = new unsigned int[SceneNodes::capacity()];
    for (unsigned int i = 0; i < SceneNodes::capacity(); ++i)
        visits[i] = 0u;
    n2.apply_to_children_recursively([&](SceneNodeID id) {
        ++visits[id];
    });
    EXPECT_EQ(0u, visits[n0.get_ID()]);
    EXPECT_EQ(0u, visits[n1.get_ID()]);
    EXPECT_EQ(0u, visits[n2.get_ID()]);
    EXPECT_EQ(0u, visits[n3.get_ID()]);
    EXPECT_EQ(0u, visits[n4.get_ID()]);
    EXPECT_EQ(0u, visits[n5.get_ID()]);
    EXPECT_EQ(0u, visits[n6.get_ID()]);

    n4.apply_recursively([&](SceneNodeID id) {
        ++visits[id];
    });
    EXPECT_EQ(0u, visits[n0.get_ID()]);
    EXPECT_EQ(0u, visits[n1.get_ID()]);
    EXPECT_EQ(1u, visits[n2.get_ID()]);
    EXPECT_EQ(0u, visits[n3.get_ID()]);
    EXPECT_EQ(1u, visits[n4.get_ID()]);
    EXPECT_EQ(1u, visits[n5.get_ID()]);
    EXPECT_EQ(0u, visits[n6.get_ID()]);

    n6.apply_to_children_recursively([&](SceneNodeID id) {
        ++visits[id];
    });
    EXPECT_EQ(0u, visits[n0.get_ID()]);
    EXPECT_EQ(1u, visits[n1.get_ID()]);
    EXPECT_EQ(1u, visits[n2.get_ID()]);
    EXPECT_EQ(0u, visits[n3.get_ID()]);
    EXPECT_EQ(1u, visits[n4.get_ID()]);
    EXPECT_EQ(1u, visits[n5.get_ID()]);
    EXPECT_EQ(0u, visits[n6.get_ID()]);

    n3.apply_recursively([&](SceneNodeID id) {
        ++visits[id];
    });
    EXPECT_EQ(1u, visits[n0.get_ID()]);
    EXPECT_EQ(2u, visits[n1.get_ID()]);
    EXPECT_EQ(2u, visits[n2.get_ID()]);
    EXPECT_EQ(1u, visits[n3.get_ID()]);
    EXPECT_EQ(2u, visits[n4.get_ID()]);
    EXPECT_EQ(2u, visits[n5.get_ID()]);
    EXPECT_EQ(1u, visits[n6.get_ID()]);

    SceneNodes::deallocate();
}

} // NS Scene
} // NS Bifrost

#endif // _BIFROST_SCENE_SCENE_NODE_TEST_H_
