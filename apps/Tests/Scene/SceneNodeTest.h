// Test Cogwheel Scene Nodes.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_SCENE_NODE_TEST_H_
#define _COGWHEEL_SCENE_SCENE_NODE_TEST_H_

#include <Scene/SceneNode.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Scene {

GTEST_TEST(Scene_SceneNode, resizing) {
    SceneNodes::allocate(8u);
    EXPECT_GE(SceneNodes::capacity(), 8u);

    // Test that capacity can be increased.
    unsigned int largerCapacity = SceneNodes::capacity() + 4u;
    SceneNodes::reserve(largerCapacity);
    EXPECT_GE(SceneNodes::capacity(), largerCapacity);

    // Test that capacity won't be decreased.
    SceneNodes::reserve(5u);
    EXPECT_GE(SceneNodes::capacity(), largerCapacity);

    SceneNodes::deallocate();
    EXPECT_LT(SceneNodes::capacity(), largerCapacity);
}

GTEST_TEST(Scene_SceneNode, creating) {
    SceneNodes::allocate(2u);
    SceneNodes::UID id = SceneNodes::create("Foo");
    EXPECT_TRUE(SceneNodes::has(id));
    
    EXPECT_EQ(SceneNodes::getName(id), "Foo");

    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneNode, sentinel_node) {
    SceneNodes::allocate(1u);

    SceneNode sentinel = SceneNodes::UID::invalid_UID();
    SceneNode node = SceneNodes::create("Foo");

    // Test that sentinel node cannot have it's parent set.
    sentinel.setParent(node);
    SceneNode parentSet = sentinel.getParent();
    EXPECT_NE(parentSet, node);
    EXPECT_EQ(parentSet, sentinel);

    EXPECT_EQ(sentinel.getChildren().size(), 0u);

    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneNode, parenting) {
    SceneNodes::allocate(1u);
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");
    SceneNode n2 = SceneNodes::create("n2");

    EXPECT_FALSE(n1.getParent().exists());
    
    // Set n0 to the parent and check the parent-child relationship.
    n1.setParent(n0);
    EXPECT_TRUE(n1.getParent().exists());
    EXPECT_EQ(n1.getParent(), n0);
    EXPECT_EQ(n0.getChildren().size(), 1u);
    EXPECT_TRUE(n0.hasChild(n1));

    // Set n2 to the parent and check the parent-child relationship.
    n1.setParent(n2);
    EXPECT_TRUE(n1.getParent().exists());
    EXPECT_EQ(n1.getParent(), n2);
    EXPECT_EQ(n2.getChildren().size(), 1u);
    EXPECT_TRUE(n2.hasChild(n1));
    // ... also check that n0 no longer has any children.
    EXPECT_EQ(0u, n0.getChildren().size());
    
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

    n0.setParent(n3);
    n4.setParent(n3);
    n6.setParent(n3);
    n2.setParent(n4);
    n5.setParent(n4);
    n1.setParent(n6);

    EXPECT_EQ(n0.getParent(), n3);
    EXPECT_TRUE(n3.hasChild(n0));
    EXPECT_EQ(n4.getParent(), n3);
    EXPECT_TRUE(n3.hasChild(n4));
    EXPECT_EQ(n6.getParent(), n3);
    EXPECT_TRUE(n3.hasChild(n6));
    EXPECT_EQ(n2.getParent(), n4);
    EXPECT_TRUE(n4.hasChild(n2));
    EXPECT_EQ(n5.getParent(), n4);
    EXPECT_TRUE(n4.hasChild(n5));
    EXPECT_EQ(n1.getParent(), n6);
    EXPECT_TRUE(n6.hasChild(n1));

    // Now parent id4 below id0, just for fun and profit.
    //     n3
    //    /  \
    //   n0  n6
    //   |    |
    //   n4  n1
    //  / \
    // n2 n5
    n4.setParent(n0);

    EXPECT_FALSE(n3.getParent().exists());
    EXPECT_EQ(n0.getParent(), n3);
    EXPECT_TRUE(n3.hasChild(n0));
    EXPECT_EQ(n6.getParent(), n3);
    EXPECT_TRUE(n3.hasChild(n6));
    EXPECT_EQ(n4.getParent(), n0);
    EXPECT_TRUE(n0.hasChild(n4));
    EXPECT_EQ(n1.getParent(), n6);
    EXPECT_TRUE(n6.hasChild(n1));
    EXPECT_EQ(n2.getParent(), n4);
    EXPECT_TRUE(n4.hasChild(n2));
    EXPECT_EQ(n5.getParent(), n4);
    EXPECT_TRUE(n4.hasChild(n5));

    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneNode, grap_traversal) {
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

    n0.setParent(n3);
    n4.setParent(n3);
    n6.setParent(n3);
    n2.setParent(n4);
    n5.setParent(n4);
    n1.setParent(n6);

    unsigned int* visits = new unsigned int[SceneNodes::capacity()];
    for (unsigned int i = 0; i < SceneNodes::capacity(); ++i)
        visits[i] = 0u;
    n2.traverseAllChildren([&](SceneNodes::UID id) {
        ++visits[id];
    });
    EXPECT_EQ(visits[n0.getID()], 0u);
    EXPECT_EQ(visits[n1.getID()], 0u);
    EXPECT_EQ(visits[n2.getID()], 0u);
    EXPECT_EQ(visits[n3.getID()], 0u);
    EXPECT_EQ(visits[n4.getID()], 0u);
    EXPECT_EQ(visits[n5.getID()], 0u);
    EXPECT_EQ(visits[n6.getID()], 0u);

    n4.traverseGraph([&](SceneNodes::UID id) {
        ++visits[id];
    });
    EXPECT_EQ(visits[n0.getID()], 0u);
    EXPECT_EQ(visits[n1.getID()], 0u);
    EXPECT_EQ(visits[n2.getID()], 1u);
    EXPECT_EQ(visits[n3.getID()], 0u);
    EXPECT_EQ(visits[n4.getID()], 1u);
    EXPECT_EQ(visits[n5.getID()], 1u);
    EXPECT_EQ(visits[n6.getID()], 0u);

    n6.traverseAllChildren([&](SceneNodes::UID id) {
        ++visits[id];
    });
    EXPECT_EQ(visits[n0.getID()], 0u);
    EXPECT_EQ(visits[n1.getID()], 1u);
    EXPECT_EQ(visits[n2.getID()], 1u);
    EXPECT_EQ(visits[n3.getID()], 0u);
    EXPECT_EQ(visits[n4.getID()], 1u);
    EXPECT_EQ(visits[n5.getID()], 1u);
    EXPECT_EQ(visits[n6.getID()], 0u);

    n3.traverseGraph([&](SceneNodes::UID id) {
        ++visits[id];
    });
    EXPECT_EQ(visits[n0.getID()], 1u);
    EXPECT_EQ(visits[n1.getID()], 2u);
    EXPECT_EQ(visits[n2.getID()], 2u);
    EXPECT_EQ(visits[n3.getID()], 1u);
    EXPECT_EQ(visits[n4.getID()], 2u);
    EXPECT_EQ(visits[n5.getID()], 2u);
    EXPECT_EQ(visits[n6.getID()], 1u);

    SceneNodes::deallocate();
}

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_SCENE_NODE_TEST_H_