// Test Cogwheel Light sources.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_LIGHT_SOURCE_TEST_H_
#define _COGWHEEL_SCENE_LIGHT_SOURCE_TEST_H_

#include <Cogwheel/Scene/LightSource.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Scene {

class Scene_LightSource : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        SceneNodes::allocate(8u);
    }
    virtual void TearDown() {
        SceneNodes::deallocate();
    }
};

TEST_F(Scene_LightSource, resizing) {
    LightSources::allocate(8u);
    EXPECT_GE(LightSources::capacity(), 8u);

    // Test that capacity can be increased.
    unsigned int larger_capacity = LightSources::capacity() + 4u;
    LightSources::reserve(larger_capacity);
    EXPECT_GE(LightSources::capacity(), larger_capacity);

    // Test that capacity won't be decreased.
    LightSources::reserve(5u);
    EXPECT_GE(LightSources::capacity(), larger_capacity);

    LightSources::deallocate();
    EXPECT_LT(LightSources::capacity(), larger_capacity);
}

TEST_F(Scene_LightSource, sentinel_node) {
    LightSources::allocate(1u);

    LightSources::UID sentinel_ID = LightSources::UID::invalid_UID();

    EXPECT_FALSE(LightSources::has(sentinel_ID));

    EXPECT_EQ(LightSources::get_node_ID(sentinel_ID), SceneNodes::UID::invalid_UID());
    EXPECT_EQ(LightSources::get_power(sentinel_ID), Math::RGB::black());

    LightSources::deallocate();
}

TEST_F(Scene_LightSource, create) {
    LightSources::allocate(2u);

    SceneNodes::UID light_node_ID = SceneNodes::create("Light");

    const Math::RGB light_power(100.0f);

    LightSources::allocate(2u);
    LightSources::UID light_ID = LightSources::create_sphere_light(light_node_ID, light_power, 0.0f);
    EXPECT_TRUE(LightSources::has(light_ID));

    EXPECT_EQ(LightSources::get_node_ID(light_ID), light_node_ID);
    EXPECT_EQ(LightSources::get_power(light_ID), light_power);

    LightSources::deallocate();
}

TEST_F(Scene_LightSource, destroy) {
    LightSources::allocate(2u);
    
    SceneNodes::UID node_ID = SceneNodes::create("Foo");
    LightSources::UID light_ID = LightSources::create_sphere_light(node_ID, Math::RGB::white(), 0.0f);
    EXPECT_TRUE(LightSources::has(light_ID));

    LightSources::reset_change_notifications();

    LightSources::destroy(light_ID);
    EXPECT_FALSE(LightSources::has(light_ID));

    // Test scene node destroyed notification.
    Core::Iterable<LightSources::light_destroyed_iterator> destroyed_lights = LightSources::get_destroyed_lights();
    EXPECT_EQ(destroyed_lights.end() - destroyed_lights.begin(), 1);
    EXPECT_EQ(*destroyed_lights.begin(), light_ID);

    LightSources::deallocate();
}

TEST_F(Scene_LightSource, create_and_destroy_notifications) {
    LightSources::allocate(8u);

    SceneNodes::UID node_ID = SceneNodes::create("Foo");

    LightSources::UID light_ID0 = LightSources::create_sphere_light(node_ID, Math::RGB(0.0f), 0.0f);
    LightSources::UID light_ID1 = LightSources::create_sphere_light(node_ID, Math::RGB(1.0f), 0.0f);
    EXPECT_TRUE(LightSources::has(light_ID0));
    EXPECT_TRUE(LightSources::has(light_ID1));

    { // Test scene node create notifications.
        Core::Iterable<LightSources::light_created_iterator> created_lights = LightSources::get_created_lights();
        EXPECT_EQ(created_lights.end() - created_lights.begin(), 2);
        Core::Iterable<LightSources::light_destroyed_iterator> destroyed_lights = LightSources::get_destroyed_lights();
        EXPECT_EQ(destroyed_lights.end() - destroyed_lights.begin(), 0);

        bool node0_created = false;
        bool node1_created = false;
        for (const LightSources::UID light_ID : created_lights) {
            if (light_ID == light_ID0)
                node0_created = true;
            if (light_ID == light_ID1)
                node1_created = true;
        }

        EXPECT_TRUE(node0_created);
        EXPECT_TRUE(node1_created);
    }

    LightSources::reset_change_notifications();

    { // Test destroy.
        LightSources::destroy(light_ID0);
        EXPECT_FALSE(LightSources::has(light_ID0));

        Core::Iterable<LightSources::light_created_iterator> created_lights = LightSources::get_created_lights();
        EXPECT_EQ(created_lights.end() - created_lights.begin(), 0);
        Core::Iterable<LightSources::light_destroyed_iterator> destroyed_lights = LightSources::get_destroyed_lights();
        EXPECT_EQ(destroyed_lights.end() - destroyed_lights.begin(), 1);

        bool node0_destroyed = false;
        bool node1_destroyed = false;
        for (const LightSources::UID light_ID : destroyed_lights) {
            if (light_ID == light_ID0)
                node0_destroyed = true;
            if (light_ID == light_ID1)
                node1_destroyed = true;
        }

        EXPECT_TRUE(node0_destroyed);
        EXPECT_FALSE(node1_destroyed);
    }

    LightSources::reset_change_notifications();

    { // Test that destroyed node cannot be destroyed again.
        LightSources::destroy(light_ID0);
        EXPECT_FALSE(LightSources::has(light_ID0));

        Core::Iterable<LightSources::light_created_iterator> created_lights = LightSources::get_created_lights();
        EXPECT_EQ(created_lights.end() - created_lights.begin(), 0);
        Core::Iterable<LightSources::light_destroyed_iterator> destroyed_lights = LightSources::get_destroyed_lights();
        EXPECT_EQ(destroyed_lights.end() - destroyed_lights.begin(), 0);
    }

    LightSources::deallocate();
}

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_LIGHT_SOURCE_TEST_H_