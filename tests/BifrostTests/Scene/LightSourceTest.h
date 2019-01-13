// Test Bifrost Light sources.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_SCENE_LIGHT_SOURCE_TEST_H_
#define _BIFROST_SCENE_LIGHT_SOURCE_TEST_H_

#include <Bifrost/Scene/LightSource.h>

#include <gtest/gtest.h>

namespace Bifrost {
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

    EXPECT_EQ(SceneNodes::UID::invalid_UID(), LightSources::get_node_ID(sentinel_ID));
    EXPECT_EQ(Math::RGB::black(), LightSources::get_sphere_light_power(sentinel_ID));

    LightSources::deallocate();
}

TEST_F(Scene_LightSource, create_sphere_light) {
    LightSources::allocate(2u);

    SceneNode light_node = SceneNodes::create("Light");

    const Math::RGB light_power(100.0f);
    const float light_radius = 2.0f;

    LightSources::allocate(2u);
    SphereLight light = LightSources::create_sphere_light(light_node.get_ID(), light_power, light_radius);
    EXPECT_TRUE(light.exists());

    EXPECT_EQ(LightSources::Type::Sphere, LightSources::get_type(light.get_ID()));
    EXPECT_EQ(light_node, light.get_node());
    EXPECT_EQ(light_power, light.get_power());
    EXPECT_EQ(light_radius, light.get_radius());

    // Test scene node created notification.
    Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
    EXPECT_EQ(1, changed_lights.end() - changed_lights.begin());
    EXPECT_EQ(light.get_ID(), *changed_lights.begin());
    EXPECT_EQ(LightSources::Change::Created, light.get_changes());

    LightSources::deallocate();
}

TEST_F(Scene_LightSource, create_directional_light) {
    LightSources::allocate(2u);

    SceneNode light_node = SceneNodes::create("Light");

    const Math::RGB light_radiance(2.0f);

    LightSources::allocate(2u);
    DirectionalLight light = LightSources::create_directional_light(light_node.get_ID(), light_radiance);
    EXPECT_TRUE(light.exists());

    EXPECT_EQ(LightSources::Type::Directional, LightSources::get_type(light.get_ID()));
    EXPECT_EQ(light_node, light.get_node());
    EXPECT_EQ(light_radiance, light.get_radiance());

    // Test scene node created notification.
    Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
    EXPECT_EQ(1, changed_lights.end() - changed_lights.begin());
    EXPECT_EQ(light.get_ID(), *changed_lights.begin());
    EXPECT_EQ(LightSources::Change::Created, light.get_changes());

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
    Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
    EXPECT_EQ(1, changed_lights.end() - changed_lights.begin());
    EXPECT_EQ(light_ID, *changed_lights.begin());
    EXPECT_EQ(LightSources::Change::Destroyed, LightSources::get_changes(light_ID));

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
        Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
        EXPECT_EQ(2, changed_lights.end() - changed_lights.begin());

        bool node0_created = false;
        bool node1_created = false;
        bool other_changes = false;
        for (const LightSources::UID light_ID : changed_lights) {
            bool light_created = LightSources::get_changes(light_ID) == LightSources::Change::Created;
            if (light_ID == light_ID0 && light_created)
                node0_created = true;
            else if (light_ID == light_ID1 && light_created)
                node1_created = true;
            else
                other_changes = true;
        }

        EXPECT_TRUE(node0_created);
        EXPECT_TRUE(node1_created);
        EXPECT_FALSE(other_changes);
    }

    LightSources::reset_change_notifications();

    { // Test destroy.
        LightSources::destroy(light_ID0);
        EXPECT_FALSE(LightSources::has(light_ID0));

        Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
        EXPECT_EQ(1, changed_lights.end() - changed_lights.begin());

        bool node0_destroyed = false;
        bool other_changes = false;
        for (const LightSources::UID light_ID : changed_lights) {
            if (light_ID == light_ID0 && LightSources::get_changes(light_ID) == LightSources::Change::Destroyed)
                node0_destroyed = true;
            else
                other_changes = true;
        }

        EXPECT_TRUE(node0_destroyed);
        EXPECT_FALSE(other_changes);
    }

    LightSources::reset_change_notifications();

    { // Test that destroyed node cannot be destroyed again.
        EXPECT_FALSE(LightSources::has(light_ID0));
        
        LightSources::destroy(light_ID0);
        EXPECT_FALSE(LightSources::has(light_ID0));
        EXPECT_TRUE(LightSources::get_changed_lights().is_empty());
    }

    LightSources::deallocate();
}

} // NS Scene
} // NS Bifrost

#endif // _BIFROST_SCENE_LIGHT_SOURCE_TEST_H_
