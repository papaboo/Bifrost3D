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

TEST_F(Scene_LightSource, invalid_sphere_light_properties) {
    LightSources::allocate(1u);

    SphereLight invalid_light = SphereLight::invalid();

    EXPECT_FALSE(invalid_light.exists());

    EXPECT_EQ(SceneNode::invalid(), invalid_light.get_node());
    EXPECT_EQ(Math::RGB(100000, 0, 100000), invalid_light.get_power());

    LightSources::deallocate();
}

TEST_F(Scene_LightSource, create_sphere_light) {
    LightSources::allocate(2u);

    SceneNode light_node = SceneNode("Light");

    const Math::RGB light_power(100.0f);
    const float light_radius = 2.0f;

    LightSources::allocate(2u);
    SphereLight light = SphereLight(light_node, light_power, light_radius);
    EXPECT_TRUE(light.exists());

    EXPECT_EQ(LightSources::Type::Sphere, light.get_type());
    EXPECT_EQ(light_node, light.get_node());
    EXPECT_EQ(light_power, light.get_power());
    EXPECT_EQ(light_radius, light.get_radius());

    // Test scene node created notification.
    Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
    EXPECT_EQ(1, changed_lights.end() - changed_lights.begin());
    EXPECT_EQ(light, *changed_lights.begin());
    EXPECT_EQ(LightSources::Change::Created, light.get_changes());

    LightSources::deallocate();
}

TEST_F(Scene_LightSource, create_directional_light) {
    LightSources::allocate(2u);

    SceneNode light_node = SceneNode("Light");

    const Math::RGB light_radiance(2.0f);

    LightSources::allocate(2u);
    DirectionalLight light = DirectionalLight(light_node, light_radiance);
    EXPECT_TRUE(light.exists());

    EXPECT_EQ(LightSources::Type::Directional, light.get_type());
    EXPECT_EQ(light_node, light.get_node());
    EXPECT_EQ(light_radiance, light.get_radiance());

    // Test scene node created notification.
    Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
    EXPECT_EQ(1, changed_lights.end() - changed_lights.begin());
    EXPECT_EQ(light, *changed_lights.begin());
    EXPECT_EQ(LightSources::Change::Created, light.get_changes());

    LightSources::deallocate();
}

TEST_F(Scene_LightSource, destroy) {
    LightSources::allocate(2u);

    SceneNode node = SceneNode("Foo");
    SphereLight light = SphereLight(node, Math::RGB::white(), 0.0f);
    EXPECT_TRUE(light.exists());

    LightSources::reset_change_notifications();

    light.destroy();
    EXPECT_FALSE(light.exists());

    // Test scene node destroyed notification.
    Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
    EXPECT_EQ(1, changed_lights.end() - changed_lights.begin());
    EXPECT_EQ(light, *changed_lights.begin());
    EXPECT_EQ(LightSources::Change::Destroyed, light.get_changes());

    LightSources::deallocate();
}

TEST_F(Scene_LightSource, create_and_destroy_notifications) {
    LightSources::allocate(8u);

    SceneNode node = SceneNode("Foo");

    SphereLight light0 = SphereLight(node, Math::RGB(0.0f), 0.0f);
    SpotLight light1 = SpotLight(node, Math::RGB(1.0f), 0.0f, 0.0f);
    EXPECT_TRUE(light0.exists());
    EXPECT_TRUE(light1.exists());

    { // Test scene node create notifications.
        Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
        EXPECT_EQ(2, changed_lights.end() - changed_lights.begin());

        bool node0_created = false;
        bool node1_created = false;
        bool other_changes = false;
        for (const LightSource light : changed_lights) {
            bool light_created = light.get_changes() == LightSources::Change::Created;
            if (light0 == light && light_created)
                node0_created = true;
            else if (light1 == light && light_created)
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
        light0.destroy();
        EXPECT_FALSE(light0.exists());

        Core::Iterable<LightSources::ChangedIterator> changed_lights = LightSources::get_changed_lights();
        EXPECT_EQ(1, changed_lights.end() - changed_lights.begin());

        bool node0_destroyed = false;
        bool other_changes = false;
        for (const LightSource light : changed_lights) {
            if (light0 == light && light.get_changes() == LightSources::Change::Destroyed)
                node0_destroyed = true;
            else
                other_changes = true;
        }

        EXPECT_TRUE(node0_destroyed);
        EXPECT_FALSE(other_changes);
    }

    LightSources::reset_change_notifications();

    { // Test that destroyed node cannot be destroyed again.
        EXPECT_FALSE(light0.exists());
        
        light0.destroy();
        EXPECT_FALSE(light0.exists());
        EXPECT_TRUE(LightSources::get_changed_lights().is_empty());
    }

    LightSources::deallocate();
}

} // NS Scene
} // NS Bifrost

#endif // _BIFROST_SCENE_LIGHT_SOURCE_TEST_H_
