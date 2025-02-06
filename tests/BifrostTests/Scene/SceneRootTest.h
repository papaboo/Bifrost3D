// Test Bifrost Scene Roots.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_SCENE_SCENE_ROOT_TEST_H_
#define _BIFROST_SCENE_SCENE_ROOT_TEST_H_

#include <Bifrost/Scene/SceneRoot.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Scene {

GTEST_TEST(Scene_SceneRoot, resizing) {
    SceneRoots::allocate(8u);
    EXPECT_GE(SceneRoots::capacity(), 8u);

    // Test that capacity can be increased.
    unsigned int larger_capacity = SceneRoots::capacity() + 4u;
    SceneRoots::reserve(larger_capacity);
    EXPECT_GE(SceneRoots::capacity(), larger_capacity);

    // Test that capacity won't be decreased.
    SceneRoots::reserve(5u);
    EXPECT_GE(SceneRoots::capacity(), larger_capacity);

    SceneRoots::deallocate();
    EXPECT_LT(SceneRoots::capacity(), larger_capacity);
}

GTEST_TEST(Scene_SceneRoot, invalid_scene_root_properties) {
    SceneRoots::allocate(1u);

    SceneRoot invalid_scene_root = SceneRoot::invalid();

    EXPECT_EQ(Math::RGB::black(), invalid_scene_root.get_environment_tint());
    EXPECT_EQ(Assets::Texture::invalid(), invalid_scene_root.get_environment_map());

    SceneRoots::deallocate();
}

GTEST_TEST(Scene_SceneRoot, create) {
    SceneNodes::allocate(1u);
    SceneRoots::allocate(1u);

    SceneRoot scene = SceneRoot("Foo", Math::RGB::blue());
    EXPECT_TRUE(scene.exists());

    EXPECT_EQ("Foo", scene.get_name());
    EXPECT_EQ(Math::RGB::blue(), scene.get_environment_tint());
    EXPECT_EQ(Assets::Texture::invalid(), scene.get_environment_map());

    // Test scene root created notification.
    Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
    EXPECT_EQ(1, changed_scenes.end() - changed_scenes.begin());
    EXPECT_EQ(scene, *changed_scenes.begin());
    EXPECT_EQ(SceneRoots::Change::Created, scene.get_changes());

    SceneRoots::deallocate();
    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneRoot, destroy) {
    SceneNodes::allocate(1u);
    SceneRoots::allocate(1u);
    SceneRoot scene = SceneRoot("Foo", Math::RGB::blue());
    EXPECT_TRUE(scene.exists());

    SceneRoots::reset_change_notifications();

    scene.destroy();
    EXPECT_FALSE(scene.exists());

    // Test scene node destroyed notification.
    Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
    EXPECT_EQ(1, changed_scenes.end() - changed_scenes.begin());
    EXPECT_EQ(scene, *changed_scenes.begin());
    EXPECT_EQ(SceneRoots::Change::Destroyed, scene.get_changes());

    SceneRoots::deallocate();
    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneRoot, create_and_destroy_notifications) {
    SceneNodes::allocate(2u);
    SceneRoots::allocate(2u);

    SceneRoot scene0 = SceneRoot("Foo", Math::RGB::blue());
    SceneRoot scene1 = SceneRoot("Bar", Math::RGB::blue());
    EXPECT_TRUE(scene0.exists());
    EXPECT_TRUE(scene1.exists());

    { // Test scene root create notifications.
        Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
        EXPECT_EQ(changed_scenes.end() - changed_scenes.begin(), 2);

        bool scene0_created = false;
        bool scene1_created = false;
        for (const SceneRoot scene : changed_scenes) {
            bool scene_created = scene.get_changes() == SceneRoots::Change::Created;
            scene0_created |= scene == scene0 && scene_created;
            scene1_created |= scene == scene1 && scene_created;
        }

        EXPECT_TRUE(scene0_created);
        EXPECT_TRUE(scene1_created);
    }

    SceneRoots::reset_change_notifications();

    { // Test destroy.
        scene0.destroy();
        EXPECT_FALSE(scene0.exists());

        Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
        EXPECT_EQ(changed_scenes.end() - changed_scenes.begin(), 1);

        SceneRoot changed_scene = *changed_scenes.begin();
        bool scene0_destroyed = changed_scene == scene0 && changed_scene.get_changes() == SceneRoots::Change::Destroyed;
        EXPECT_TRUE(scene0_destroyed);
    }

    SceneRoots::reset_change_notifications();

    { // Test that destroyed scene cannot be destroyed again.
        EXPECT_FALSE(scene0.exists());

        scene0.destroy();
        EXPECT_FALSE(scene0.exists());
        EXPECT_TRUE(SceneRoots::get_changed_scenes().is_empty());
    }

    SceneRoots::deallocate();
    SceneNodes::deallocate();
}

GTEST_TEST(Scene_SceneRoot, update_notifications) {
    SceneNodes::allocate(1u);
    SceneRoots::allocate(1u);

    SceneRoot scene = SceneRoot("Foo", Math::RGB::blue());
    
    SceneRoots::reset_change_notifications();

    { // Background color change notification.
        scene.set_environment_tint(Math::RGB::green());

        Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
        EXPECT_EQ(changed_scenes.end() - changed_scenes.begin(), 1);
        EXPECT_EQ(scene, *changed_scenes.begin());
        EXPECT_EQ(scene.get_changes(), SceneRoots::Change::EnvironmentTint);
    }

    SceneRoots::reset_change_notifications();

    { // Environment map change notification.
        Assets::Textures::allocate(2u);

        Assets::Texture map = Assets::Texture::create2D(Assets::Image::invalid());

        scene.set_environment_map(map);

        Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
        EXPECT_EQ(changed_scenes.end() - changed_scenes.begin(), 1);
        EXPECT_EQ(scene, *changed_scenes.begin());
        EXPECT_EQ(scene.get_changes(), SceneRoots::Change::EnvironmentMap);

        Assets::Textures::deallocate();
    }

    SceneRoots::deallocate();
    SceneNodes::deallocate();
}

} // NS Scene
} // NS Bifrost

#endif // _BIFROST_SCENE_SCENE_ROOT_TEST_H_
