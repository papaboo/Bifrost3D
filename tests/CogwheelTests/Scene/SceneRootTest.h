// Test Cogwheel Scene Roots.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_SCENE_ROOT_TEST_H_
#define _COGWHEEL_SCENE_SCENE_ROOT_TEST_H_

#include <Cogwheel/Scene/SceneRoot.h>

#include <gtest/gtest.h>

namespace Cogwheel {
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

GTEST_TEST(Scene_SceneRoot, sentinel_scene) {
    SceneRoots::allocate(1u);

    SceneRoot sentinel = SceneRoots::UID::invalid_UID();

    EXPECT_EQ(Math::RGB::black(), sentinel.get_environment_tint());
    EXPECT_EQ(Assets::Textures::UID::invalid_UID(), sentinel.get_environment_map());

    SceneRoots::deallocate();
}

GTEST_TEST(Scene_SceneRoot, create) {
    SceneRoots::allocate(2u);
    SceneRoots::UID scene_ID = SceneRoots::create("Foo", SceneNodes::UID::invalid_UID(), Math::RGB::blue());
    EXPECT_TRUE(SceneRoots::has(scene_ID));
    
    EXPECT_EQ("Foo", SceneRoots::get_name(scene_ID));
    EXPECT_EQ(Math::RGB::blue(), SceneRoots::get_environment_tint(scene_ID));
    EXPECT_EQ(Assets::Textures::UID::invalid_UID(), SceneRoots::get_environment_map(scene_ID));

    // Test scene root created notification.
    Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
    EXPECT_EQ(1, changed_scenes.end() - changed_scenes.begin());
    EXPECT_EQ(scene_ID, *changed_scenes.begin());
    EXPECT_EQ(SceneRoots::Change::Created, SceneRoots::get_changes(scene_ID));

    SceneRoots::deallocate();
}

GTEST_TEST(Scene_SceneRoot, destroy) {
    SceneRoots::allocate(2u);
    SceneRoot scene = SceneRoots::create("Foo", SceneNodes::UID::invalid_UID(), Math::RGB::blue());
    EXPECT_TRUE(scene.exists());

    SceneRoots::reset_change_notifications();

    SceneRoots::destroy(scene.get_ID());
    EXPECT_FALSE(scene.exists());

    // Test scene node destroyed notification.
    Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
    EXPECT_EQ(1, changed_scenes.end() - changed_scenes.begin());
    EXPECT_EQ(scene.get_ID(), *changed_scenes.begin());
    EXPECT_EQ(SceneRoots::Change::Destroyed, scene.get_changes());

    SceneRoots::deallocate();
}

GTEST_TEST(Scene_SceneRoot, create_and_destroy_notifications) {
    SceneRoots::allocate(8u);

    SceneRoots::UID scene_ID0 = SceneRoots::create("Foo", SceneNodes::UID::invalid_UID(), Math::RGB::blue());
    SceneRoots::UID scene_ID1 = SceneRoots::create("Bar", SceneNodes::UID::invalid_UID(), Math::RGB::blue());
    EXPECT_TRUE(SceneRoots::has(scene_ID0));
    EXPECT_TRUE(SceneRoots::has(scene_ID1));

    { // Test scene scene create notifications.
        Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
        EXPECT_EQ(changed_scenes.end() - changed_scenes.begin(), 2);

        bool scene0_created = false;
        bool scene1_created = false;
        bool other_changes = false;
        for (const SceneRoots::UID scene_ID : changed_scenes) {
            bool scene_created = SceneRoots::get_changes(scene_ID) == SceneRoots::Change::Created;
            if (scene_ID == scene_ID0 && scene_created)
                scene0_created = true;
            else if (scene_ID == scene_ID1 && scene_created)
                scene1_created = true;
            else
                other_changes = true;
        }

        EXPECT_TRUE(scene0_created);
        EXPECT_TRUE(scene1_created);
        EXPECT_FALSE(other_changes);
    }

    SceneRoots::reset_change_notifications();

    { // Test destroy.
        SceneRoots::destroy(scene_ID0);
        EXPECT_FALSE(SceneRoots::has(scene_ID0));

        Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
        EXPECT_EQ(changed_scenes.end() - changed_scenes.begin(), 1);

        SceneRoot changed_scene = *changed_scenes.begin();
        bool scene0_destroyed = changed_scene.get_ID() == scene_ID0 && changed_scene.get_changes() == SceneRoots::Change::Destroyed;
        EXPECT_TRUE(scene0_destroyed);
    }

    SceneRoots::reset_change_notifications();

    { // Test that destroyed scene cannot be destroyed again.
        EXPECT_FALSE(SceneRoots::has(scene_ID0));

        SceneRoots::destroy(scene_ID0);
        EXPECT_FALSE(SceneRoots::has(scene_ID0));
        EXPECT_TRUE(SceneRoots::get_changed_scenes().is_empty());
    }

    SceneRoots::deallocate();
}

GTEST_TEST(Scene_SceneRoot, update_notifications) {
    SceneRoots::allocate(8u);

    SceneRoot scene = SceneRoots::create("Foo", SceneNodes::UID::invalid_UID(), Math::RGB::blue());
    
    SceneRoots::reset_change_notifications();

    { // Background color change notification.
        scene.set_environment_tint(Math::RGB::green());

        Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
        EXPECT_EQ(changed_scenes.end() - changed_scenes.begin(), 1);

        EXPECT_TRUE(scene.get_ID() == *changed_scenes.begin());
        EXPECT_TRUE(scene.get_changes() == SceneRoots::Change::EnvironmentTint);
    }

    SceneRoots::reset_change_notifications();

    { // Environment map change notification.
        Assets::Textures::allocate(2u);

        Assets::Textures::UID map = Assets::Textures::create2D(Assets::Images::UID::invalid_UID());

        scene.set_environment_map(map);

        Core::Iterable<SceneRoots::ChangedIterator> changed_scenes = SceneRoots::get_changed_scenes();
        EXPECT_EQ(changed_scenes.end() - changed_scenes.begin(), 1);

        EXPECT_TRUE(scene.get_ID() == *changed_scenes.begin());
        EXPECT_TRUE(scene.get_changes() == SceneRoots::Change::EnvironmentMap);

        Assets::Textures::deallocate();
    }

    SceneRoots::deallocate();
}

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_SCENE_ROOT_TEST_H_