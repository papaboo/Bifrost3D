// Test Cogwheel visual materials.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_MATERIAL_TEST_H_
#define _COGWHEEL_ASSETS_MATERIAL_TEST_H_

#include <Cogwheel/Assets/Material.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Assets {

class Assets_Material : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Materials::allocate(8u);
    }
    virtual void TearDown() {
        Materials::deallocate();
    }
};

TEST_F(Assets_Material, resizing) {
    // Test that capacity can be increased.
    unsigned int largerCapacity = Materials::capacity() + 4u;
    Materials::reserve(largerCapacity);
    EXPECT_GE(Materials::capacity(), largerCapacity);

    // Test that capacity won't be decreased.
    Materials::reserve(5u);
    EXPECT_GE(Materials::capacity(), largerCapacity);

    Materials::deallocate();
    EXPECT_LT(Materials::capacity(), largerCapacity);
}

TEST_F(Assets_Material, sentinel_material) {
    Materials::UID sentinel_ID = Materials::UID::invalid_UID();

    EXPECT_FALSE(Materials::has(sentinel_ID));
    EXPECT_EQ(Materials::get_base_tint(sentinel_ID), Math::RGB::red());
    EXPECT_EQ(Materials::get_base_roughness(sentinel_ID), 0.0f);
    EXPECT_EQ(Materials::get_metallic(sentinel_ID), 0.0f);
    EXPECT_EQ(Materials::get_specularity(sentinel_ID), 0.0f);
}

TEST_F(Assets_Material, create) {
    Materials::Data data;
    data.base_tint = Math::RGB::red();
    data.base_roughness = 0.5f;
    data.metallic = 1.0f;
    data.specularity = 0.04f;
    Materials::UID material_ID = Materials::create("TestMaterial", data);

    EXPECT_TRUE(Materials::has(material_ID));
    EXPECT_EQ(Materials::get_base_tint(material_ID), Math::RGB::red());
    EXPECT_EQ(Materials::get_base_roughness(material_ID), 0.5f);
    EXPECT_EQ(Materials::get_metallic(material_ID), 1.0f);
    EXPECT_EQ(Materials::get_specularity(material_ID), 0.04f);

    // Test material created notification.
    Core::Iterable<Materials::material_created_iterator> created_materials = Materials::get_created_materials();
    EXPECT_EQ(created_materials.end() - created_materials.begin(), 1);
    EXPECT_EQ(*created_materials.begin(), material_ID);
}

TEST_F(Assets_Material, destroy) {
    Materials::Data data = {};
    Materials::UID material_ID = Materials::create("TestMaterial", data);
    EXPECT_TRUE(Materials::has(material_ID));

    Materials::reset_change_notifications();

    Materials::destroy(material_ID);
    EXPECT_FALSE(Materials::has(material_ID));

    // Test material destroyed notification.
    Core::Iterable<Materials::material_destroyed_iterator> destroyed_materials = Materials::get_destroyed_materials();
    EXPECT_EQ(destroyed_materials.end() - destroyed_materials.begin(), 1);
    EXPECT_EQ(*destroyed_materials.begin(), material_ID);
}

TEST_F(Assets_Material, create_and_destroy_notifications) {
    Materials::Data data = {};
    Materials::UID material_ID0 = Materials::create("TestMaterial0", data);
    Materials::UID material_ID1 = Materials::create("TestMaterial1", data);
    EXPECT_TRUE(Materials::has(material_ID0));
    EXPECT_TRUE(Materials::has(material_ID1));

    { // Test material create notifications.
        Core::Iterable<Materials::material_created_iterator> created_materials = Materials::get_created_materials();
        EXPECT_EQ(created_materials.end() - created_materials.begin(), 2);
        Core::Iterable<Materials::material_destroyed_iterator> destroyed_materials = Materials::get_destroyed_materials();
        EXPECT_EQ(destroyed_materials.end() - destroyed_materials.begin(), 0);

        bool material0_created = false;
        bool material1_created = false;
        for (const Materials::UID material_ID : created_materials) {
            if (material_ID == material_ID0)
                material0_created = true;
            if (material_ID == material_ID1)
                material1_created = true;
        }

        EXPECT_TRUE(material0_created);
        EXPECT_TRUE(material1_created);
        EXPECT_TRUE(Materials::has_events(material_ID0, Materials::Events::Created));
        EXPECT_TRUE(Materials::has_events(material_ID1, Materials::Events::Created));
    }

    Materials::reset_change_notifications();

    { // Test destroy.
        Materials::destroy(material_ID0);
        EXPECT_FALSE(Materials::has(material_ID0));

        Core::Iterable<Materials::material_created_iterator> created_materials = Materials::get_created_materials();
        EXPECT_EQ(created_materials.end() - created_materials.begin(), 0);
        Core::Iterable<Materials::material_destroyed_iterator> destroyed_materials = Materials::get_destroyed_materials();
        EXPECT_EQ(destroyed_materials.end() - destroyed_materials.begin(), 1);

        bool material0_destroyed = false;
        bool material1_destroyed = false;
        for (const Materials::UID material_ID : destroyed_materials) {
            if (material_ID == material_ID0)
                material0_destroyed = true;
            if (material_ID == material_ID1)
                material1_destroyed = true;
        }

        EXPECT_TRUE(material0_destroyed);
        EXPECT_FALSE(material1_destroyed);
        EXPECT_TRUE(Materials::has_events(material_ID0, Materials::Events::Destroyed));
        EXPECT_FALSE(Materials::has_events(material_ID1, Materials::Events::Destroyed));
    }

    Materials::reset_change_notifications();

    { // Test that destroyed material cannot be destroyed again.
        Materials::destroy(material_ID0);
        EXPECT_FALSE(Materials::has(material_ID0));

        Core::Iterable<Materials::material_created_iterator> created_materials = Materials::get_created_materials();
        EXPECT_EQ(created_materials.end() - created_materials.begin(), 0);
        Core::Iterable<Materials::material_destroyed_iterator> destroyed_materials = Materials::get_destroyed_materials();
        EXPECT_EQ(destroyed_materials.end() - destroyed_materials.begin(), 0);
    }
}

TEST_F(Assets_Material, change_notifications) {
    Materials::Data data = {};
    Material material = Materials::create("TestMaterial", data);

    // Test that no materials are initially changed and that a creation event doesn't trigger a change notification as well.
    Core::Iterable<Materials::material_changed_iterator> changed_materials = Materials::get_changed_materials();
    EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 0);
    EXPECT_FALSE(material.has_events(Materials::Events::Changed));

    { // Change base tint.
        Math::RGB new_tint = Math::RGB::red();
        material.set_base_tint(new_tint);

        // Test that only the material has changed.
        Core::Iterable<Materials::material_changed_iterator> changed_materials = Materials::get_changed_materials();
        for (const Materials::UID material_ID : changed_materials) {
            EXPECT_EQ(material_ID, material.get_ID());
            EXPECT_EQ(Materials::get_base_tint(material_ID), new_tint);
            EXPECT_TRUE(Materials::has_events(material_ID, Materials::Events::Changed));
        }
    }

    {
        Materials::reset_change_notifications();

        // Check that change notifications have been properly reset.
        Core::Iterable<Materials::material_changed_iterator> changed_materials = Materials::get_changed_materials();
        EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 0);
        EXPECT_FALSE(material.has_events(Materials::Events::Changed));
    }

    { // Change base roughness.
        float new_roughness = 0.4f;
        material.set_base_roughness(new_roughness);

        // Test that only the material has changed.
        Core::Iterable<Materials::material_changed_iterator> changed_materials = Materials::get_changed_materials();
        for (const Materials::UID material_ID : changed_materials) {
            EXPECT_EQ(material_ID, material.get_ID());
            EXPECT_EQ(Materials::get_base_roughness(material_ID), new_roughness);
            EXPECT_TRUE(Materials::has_events(material_ID, Materials::Events::Changed));
        }
    }
}

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_MATERIAL_TEST_H_