// Test Bifrost visual materials.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_MATERIAL_TEST_H_
#define _BIFROST_ASSETS_MATERIAL_TEST_H_

#include <Bifrost/Assets/Material.h>

#include <gtest/gtest.h>

namespace Bifrost {
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
    MaterialID sentinel_ID = MaterialID::invalid_UID();

    EXPECT_FALSE(Materials::has(sentinel_ID));
    EXPECT_EQ(Materials::get_tint(sentinel_ID), Math::RGB::red());
    EXPECT_EQ(Materials::get_roughness(sentinel_ID), 0.0f);
    EXPECT_EQ(Materials::get_metallic(sentinel_ID), 0.0f);
    EXPECT_EQ(Materials::get_specularity(sentinel_ID), 0.0f);
    EXPECT_EQ(Materials::get_coverage(sentinel_ID), 1.0f);
    EXPECT_EQ(Materials::get_transmission(sentinel_ID), 0.0f);
}

TEST_F(Assets_Material, create) {
    Materials::Data data;
    data.tint = Math::RGB::red();
    data.roughness = 0.5f;
    data.metallic = 1.0f;
    data.specularity = 0.04f;
    data.coverage = 0.25f;
    data.transmission = 0.5f;
    MaterialID material_ID = Materials::create("TestMaterial", data);

    EXPECT_TRUE(Materials::has(material_ID));
    EXPECT_EQ(Materials::get_tint(material_ID), Math::RGB::red());
    EXPECT_EQ(Materials::get_roughness(material_ID), 0.5f);
    EXPECT_EQ(Materials::get_metallic(material_ID), 1.0f);
    EXPECT_EQ(Materials::get_specularity(material_ID), 0.04f);
    EXPECT_EQ(Materials::get_coverage(material_ID), 0.25f);
    EXPECT_EQ(Materials::get_transmission(material_ID), 0.5f);

    // Test material created notification.
    Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
    EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 1);
    EXPECT_EQ(*changed_materials.begin(), material_ID);
    EXPECT_TRUE(Materials::get_changes(material_ID).is_set(Materials::Change::Created));
    EXPECT_FALSE(Materials::get_changes(material_ID).is_set(Materials::Change::Updated));
}

TEST_F(Assets_Material, destroy) {
    Materials::Data data = {};
    MaterialID material_ID = Materials::create("TestMaterial", data);
    EXPECT_TRUE(Materials::has(material_ID));

    Materials::reset_change_notifications();

    Materials::destroy(material_ID);
    EXPECT_FALSE(Materials::has(material_ID));

    // Test material destroyed notification.
    Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
    EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 1);
    EXPECT_EQ(*changed_materials.begin(), material_ID);
    EXPECT_TRUE(Materials::get_changes(material_ID).is_set(Materials::Change::Destroyed));
}

TEST_F(Assets_Material, create_and_change) {
    Materials::Data data = {};
    Material material = Materials::create("TestMaterial", data);

    Math::RGB new_tint = Math::RGB::green();
    material.set_tint(new_tint);

    // Test that creating and changing the material creates a single change.
    Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
    EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 1);
    EXPECT_EQ(*changed_materials.begin(), material.get_ID());
    EXPECT_TRUE(material.get_changes().is_set(Materials::Change::Created));
    EXPECT_TRUE(material.get_changes().is_set(Materials::Change::Updated));
}

TEST_F(Assets_Material, create_and_destroy_notifications) {
    Materials::Data data = {};
    MaterialID material_ID0 = Materials::create("TestMaterial0", data);
    MaterialID material_ID1 = Materials::create("TestMaterial1", data);
    EXPECT_TRUE(Materials::has(material_ID0));
    EXPECT_TRUE(Materials::has(material_ID1));

    { // Test material create notifications.
        Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
        EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 2);

        bool material0_created = false;
        bool material1_created = false;
        for (const MaterialID material_ID : changed_materials) {
            if (material_ID == material_ID0)
                material0_created = Materials::get_changes(material_ID) == Materials::Change::Created;
            if (material_ID == material_ID1)
                material1_created = Materials::get_changes(material_ID) == Materials::Change::Created;
        }

        EXPECT_TRUE(material0_created);
        EXPECT_TRUE(material1_created);
        EXPECT_TRUE(Materials::get_changes(material_ID0).is_set(Materials::Change::Created));
        EXPECT_TRUE(Materials::get_changes(material_ID1).is_set(Materials::Change::Created));
    }

    Materials::reset_change_notifications();

    { // Test destroy.
        Materials::destroy(material_ID0);
        EXPECT_FALSE(Materials::has(material_ID0));

        Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
        EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 1);

        bool material0_destroyed = false;
        bool material1_destroyed = false;
        for (const MaterialID material_ID : changed_materials) {
            if (material_ID == material_ID0)
                material0_destroyed = Materials::get_changes(material_ID) == Materials::Change::Destroyed;
            if (material_ID == material_ID1)
                material1_destroyed = Materials::get_changes(material_ID) == Materials::Change::Destroyed;
        }

        EXPECT_TRUE(material0_destroyed);
        EXPECT_FALSE(material1_destroyed);
        EXPECT_TRUE(Materials::get_changes(material_ID0).is_set(Materials::Change::Destroyed));
        EXPECT_FALSE(Materials::get_changes(material_ID1).is_set(Materials::Change::Destroyed));
    }

    Materials::reset_change_notifications();

    { // Test that destroyed material cannot be destroyed again.
        Materials::destroy(material_ID0);
        EXPECT_FALSE(Materials::has(material_ID0));

        Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
        EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 0);
    }
}

TEST_F(Assets_Material, change_notifications) {
    Materials::Data data = {};
    Material material = Materials::create("TestMaterial", data);

    // Test that no materials are initially changed and that a creation doesn't trigger a change notification as well.
    Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
    EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 1);
    EXPECT_TRUE(material.get_changes().is_set(Materials::Change::Created));
    EXPECT_FALSE(material.get_changes().is_set(Materials::Change::Updated));

    Materials::reset_change_notifications();

    { // Change tint.
        Math::RGB new_tint = Math::RGB::red();
        material.set_tint(new_tint);

        // Test that only the material has changed.
        for (const MaterialID material_ID : Materials::get_changed_materials()) {
            EXPECT_EQ(material_ID, material.get_ID());
            EXPECT_EQ(Materials::get_tint(material_ID), new_tint);
            EXPECT_TRUE(Materials::get_changes(material_ID).is_set(Materials::Change::Updated));
        }
    }

    {
        Materials::reset_change_notifications();

        // Check that change notifications have been properly reset.
        Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
        EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 0);
        EXPECT_FALSE(material.get_changes().is_set(Materials::Change::Updated));
    }

    { // Change roughness.
        float new_roughness = 0.4f;
        material.set_roughness(new_roughness);

        // Test that only the material has changed.
        for (const MaterialID material_ID : Materials::get_changed_materials()) {
            EXPECT_EQ(material_ID, material.get_ID());
            EXPECT_EQ(Materials::get_roughness(material_ID), new_roughness);
            EXPECT_TRUE(Materials::get_changes(material_ID).is_set(Materials::Change::Updated));
        }
    }
}

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_MATERIAL_TEST_H_
