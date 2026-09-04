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

namespace Bifrost::Assets {

class Assets_Material : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Materials::allocate(8u);
        Textures::allocate(8u);
        Images::allocate(8u);
    }
    virtual void TearDown() {
        Materials::deallocate();
        Textures::deallocate();
        Images::deallocate();
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

TEST_F(Assets_Material, invalid_material_properties) {
    Material invalid_material = Material::invalid();

    EXPECT_FALSE(invalid_material.exists());
    EXPECT_EQ(invalid_material.get_tint(), Math::RGB::red());
    EXPECT_EQ(invalid_material.get_roughness(), 0.0f);
    EXPECT_EQ(invalid_material.get_metallic(), 0.0f);
    EXPECT_EQ(invalid_material.get_specularity(), 0.0f);
    EXPECT_EQ(invalid_material.get_coverage(), 1.0f);
    EXPECT_EQ(invalid_material.get_transmission(), 0.0f);
}

TEST_F(Assets_Material, create) {
    Materials::Data data = {};
    data.tint = Math::RGB::red();
    data.roughness = 0.5f;
    data.metallic = 1.0f;
    data.specularity = 0.04f;
    data.coverage = 0.25f;
    data.transmission = 0.5f;
    Material material = Material("TestMaterial", data);

    EXPECT_TRUE(material.exists());
    EXPECT_EQ(material.get_tint(), Math::RGB::red());
    EXPECT_EQ(material.get_roughness(), 0.5f);
    EXPECT_EQ(material.get_metallic(), 1.0f);
    EXPECT_EQ(material.get_specularity(), 0.04f);
    EXPECT_EQ(material.get_coverage(), 0.25f);
    EXPECT_EQ(material.get_transmission(), 0.5f);

    // Test material created notification.
    Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
    EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 1);
    EXPECT_EQ(material, *changed_materials.begin());
    EXPECT_TRUE(material.get_changes().is_set(Materials::Change::Created));
    EXPECT_FALSE(material.get_changes().is_set(Materials::Change::Updated));
}

TEST_F(Assets_Material, destroy) {
    Materials::Data data = {};
    Material material = Material("TestMaterial", data);
    EXPECT_TRUE(material.exists());

    Materials::reset_change_notifications();

    material.destroy();
    EXPECT_FALSE(material.exists());

    // Test material destroyed notification.
    Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
    EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 1);
    EXPECT_EQ(material, *changed_materials.begin());
    EXPECT_TRUE(material.get_changes().is_set(Materials::Change::Destroyed));
}

TEST_F(Assets_Material, create_and_change) {
    Materials::Data data = {};
    Material material = Material("TestMaterial", data);

    Math::RGB new_tint = Math::RGB::green();
    material.set_tint(new_tint);

    // Test that creating and changing the material creates a single change.
    Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
    EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 1);
    EXPECT_EQ(material, *changed_materials.begin());
    EXPECT_TRUE(material.get_changes().is_set(Materials::Change::Created));
    EXPECT_TRUE(material.get_changes().is_set(Materials::Change::Updated));
}

TEST_F(Assets_Material, create_and_destroy_notifications) {
    Materials::Data data = {};
    Material material0 = Material("TestMaterial0", data);
    Material material1 = Material("TestMaterial1", data);
    EXPECT_TRUE(material0.exists());
    EXPECT_TRUE(material1.exists());

    { // Test material create notifications.
        Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
        EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 2);

        bool material0_created = false;
        bool material1_created = false;
        for (const Material material : changed_materials) {
            if (material == material0)
                material0_created = material.get_changes() == Materials::Change::Created;
            if (material == material1)
                material1_created = material.get_changes() == Materials::Change::Created;
        }

        EXPECT_TRUE(material0_created);
        EXPECT_TRUE(material1_created);
        EXPECT_TRUE(material0.get_changes().is_set(Materials::Change::Created));
        EXPECT_TRUE(material1.get_changes().is_set(Materials::Change::Created));
    }

    Materials::reset_change_notifications();

    { // Test destroy.
        material0.destroy();
        EXPECT_FALSE(material0.exists());

        Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
        EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 1);

        bool material0_destroyed = false;
        bool material1_destroyed = false;
        for (const Material material : changed_materials) {
            if (material == material0)
                material0_destroyed = material.get_changes() == Materials::Change::Destroyed;
            if (material == material1)
                material1_destroyed = material.get_changes() == Materials::Change::Destroyed;
        }

        EXPECT_TRUE(material0_destroyed);
        EXPECT_FALSE(material1_destroyed);
        EXPECT_TRUE(material0.get_changes().is_set(Materials::Change::Destroyed));
        EXPECT_FALSE(material1.get_changes().is_set(Materials::Change::Destroyed));
    }

    Materials::reset_change_notifications();

    { // Test that destroyed material cannot be destroyed again.
        material0.destroy();
        EXPECT_FALSE(material0.exists());

        Core::Iterable<Materials::ChangedIterator> changed_materials = Materials::get_changed_materials();
        EXPECT_EQ(changed_materials.end() - changed_materials.begin(), 0);
    }
}

TEST_F(Assets_Material, change_notifications) {
    Materials::Data data = {};
    Material material = Material("TestMaterial", data);

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
        for (const Material changed_material : Materials::get_changed_materials()) {
            EXPECT_EQ(changed_material, material);
            EXPECT_EQ(changed_material.get_tint(), new_tint);
            EXPECT_TRUE(changed_material.get_changes().is_set(Materials::Change::Updated));
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
        for (const Material changed_material : Materials::get_changed_materials()) {
            EXPECT_EQ(changed_material, material);
            EXPECT_EQ(changed_material.get_roughness(), new_roughness);
            EXPECT_TRUE(changed_material.get_changes().is_set(Materials::Change::Updated));
        }
    }
}

TEST_F(Assets_Material, tint_sampling) {
    using namespace Bifrost::Math;

    RGB base_tint = { 0.1f, 0.5f, 0.9f };
    Vector2f texcoords = { 0.5f, 0.5f };

    { // No texture
        Material no_texture_mat = Material::create_dielectric("no_texture", base_tint, 0.5f);
        EXPECT_FALSE(no_texture_mat.has_tint_texture());
        EXPECT_RGB_EQ(base_tint, no_texture_mat.get_tint(texcoords));
    }

    { // Intensity texture
        float image_tint = 0.5f;
        Image image = Image::create2D("intensity", PixelFormat::Intensity_Float, false, { 1, 1 });
        image.set_pixel(Math::RGB(image_tint), 0);
        Texture texture = Texture::create2D(image);
        Material intensity_textured_mat = Material::create_dielectric("intensity_textured", base_tint, 0.5f);
        intensity_textured_mat.set_tint_roughness_texture(texture);
        EXPECT_TRUE(intensity_textured_mat.has_tint_texture());
        EXPECT_RGB_EQ(base_tint * image_tint, intensity_textured_mat.get_tint(texcoords));
    }

    { // RGB texture
        RGB image_tint = { 0.4f, 0.5f, 1.0f };
        Image image = Image::create2D("color", PixelFormat::RGB_Float, false, { 1, 1 });
        image.set_pixel(image_tint, 0);
        Texture texture = Texture::create2D(image);
        Material color_textured_mat = Material::create_dielectric("color_textured", base_tint, 0.5f);
        color_textured_mat.set_tint_roughness_texture(texture);
        EXPECT_TRUE(color_textured_mat.has_tint_texture());
        EXPECT_RGB_EQ(base_tint * image_tint, color_textured_mat.get_tint(texcoords));
    }

    { // Roughness only texture should not affect tint
        float image_roughness = 0.5f;
        Image image = Image::create2D("roughness", PixelFormat::Roughness8, false, { 1, 1 });
        image.set_pixel(Math::RGBA(RGB::white(), image_roughness), 0);
        Texture texture = Texture::create2D(image);
        Material roughness_textured_mat = Material::create_dielectric("roughness_textured", base_tint, 0.5f);
        roughness_textured_mat.set_tint_roughness_texture(texture);
        EXPECT_FALSE(roughness_textured_mat.has_tint_texture());
        EXPECT_TRUE(roughness_textured_mat.has_roughness_texture());
        EXPECT_RGB_EQ(base_tint, roughness_textured_mat.get_tint(texcoords));
    }
}

TEST_F(Assets_Material, roughness_sampling) {
    using namespace Bifrost::Math;

    float base_roughness = 0.75f;
    Vector2f texcoords = { 0.5f, 0.5f };

    { // No texture
        Material no_texture_mat = Material::create_dielectric("no_texture", RGB::white(), base_roughness);
        EXPECT_FALSE(no_texture_mat.has_roughness_texture());
        EXPECT_FLOAT_EQ(base_roughness, no_texture_mat.get_roughness(texcoords));
    }

    { // Roughness texture
        float image_roughness = 0.5f;
        Image image = Image::create2D("roughness", PixelFormat::Roughness8, false, { 1, 1 });
        image.set_pixel(Math::RGBA(RGB::white(), image_roughness), 0);
        Texture texture = Texture::create2D(image);
        Material roughness_textured_mat = Material::create_dielectric("roughness_textured", RGB::white(), base_roughness);
        roughness_textured_mat.set_tint_roughness_texture(texture);
        EXPECT_TRUE(roughness_textured_mat.has_roughness_texture());
        EXPECT_FLOAT_EQ_EPS(base_roughness * image_roughness, roughness_textured_mat.get_roughness(texcoords), UNorm8::max_precision());
    }

    { // Intensity texture should not affect roughness
        float image_tint = 0.5f;
        Image image = Image::create2D("intensity", PixelFormat::Intensity_Float, false, { 1, 1 });
        image.set_pixel(Math::RGB(image_tint), 0);
        Texture texture = Texture::create2D(image);
        Material intensity_textured_mat = Material::create_dielectric("intensity_textured", RGB::white(), base_roughness);
        intensity_textured_mat.set_tint_roughness_texture(texture);
        EXPECT_TRUE(intensity_textured_mat.has_tint_texture());
        EXPECT_FALSE(intensity_textured_mat.has_roughness_texture());
        EXPECT_FLOAT_EQ(base_roughness, intensity_textured_mat.get_roughness(texcoords));
    }
}

TEST_F(Assets_Material, tint_roughness_sampling) {
    using namespace Bifrost::Math;

    RGB base_tint = { 0.1f, 0.5f, 0.9f };
    float base_roughness = 0.75f;
    RGB image_tint = { 0.4f, 1.0f, 0.2f };
    float image_roughness = 0.5f;
    Vector2f texcoords = { 0.5f, 0.5f };

    Image image = Image::create2D("color", PixelFormat::RGBA_Float, false, { 1, 1 });
    image.set_pixel({ image_tint, image_roughness }, 0);
    Texture texture = Texture::create2D(image);
    Material textured_mat = Material::create_dielectric("color_roughness_textured", base_tint, base_roughness);
    textured_mat.set_tint_roughness_texture(texture);
    
    EXPECT_TRUE(textured_mat.has_tint_texture());
    EXPECT_TRUE(textured_mat.has_roughness_texture());

    // Sample properties individually
    EXPECT_RGB_EQ(base_tint * image_tint, textured_mat.get_tint(texcoords));
    EXPECT_FLOAT_EQ(base_roughness * image_roughness, textured_mat.get_roughness(texcoords));

    // Sample properties at the same time.
    auto [actual_tint, actual_roughness] = textured_mat.get_tint_roughness(texcoords);
    EXPECT_RGB_EQ(base_tint * image_tint, actual_tint);
    EXPECT_FLOAT_EQ(base_roughness * image_roughness, actual_roughness);
}

} // NS Bifrost::Assets

#endif // _BIFROST_ASSETS_MATERIAL_TEST_H_
