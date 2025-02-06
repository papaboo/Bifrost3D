// Test Bifrost textures.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_TEXTURE_TEST_H_
#define _BIFROST_ASSETS_TEXTURE_TEST_H_

#include <Bifrost/Assets/Texture.h>
#include <Expects.h>

namespace Bifrost {
namespace Assets {

class Assets_Textures : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Textures::allocate(8u);
        Images::allocate(1u);
    }
    virtual void TearDown() {
        Images::deallocate();
        Textures::deallocate();
    }
};

TEST_F(Assets_Textures, resizing) {
    // Test that capacity can be increased.
    unsigned int largerCapacity = Textures::capacity() + 4u;
    Textures::reserve(largerCapacity);
    EXPECT_GE(Textures::capacity(), largerCapacity);

    // Test that capacity won't be decreased.
    Textures::reserve(5u);
    EXPECT_GE(Textures::capacity(), largerCapacity);

    Textures::deallocate();
    EXPECT_LT(Textures::capacity(), largerCapacity);
}

TEST_F(Assets_Textures, invalid_texture_properties) {
    Texture invalid_texture = Texture::invalid();

    EXPECT_FALSE(invalid_texture.exists());
    EXPECT_EQ(invalid_texture.get_image(), Image::invalid());
}

TEST_F(Assets_Textures, create) {
    Image image = Image::create2D("Test image", PixelFormat::RGBA32, 2.2f, Math::Vector2ui(3, 3));
    Texture texture = Texture::create2D(image);

    EXPECT_TRUE(texture.exists());
    EXPECT_EQ(image, texture.get_image());

    // Test texture created notification.
    Core::Iterable<Textures::ChangedIterator> changed_textures = Textures::get_changed_textures();
    EXPECT_EQ(changed_textures.end() - changed_textures.begin(), 1);
    EXPECT_EQ(texture, *changed_textures.begin());
    EXPECT_EQ(texture.get_changes(), Textures::Change::Created);
}

TEST_F(Assets_Textures, destroy) {
    Image image = Image::create2D("Test image", PixelFormat::RGBA32, 2.2f, Math::Vector2ui(1, 1));

    Texture texture = Texture::create2D(image);
    EXPECT_TRUE(texture.exists());

    Textures::reset_change_notifications();

    texture.destroy();
    EXPECT_FALSE(texture.exists());

    // Test texture destroyed notification.
    Core::Iterable<Textures::ChangedIterator> changed_textures = Textures::get_changed_textures();
    EXPECT_EQ(changed_textures.end() - changed_textures.begin(), 1);
    EXPECT_EQ(texture, *changed_textures.begin());
    EXPECT_EQ(texture.get_changes(), Textures::Change::Destroyed);
}

TEST_F(Assets_Textures, create_and_destroy_notifications) {
    Image image = Image::create2D("Test image", PixelFormat::RGBA32, 2.2f, Math::Vector2ui(1, 1));

    Texture texture0 = Texture::create2D(image);
    Texture texture1 = Texture::create2D(image);
    EXPECT_TRUE(texture0.exists());
    EXPECT_TRUE(texture1.exists());

    { // Test texture create notifications.
        Core::Iterable<Textures::ChangedIterator> changed_textures = Textures::get_changed_textures();
        EXPECT_EQ(changed_textures.end() - changed_textures.begin(), 2);

        bool texture0_created = false;
        bool texture1_created = false;
        bool other_event = false;
        for (const Texture texture : changed_textures) {
            bool texture_created = texture.get_changes() == Textures::Change::Created;
            if (texture == texture0 && texture_created)
                texture0_created = true;
            else if (texture == texture1 && texture_created)
                texture1_created = true;
            else
                other_event = true;
        }

        EXPECT_TRUE(texture0_created);
        EXPECT_TRUE(texture1_created);
        EXPECT_FALSE(other_event);
    }

    Textures::reset_change_notifications();

    { // Test destroy.
        texture0.destroy();
        EXPECT_FALSE(texture0.exists());

        Core::Iterable<Textures::ChangedIterator> changed_textures = Textures::get_changed_textures();
        EXPECT_EQ(changed_textures.end() - changed_textures.begin(), 1);

        bool texture0_destroyed = false;
        bool other_change = false;
        for (const Texture texture : changed_textures) {
            if (texture == texture0 && texture.get_changes() == Textures::Change::Destroyed)
                texture0_destroyed = true;
            else
                other_change = true;
        }

        EXPECT_TRUE(texture0_destroyed);
        EXPECT_FALSE(other_change);
    }

    Textures::reset_change_notifications();

    { // Test that destroyed texture cannot be destroyed again.
        EXPECT_FALSE(texture0.exists());

        texture0.destroy();
        EXPECT_FALSE(texture0.exists());
        EXPECT_TRUE(Textures::get_changed_textures().is_empty());
    }
}

TEST_F(Assets_Textures, sample2D) {
    using namespace Bifrost::Math;

    unsigned int size = 4;
    Image image = Image::create2D("Test", PixelFormat::RGBA_Float, 1.0f, Vector2ui(size));
    for (unsigned int y = 0; y < size; ++y)
        for (unsigned int x = 0; x < size; ++x)
            image.set_pixel(RGBA(x / float(size), y / float(size), 0, 1), Vector2ui(x, y));

    { // Test with no filter and clamp.
        Texture texture = Texture::create2D(image, MagnificationFilter::None, MinificationFilter::None, WrapMode::Clamp, WrapMode::Clamp);

        { // Sample lower left corner.
            RGBA color = sample2D(texture, Vector2f(0, 0));
            EXPECT_RGBA_EQ(RGBA(0, 0, 0, 1), color);
        }

        { // Sample upper right corner.
            RGBA color = sample2D(texture, Vector2f(1, 1));
            EXPECT_RGBA_EQ(RGBA(0.75f, 0.75f, 0, 1), color);
        }

        { // Sample outside upper left corner.
            RGBA color = sample2D(texture, Vector2f(-2, 2));
            EXPECT_RGBA_EQ(RGBA(0, 0.75f, 0, 1), color);
        }

        { // Sample outside lower right corner.
            RGBA color = sample2D(texture, Vector2f(2, -2));
            EXPECT_RGBA_EQ(RGBA(0.75f, 0, 0, 1), color);
        }
    }

    { // No filter and repeat.
        // Test that pixel borders are where they are expected to beby sampling around (2, 2).
        Texture texture = Texture::create2D(image, MagnificationFilter::None, MinificationFilter::None, WrapMode::Repeat, WrapMode::Repeat);

        RGBA color = sample2D(texture, Vector2f(0.49f, 0.49f));
        EXPECT_RGBA_EQ(RGBA(0.25f, 0.25f, 0, 1), color);

        color = sample2D(texture, Vector2f(0.49f, 0.51f));
        EXPECT_RGBA_EQ(RGBA(0.25f, 0.5f, 0, 1), color);

        color = sample2D(texture, Vector2f(0.51f, 0.49f));
        EXPECT_RGBA_EQ(RGBA(0.5f, 0.25f, 0, 1), color);

        color = sample2D(texture, Vector2f(0.51f, 0.51f));
        EXPECT_RGBA_EQ(RGBA(0.5f, 0.5f, 0, 1), color);
    }

    { // Test with linear filtering and repeat wrap mode.
        Texture texture = Texture::create2D(image, MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Repeat);

        { // Sample lower left corner.
            RGBA color = sample2D(texture, Vector2f(0.125f, 0.125f));
            EXPECT_RGBA_EQ(RGBA(0, 0, 0, 1), color);

            color = sample2D(texture, Vector2f(0.125f, 1.125f));
            EXPECT_RGBA_EQ(RGBA(0, 0, 0, 1), color);

            color = sample2D(texture, Vector2f(0.125f, -0.875f));
            EXPECT_RGBA_EQ(RGBA(0, 0, 0, 1), color);

            color = sample2D(texture, Vector2f(0.125f, -0.9375f));
            EXPECT_RGBA_EQ(RGBA(0, 0.1875f, 0, 1), color);
        }

        { // Sample upper left corner.
            RGBA color = sample2D(texture, Vector2f(0.125f, 0.875f));
            EXPECT_RGBA_EQ(RGBA(0, 0.75f, 0, 1), color);

            color = sample2D(texture, Vector2f(0.125f, -0.125f));
            EXPECT_RGBA_EQ(RGBA(0, 0.75f, 0, 1), color);

            color = sample2D(texture, Vector2f(0.3125f, -0.125f));
            EXPECT_RGBA_EQ(RGBA(0.1875f, 0.75f, 0, 1), color);
        }
    }
}

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_TEXTURE_TEST_H_
