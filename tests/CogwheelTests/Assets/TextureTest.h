// Test Cogwheel textures.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_TEXTURE_TEST_H_
#define _COGWHEEL_ASSETS_TEXTURE_TEST_H_

#include <Cogwheel/Assets/Texture.h>
#include <Expects.h>

namespace Cogwheel {
namespace Assets {

class Assets_Textures : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Textures::allocate(8u);
    }
    virtual void TearDown() {
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

TEST_F(Assets_Textures, sentinel_mesh) {
    Textures::UID sentinel_ID = Textures::UID::invalid_UID();

    EXPECT_FALSE(Textures::has(sentinel_ID));
    EXPECT_EQ(Textures::get_image_ID(sentinel_ID), Images::UID::invalid_UID());
}

TEST_F(Assets_Textures, create) {
    Images::allocate(1u);
    Images::UID image_ID = Images::create("Test image", PixelFormat::RGBA32, 2.2f, Math::Vector2ui(3, 3));
    Textures::UID texture_ID = Textures::create2D(image_ID);

    EXPECT_TRUE(Textures::has(texture_ID));
    EXPECT_EQ(Textures::get_image_ID(texture_ID), image_ID);

    // Test texture created notification.
    Core::Iterable<Textures::ChangedIterator> changed_textures = Textures::get_changed_textures();
    EXPECT_EQ(changed_textures.end() - changed_textures.begin(), 1);
    EXPECT_EQ(*changed_textures.begin(), texture_ID);
    EXPECT_EQ(Textures::get_changes(texture_ID), Textures::Changes::Created);

    Images::deallocate();
}

TEST_F(Assets_Textures, destroy) {
    Textures::UID texture_ID = Textures::create2D(Images::UID::invalid_UID());
    EXPECT_TRUE(Textures::has(texture_ID));

    Textures::reset_change_notifications();

    Textures::destroy(texture_ID);
    EXPECT_FALSE(Textures::has(texture_ID));

    // Test texture destroyed notification.
    Core::Iterable<Textures::ChangedIterator> changed_textures = Textures::get_changed_textures();
    EXPECT_EQ(changed_textures.end() - changed_textures.begin(), 1);
    EXPECT_EQ(*changed_textures.begin(), texture_ID);
    EXPECT_EQ(Textures::get_changes(texture_ID), Textures::Changes::Destroyed);
}

TEST_F(Assets_Textures, create_and_destroy_notifications) {
    Textures::UID texture_ID0 = Textures::create2D(Images::UID::invalid_UID());
    Textures::UID texture_ID1 = Textures::create2D(Images::UID::invalid_UID());
    EXPECT_TRUE(Textures::has(texture_ID0));
    EXPECT_TRUE(Textures::has(texture_ID1));

    { // Test texture create notifications.
        Core::Iterable<Textures::ChangedIterator> changed_textures = Textures::get_changed_textures();
        EXPECT_EQ(changed_textures.end() - changed_textures.begin(), 2);

        bool texture0_created = false;
        bool texture1_created = false;
        bool other_event = false;
        for (const Textures::UID texture_ID : changed_textures) {
            bool texture_created = Textures::get_changes(texture_ID) == Textures::Changes::Created;
            if (texture_ID == texture_ID0 && texture_created)
                texture0_created = true;
            else if (texture_ID == texture_ID1 && texture_created)
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
        Textures::destroy(texture_ID0);
        EXPECT_FALSE(Textures::has(texture_ID0));

        Core::Iterable<Textures::ChangedIterator> changed_textures = Textures::get_changed_textures();
        EXPECT_EQ(changed_textures.end() - changed_textures.begin(), 1);

        bool texture0_destroyed = false;
        bool other_change = false;
        for (const Textures::UID texture_ID : changed_textures) {
            if (texture_ID == texture_ID0 && Textures::get_changes(texture_ID) == Textures::Changes::Destroyed)
                texture0_destroyed = true;
            else
                other_change = true;
        }

        EXPECT_TRUE(texture0_destroyed);
        EXPECT_FALSE(other_change);
    }

    Textures::reset_change_notifications();

    { // Test that destroyed texture cannot be destroyed again.
        EXPECT_FALSE(Textures::has(texture_ID0));

        Textures::destroy(texture_ID0);
        EXPECT_FALSE(Textures::has(texture_ID0));
        EXPECT_TRUE(Textures::get_changed_textures().is_empty());
    }
}

TEST_F(Assets_Textures, sample2D) {
    using namespace Cogwheel::Math;

    Images::allocate(1u);

    Image image = Images::create("Test", PixelFormat::RGBA_Float, 1.0f, Vector2ui(2, 2));
    image.set_pixel(RGBA(0, 0, 0, 1), Vector2ui(0, 0));
    image.set_pixel(RGBA(0, 1, 0, 1), Vector2ui(0, 1));
    image.set_pixel(RGBA(1, 0, 0, 1), Vector2ui(1, 0));
    image.set_pixel(RGBA(1, 1, 0, 1), Vector2ui(1, 1));

    { // Test with no filter and clamp.
        Textures::UID texture_ID = Textures::create2D(image.get_ID(), MagnificationFilter::None, MinificationFilter::None, WrapMode::Clamp, WrapMode::Clamp);

        { // Sample lower left corner.
            RGBA color = sample2D(texture_ID, Vector2f(0, 0));
            EXPECT_RGBA_EQ(RGBA(0, 0, 0, 1), color);
        }

        { // Sample upper right corner.
            RGBA color = sample2D(texture_ID, Vector2f(1, 1));
            EXPECT_RGBA_EQ(RGBA(1, 1, 0, 1), color);
        }

        { // Sample outside upper left corner.
            RGBA color = sample2D(texture_ID, Vector2f(-2, 2));
            EXPECT_RGBA_EQ(RGBA(0, 1, 0, 1), color);
        }

        { // Sample outside lower right corner.
            RGBA color = sample2D(texture_ID, Vector2f(2, -2));
            EXPECT_RGBA_EQ(RGBA(1, 0, 0, 1), color);
        }
    }

    { // Test with linear filtering and repeat wrap mode.
        Textures::UID texture_ID = Textures::create2D(image.get_ID());

        { // Sample lower left corner.
            RGBA color = sample2D(texture_ID, Vector2f(0.25f, 0.25f));
            EXPECT_RGBA_EQ(RGBA(0, 0, 0, 1), color);

            color = sample2D(texture_ID, Vector2f(0.25f, 1.25f));
            EXPECT_RGBA_EQ(RGBA(0, 0, 0, 1), color);

            color = sample2D(texture_ID, Vector2f(0.25f, -0.75f));
            EXPECT_RGBA_EQ(RGBA(0, 0, 0, 1), color);

            color = sample2D(texture_ID, Vector2f(0.25f, -0.95f));
            EXPECT_RGBA_EQ(RGBA(0, 0.4f, 0, 1), color);
        }

        { // Sample upper left corner.
            RGBA color = sample2D(texture_ID, Vector2f(0.25f, 0.75f));
            EXPECT_RGBA_EQ(RGBA(0, 1, 0, 1), color);

            color = sample2D(texture_ID, Vector2f(0.25f, -0.25f));
            EXPECT_RGBA_EQ(RGBA(0, 1, 0, 1), color);

            color = sample2D(texture_ID, Vector2f(0.40f, -0.25f));
            EXPECT_RGBA_EQ(RGBA(0.3f, 1, 0, 1), color);
        }
    }

    Images::deallocate();
}

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_TEXTURE_TEST_H_