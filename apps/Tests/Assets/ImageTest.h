// Test Cogwheel images.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_IMAGE_TEST_H_
#define _COGWHEEL_ASSETS_IMAGE_TEST_H_

#include <Cogwheel/Assets/Image.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Assets {

class Assets_Images : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Images::allocate(8u);
    }
    virtual void TearDown() {
        Images::deallocate();
    }
};

TEST_F(Assets_Images, resizing) {
    // Test that capacity can be increased.
    unsigned int largerCapacity = Images::capacity() + 4u;
    Images::reserve(largerCapacity);
    EXPECT_GE(Images::capacity(), largerCapacity);

    // Test that capacity won't be decreased.
    Images::reserve(5u);
    EXPECT_GE(Images::capacity(), largerCapacity);

    Images::deallocate();
    EXPECT_LT(Images::capacity(), largerCapacity);
}

TEST_F(Assets_Images, sentinel_material) {
    Images::UID sentinel_ID = Images::UID::invalid_UID();

    EXPECT_FALSE(Images::has(sentinel_ID));
    EXPECT_EQ(Images::get_pixel_format(sentinel_ID), PixelFormat::Unknown);
    EXPECT_EQ(Images::get_mipmap_count(sentinel_ID), 0u);
    EXPECT_EQ(Images::get_width(sentinel_ID), 1u);
    EXPECT_EQ(Images::get_height(sentinel_ID), 1u);
    EXPECT_EQ(Images::get_depth(sentinel_ID), 1u);
    EXPECT_EQ(Images::get_pixels(sentinel_ID), nullptr);
}

TEST_F(Assets_Images, create) {
    Images::UID image_ID = Images::create("Test image", PixelFormat::RGBA32, Math::Vector3ui(1,2,3));

    EXPECT_TRUE(Images::has(image_ID));
    EXPECT_EQ(Images::get_pixel_format(image_ID), PixelFormat::RGBA32);
    EXPECT_EQ(Images::get_mipmap_count(image_ID), 1u);
    EXPECT_EQ(Images::get_width(image_ID), 1u);
    EXPECT_EQ(Images::get_height(image_ID), 2u);
    EXPECT_EQ(Images::get_depth(image_ID), 3u);
    EXPECT_NE(Images::get_pixels(image_ID), nullptr);

    // Test image created notification.
    Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
    EXPECT_EQ(changed_images.end() - changed_images.begin(), 1);
    EXPECT_EQ(*changed_images.begin(), image_ID);
    EXPECT_TRUE(Images::has_changes(image_ID, Images::Changes::Created));
    EXPECT_FALSE(Images::has_changes(image_ID, Images::Changes::PixelsUpdated));
}

TEST_F(Assets_Images, destroy) {
    Images::UID image_ID = Images::create("Test image", PixelFormat::RGBA32, Math::Vector3ui(1, 2, 3));
    EXPECT_TRUE(Images::has(image_ID));

    Images::reset_change_notifications();

    Images::destroy(image_ID);
    EXPECT_FALSE(Images::has(image_ID));

    // Test image destroyed notification.
    Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
    EXPECT_EQ(changed_images.end() - changed_images.begin(), 1);
    EXPECT_EQ(*changed_images.begin(), image_ID);
    EXPECT_TRUE(Images::has_changes(image_ID, Images::Changes::Destroyed));
}

TEST_F(Assets_Images, create_and_destroy_notifications) {
    Images::UID image_ID0 = Images::create("Test image 0", PixelFormat::RGBA32, Math::Vector3ui(1, 2, 3));
    Images::UID image_ID1 = Images::create("Test image 1", PixelFormat::RGBA32, Math::Vector3ui(3, 2, 1));
    EXPECT_TRUE(Images::has(image_ID0));
    EXPECT_TRUE(Images::has(image_ID1));

    { // Test image create notifications.
        Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
        EXPECT_EQ(changed_images.end() - changed_images.begin(), 2);

        bool image0_created = false;
        bool image1_created = false;
        bool other_events = false;
        for (const Images::UID image_ID : changed_images) {
            bool image_created = Images::get_changes(image_ID) == Images::Changes::Created;
            if (image_ID == image_ID0 && image_created)
                image0_created = true;
            else if (image_ID == image_ID1 && image_created)
                image1_created = true;
            else
                other_events = true;
        }

        EXPECT_TRUE(image0_created);
        EXPECT_TRUE(image1_created);
        EXPECT_FALSE(other_events);
    }

    Images::reset_change_notifications();

    { // Test destroy.
        Images::destroy(image_ID0);
        EXPECT_FALSE(Images::has(image_ID0));

        Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
        EXPECT_EQ(changed_images.end() - changed_images.begin(), 1);

        bool image0_destroyed = false;
        bool other_events = false;
        for (const Images::UID image_ID : changed_images) {
            if (image_ID == image_ID0 && Images::get_changes(image_ID) == Images::Changes::Destroyed)
                image0_destroyed = true;
            else
                other_events = true;
        }

        EXPECT_TRUE(image0_destroyed);
        EXPECT_FALSE(other_events);
        EXPECT_TRUE(Images::has_changes(image_ID0, Images::Changes::Destroyed));
        EXPECT_FALSE(Images::has_changes(image_ID1, Images::Changes::Destroyed));
    }

    Images::reset_change_notifications();

    { // Test that destroyed images cannot be destroyed again.
        EXPECT_FALSE(Images::has(image_ID0));
        
        Images::destroy(image_ID0);
        EXPECT_FALSE(Images::has(image_ID0));
        EXPECT_TRUE(Images::get_changed_images().is_empty());
    }
}

TEST_F(Assets_Images, create_and_change) {
    Images::UID image_ID = Images::create("Test image", PixelFormat::RGBA32, Math::Vector3ui(1, 2, 3));

    Images::set_pixel(image_ID, Math::RGBA::yellow(), Math::Vector2ui(0,1));

    // Test that creating and changing an image creates a single change.
    Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
    EXPECT_EQ(changed_images.end() - changed_images.begin(), 1);
    EXPECT_EQ(*changed_images.begin(), image_ID);
    EXPECT_TRUE(Images::has_changes(image_ID, Images::Changes::Created));
    EXPECT_TRUE(Images::has_changes(image_ID, Images::Changes::PixelsUpdated));
}

TEST_F(Assets_Images, pixel_updates) {
    Images::UID image_ID = Images::create("Test image", PixelFormat::RGBA_Float, Math::Vector2ui(3, 2));

    Images::set_pixel(image_ID, Math::RGBA(1, 2, 3, 1), Math::Vector2ui(0, 0));
    Images::set_pixel(image_ID, Math::RGBA(4, 5, 6, 1), Math::Vector2ui(1, 0));
    Images::set_pixel(image_ID, Math::RGBA(7, 8, 9, 1), Math::Vector2ui(2, 0));
    Images::set_pixel(image_ID, Math::RGBA(11, 12, 13, 1), Math::Vector2ui(0, 1));
    Images::set_pixel(image_ID, Math::RGBA(14, 15, 16, 1), Math::Vector2ui(1, 1));
    Images::set_pixel(image_ID, Math::RGBA(17, 18, 19, 1), Math::Vector2ui(2, 1));

    // Test that editing multiple pixels create a single change.
    Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
    EXPECT_EQ(changed_images.end() - changed_images.begin(), 1);

    // Test that the pixels have the correct colors.
    EXPECT_EQ(Images::get_pixel(image_ID, Math::Vector2ui(0, 0)), Math::RGBA(1, 2, 3, 1));
    EXPECT_EQ(Images::get_pixel(image_ID, Math::Vector2ui(1, 0)), Math::RGBA(4, 5, 6, 1));
    EXPECT_EQ(Images::get_pixel(image_ID, Math::Vector2ui(2, 0)), Math::RGBA(7, 8, 9, 1));
    EXPECT_EQ(Images::get_pixel(image_ID, Math::Vector2ui(0, 1)), Math::RGBA(11, 12, 13, 1));
    EXPECT_EQ(Images::get_pixel(image_ID, Math::Vector2ui(1, 1)), Math::RGBA(14, 15, 16, 1));
    EXPECT_EQ(Images::get_pixel(image_ID, Math::Vector2ui(2, 1)), Math::RGBA(17, 18, 19, 1));
}

TEST_F(Assets_Images, mipmap_size) {
    unsigned int mipmap_count = 4u;
    Images::UID image_ID = Images::create("Test image", PixelFormat::RGBA32, Math::Vector2ui(8, 6), mipmap_count);

    EXPECT_EQ(Images::get_mipmap_count(image_ID), mipmap_count);

    EXPECT_EQ(Images::get_width(image_ID, 0), 8u);
    EXPECT_EQ(Images::get_width(image_ID, 1), 4u);
    EXPECT_EQ(Images::get_width(image_ID, 2), 2u);
    EXPECT_EQ(Images::get_width(image_ID, 3), 1u);

    EXPECT_EQ(Images::get_height(image_ID, 0), 6u);
    EXPECT_EQ(Images::get_height(image_ID, 1), 3u);
    EXPECT_EQ(Images::get_height(image_ID, 2), 1u);
    EXPECT_EQ(Images::get_height(image_ID, 3), 1u);

    EXPECT_EQ(Images::get_depth(image_ID, 0), 1u);
    EXPECT_EQ(Images::get_depth(image_ID, 1), 1u);
    EXPECT_EQ(Images::get_depth(image_ID, 2), 1u);
    EXPECT_EQ(Images::get_depth(image_ID, 3), 1u);
}

} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_IMAGE_TEST_H_