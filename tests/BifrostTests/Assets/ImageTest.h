// Test Bifrost images.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_IMAGE_TEST_H_
#define _BIFROST_ASSETS_IMAGE_TEST_H_

#include <Bifrost/Assets/Image.h>
#include <Expects.h>

namespace Bifrost {
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
    ImageID sentinel_ID = ImageID::invalid_UID();

    EXPECT_FALSE(Images::has(sentinel_ID));
    EXPECT_EQ(PixelFormat::Unknown, Images::get_pixel_format(sentinel_ID));
    EXPECT_EQ(0u, Images::get_mipmap_count(sentinel_ID));
    EXPECT_EQ(1u, Images::get_width(sentinel_ID));
    EXPECT_EQ(1u, Images::get_height(sentinel_ID));
    EXPECT_EQ(1u, Images::get_depth(sentinel_ID));
    EXPECT_EQ(nullptr, Images::get_pixels(sentinel_ID));
}

TEST_F(Assets_Images, create) {
    ImageID image_ID = Images::create3D("Test image", PixelFormat::RGBA32, 2.2f, Math::Vector3ui(1,2,3), 2);

    EXPECT_TRUE(Images::has(image_ID));
    EXPECT_EQ(PixelFormat::RGBA32, Images::get_pixel_format(image_ID));
    EXPECT_EQ(2.2f, Images::get_gamma(image_ID));
    EXPECT_EQ(2u, Images::get_mipmap_count(image_ID));
    EXPECT_EQ(1u, Images::get_width(image_ID));
    EXPECT_EQ(2u, Images::get_height(image_ID));
    EXPECT_EQ(3u, Images::get_depth(image_ID));
    EXPECT_NE(nullptr, Images::get_pixels(image_ID));

    // Test image created notification.
    Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
    EXPECT_EQ(1u, changed_images.end() - changed_images.begin());
    EXPECT_EQ(image_ID, *changed_images.begin());
    EXPECT_TRUE(Images::get_changes(image_ID).is_set(Images::Change::Created));
    EXPECT_FALSE(Images::get_changes(image_ID).is_set(Images::Change::PixelsUpdated));
}

TEST_F(Assets_Images, destroy) {
    ImageID image_ID = Images::create3D("Test image", PixelFormat::RGBA32, 2.2f, Math::Vector3ui(1, 2, 3));
    EXPECT_TRUE(Images::has(image_ID));

    Images::reset_change_notifications();

    Images::destroy(image_ID);
    EXPECT_FALSE(Images::has(image_ID));

    // Test image destroyed notification.
    Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
    EXPECT_EQ(1u, changed_images.end() - changed_images.begin());
    EXPECT_EQ(image_ID, *changed_images.begin());
    EXPECT_EQ(Images::Change::Destroyed, Images::get_changes(image_ID));
}

TEST_F(Assets_Images, create_and_destroy_notifications) {
    ImageID image_ID0 = Images::create3D("Test image 0", PixelFormat::RGBA32, 2.2f, Math::Vector3ui(1, 2, 3));
    ImageID image_ID1 = Images::create3D("Test image 1", PixelFormat::RGBA32, 2.2f, Math::Vector3ui(3, 2, 1));
    EXPECT_TRUE(Images::has(image_ID0));
    EXPECT_TRUE(Images::has(image_ID1));

    { // Test image create notifications.
        EXPECT_EQ(Images::Change::Created, Images::get_changes(image_ID0));
        EXPECT_EQ(Images::Change::Created, Images::get_changes(image_ID1));
        
        Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
        EXPECT_EQ(2u, changed_images.end() - changed_images.begin());

        bool image0_created = false;
        bool image1_created = false;
        bool other_events = false;
        for (const ImageID image_ID : changed_images) {
            bool image_created = Images::get_changes(image_ID) == Images::Change::Created;
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
        EXPECT_EQ(1u, changed_images.end() - changed_images.begin());

        bool image0_destroyed = false;
        bool other_events = false;
        for (const ImageID image_ID : changed_images) {
            if (image_ID == image_ID0 && Images::get_changes(image_ID) == Images::Change::Destroyed)
                image0_destroyed = true;
            else
                other_events = true;
        }

        EXPECT_TRUE(image0_destroyed);
        EXPECT_FALSE(other_events);
        EXPECT_EQ(Images::Change::Destroyed, Images::get_changes(image_ID0));
        EXPECT_FALSE(Images::get_changes(image_ID1).is_set(Images::Change::Destroyed));
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
    ImageID image_ID = Images::create3D("Test image", PixelFormat::RGBA32, 2.2f, Math::Vector3ui(1, 2, 3));

    Images::set_pixel(image_ID, Math::RGBA::yellow(), Math::Vector2ui(0,1));

    // Test that creating and changing an image creates a single change.
    Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
    EXPECT_EQ(1u, changed_images.end() - changed_images.begin());
    EXPECT_EQ(image_ID, *changed_images.begin());
    EXPECT_TRUE(Images::get_changes(image_ID).is_set(Images::Change::Created));
    EXPECT_TRUE(Images::get_changes(image_ID).is_set(Images::Change::PixelsUpdated));
}

TEST_F(Assets_Images, pixel_updates) {
    ImageID image_ID = Images::create2D("Test image", PixelFormat::RGBA_Float, 2.2f, Math::Vector2ui(3, 2), 2);

    Images::set_pixel(image_ID, Math::RGBA(1, 2, 3, 1), Math::Vector2ui(0, 0));
    Images::set_pixel(image_ID, Math::RGBA(4, 5, 6, 1), Math::Vector2ui(1, 0));
    Images::set_pixel(image_ID, Math::RGBA(7, 8, 9, 1), Math::Vector2ui(2, 0));
    Images::set_pixel(image_ID, Math::RGBA(11, 12, 13, 1), Math::Vector2ui(0, 1));
    Images::set_pixel(image_ID, Math::RGBA(14, 15, 16, 1), Math::Vector2ui(1, 1));
    Images::set_pixel(image_ID, Math::RGBA(17, 18, 19, 1), Math::Vector2ui(2, 1));
    Images::set_pixel(image_ID, Math::RGBA(20, 21, 22, 1), Math::Vector2ui(0, 0), 1);

    // Test that editing multiple pixels create a single change.
    Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
    EXPECT_EQ(1u, changed_images.end() - changed_images.begin());

    // Test that the pixels have the correct colors.
    EXPECT_RGBA_EQ(Math::RGBA(1, 2, 3, 1), Images::get_pixel(image_ID, Math::Vector2ui(0, 0)));
    EXPECT_RGBA_EQ(Math::RGBA(4, 5, 6, 1), Images::get_pixel(image_ID, Math::Vector2ui(1, 0)));
    EXPECT_RGBA_EQ(Math::RGBA(7, 8, 9, 1), Images::get_pixel(image_ID, Math::Vector2ui(2, 0)));
    EXPECT_RGBA_EQ(Math::RGBA(11, 12, 13, 1), Images::get_pixel(image_ID, Math::Vector2ui(0, 1)));
    EXPECT_RGBA_EQ(Math::RGBA(14, 15, 16, 1), Images::get_pixel(image_ID, Math::Vector2ui(1, 1)));
    EXPECT_RGBA_EQ(Math::RGBA(17, 18, 19, 1), Images::get_pixel(image_ID, Math::Vector2ui(2, 1)));
    EXPECT_RGBA_EQ(Math::RGBA(20, 21, 22, 1), Images::get_pixel(image_ID, Math::Vector2ui(0, 0), 1));
}

TEST_F(Assets_Images, mipmap_size) {
    unsigned int mipmap_count = 4u;
    ImageID image_ID = Images::create2D("Test image", PixelFormat::RGBA32, 2.2f, Math::Vector2ui(8, 6), mipmap_count);

    EXPECT_EQ(mipmap_count, Images::get_mipmap_count(image_ID));

    EXPECT_EQ(8u, Images::get_width(image_ID, 0));
    EXPECT_EQ(4u, Images::get_width(image_ID, 1));
    EXPECT_EQ(2u, Images::get_width(image_ID, 2));
    EXPECT_EQ(1u, Images::get_width(image_ID, 3));

    EXPECT_EQ(6u, Images::get_height(image_ID, 0));
    EXPECT_EQ(3u, Images::get_height(image_ID, 1));
    EXPECT_EQ(1u, Images::get_height(image_ID, 2));
    EXPECT_EQ(1u, Images::get_height(image_ID, 3));

    EXPECT_EQ(1u, Images::get_depth(image_ID, 0));
    EXPECT_EQ(1u, Images::get_depth(image_ID, 1));
    EXPECT_EQ(1u, Images::get_depth(image_ID, 2));
    EXPECT_EQ(1u, Images::get_depth(image_ID, 3));
}

TEST_F(Assets_Images, mipmapable_events) {

    unsigned int width = 2, height = 2;
    Image image = Images::create2D("Test image", PixelFormat::RGBA_Float, 1.0f, Math::Vector2ui(width, height));
    EXPECT_FALSE(image.is_mipmapable());

    Images::reset_change_notifications();

    { // Test mipmapable change event.
        image.set_mipmapable(true);
        Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
        EXPECT_EQ(1u, changed_images.end() - changed_images.begin());
        Image changed_image = *changed_images.begin();
        EXPECT_EQ(changed_image, image);
        EXPECT_EQ(Images::Change::Mipmapable, changed_image.get_changes());
    }

    { // Test that destroying an image keep the mipmapable change event.
        Images::destroy(image.get_ID());
        Core::Iterable<Images::ChangedIterator> changed_images = Images::get_changed_images();
        EXPECT_EQ(1u, changed_images.end() - changed_images.begin());
        Image destroyed_image = *changed_images.begin();
        EXPECT_EQ(destroyed_image, image);
        Core::Bitmask<Images::Change> mipmapable_destroyed = { Images::Change::Mipmapable, Images::Change::Destroyed };
        EXPECT_EQ(mipmapable_destroyed, destroyed_image.get_changes());
    }
}

// ------------------------------------------------------------------------------------------------
// Image utils tests.
// ------------------------------------------------------------------------------------------------

class Assets_ImageUtils : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Images::allocate(8u);
    }
    virtual void TearDown() {
        Images::deallocate();
    }
};

TEST_F(Assets_ImageUtils, fill_mipmaps) {
    using namespace Bifrost::Math;

    unsigned int width = 7, height = 5, mipmap_count = 3u;
    Image image = Images::create2D("Test image", PixelFormat::RGBA_Float, 1.0f, Vector2ui(width, height), mipmap_count);

    EXPECT_EQ(7u, image.get_width(0));
    EXPECT_EQ(3u, image.get_width(1));
    EXPECT_EQ(1u, image.get_width(2));

    EXPECT_EQ(5u, image.get_height(0));
    EXPECT_EQ(2u, image.get_height(1));
    EXPECT_EQ(1u, image.get_height(2));

    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x)
            image.set_pixel(RGBA(float(x), float(y), 0.0f, 1.0f), Vector2ui(x, y));

    ImageUtils::fill_mipmap_chain(image.get_ID());

    EXPECT_RGBA_EQ(RGBA(0.5f, 0.5f, 0.0f, 1.0f), image.get_pixel(Vector2ui(0, 0), 1));
    EXPECT_RGBA_EQ(RGBA(2.5f, 0.5f, 0.0f, 1.0f), image.get_pixel(Vector2ui(1, 0), 1));
    EXPECT_RGBA_EQ(RGBA(5.0f, 0.5f, 0.0f, 1.0f), image.get_pixel(Vector2ui(2, 0), 1));
    EXPECT_RGBA_EQ(RGBA(0.5f, 3.0f, 0.0f, 1.0f), image.get_pixel(Vector2ui(0, 1), 1));
    EXPECT_RGBA_EQ(RGBA(2.5f, 3.0f, 0.0f, 1.0f), image.get_pixel(Vector2ui(1, 1), 1));
    EXPECT_RGBA_EQ(RGBA(5.0f, 3.0f, 0.0f, 1.0f), image.get_pixel(Vector2ui(2, 1), 1));
    // EXPECT_RGBA_EQ(RGBA(3.0f, 2.0f, 0.0f, 1.0f), image.get_pixel(Vector2ui(0, 0), 2)); // NOTE The curent mipmap chain fill can tend to scew the result if textures are non-power-of-two.
}

TEST_F(Assets_ImageUtils, summed_area_table_from_image) {
    using namespace Bifrost::Math;

    unsigned int width = 5, height = 3;
    Image image = Images::create2D("Test image", PixelFormat::RGBA_Float, 1.0f, Vector2ui(width, height));

    // Fill image
    // [0, -1, 0, 0], [1, -1, 0, 0], [2, -1, 0, 0], [3, -1, 0, 0], [4, -1, 0, 0]
    // [0, 0, 0, -1], [1, 0, 1, -1], [2, 0, 8, -1], [3, 0, 27, -1], [4, 0, 64, -1]
    // [0, 0, 0, -2], [1, 0, 8, -2], [2, 0, 64, -2], [3, 0, 216, -2], [4, 0, 512, -2]
    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x)
            image.set_pixel(RGBA(float(x), float(y) - 1, float(x * x * x * y * y * y), -float(y)), Vector2ui(x, y));

    RGBA* sat = ImageUtils::compute_summed_area_table(image.get_ID());

    // Verify that each entry in the sat is the sum of all the pixels above and to the left of that entry, inclusive.
    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x) {
            RGBA sat_value = sat[x + y * width];

            RGBA upper_left_sum = RGBA(0.0f, 0.0f, 0.0f, 0.0f);
            for (unsigned int yy = 0; yy <= y; ++yy)
                for (unsigned int xx = 0; xx <= x; ++xx) {
                    RGBA p = image.get_pixel(Vector2ui(xx, yy));
                    upper_left_sum.rgb() += p.rgb();
                    upper_left_sum.a += p.a;
                }

            EXPECT_RGBA_EQ(upper_left_sum, sat_value);
        }
}

TEST_F(Assets_ImageUtils, combine_tint_and_roughness) {
    using namespace Bifrost::Math;

    int width = 2, height = 2;
    Vector2ui size = Vector2ui(width, height);
    int pixel_count = int(size.x * size.y);

    Image intensity8 = Images::create2D("Intensity8", PixelFormat::Intensity8, 2.2f, size, 1);
    Image rgb24 = Images::create2D("RGB24", PixelFormat::RGB24, 2.2f, size, 1);
    Image rgba32 = Images::create2D("RGBA32", PixelFormat::RGBA32, 2.2f, size, 1);
    Image roughness8 = Images::create2D("Roughness8", PixelFormat::Roughness8, 2.2f, size, 1);
    Image rgb_float = Images::create2D("RGB_Float", PixelFormat::RGB_Float, 1.0f, size, 1);
    Image rgba_float = Images::create2D("RGBA_Float", PixelFormat::RGBA_Float, 1.0f, size, 1);
    Image intensity_float = Images::create2D("Intensity_Float", PixelFormat::Intensity_Float, 1.0f, size, 1);

    for (int p = 0; p < pixel_count; ++p) {
        float v = p / float(pixel_count - 1);
        RGBA pixel(v, v * 0.4f, v * 0.2f, 0.5f);
        intensity8.set_pixel(pixel, p);
        rgb24.set_pixel(pixel, p);
        rgba32.set_pixel(pixel, p);
        roughness8.set_pixel(pixel, p);
        rgb_float.set_pixel(pixel, p);
        rgba_float .set_pixel(pixel, p);
        intensity_float.set_pixel(pixel, p);
    }

    { // Test simple cases where only one image exists
        Image combined_intensity8_none = ImageUtils::combine_tint_roughness(intensity8, ImageID::invalid_UID());
        ASSERT_EQ(intensity8, combined_intensity8_none);

        Image combined_rgb24_none = ImageUtils::combine_tint_roughness(rgb24, ImageID::invalid_UID());
        ASSERT_EQ(rgb24, combined_rgb24_none);

        Image combined_none_roughness8 = ImageUtils::combine_tint_roughness(ImageID::invalid_UID(), roughness8);
        ASSERT_EQ(roughness8, combined_none_roughness8);

        // Combining a four channel tint image with no roughness should still return a new image to ensure that the roughness channel is all ones.
        Image combined_rgba32_none = ImageUtils::combine_tint_roughness(rgba32, ImageID::invalid_UID());
        ASSERT_NE(combined_rgba32_none, rgba32);
        for (int p = 0; p < pixel_count; ++p) {
            RGBA expected_pixel = RGBA(rgba32.get_pixel(p).rgb(), 1);
            RGBA actual_pixel = combined_rgba32_none.get_pixel(p);
            ASSERT_EQ(expected_pixel, actual_pixel);
        }
    }

    auto test_combination = [](const Image tint, const Image roughness, int roughness_channel) {
        Image combined_tint_roughness = ImageUtils::combine_tint_roughness(tint, roughness, roughness_channel);
        int pixel_count = int(combined_tint_roughness.get_pixel_count());

        // Assert on pixels
        // Quantize values to byte precision.
        for (int p = 0; p < pixel_count; ++p) {
            int input_red = int(tint.get_pixel(p).r * 255.0f + 0.5f);
            int combined_red = int(combined_tint_roughness.get_pixel(p).r * 255.0f + 0.5f);
            ASSERT_EQ(input_red, combined_red);
            int input_green = int(tint.get_pixel(p).g * 255.0f + 0.5f);
            int combined_green = int(combined_tint_roughness.get_pixel(p).g * 255.0f + 0.5f);
            ASSERT_EQ(input_green, combined_green);
            int input_blue = int(tint.get_pixel(p).b * 255.0f + 0.5f);
            int combined_blue = int(combined_tint_roughness.get_pixel(p).b * 255.0f + 0.5f);
            ASSERT_EQ(input_blue, combined_blue);

            int input_roughness = int(roughness.get_pixel(p)[roughness_channel] * 255.0f + 0.5f);
            int combined_roughness = int(combined_tint_roughness.get_pixel(p)[roughness_channel] * 255.0f + 0.5f);
            ASSERT_EQ(input_roughness, combined_roughness);
        }
    };

    { // Test basic tint RGB24/RGBA32 and Roughness8/intensity8 combination
        test_combination(rgb24, intensity8, 0);
        test_combination(rgb24, roughness8, 3);
        test_combination(rgba32, intensity8, 0);
        test_combination(rgba32, roughness8, 3);
    }

    { // Test with weird roughness, fx RGBA_float, and extract different channels
        test_combination(rgb_float, intensity_float, 0);
        test_combination(rgba_float, intensity_float, 0);
        test_combination(rgb_float, intensity8, 0);
        test_combination(rgba_float, roughness8, 3);
        test_combination(rgb_float, rgba_float, 2);
        test_combination(rgba_float, rgb_float, 1);
    }
}

} // NS Assets
} // NS Bifrost

#endif // _BIFROST_ASSETS_IMAGE_TEST_H_
