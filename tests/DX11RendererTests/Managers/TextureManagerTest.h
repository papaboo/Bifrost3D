// DX11Renderer texture manager test.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_TEXTURE_MANAGER_TEST_H_
#define _DX11RENDERER_TEXTURE_MANAGER_TEST_H_

#include <gtest/gtest.h>
#include <Utils.h>

#include <DX11Renderer/Managers/TextureManager.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Texture.h>

namespace DX11Renderer::Managers {

class TextureManagerFixture : public ::testing::Test {
protected:
    ODevice1 device;
    ODeviceContext1 context;

    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Bifrost::Assets::Images::allocate(8u);
        Bifrost::Assets::Textures::allocate(8u);
        device = create_test_device();
        context = get_immidiate_context1(device);
    }
    virtual void TearDown() {
        Bifrost::Assets::Images::deallocate();
        Bifrost::Assets::Textures::deallocate();
    }
};

inline Bifrost::Assets::Image create_image() {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;

    Image image = Image::create2D("Test image", PixelFormat::RGB_Float, 1.0f, Vector2ui(2, 2));
    image.set_pixel(RGBA(0, 0, 0, 1), Vector2ui(0, 0));
    image.set_pixel(RGBA(0, 1, 0, 1), Vector2ui(0, 1));
    image.set_pixel(RGBA(1, 0, 0, 1), Vector2ui(1, 0));
    image.set_pixel(RGBA(1, 1, 0, 1), Vector2ui(1, 1));
    return image;
}

inline Bifrost::Assets::Texture create_texture(Bifrost::Assets::Image image) {
    return Bifrost::Assets::Texture::create2D(image);
}

TEST_F(TextureManagerFixture, create_dx_image_and_texture_representation) {
    TextureManager texture_manager(device);
    
    auto image = create_image();
    auto texture = create_texture(image);

    { // Assert that no images or textures have been allocated.
        auto& dx_image = texture_manager.get_image(image.get_ID());
        EXPECT_EQ(dx_image.srv, nullptr);

        auto& dx_texture = texture_manager.get_texture(texture.get_ID());
        EXPECT_EQ(dx_texture.image, nullptr);
        EXPECT_EQ(dx_texture.sampler, nullptr);
    }

    texture_manager.handle_updates(device, context);

    { // Assert that image and texture has been allocated.
        auto& dx_image = texture_manager.get_image(image.get_ID());
        EXPECT_NE(dx_image.srv, nullptr);

        auto& dx_texture = texture_manager.get_texture(texture.get_ID());
        EXPECT_NE(dx_texture.image, nullptr);
        EXPECT_NE(dx_texture.image->srv, nullptr);
        EXPECT_EQ(dx_texture.image, &dx_image);
        EXPECT_NE(dx_texture.sampler, nullptr);
    }
}

TEST_F(TextureManagerFixture, created_and_destroyed_image_is_ignored) {
    TextureManager texture_manager(device);

    auto image = create_image();
    auto texture = create_texture(image);
    texture.destroy();
    image.destroy();

    texture_manager.handle_updates(device, context);

    { // Assert that the image and texture aren't allocated.
        auto& dx_image = texture_manager.get_image(image.get_ID());
        EXPECT_EQ(dx_image.srv, nullptr);

        auto& dx_texture = texture_manager.get_texture(texture.get_ID());
        EXPECT_EQ(dx_texture.image, nullptr);
        EXPECT_EQ(dx_texture.sampler, nullptr);
    }
}

TEST_F(TextureManagerFixture, destroyed_assets_are_cleared) {
    TextureManager texture_manager(device);

    auto image = create_image();
    auto texture = create_texture(image);
    texture_manager.handle_updates(device, context);

    { // Assert that image and texture has been allocated.
        auto& dx_image = texture_manager.get_image(image.get_ID());
        EXPECT_NE(dx_image.srv, nullptr);

        auto& dx_texture = texture_manager.get_texture(texture.get_ID());
        EXPECT_NE(dx_texture.image, nullptr);
        EXPECT_NE(dx_texture.sampler, nullptr);
    }

    image.destroy();
    texture.destroy();
    texture_manager.handle_updates(device, context);

    { // Test that the assets are cleared
        auto& dx_image = texture_manager.get_image(image.get_ID());
        EXPECT_EQ(dx_image.srv, nullptr);

        auto& dx_texture = texture_manager.get_texture(texture.get_ID());
        EXPECT_EQ(dx_texture.image, nullptr);
        EXPECT_EQ(dx_texture.sampler, nullptr);
    }
}

TEST_F(TextureManagerFixture, unreferenced_images_are_not_uploaded) {
    TextureManager texture_manager(device);

    auto image = create_image();
    texture_manager.handle_updates(device, context);

    { // Assert that image has not been allocated
        auto& dx_image = texture_manager.get_image(image.get_ID());
        EXPECT_EQ(dx_image.srv, nullptr);
    }

    auto texture = create_texture(image);
    texture_manager.handle_updates(device, context);

    { // Assert that image and texture has been allocated.
        auto& dx_image = texture_manager.get_image(image.get_ID());
        EXPECT_NE(dx_image.srv, nullptr);

        auto& dx_texture = texture_manager.get_texture(texture.get_ID());
        EXPECT_NE(dx_texture.image, nullptr);
        EXPECT_NE(dx_texture.sampler, nullptr);
    }
}

} // NS DX11Renderer::Managers

#endif // _DX11RENDERER_TEXTURE_MANAGER_TEST_H_