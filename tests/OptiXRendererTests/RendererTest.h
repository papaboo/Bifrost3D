// OptiXRenderer renderer test.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RENDERER_TEST_H_
#define _OPTIXRENDERER_RENDERER_TEST_H_

#include <Utils.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Core/Renderer.h>
#include <Bifrost/Math/FixedPointTypes.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>

#include <OptiXRenderer/Renderer.h>

#include <filesystem>

#include <optixu/optixpp_namespace.h>

namespace OptiXRenderer {

struct half4 { __half r, g, b, a; };

class RendererFixture : public ::testing::Test {
protected:
    Renderer* renderer;

    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        Bifrost::Assets::Images::allocate(8u);
        Bifrost::Assets::Materials::allocate(8u);
        Bifrost::Assets::Meshes::allocate(8u);
        Bifrost::Assets::MeshModels::allocate(8u);
        Bifrost::Assets::Textures::allocate(8u);
        Bifrost::Core::Renderers::allocate(1u);
        Bifrost::Scene::Cameras::allocate(8u);
        Bifrost::Scene::LightSources::allocate(8u);
        Bifrost::Scene::SceneRoots::allocate(8u);
        Bifrost::Scene::SceneNodes::allocate(8u);

        renderer = Renderer::initialize(0, get_data_directory());
    }
    virtual void TearDown() {
        delete renderer;

        Bifrost::Assets::Images::deallocate();
        Bifrost::Assets::Materials::deallocate();
        Bifrost::Assets::Meshes::deallocate();
        Bifrost::Assets::MeshModels::deallocate();
        Bifrost::Assets::Textures::deallocate();
        Bifrost::Core::Renderers::deallocate();
        Bifrost::Scene::Cameras::deallocate();
        Bifrost::Scene::LightSources::deallocate();
        Bifrost::Scene::SceneRoots::deallocate();
        Bifrost::Scene::SceneNodes::deallocate();
    }

    Bifrost::Scene::CameraID create_ortho_camera(Bifrost::Math::Vector2i size, optix::float3 environment_tint = optix::make_float3(1, 1, 1)) {
        using namespace Bifrost;

        Math::RGB env_tint = Math::RGB(environment_tint.x, environment_tint.y, environment_tint.z);
        Scene::SceneRoot scene = Scene::SceneRoot("Test", env_tint);

        float depth = 1000;
        Math::Matrix4x4f orthographic_matrix, inverse_orthographic_matrix;
        Scene::CameraUtils::compute_orthographic_projection(float(size.x), float(size.y), depth, orthographic_matrix, inverse_orthographic_matrix);

        Scene::CameraID camera_ID = Scene::Cameras::create("Test", scene.get_ID(), orthographic_matrix, inverse_orthographic_matrix);
        Scene::Cameras::set_renderer_ID(camera_ID, renderer->get_renderer_ID());
        return camera_ID;
    }

    // Creates an orthographic camera and a scene that contains a quad that fully covers the camera.
    // The vertices are aligned with the camera corners and are colored relative to their position in the frame.
    Bifrost::Scene::CameraID create_ortho_camera_with_quad_scene(Bifrost::Math::Vector2i size, optix::float3 environment_tint = optix::make_float3(1, 1, 1)) {
        using namespace Bifrost::Assets;
        using namespace Bifrost::Scene;

        CameraID camera_ID = create_ortho_camera(size, environment_tint);
        SceneRoot root = Cameras::get_scene_ID(camera_ID);

        Mesh mesh = Mesh("Triangle", 2, 4, { MeshFlag::Position, MeshFlag::TintAndRoughness });
        mesh.get_primitives()[0] = { 0, 1, 2 };
        mesh.get_primitives()[1] = { 1, 2, 3 };
        mesh.get_positions()[0] = { -0.5f * size.x, -0.5f * size.y, 1.0f };
        mesh.get_positions()[1] = { -0.5f * size.x, 0.5f * size.y, 1.0f };
        mesh.get_positions()[2] = { 0.5f * size.x, -0.5f * size.y, 1.0f };
        mesh.get_positions()[3] = { 0.5f * size.x, 0.5f * size.y, 1.0f };
        mesh.get_tint_and_roughness()[0] = { 0.0f, 0.0f, 1.0f, 0.0f };
        mesh.get_tint_and_roughness()[1] = { 0.0f, 1.0f, 1.0f, 0.0f };
        mesh.get_tint_and_roughness()[2] = { 1.0f, 0.0f, 1.0f, 0.0f };
        mesh.get_tint_and_roughness()[3] = { 1.0f, 1.0f, 1.0f, 0.0f };

        auto material = Bifrost::Assets::Material::create_dielectric("Material", Bifrost::Math::RGB::white(), 0.0f);
        material.set_flags(MaterialFlag::ThinWalled);
        material.set_shading_model(ShadingModel::Diffuse);

        SceneNode node = SceneNode("Node");
        node.set_parent(root.get_root_node());

        MeshModel(node, mesh, material);

        return camera_ID;
    }

    optix::Buffer create_render_target(Renderer* renderer, Bifrost::Math::Vector2i size) {
        return renderer->get_context()->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_HALF4, size.x, size.y);
    }

    void render(Bifrost::Scene::CameraID camera_ID, Bifrost::Math::Vector2i size, Backend backend, std::function<void(half4*)> render_callback) {
        renderer->set_backend(camera_ID, backend);

        renderer->handle_updates();

        auto render_target = create_render_target(renderer, size);
        renderer->render(camera_ID, render_target, size);
        half4* frame = (half4*)render_target->map();
        render_callback(frame);
        render_target->unmap();
    }

    void render(Bifrost::Scene::CameraID camera_ID, Bifrost::Math::Vector2i size, std::function<void(half4*)> render_callback) {
        render(camera_ID, size, Backend::PathTracing, render_callback);
    }

    Bifrost::Assets::Image render_auxiliary(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::Screenshot::Content content, Bifrost::Math::Vector2i size) {
        renderer->handle_updates();
        auto image = renderer->request_auxiliary_buffers(camera_ID, content, size)[0];
        return Bifrost::Assets::Image::create2D("auxiliary buffer", image.format, false, Bifrost::Math::Vector2ui(image.width, image.height), image.pixels);
    }
};

TEST_F(RendererFixture, render_background_color) {
    optix::float3 background_color = { 0.1f, 0.5f, 2.0f };
    auto frame_size = Bifrost::Math::Vector2i(16, 12);
    auto camera_ID = create_ortho_camera(frame_size, background_color);

    render(camera_ID, frame_size, [=](half4* pixels) {
        for (int i = 0; i < frame_size.x * frame_size.y; ++i) {
            optix::float3 actual_background_color = optix::make_float3(pixels[i].r, pixels[i].g, pixels[i].b);
            EXPECT_FLOAT3_EQ_EPS(background_color, actual_background_color, 1e-4f);
        }
    });
}

TEST_F(RendererFixture, render_tint) {
    using namespace Bifrost;

    auto frame_size = Math::Vector2i(4, 3);
    auto camera_ID = create_ortho_camera_with_quad_scene(frame_size);

    render(camera_ID, frame_size, Backend::TintVisualization, [=](half4* pixels) {
        for (int y = 0; y < frame_size.y; ++y) {
            float green_tint = (0.5f + y) / frame_size.y;
            for (int x = 0; x < frame_size.x; ++x) {
                float red_tint = (0.5f + x) / frame_size.x;

                half4 pixel = pixels[x + y * frame_size.x];
                EXPECT_FLOAT_EQ_EPS(red_tint, pixel.r, 0.003f) << " at pixel (" << x << ", " << y << ")";
                EXPECT_FLOAT_EQ_EPS(green_tint, pixel.g, 0.003f) << " at pixel (" << x << ", " << y << ")";
            }
        }});
}

TEST_F(RendererFixture, render_auxiliary_tint) {
    using namespace Bifrost;

    auto frame_size = Math::Vector2i(4, 3);
    auto camera_ID = create_ortho_camera_with_quad_scene(frame_size);

    auto tint_image = render_auxiliary(camera_ID, Scene::Screenshot::Content::Tint, frame_size);

    // Slight increase in eps, as floating point errors keep the test from being exact.
    float unorm8_eps = Math::UNorm8::max_precision() * 1.001f;
    for (int y = 0; y < frame_size.y; ++y) {
        float green_tint = (0.5f + y) / frame_size.y;
        for (int x = 0; x < frame_size.x; ++x) {
            float red_tint = (0.5f + x) / frame_size.x;

            auto pixel = tint_image.get_pixel(Math::Vector2ui(x, y));
            EXPECT_FLOAT_EQ_EPS(red_tint, pixel.r, unorm8_eps) << " at pixel (" << x << ", " << y << ")";
            EXPECT_FLOAT_EQ_EPS(green_tint, pixel.g, unorm8_eps) << " at pixel (" << x << ", " << y << ")";
        }
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RENDERER_TEST_H_