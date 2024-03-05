// SimpleViewer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scenes/CornellBox.h>
#include <Scenes/Material.h>
#include <Scenes/Opacity.h>
#include <Scenes/Sphere.h>
#include <Scenes/SphereLight.h>
#include <Scenes/Test.h>
#include <Scenes/Veach.h>

#include <GUI/RenderingGUI.h>

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Core/Engine.h>
#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Input/Mouse.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>
#include <Bifrost/Scene/SceneRoot.h>

#include <ImGui/ImGUIAdaptor.h>
#include <ImGui/Renderers/DX11Renderer.h>

#include <Win32Driver.h>
#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/Renderer.h>
#ifdef OPTIX_FOUND
#include <DX11OptiXAdaptor/Adaptor.h>
#include <OptiXRenderer/Renderer.h>
#endif

#include <glTFLoader/glTFLoader.h>
#include <ObjLoader/ObjLoader.h>
#include <StbImageLoader/StbImageLoader.h>

#include <iostream>
#include <io.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Input;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

static std::string g_scene;
static std::string g_environment;
static RGB g_environment_color = RGB(0.68f, 0.92f, 1.0f);
static float g_scene_size;
static DX11Renderer::Compositor* compositor = nullptr;
static DX11Renderer::Renderer* dx11_renderer = nullptr;
static ImGui::ImGuiAdaptor* imgui = nullptr;
static Vector2ui g_window_size = Vector2ui(640, 480);
static Vector3f g_camera_translation = Vector3f(nanf(""));
static float g_camera_horizontal_rotation = nanf("");
static float g_camera_vertical_rotation = nanf("");

#ifdef OPTIX_FOUND
bool optix_enabled = true;
bool rasterizer_enabled = true;
OptiXRenderer::Renderer* optix_renderer = nullptr;
#endif

class Navigation final {
public:

    Navigation(CameraID camera_ID, float velocity, Vector3f camera_translation, float camera_vertical_rotation, float camera_horizontal_rotation)
        : m_camera_ID(camera_ID)
        , m_velocity(velocity)
    {
        Transform transform = Cameras::get_transform(m_camera_ID);

        Vector3f forward = transform.rotation.forward();
        m_vertical_rotation = !isnan(camera_vertical_rotation) ? camera_vertical_rotation : std::atan2(forward.x, forward.z);
        m_horizontal_rotation = !isnan(camera_horizontal_rotation) ? camera_horizontal_rotation : std::asin(forward.y);

        if (!isnan(camera_translation.x))
            transform.translation = camera_translation;
        transform.rotation = Quaternionf::from_angle_axis(m_vertical_rotation, Vector3f::up()) * Quaternionf::from_angle_axis(m_horizontal_rotation, -Vector3f::right());

        Cameras::set_transform(m_camera_ID, transform);
    }

    void navigate(Engine& engine) {
        const Keyboard* keyboard = engine.get_keyboard();
        const Mouse* mouse = engine.get_mouse();

        Transform transform = Cameras::get_transform(m_camera_ID);

        { // Translation
            float strafing = 0.0f;
            if (keyboard->is_pressed(Keyboard::Key::D) || keyboard->is_pressed(Keyboard::Key::Right))
                strafing = 1.0f;
            if (keyboard->is_pressed(Keyboard::Key::A) || keyboard->is_pressed(Keyboard::Key::Left))
                strafing -= 1.0f;

            float forward = 0.0f;
            if (keyboard->is_pressed(Keyboard::Key::W) || keyboard->is_pressed(Keyboard::Key::Up))
                forward = 1.0f;
            if (keyboard->is_pressed(Keyboard::Key::S) || keyboard->is_pressed(Keyboard::Key::Down))
                forward -= 1.0f;

            float velocity = m_velocity;
            if (keyboard->is_pressed(Keyboard::Key::LeftShift) || keyboard->is_pressed(Keyboard::Key::RightShift))
                velocity *= 5.0f;

            if (strafing != 0.0f || forward != 0.0f) {
                Vector3f translation_offset = transform.rotation * Vector3f(strafing, 0.0f, forward);
                float dt = engine.get_time().is_paused() ? engine.get_time().get_raw_delta_time() : engine.get_time().get_smooth_delta_time();
                transform.translation += normalize(translation_offset) * velocity * dt;
            }
        }

        { // Rotation
            if (mouse->is_pressed(Mouse::Button::Left)) {

                m_vertical_rotation += degrees_to_radians(float(mouse->get_delta().x));

                // Clamp horizontal rotation to -89 and 89 degrees to avoid turning the camera on it's head and the singularities of cross products at the poles.
                m_horizontal_rotation -= degrees_to_radians(float(mouse->get_delta().y));
                m_horizontal_rotation = clamp(m_horizontal_rotation, -PI<float>() * 0.49f, PI<float>() * 0.49f);

                transform.rotation = Quaternionf::from_angle_axis(m_vertical_rotation, Vector3f::up()) * Quaternionf::from_angle_axis(m_horizontal_rotation, -Vector3f::right());
            }
        }

        if (transform != Cameras::get_transform(m_camera_ID))
            Cameras::set_transform(m_camera_ID, transform);

        if (keyboard->was_pressed(Keyboard::Key::Space)) {
            float new_time_scale = engine.get_time().is_paused() ? 1.0f : 0.0f;
            engine.get_time().set_time_scale(new_time_scale);
        }
    }

private:
    CameraID m_camera_ID;
    float m_vertical_rotation;
    float m_horizontal_rotation;
    float m_velocity;
};

class CameraHandler final {
public:
    CameraHandler(CameraID camera_ID, float aspect_ratio, float near, float far)
        : m_camera_ID(camera_ID), m_aspect_ratio(aspect_ratio), m_FOV(PI<float>() / 4.0f)
        , m_near(near), m_far(far) {
        Matrix4x4f perspective_matrix, inverse_perspective_matrix;
        CameraUtils::compute_perspective_projection(m_near, m_far, m_FOV, m_aspect_ratio,
            perspective_matrix, inverse_perspective_matrix);

        Cameras::set_projection_matrices(m_camera_ID, perspective_matrix, inverse_perspective_matrix);
    }

    void handle(const Engine& engine) {

        const Mouse* mouse = engine.get_mouse();
        float new_FOV = m_FOV - mouse->get_scroll_delta() * engine.get_time().get_smooth_delta_time(); // TODO Non-linear increased / decrease. Especially that it can become negative is an issue.

        float window_aspect_ratio = engine.get_window().get_aspect_ratio();
        if (window_aspect_ratio != m_aspect_ratio || new_FOV != m_FOV) {
            Matrix4x4f perspective_matrix, inverse_perspective_matrix;
            CameraUtils::compute_perspective_projection(m_near, m_far, new_FOV, window_aspect_ratio,
                perspective_matrix, inverse_perspective_matrix);

            Cameras::set_projection_matrices(m_camera_ID, perspective_matrix, inverse_perspective_matrix);
            m_aspect_ratio = window_aspect_ratio;
            m_FOV = new_FOV;
        }
    }

    void set_near_and_far(float near, float far) {
        m_near = near;
        m_far = far;

        Matrix4x4f perspective_matrix, inverse_perspective_matrix;
        CameraUtils::compute_perspective_projection(m_near, m_far, m_FOV, m_aspect_ratio,
            perspective_matrix, inverse_perspective_matrix);

        Cameras::set_projection_matrices(m_camera_ID, perspective_matrix, inverse_perspective_matrix);
    }

private:
    CameraID m_camera_ID;
    float m_aspect_ratio;
    float m_FOV;
    float m_near, m_far;
};

class RenderSwapper final {
public:
    RenderSwapper(CameraID camera_ID)
        : m_camera_ID(camera_ID) { }

    void handle(const Engine& engine) {
        const Keyboard* keyboard = engine.get_keyboard();

        if (keyboard->was_released(Keyboard::Key::P) && !keyboard->is_modifiers_pressed()) {
            Renderers::Iterator renderer_itr = Renderers::get_iterator(Cameras::get_renderer_ID(m_camera_ID));
            ++renderer_itr;
            Renderers::Iterator new_renderer_itr = (renderer_itr == Renderers::end()) ? Renderers::begin() : renderer_itr;
            Cameras::set_renderer_ID(m_camera_ID, *new_renderer_itr);
        }
    }

private:
    CameraID m_camera_ID;
};

static inline void update_FPS(Engine& engine) {
    static const int COUNT = 8;
    static float delta_times[COUNT] = { 1e30f, 1e30f, 1e30f, 1e30f, 1e30f, 1e30f, 1e30f, 1e30f };
    static int next_index = 0;

    delta_times[next_index] = engine.get_time().get_raw_delta_time();
    next_index = (next_index + 1) % COUNT;

    float summed_deltas = 0.0f;
    for (int i = 0; i < COUNT; ++i)
        summed_deltas += delta_times[i];
    float fps = COUNT / summed_deltas;

    std::ostringstream title;
    title << "SimpleViewer - FPS " << fps;
    engine.get_window().set_name(title.str().c_str());
}

ImageID load_image(const std::string& path) {
    const int read_only_flag = 4;
    if (_access(path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(path);
    std::string new_path = path;

    // Test tga.
    new_path[path.size() - 3] = 't'; new_path[path.size() - 2] = 'g'; new_path[path.size() - 1] = 'a';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(new_path);

    // Test png.
    new_path[path.size() - 3] = 'p'; new_path[path.size() - 2] = 'n'; new_path[path.size() - 1] = 'g';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(new_path);

    // Test jpg.
    new_path[path.size() - 3] = 'j'; new_path[path.size() - 2] = 'p'; new_path[path.size() - 1] = 'g';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(new_path);

    // No dice. Report error and return an invalid ID.
    printf("No image found at '%s'\n", path.c_str());
    return ImageID::invalid_UID();
}

// Merges all nodes in the scene sharing the same material and destroys all other nodes.
// Future work
// * Only combine meshes within some max distance to each other, fx the diameter of their bounds.
//   This avoids their bounding boxes containing mostly empty space and messing up ray tracing, 
//   which would be the case if two models on opposite sides of the scene were to be combined.
//   It also avoids combining leafs on a tree acros the entire scene.
void mesh_combine_whole_scene(SceneNodeID scene_root) {

    // Asserts of properties used when combining UIDs and mesh flags in one uint key.
    assert(MeshModelID::MAX_IDS <= 0xFFFFFF);
    assert((int)MeshFlag::Position <= 0xFF);
    assert((int)MeshFlag::Normal <= 0xFF);
    assert((int)MeshFlag::Texcoord <= 0xFF);
    
    std::vector<bool> used_meshes = std::vector<bool>(Meshes::capacity());
    for (MeshID mesh_ID : Meshes::get_iterable())
        used_meshes[mesh_ID] = false;

    struct OrderedModel {
        unsigned int key;
        MeshModelID model_ID;

        inline bool operator<(OrderedModel lhs) const { return key < lhs.key; }
    };

    // Sort models based on material ID and mesh flags.
    std::vector<OrderedModel> ordered_models = std::vector<OrderedModel>();
    ordered_models.reserve(MeshModels::capacity());
    for (MeshModelID model_ID : MeshModels::get_iterable()) {
        unsigned int key = MeshModels::get_material_ID(model_ID).get_index() << 8u;

        // Least significant bits in key consist of mesh flags.
        Mesh mesh = MeshModels::get_mesh_ID(model_ID);
        key |= int(mesh.get_positions() ? MeshFlag::Position : MeshFlag::None);
        key |= int(mesh.get_normals() ? MeshFlag::Normal : MeshFlag::None);
        key |= int(mesh.get_texcoords() ? MeshFlag::Texcoord : MeshFlag::None);

        OrderedModel model = { key, model_ID };
        ordered_models.push_back(model);
    }

    std::sort(ordered_models.begin(), ordered_models.end());

    { // Loop through models, merging all models in a segment with same material and flags.
        auto segment_begin = ordered_models.begin();
        for (auto itr = ordered_models.begin(); itr < ordered_models.end(); ++itr) {
            bool next_material_found = itr->key != segment_begin->key;
            bool last_model = (itr + 1) == ordered_models.end();
            if (next_material_found || last_model) {
                auto segment_end = itr;
                if (last_model)
                    ++segment_end;
                // Combine the meshes in the segment if there are more than one.
                auto model_count = segment_end - segment_begin;
                if (model_count == 1) {
                    MeshID mesh_ID = MeshModels::get_mesh_ID(segment_begin->model_ID);
                    used_meshes[mesh_ID] = true;
                }

                if (model_count > 1) {
                    Material material = MeshModels::get_material_ID(segment_begin->model_ID);

                    // Create new scene node to hold the combined model.
                    SceneNode node0 = MeshModels::get_scene_node_ID(segment_begin->model_ID);
                    Transform transform = node0.get_global_transform();
                    SceneNode merged_node = SceneNodes::create(material.get_name() + "_combined", transform);
                    merged_node.set_parent(scene_root);

                    std::vector<MeshUtils::TransformedMesh> transformed_meshes = std::vector<MeshUtils::TransformedMesh>();
                    for (auto model = segment_begin; model < segment_end; ++model) {
                        MeshID mesh_ID = MeshModels::get_mesh_ID(model->model_ID);
                        SceneNode node = MeshModels::get_scene_node_ID(model->model_ID);
                        MeshUtils::TransformedMesh meshie = { mesh_ID, node.get_global_transform() };
                        transformed_meshes.push_back(meshie);
                    }

                    std::string mesh_name = material.get_name() + "_combined_mesh";
                    unsigned int mesh_flags = segment_begin->key; // The mesh flags are contained in the key.
                    MeshID merged_mesh_ID = MeshUtils::combine(mesh_name, transformed_meshes.data(), transformed_meshes.data() + transformed_meshes.size(), mesh_flags);

                    // Create new model.
                    MeshModelID merged_model = MeshModels::create(merged_node.get_ID(), merged_mesh_ID, material.get_ID());
                    if (merged_mesh_ID.get_index() < used_meshes.size())
                        used_meshes[merged_model] = true;
                }

                segment_begin = itr;
            }
        }
    }

    // Destroy meshes that are no longer used.
    // NOTE Reference counting on the mesh UIDs would be really handy here.
    for (MeshID mesh_ID : Meshes::get_iterable())
        if (mesh_ID.get_index() < used_meshes.size() && used_meshes[mesh_ID] == false)
            Meshes::destroy(mesh_ID);

    // Destroy old models and scene nodes that no longer connect to a mesh.
    // TODO Delete parents as well.
    for (OrderedModel ordered_model : ordered_models) {
        MeshModel model = ordered_model.model_ID;
        if (!model.get_mesh().exists()) {
            SceneNodes::destroy(model.get_scene_node().get_ID());
            MeshModels::destroy(model.get_ID());
        }
    }
}

void detect_and_flag_cutout_materials() {
    // A cutout is a black / white alpha mask. In order to allow for textures with 'soft edges' to be flagged as cut outs 
    // (because transparency is a pain) we allow soft borders.
    // These are detected by grouping pixels in 2x2 groups. If a single pixel in that group is non-grey, then the group is considered a cutout.

    enum State : unsigned char { Unprocessed, Cutout, Transparent};

    std::vector<State> image_states = std::vector<State>();
    image_states.resize(Images::capacity());
    memset(image_states.data(), Unprocessed, image_states.capacity());

    for (MeshModel model : MeshModels::get_iterable()) {
        Material material = model.get_material();
        if (material.get_coverage_texture().exists()) {
            Image coverage_img = material.get_coverage_texture().get_image();
            assert(coverage_img.get_pixel_format() == PixelFormat::Alpha8);

            State& image_state = image_states[coverage_img.get_ID()];
            if (image_state == Unprocessed)
            {
                int width = coverage_img.get_width(), height = coverage_img.get_height();
                unsigned char* pixels = coverage_img.get_pixels<unsigned char>();

                auto is_cutout_opacity = [](unsigned char intensity) -> bool { return intensity < 2 || 253 < intensity; };

                image_state = Cutout;
                for (int y = 0; y < height - 1; ++y)
                    for (int x = 0; x < width - 1; ++x) {
                        unsigned char intensity = pixels[x + y * width];
                        if (!is_cutout_opacity(intensity)) {
                            // Intensity is not black / white.
                            // Check if the pixel is part of a border or if its part of a larger 'greyish blob'.

                            bool cutout_border = is_cutout_opacity(pixels[(x + 1) + y * width])
                                || is_cutout_opacity(pixels[x + (y + 1) * width])
                                || is_cutout_opacity(pixels[(x + 1) + (y + 1) * width]);
                            if (!cutout_border)
                                image_state = Transparent;
                        }
                    }
            }

            if (image_state == Cutout)
                material.set_flags(MaterialFlag::Cutout);
        }
    }
}

static inline void miniheaps_cleanup_callback() {
    Cameras::reset_change_notifications();
    Images::reset_change_notifications();
    LightSources::reset_change_notifications();
    Materials::reset_change_notifications();
    Meshes::reset_change_notifications();
    MeshModels::reset_change_notifications();
    SceneNodes::reset_change_notifications();
    SceneRoots::reset_change_notifications();
    Textures::reset_change_notifications();
}

int initializer(Engine& engine) {
    engine.get_window().set_name("SimpleViewer");

    Cameras::allocate(1u);
    Images::allocate(8u);
    LightSources::allocate(8u);
    Materials::allocate(8u);
    Meshes::allocate(8u);
    MeshModels::allocate(8u);
    Renderers::allocate(2u);
    SceneNodes::allocate(8u);
    SceneRoots::allocate(1u);
    Textures::allocate(8u);

    engine.add_tick_cleanup_callback(miniheaps_cleanup_callback);

    return 0;
}

int initialize_scene(Engine& engine) {
    // Setup scene.
    SceneRootID scene_ID = SceneRootID::invalid_UID();
    if (!g_environment.empty()) {
        Image image = StbImageLoader::load(g_environment);
        if (image.exists()) {
            if (channel_count(image.get_pixel_format()) != 4)
                image.change_format(PixelFormat::RGBA_Float, 1.0f);
            TextureID env_ID = Textures::create2D(image.get_ID(), MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
            scene_ID = SceneRoots::create("Model scene", env_ID);
        } else
            scene_ID = SceneRoots::create("Model scene", g_environment_color);
    } else
        scene_ID = SceneRoots::create("Model scene", g_environment_color);
    SceneNodeID root_node_ID = SceneRoots::get_root_node(scene_ID);

    // Create camera
    CameraID cam_ID = Cameras::create("Camera", scene_ID, Matrix4x4f::identity(), Matrix4x4f::identity()); // Matrices will be set up by the CameraHandler.
    CameraHandler* camera_handler = new CameraHandler(cam_ID, engine.get_window().get_aspect_ratio(), 0.1f, 100.0f);
    engine.add_mutating_callback([=, &engine] { camera_handler->handle(engine); });

    // Load model
    bool load_model_from_file = false;
    if (g_scene.empty() || g_scene.compare("CornellBox") == 0)
        Scenes::create_cornell_box(cam_ID, root_node_ID);
    else if (g_scene.compare("MaterialScene") == 0)
        Scenes::create_material_scene(cam_ID, root_node_ID, imgui);
    else if (g_scene.compare("OpacityScene") == 0)
        Scenes::create_opacity_scene(engine, cam_ID, root_node_ID);
    else if (g_scene.compare("SphereScene") == 0)
        Scenes::create_sphere_scene(cam_ID, root_node_ID);
    else if (g_scene.compare("SphereLightScene") == 0)
        Scenes::SphereLightScene::create(engine, cam_ID, scene_ID);
    else if (g_scene.compare("TestScene") == 0)
        Scenes::create_test_scene(engine, cam_ID, root_node_ID);
    else if (g_scene.compare("VeachScene") == 0)
        Scenes::create_veach_scene(engine, cam_ID, scene_ID);
    else {
        printf("Loading scene: '%s'\n", g_scene.c_str());
        SceneNodeID obj_root_ID = SceneNodeID::invalid_UID();
        if (ObjLoader::file_supported(g_scene))
            obj_root_ID = ObjLoader::load(g_scene, load_image);
        else if (glTFLoader::file_supported(g_scene))
            obj_root_ID = glTFLoader::load(g_scene);
        SceneNodes::set_parent(obj_root_ID, root_node_ID);
        // mesh_combine_whole_scene(root_node_ID);
        detect_and_flag_cutout_materials();
        load_model_from_file = true;
    }

    if (SceneNodes::get_children_IDs(root_node_ID).size() == 0u) {
        printf("SimpleViewer error: No objects in scene.\n");
        return -1;
    }

    // Rough approximation of the scene bounds using bounding spheres.
    AABB scene_bounds = AABB::invalid();
    for (MeshModel model : MeshModels::get_iterable()) {
        AABB mesh_aabb = model.get_mesh().get_bounds();
        Transform transform = model.get_scene_node().get_global_transform();
        Vector3f bounding_sphere_center = transform * mesh_aabb.center();
        float bounding_sphere_radius = magnitude(mesh_aabb.size()) * 0.5f;
        AABB global_mesh_aabb = AABB(bounding_sphere_center - bounding_sphere_radius, bounding_sphere_center + bounding_sphere_radius);
        scene_bounds.grow_to_contain(global_mesh_aabb);
    }
    g_scene_size = magnitude(scene_bounds.size());
    camera_handler->set_near_and_far(g_scene_size / 10000.0f, g_scene_size * 3.0f);

    if (load_model_from_file) {
        Transform cam_transform = Cameras::get_transform(cam_ID);
        cam_transform.translation = scene_bounds.center() + scene_bounds.size();
        cam_transform.look_at(scene_bounds.center());
        Cameras::set_transform(cam_ID, cam_transform);
    }

    // Add a light source if none were added yet.
    bool no_light_sources = LightSources::begin() == LightSources::end() && g_environment.empty();
    if (no_light_sources && load_model_from_file) {
        Quaternionf light_direction = Quaternionf::look_in(normalize(Vector3f(-0.1f, -10.0f, -0.1f)));
        Transform light_transform = Transform(Vector3f::zero(), light_direction);
        SceneNodeID light_node_ID = SceneNodes::create("Light", light_transform);
        LightSourceID light_ID = LightSources::create_directional_light(light_node_ID, RGB(15.0f));
        SceneNodes::set_parent(light_node_ID, root_node_ID);
    }

    float camera_velocity = g_scene_size * 0.1f;
    Navigation* camera_navigation = new Navigation(cam_ID, camera_velocity, g_camera_translation, g_camera_vertical_rotation, g_camera_horizontal_rotation);
    engine.add_mutating_callback([=, &engine] { camera_navigation->navigate(engine); });
    RenderSwapper* render_swapper = new RenderSwapper(cam_ID);
    engine.add_mutating_callback([=, &engine] { render_swapper->handle(engine); });
    engine.add_mutating_callback([&engine] { update_FPS(engine); });

    return 0;
}

int win32_window_initialized(Engine& engine, Window& window, HWND& hwnd) {
    using namespace DX11Renderer;

    if (window.get_width() != g_window_size.x || window.get_height() != g_window_size.y)
        window.resize(g_window_size.x, g_window_size.y);

    compositor = Compositor::initialize(hwnd, window);

#ifdef OPTIX_FOUND
    if (rasterizer_enabled)
        dx11_renderer = (Renderer*)compositor->add_renderer(Renderer::initialize).get();

    if (optix_enabled) {
        auto* optix_adaptor = (DX11OptiXAdaptor::Adaptor*)compositor->add_renderer(DX11OptiXAdaptor::Adaptor::initialize).get();
        if (optix_adaptor != nullptr)
            optix_renderer = optix_adaptor->get_renderer();
    }

#else
    compositor->add_renderer(Renderer::initialize);
#endif

    { // Setup GUI
        imgui = new ImGui::ImGuiAdaptor();
#ifdef OPTIX_FOUND
        imgui->add_frame(std::make_unique<GUI::RenderingGUI>(compositor, dx11_renderer, optix_renderer));
#else
        imgui->add_frame(std::make_unique<GUI::RenderingGUI>(compositor, dx11_renderer));
#endif
        engine.add_mutating_callback([=, &engine] {
            auto* keyboard = engine.get_keyboard();
            auto* imgui_adaptor = static_cast<ImGui::ImGuiAdaptor*>(imgui);

            bool control_pressed = keyboard->is_pressed(Keyboard::Key::LeftControl) || keyboard->is_pressed(Keyboard::Key::RightControl);
            bool g_was_released = keyboard->was_released(Keyboard::Key::G);
            imgui_adaptor->set_enabled(imgui_adaptor->is_enabled() ^ (g_was_released && control_pressed));

            imgui_adaptor->new_frame(engine);
        });

        compositor->add_GUI_renderer(ImGui::Renderers::DX11Renderer::initialize);
    }

    engine.add_non_mutating_callback([=] { compositor->render(); });

    return initialize_scene(engine);
}

void print_usage() {
    char* usage =
        "usage simpleviewer:\n"
        "  -h  | --help: Show command line usage for simpleviewer.\n"
        "  -s  | --scene <model>: Loads the model specified. Reserved names are 'CornellBox', 'MaterialScene', 'SphereScene', 'SphereLightScene', 'TestScene' and 'VeachScene', which loads the corresponding builtin scenes.\n"
#ifdef OPTIX_FOUND
        "  -p | --path-tracing-only: Launches with the path tracer as the only avaliable renderer.\n"
        "  -r | --rasterizer-only: Launches with the rasterizer as the only avaliable renderer.\n"
#endif
        "  -e  | --environment-map <image>: Loads the specified image for the environment.\n"
        "  -c  | --environment-tint [R,G,B]: Tint the environment by the specified value.\n"
        "      | --window-size [width, height]: Size of the window.\n"
        "      | --camera-position [X,Y,Z]: Position of the camera.\n"
        "      | --camera-rotation [vertical, horizontal]: Orientation of the camera in radians.\n";
    printf("%s", usage);
}

// String representation is assumed to be "[x, y]".
inline Vector2f parse_vector2f(const char* const vec2_str) {
    Vector2f result;

    const char* x_begin = vec2_str + 1; // Skip [
    char* channel_end;
    result.x = strtof(x_begin, &channel_end);

    char* y_begin = channel_end + 1; // Skip ,
    result.y = strtof(y_begin, &channel_end);

    return result;
}

// String representation is assumed to be "[x, y, z]".
inline Vector3f parse_vector3f(const char* const vec3_str) {
    Vector3f result;

    const char* x_begin = vec3_str + 1; // Skip [
    char* channel_end;
    result.x = strtof(x_begin, &channel_end);

    char* y_begin = channel_end + 1; // Skip ,
    result.y = strtof(y_begin, &channel_end);

    char* z_begin = channel_end + 1; // Skip ,
    result.z = strtof(z_begin, &channel_end);

    return result;
}

// String representation is assumed to be "[r, g, b]".
inline RGB parse_RGB(const char* const rgb_str) {
    Vector3f vec = parse_vector3f(rgb_str);
    return RGB(vec.x, vec.y, vec.z);
}

int main(int argc, char** argv) {

    std::string command = argc >= 2 ? std::string(argv[1]) : "";
    if (command.compare("-h") == 0 || command.compare("--help") == 0) {
        print_usage();
        return 0;
    }

    // Parse command line arguments.
    int argument = 1;
    while (argument < argc) {
        if (strcmp(argv[argument], "--scene") == 0 || strcmp(argv[argument], "-s") == 0)
            g_scene = std::string(argv[++argument]);
        else if (strcmp(argv[argument], "--environment-map") == 0 || strcmp(argv[argument], "-e") == 0)
            g_environment = std::string(argv[++argument]);
        else if (strcmp(argv[argument], "--environment-tint") == 0 || strcmp(argv[argument], "-c") == 0)
            g_environment_color = parse_RGB(argv[++argument]);
        else if (strcmp(argv[argument], "--window-size") == 0)
            g_window_size = (Vector2ui)parse_vector2f(argv[++argument]);
        else if (strcmp(argv[argument], "--camera-position") == 0)
            g_camera_translation = parse_vector3f(argv[++argument]);
        else if (strcmp(argv[argument], "--camera-rotation") == 0) {
            Vector2f camera_rotation = parse_vector2f(argv[++argument]);
            g_camera_vertical_rotation = camera_rotation.x;
            g_camera_horizontal_rotation = camera_rotation.y;
        }
#ifdef OPTIX_FOUND
        else if (strcmp(argv[argument], "--path-tracing-only") == 0 || strcmp(argv[argument], "-p") == 0) {
            optix_enabled = true;
            rasterizer_enabled = false;
        } else if (strcmp(argv[argument], "--rasterizer-only") == 0 || strcmp(argv[argument], "-r") == 0) {
            optix_enabled = false;
            rasterizer_enabled = true;
        }
#endif
        else
            printf("Unknown argument: '%s'\n", argv[argument]);
        ++argument;
    }

    if (g_scene.empty())
        printf("SimpleViewer will display the Cornell Box scene.\n");

    int error_code = Win32Driver::run(initializer, win32_window_initialized);

    delete compositor;

    return error_code;
}
