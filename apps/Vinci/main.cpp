// Vinci - Image generator
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <SceneGenerator.h>

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Core/Engine.h>
#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Input/Mouse.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>
#include <Bifrost/Scene/SceneRoot.h>

#include <DX11OptiXAdaptor/Adaptor.h>
#include <DX11Renderer/Compositor.h>
#include <OptiXRenderer/Renderer.h>

#include <ObjLoader/ObjLoader.h>
#include <glTFLoader/glTFLoader.h>
#include <StbImageLoader/StbImageLoader.h>
#include <StbImageWriter/StbImageWriter.h>

#include <Win32Driver.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Input;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;
using namespace OptiXRenderer;

// ------------------------------------------------------------------------------------------------
// Parsing utilities
// ------------------------------------------------------------------------------------------------

inline void parse_float_array(const char* array_str, float* elements_begin, float* elements_end) {
    const char* str = array_str + 1; // Skip [

    while (elements_begin != elements_end) {
        if (*str == ',')
            ++str;

        char* element_str_end;
        *elements_begin = strtof(str, &element_str_end);
        str = element_str_end;
    }
}

template<typename T>
inline T parse_array_type(const char* rgb_str) {
    T t;
    parse_float_array(rgb_str, t.begin(), t.end());
    return t;
}

inline Vector2ui parse_vector2ui(const char* rgb_str) { return (Vector2ui)parse_array_type<Vector2f>(rgb_str); }
inline Vector3f parse_vector3f(const char* rgb_str) { return parse_array_type<Vector3f>(rgb_str); }
inline Vector4f parse_vector4f(const char* rgb_str) { return parse_array_type<Vector4f>(rgb_str); }
inline RGB parse_RGB(const char* rgb_str) { return parse_array_type<RGB>(rgb_str); }

Images::UID load_image(const std::string& filename) {
    return StbImageLoader::load(filename);
}

// ------------------------------------------------------------------------------------------------
// Program options
// ------------------------------------------------------------------------------------------------

struct Options {
    std::string scene;

    int random_seed;
    int random_scene_images;
    std::string texture_directory;

    Texture environment_map;
    RGB environment_tint;
    Vector2ui window_size;
    std::string output;

    // Parse command line arguments.
    static inline Options parse(int argc, char** argv) {
        Options options;
        options.output = "";
        options.window_size = Vector2ui(640, 480);

        options.scene = "";
        options.random_seed = 45678907;
        options.random_scene_images = 32;

        options.environment_map = Textures::UID::invalid_UID();
        options.environment_tint = RGB::white();

        int argument = 1;
        while (argument < argc) {
            if (strcmp(argv[argument], "--scene") == 0 || strcmp(argv[argument], "-s") == 0)
                options.scene = std::string(argv[++argument]);
            else if (strcmp(argv[argument], "--output") == 0 || strcmp(argv[argument], "-o") == 0)
                options.output = std::string(argv[++argument]);
            else if (strcmp(argv[argument], "--textures") == 0 || strcmp(argv[argument], "-t") == 0)
                options.texture_directory = std::string(argv[++argument]);
            else if (strcmp(argv[argument], "--environment-map") == 0 || strcmp(argv[argument], "-e") == 0) {
                std::string environment_path = std::string(argv[++argument]);
                Image image = load_image(environment_path);
                if (image.exists()) {
                    if (channel_count(image.get_pixel_format()) != 4) {
                        Image new_image = ImageUtils::change_format(image.get_ID(), PixelFormat::RGBA_Float, 1.0f);
                        Images::destroy(image.get_ID());
                        image = new_image;
                    }
                    options.environment_map = Textures::create2D(image.get_ID(), MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
                }
            } else if (strcmp(argv[argument], "--environment-tint") == 0)
                options.environment_tint = parse_RGB(argv[++argument]);
            else if (strcmp(argv[argument], "--window-size") == 0)
                options.window_size = parse_vector2ui(argv[++argument]);
            else
                printf("Unknown argument: '%s'\n", argv[argument]);
            ++argument;
        }

        return options;
    }
};

// ------------------------------------------------------------------------------------------------
// Utilities
// ------------------------------------------------------------------------------------------------

class Navigation final {
public:

    Navigation(Cameras::UID camera_ID, float velocity)
        : m_camera_ID(camera_ID)
        , m_velocity(velocity)
    {
        Transform transform = Cameras::get_transform(m_camera_ID);

        Vector3f forward = transform.rotation.forward();
        m_vertical_rotation = std::atan2(forward.x, forward.z);
        m_horizontal_rotation = std::asin(forward.y);
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
    }

private:
    Cameras::UID m_camera_ID;
    float m_vertical_rotation;
    float m_horizontal_rotation;
    float m_velocity;
};

class SceneRefresher final {
public:

    SceneRefresher(SceneGenerator::RandomScene& scene)
        : m_scene(scene) {}

    void refresh(Engine& engine) {
        if (engine.get_keyboard()->was_pressed(Keyboard::Key::N))
            m_scene.new_scene();
    }

private:
    SceneGenerator::RandomScene& m_scene;
};

// ------------------------------------------------------------------------------------------------
// Vinci
// ------------------------------------------------------------------------------------------------

Options g_options;

DX11Renderer::Compositor* g_compositor = nullptr;
DX11OptiXAdaptor::Adaptor* g_optix_adaptor = nullptr;
SceneGenerator::RandomScene* g_random_scene = nullptr;

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

int setup_scene(Engine& engine, Options& options) {
    // Setup scene root.
    SceneRoot scene_root = SceneRoots::create("Model scene", options.environment_map.get_ID(), options.environment_tint);
    SceneNode root_node = scene_root.get_root_node();

    // Generate scene
    if (!options.scene.empty()) {
        printf("Loading scene: '%s'\n", options.scene.c_str());
        SceneNode scene_root;
        if (ObjLoader::file_supported(options.scene))
            scene_root = ObjLoader::load(options.scene, load_image);
        else if (glTFLoader::file_supported(options.scene))
            scene_root = glTFLoader::load(options.scene);
        scene_root.set_parent(root_node);
    } else {
        // Generate random scene primitives
        g_random_scene = new SceneGenerator::RandomScene(options.random_seed, options.texture_directory);
        g_random_scene->get_root_node().set_parent(root_node);

        auto* scene_refresher = new SceneRefresher(*g_random_scene);
        engine.add_mutating_callback([=, &engine] { scene_refresher->refresh(engine); });
    }

    AABB scene_bounds = AABB::invalid();
    for (MeshModel model : MeshModels::get_iterable()) {
        AABB mesh_aabb = model.get_mesh().get_bounds();
        Transform transform = model.get_scene_node().get_global_transform();
        Vector3f bounding_sphere_center = transform * mesh_aabb.center();
        float bounding_sphere_radius = magnitude(mesh_aabb.size()) * 0.5f;
        AABB global_mesh_aabb = AABB(bounding_sphere_center - bounding_sphere_radius, bounding_sphere_center + bounding_sphere_radius);
        scene_bounds.grow_to_contain(global_mesh_aabb);
    }
    float scene_size = magnitude(scene_bounds.size());

    // Setup camera
    Matrix4x4f perspective_matrix, inverse_perspective_matrix;
    float near = scene_size / 10000.0f;
    float far = scene_size * 3.0f;
    float field_of_view = PI<float>() / 4.0f;
    CameraUtils::compute_perspective_projection(near, far, field_of_view, engine.get_window().get_aspect_ratio(),
        perspective_matrix, inverse_perspective_matrix);
    Cameras::UID camera_ID = Cameras::create("Camera", scene_root.get_ID(), perspective_matrix, inverse_perspective_matrix);
    float camera_velocity = scene_size * 0.1f;
    Transform cam_transform = Cameras::get_transform(camera_ID);
    cam_transform.translation = scene_bounds.center() + scene_bounds.size() * 0.5f;
    cam_transform.look_at(scene_bounds.center());
    Cameras::set_transform(camera_ID, cam_transform);

    // Disable screen space effects two keep the data in a linear color space.
    auto effects_settings = Cameras::get_effects_settings(camera_ID);
    effects_settings.exposure.mode = CameraEffects::ExposureMode::Fixed;
    effects_settings.tonemapping.mode = CameraEffects::TonemappingMode::Linear;
    effects_settings.vignette = 0.0f;
    Cameras::set_effects_settings(camera_ID, effects_settings);

    Navigation* camera_navigation = new Navigation(camera_ID, camera_velocity);
    engine.add_mutating_callback([=, &engine] { camera_navigation->navigate(engine); });

    return 0;
}

int initializer(Engine& engine) {
    engine.get_window().set_name("Vinci");

    engine.add_tick_cleanup_callback(miniheaps_cleanup_callback);

    return 0;
}

int win32_window_initialized(Engine& engine, Window& window, HWND& hwnd) {
    if (window.get_width() != g_options.window_size.x || window.get_height() != g_options.window_size.y)
        window.resize(g_options.window_size.x, g_options.window_size.y);

    g_compositor = DX11Renderer::Compositor::initialize(hwnd, window, engine.data_directory());

    g_optix_adaptor = (DX11OptiXAdaptor::Adaptor*)g_compositor->add_renderer(DX11OptiXAdaptor::Adaptor::initialize).get();

    Renderers::UID default_renderer = *Renderers::begin();
    for (auto camera_ID : Cameras::get_iterable())
        Cameras::set_renderer_ID(camera_ID, default_renderer);

    engine.add_non_mutating_callback([=] { g_compositor->render(); });

    return setup_scene(engine, g_options);
}

void print_usage() {
    char* usage =
        "Vinci usage:\n"
        "  -h  | --help: Show command line usage for Vinci.\n"
        "  -s  | --scene <model file>: Loads the model specified.\n"
        "  -o  | --output <output directory>.\n"
        "  -t  | --textures <texture folder>: Root folder of physically based rendering textures.\n"
        "  -e  | --environment-map <image>: Loads the specified image for the environment.\n"
        "      | --environment-tint [R,G,B]: Tint the environment by the specified value.\n"
        "      | --window-size [width, height]: Size of the window.\n";
    printf("%s", usage);
}

int main(int argc, char** argv) {
    printf("Vinci\n");

	// Output help menu
    std::string command = argc >= 2 ? std::string(argv[1]) : "";
    if (command.compare("-h") == 0 || command.compare("--help") == 0) {
        print_usage();
        return 0;
    }

    // Initialize mini heaps
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

    g_options = Options::parse(argc, argv);

    if (g_options.scene.empty())
        printf("Vinci render arbitrary geometry.\n");

    int error_code = Win32Driver::run(initializer, win32_window_initialized);

    delete g_compositor;

    return error_code;
}
