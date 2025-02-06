// Vinci - Image generator
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <MaterialRandomizer.h>
#include <SceneGenerator.h>
#include <SceneSampler.h>

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

#include <filesystem>

namespace fs = std::filesystem;

using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Input;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;
using namespace OptiXRenderer;

// ------------------------------------------------------------------------------------------------
// Parsing utilities
// ------------------------------------------------------------------------------------------------

inline float parse_float(const char* float_str) {
    char* dummy_end;
    return strtof(float_str, &dummy_end);
}

inline void parse_float_array(const char* array_str, float* elements_begin, float* elements_end) {
    const char* str = array_str + 1; // Skip [

    while (elements_begin != elements_end) {
        if (*str == ',')
            ++str;

        char* element_str_end;
        *elements_begin++ = strtof(str, &element_str_end);
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

Image load_image(const std::string& filename) {
    return StbImageLoader::load(filename);
}

// ------------------------------------------------------------------------------------------------
// Program options
// ------------------------------------------------------------------------------------------------

struct Options {
    std::string scene;
    float scene_scale;

    int random_seed;
    int random_scene_images;
    std::string texture_directory;

    Texture environment_map;
    RGB environment_tint;
    Vector2ui window_size;
    fs::path output_directory;

    // Parse command line arguments.
    static inline Options parse(int argc, char** argv) {
        Options options;
        options.output_directory = "";
        options.window_size = Vector2ui(640, 480);

        options.scene = "";
        options.scene_scale = 1.0f;
        options.random_seed = 45678907;
        options.random_scene_images = 32;

        options.environment_map = TextureID::invalid_UID();
        options.environment_tint = RGB::black();

        int argument = 1;
        while (argument < argc) {
            if (strcmp(argv[argument], "--scene") == 0 || strcmp(argv[argument], "-s") == 0)
                options.scene = std::string(argv[++argument]);
            if (strcmp(argv[argument], "--scene-scale") == 0)
                options.scene_scale = parse_float(argv[++argument]);
            else if (strcmp(argv[argument], "--output") == 0 || strcmp(argv[argument], "-o") == 0)
                options.output_directory = std::string(argv[++argument]);
            else if (strcmp(argv[argument], "--textures") == 0 || strcmp(argv[argument], "-t") == 0)
                options.texture_directory = std::string(argv[++argument]);
            else if (strcmp(argv[argument], "--environment-map") == 0 || strcmp(argv[argument], "-e") == 0) {
                std::string environment_path = std::string(argv[++argument]);
                Image image = load_image(environment_path);
                if (image.exists()) {
                    if (channel_count(image.get_pixel_format()) != 4)
                        image.change_format(PixelFormat::RGBA_Float, 1.0f);
                    options.environment_map = Texture::create2D(image, MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp).get_ID();
                    if (options.environment_tint == RGB::black()) // Test if environment tint hasn't been set and if not then set it to white, so the environment map is shown.
                        options.environment_tint = RGB::white();
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

    Navigation(CameraID camera_ID, float velocity)
        : m_camera_ID(camera_ID)
        , m_velocity(velocity)
        , m_camera_moved(false) {
        Transform transform = Cameras::get_transform(m_camera_ID);

        Vector3f forward = transform.rotation.forward();
        m_vertical_rotation = std::atan2(forward.x, forward.z);
        m_horizontal_rotation = std::asin(forward.y);
    }

    inline CameraID camera_ID() const { return m_camera_ID; }
    inline bool camera_has_moved() const { return m_camera_moved; }

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

        if (transform != Cameras::get_transform(m_camera_ID)) {
            m_camera_moved = true;
            Cameras::set_transform(m_camera_ID, transform);
        }
    }

private:
    CameraID m_camera_ID;
    float m_vertical_rotation;
    float m_horizontal_rotation;
    float m_velocity;
    bool m_camera_moved;
};

class SceneRefresher final {
public:

    SceneRefresher(SceneGenerator::RandomScene& scene, CameraID camera_ID)
        : m_random_scene(&scene), m_scene_sampler(nullptr), m_material_randomizer(nullptr), m_camera_ID(camera_ID)
        , m_light_node(SceneNode()), m_camera_to_light_transform(Transform::identity()) {}

    SceneRefresher(SceneSampler& scene, MaterialRandomizer& material_randomizer, CameraID camera_ID)
        : m_random_scene(nullptr), m_scene_sampler(&scene), m_material_randomizer(&material_randomizer), m_camera_ID(camera_ID)
        , m_light_node(SceneNode()), m_camera_to_light_transform(Transform::identity()) {}

    void set_light_node(SceneNode light_node, Transform camera_to_light_transform) {
        m_light_node = light_node;
        m_camera_to_light_transform = camera_to_light_transform;
    }

    void refresh() {
        if (m_random_scene != nullptr)
            m_random_scene->new_scene();
        else {
            Transform camera_transform = m_scene_sampler->sample_world_transform();

            // Move camera backwards such that the focus point is in front on the nearplane.
            Vector3f forward = camera_transform.rotation.forward();
            camera_transform.translation -= forward * 50.0f; // TODO Randomize between near plane (find by projection) and 22 units further out

            Cameras::set_transform(m_camera_ID, camera_transform);

            ++m_counter;
            if (m_material_randomizer != nullptr && (m_counter % 8) == 0)
                m_material_randomizer->update_materials();
        }

        // Update light transform.
        if (m_light_node.exists()) {
            Transform camera_transform = Cameras::get_transform(m_camera_ID);
            Transform light_transform = camera_transform * m_camera_to_light_transform;
            m_light_node.set_global_transform(light_transform);

            // Unfortunately lights won't move, so we have to manually flag the light as updated.
            for (LightSource light : LightSources::get_iterable()) {
                if (light.get_node() == m_light_node) {
                    SpotLight spot_light = SpotLight(light.get_ID());
                    float radius = spot_light.get_radius();
                    spot_light.set_radius(radius);
                }
            }
        }
    }

private:
    SceneGenerator::RandomScene* m_random_scene;
    SceneSampler* m_scene_sampler;
    MaterialRandomizer* m_material_randomizer;
    CameraID m_camera_ID;
    SceneNode m_light_node;
    Transform m_camera_to_light_transform;

    int m_counter = 0;
};

class DataGeneration final {
public:
    DataGeneration(SceneRefresher& scene_refresher, Navigation& camera_navigation, const fs::path& output_directory, int max_iterations = 256)
        : m_scene_refresher(&scene_refresher), m_camera_navigation(camera_navigation), m_output_directory(output_directory), m_iteration(0), m_max_iterations(max_iterations) {
        queue_screenshot();
    }

    void tick(Engine& engine) {
        bool screenshot_resolved = false;

        if (m_camera_navigation.camera_has_moved())
            Cameras::cancel_screenshot(camera_ID());

        if (Cameras::pending_screenshots(camera_ID()).any_set()) {
            // Resolve and save screenshots.
            auto output_screenshot = [&](Screenshot::Content content, const std::string& path) {
                auto image = Cameras::resolve_screenshot(camera_ID(), content, "ss");
                if (image.exists()) {
                    if (!StbImageWriter::write(image, path))
                        printf("Failed to output screenshot to '%s'\n", path.c_str());
                    image.destroy();
                }
            };

            std::string base_path = (m_output_directory / std::to_string(m_iteration)).string();
            output_screenshot(Screenshot::Content::ColorLDR, base_path + "_color_LDR.png");
            output_screenshot(Screenshot::Content::ColorHDR, base_path + "_color_HDR.hdr");
            output_screenshot(Screenshot::Content::Depth, base_path + "_depth.hdr");
            output_screenshot(Screenshot::Content::Albedo, base_path + "_albedo.png");
            output_screenshot(Screenshot::Content::Tint, base_path + "_tint.png");
            output_screenshot(Screenshot::Content::Roughness, base_path + "_roughness.png");
            screenshot_resolved = true;

            ++m_iteration;
        }

        if (m_iteration == m_max_iterations)
            engine.request_quit();

        if (screenshot_resolved) {
            m_scene_refresher->refresh();
            queue_screenshot();
        }
    }

private:
    SceneRefresher* m_scene_refresher;

    Navigation& m_camera_navigation;
    const fs::path& m_output_directory;
    int m_iteration;
    int m_max_iterations;

    inline CameraID camera_ID() const { return m_camera_navigation.camera_ID(); }

    void queue_screenshot() {
        const Cameras::ScreenshotContent content = { Screenshot::Content::ColorLDR, Screenshot::Content::Depth, Screenshot::Content::Albedo, Screenshot::Content::Tint, Screenshot::Content::Roughness };
        int iterations = 128;
        Cameras::request_screenshot(camera_ID(), content, iterations);
    }
};

// ------------------------------------------------------------------------------------------------
// Vinci
// ------------------------------------------------------------------------------------------------

Options g_options;

DX11Renderer::Compositor* g_compositor = nullptr;
DX11OptiXAdaptor::Adaptor* g_optix_adaptor = nullptr;
SceneSampler* g_scene_sampler = nullptr;
MaterialRandomizer* g_material_randomizer = nullptr;
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
    SceneRoot scene_root = SceneRoot("Model scene", options.environment_map, options.environment_tint);
    SceneNode root_node = scene_root.get_root_node();

    // Setup camera
    Matrix4x4f perspective_matrix, inverse_perspective_matrix;
    float near = 7.887667f; // 43.180657f;
    float far = 1000;
    float cos_field_of_view = 0.9853505f;
    float field_of_view = acos(cos_field_of_view);
    CameraUtils::compute_perspective_projection(near, far, field_of_view, engine.get_window().get_aspect_ratio(),
        perspective_matrix, inverse_perspective_matrix);
    CameraID camera_ID = Cameras::create("Camera", scene_root.get_ID(), perspective_matrix, inverse_perspective_matrix);
    Transform cam_transform = Cameras::get_transform(camera_ID);
    // Disable screen space effects to keep the data in a linear color space.
    Cameras::set_effects_settings(camera_ID, CameraEffects::Settings::linear());
    auto optix_renderer = g_optix_adaptor->get_renderer();
    optix_renderer->set_backend(camera_ID, OptiXRenderer::Backend::AIDenoisedPathTracing);

    // Generate scene
    SceneRefresher* scene_refresher = nullptr;
    float camera_velocity = 5.0f;
    if (!options.scene.empty()) {
        printf("Loading scene: '%s'\n", options.scene.c_str());
        SceneNode loaded_scene_root;
        if (ObjLoader::file_supported(options.scene))
            loaded_scene_root = ObjLoader::load(options.scene, load_image);
        else if (glTFLoader::file_supported(options.scene))
            loaded_scene_root = glTFLoader::load(options.scene);

        loaded_scene_root.set_parent(root_node);
        root_node.apply_delta_transform(Transform(Vector3f::zero(), Quaternionf::identity(), options.scene_scale));

        g_scene_sampler = new SceneSampler(scene_root, options.random_seed);
        g_material_randomizer = new MaterialRandomizer(options.random_seed);
        scene_refresher = new SceneRefresher(*g_scene_sampler, *g_material_randomizer, camera_ID);

        // Rough approximation of the scene bounds using bounding spheres for the geometry.
        AABB scene_bounds = AABB::invalid();
        for (MeshModel model : MeshModels::get_iterable()) {
            AABB mesh_aabb = model.get_mesh().get_bounds();
            Transform transform = model.get_scene_node().get_global_transform();
            Vector3f bounding_sphere_center = transform * mesh_aabb.center();
            float bounding_sphere_radius = transform.scale * magnitude(mesh_aabb.size()) * 0.5f;
            AABB global_mesh_aabb = AABB(bounding_sphere_center - bounding_sphere_radius, bounding_sphere_center + bounding_sphere_radius);
            scene_bounds.grow_to_contain(global_mesh_aabb);
        }

        camera_velocity = 0.1f * magnitude(scene_bounds.size());
    } else {
        // Generate random scene primitives
        g_random_scene = new SceneGenerator::RandomScene(options.random_seed, camera_ID, options.texture_directory);
        g_random_scene->get_root_node().set_parent(root_node);

        scene_refresher = new SceneRefresher(*g_random_scene, camera_ID);
    }
    engine.add_mutating_callback([=, &engine] {
        if (engine.get_keyboard()->was_pressed(Keyboard::Key::N))
            scene_refresher->refresh();
        });

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

    // Setup lightsource colocated with camera.
    Transform light_transform = cam_transform;
    light_transform.translation += cam_transform.rotation.forward() * near * 0.999f;
    Transform camera_to_light_transform = cam_transform.inverse() * light_transform;
    SceneNode light_node = SceneNode("light node", light_transform);
    light_node.set_parent(root_node);
    SpotLight(light_node, RGB(25), 0.75f, cos_field_of_view);

    scene_refresher->set_light_node(light_node, camera_to_light_transform);

    Navigation* camera_navigation = new Navigation(camera_ID, camera_velocity);
    engine.add_mutating_callback([=, &engine] { camera_navigation->navigate(engine); });

    if (!g_options.output_directory.empty()) {
        if (!fs::exists(g_options.output_directory))
            fs::create_directories(g_options.output_directory);
        if (fs::is_directory(g_options.output_directory)) {
            DataGeneration* data_generation = new DataGeneration(*scene_refresher, *camera_navigation, g_options.output_directory);
            engine.add_mutating_callback([=, &engine] { data_generation->tick(engine); });
        }
    }

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

    g_compositor = DX11Renderer::Compositor::initialize(hwnd, window);

    g_optix_adaptor = (DX11OptiXAdaptor::Adaptor*)g_compositor->add_renderer(DX11OptiXAdaptor::Adaptor::initialize).get();

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
        "      | --scene-scale <scale>: The scale of the scene to fit with the camera options.\n"
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
    Images::allocate(128u);
    LightSources::allocate(2u);
    Materials::allocate(128u);
    Meshes::allocate(128u);
    MeshModels::allocate(128u);
    Renderers::allocate(2u);
    SceneNodes::allocate(128u);
    SceneRoots::allocate(1u);
    Textures::allocate(128u);

    g_options = Options::parse(argc, argv);

    if (g_options.scene.empty())
        printf("Vinci render arbitrary geometry.\n");

    int error_code = Win32Driver::run(initializer, win32_window_initialized);

    delete g_compositor;

    return error_code;
}
