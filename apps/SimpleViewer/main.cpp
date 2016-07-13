#include <CornellBoxScene.h>
#include <MaterialScene.h>
#include <SphereScene.h>
#include <TestScene.h>

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/LightSource.h>
#include <Cogwheel/Scene/SceneNode.h>
#include <Cogwheel/Scene/SceneRoot.h>

#include <GLFWDriver.h>

#include <ObjLoader/ObjLoader.h>
#include <StbImageLoader/StbImageLoader.h>

#include <OptiXRenderer/Renderer.h>

#include <cstdio>
#include <iostream>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Input;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

static std::string g_scene;
static std::string g_environment;
static RGB g_environment_color = RGB(0.68f, 0.92f, 1.0f);
static float g_scene_size;

class Navigation final {
public:

    Navigation(SceneNodes::UID node_ID, float velocity) 
        : m_node_ID(node_ID)
        , m_vertical_rotation(0.0f) 
        , m_horizontal_rotation(0.0f)
        , m_velocity(velocity)
    { }

    void navigate(Engine& engine) {
        const Keyboard* keyboard = engine.get_keyboard();
        const Mouse* mouse = engine.get_mouse();

        SceneNode node = m_node_ID;
        Transform transform = node.get_global_transform();

        { // Translation
            float strafing = 0.0f;
            if (keyboard->is_pressed(Keyboard::Key::D))
                strafing = 1.0f;
            if (keyboard->is_pressed(Keyboard::Key::A))
                strafing -= 1.0f;

            float forward = 0.0f;
            if (keyboard->is_pressed(Keyboard::Key::W))
                forward = 1.0f;
            if (keyboard->is_pressed(Keyboard::Key::S))
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
                m_horizontal_rotation += degrees_to_radians(float(mouse->get_delta().y));
                m_horizontal_rotation = clamp(m_horizontal_rotation, -PI<float>() * 0.49f, PI<float>() * 0.49f);

                transform.rotation = Quaternionf::from_angle_axis(m_vertical_rotation, Vector3f::up()) * Quaternionf::from_angle_axis(m_horizontal_rotation, Vector3f::right());
            }
        }

        if (transform != node.get_global_transform())
            node.set_global_transform(transform);

        if (keyboard->was_pressed(Keyboard::Key::Space)) {
            float new_time_scale = engine.get_time().is_paused() ? 1.0f : 0.0f;
            engine.get_time().set_time_scale(new_time_scale);
        }
    }

    static inline void navigate_callback(Engine& engine, void* state) {
        static_cast<Navigation*>(state)->navigate(engine);
    }

private:
    SceneNodes::UID m_node_ID;
    float m_vertical_rotation;
    float m_horizontal_rotation;
    float m_velocity;
};

class CameraHandler final {
public:
    CameraHandler(Cameras::UID camera_ID, float aspect_ratio)
        : m_camera_ID(camera_ID), m_aspect_ratio(aspect_ratio) {
    }

    void handle(Window& window) {
        float window_aspect_ratio = window.get_aspect_ratio();
        if (window_aspect_ratio != m_aspect_ratio) {
            Matrix4x4f perspective_matrix, inverse_perspective_matrix;
            CameraUtils::compute_perspective_projection(0.1f, 100.0f, PI<float>() / 4.0f, window_aspect_ratio,
                perspective_matrix, inverse_perspective_matrix);

            Cameras::set_projection_matrices(m_camera_ID, perspective_matrix, inverse_perspective_matrix);
            m_aspect_ratio = window_aspect_ratio;
        }
    }

    static inline void handle_callback(Engine& engine, void* state) {
        static_cast<CameraHandler*>(state)->handle(engine.get_window());
    }

private:
    Cameras::UID m_camera_ID;
    float m_aspect_ratio;
};

static inline void scenenode_cleanup_callback(void* dummy) {
    Images::reset_change_notifications();
    LightSources::reset_change_notifications();
    Materials::reset_change_notifications();
    Meshes::reset_change_notifications();
    MeshModels::reset_change_notifications();
    SceneNodes::reset_change_notifications();
    // Scenes::reset_change_notifications();
    Textures::reset_change_notifications();
}

void initializer(Cogwheel::Core::Engine& engine) {
    engine.get_window().set_name("SimpleViewer");

    Images::allocate(8u);
    LightSources::allocate(8u);
    Materials::allocate(8u);
    Meshes::allocate(8u);
    MeshModels::allocate(8u);
    SceneNodes::allocate(8u);
    Textures::allocate(8u);

    engine.add_tick_cleanup_callback(scenenode_cleanup_callback, nullptr);

    // TODO Scene and scene color should be set up by the test scenes themselves.
    SceneRoots::allocate(1u);
    SceneNodes::UID root_node_ID = SceneNodes::create("Root");
    SceneRoots::UID scene_ID = SceneRoots::UID::invalid_UID();
    if (!g_environment.empty()) {
        Image image = StbImageLoader::load(g_environment);
        if (channel_count(image.get_pixel_format()) != 4) {
            Image new_image = ImageUtils::change_format(image.get_ID(), PixelFormat::RGBA_Float);
            Images::destroy(image.get_ID());
            image = new_image;
        }
        Textures::UID env_ID = Textures::create2D(image.get_ID(), MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
        scene_ID = SceneRoots::create("Model scene", root_node_ID, env_ID);
    } else
        scene_ID = SceneRoots::create("Model scene", root_node_ID, g_environment_color);
    
    // Create camera
    SceneNodes::UID cam_node_ID = SceneNodes::create("Cam");

    Matrix4x4f perspective_matrix, inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(0.1f, 100.0f, PI<float>() / 4.0f, engine.get_window().get_aspect_ratio(),
        perspective_matrix, inverse_perspective_matrix);

    Cameras::allocate(1u);
    Cameras::UID cam_ID = Cameras::create(cam_node_ID, scene_ID, perspective_matrix, inverse_perspective_matrix);

    // Load model
    bool load_model_from_file = false;
    if (g_scene.empty() || g_scene.compare("CornellBox") == 0)
        create_cornell_box_scene(cam_ID, root_node_ID);
    else if (g_scene.compare("MaterialScene") == 0)
        create_material_scene(cam_ID, root_node_ID);
    else if (g_scene.compare("SphereScene") == 0)
        create_sphere_scene(cam_ID, root_node_ID);
    else if (g_scene.compare("TestScene") == 0)
        create_test_scene(engine, cam_ID, root_node_ID);
    else {
        SceneNodes::UID obj_root_ID = ObjLoader::load(g_scene, StbImageLoader::load);
        SceneNodes::set_parent(obj_root_ID, root_node_ID);
        load_model_from_file = true;
    }

    // Rough approximation of the scene bounds using bounding spheres.
    AABB scene_bounds = AABB::invalid();
    for (MeshModels::UID model_ID : MeshModels::get_iterable()) {
        MeshModel model = MeshModels::get_model(model_ID);
        AABB mesh_aabb = Meshes::get_bounds(model.mesh_ID);
        Transform transform = SceneNodes::get_global_transform(model.scene_node_ID);
        Vector3f bounding_sphere_center = transform * mesh_aabb.center();
        float bounding_sphere_radius = magnitude(mesh_aabb.size()) * 0.5f;
        AABB global_mesh_aabb = AABB(bounding_sphere_center - bounding_sphere_radius, bounding_sphere_center + bounding_sphere_radius);
        scene_bounds.grow_to_contain(global_mesh_aabb);
    }
    g_scene_size = magnitude(scene_bounds.size());

    float camera_velocity = g_scene_size * 0.1f;
    Navigation* camera_navigation = new Navigation(cam_node_ID, camera_velocity);
    engine.add_mutating_callback(Navigation::navigate_callback, camera_navigation);
    CameraHandler* camera_handler = new CameraHandler(cam_ID, engine.get_window().get_aspect_ratio());
    engine.add_mutating_callback(CameraHandler::handle_callback, camera_handler);

    if (load_model_from_file) {
        Transform cam_transform = SceneNodes::get_global_transform(cam_node_ID);
        cam_transform.translation = scene_bounds.center() + scene_bounds.size() * 1.0f;
        cam_transform.look_at(scene_bounds.center());
        SceneNodes::set_global_transform(cam_node_ID, cam_transform);
    }

    // Add a light source if none were added yet.
    bool no_light_sources = LightSources::begin() == LightSources::end() && g_environment.empty();
    if (no_light_sources && load_model_from_file) {
        Quaternionf light_direction = Quaternionf::look_in(normalize(Vector3f(-0.1f, -10.0f, -0.1f)));
        Transform light_transform = Transform(Vector3f::zero(), light_direction);
        SceneNodes::UID light_node_ID = SceneNodes::create("Light", light_transform);
        LightSources::UID light_ID = LightSources::create_directional_light(light_node_ID, RGB(15.0f));
        SceneNodes::set_parent(light_node_ID, root_node_ID);
    }
}

void initialize_window(Cogwheel::Core::Window& window) {
    OptiXRenderer::Renderer* renderer = new OptiXRenderer::Renderer();
    // renderer->set_scene_epsilon(g_scene_size * 0.00001f);
    Engine::get_instance()->add_non_mutating_callback(OptiXRenderer::render_callback, renderer);
}

void print_usage() {
    char* usage =
        "usage simpleviewer:\n"
        "  -h | --help: Show command line usage for simpleviewer.\n"
        "  -s | --scene <model>: Loads the model specified. Reserved names are 'CornellBox', 'MaterialScene', 'SphereScene' and 'TestScene', which loads the corresponding builtin scenes.\n"
        "  -e | --environment-map <image>: Loads the specified image for the environment.\n"
        "  -c | --environment-color <RGB>: Sets the background color to the specified value.\n";
    printf("%s", usage);
}

// String representation is assumed to be "[r, g, b]".
RGB parse_RGB(const std::string& rgb_str) {
    const char* red_begin = rgb_str.c_str() + 1; // Skip [
    char* channel_end;
    RGB result = RGB::black();

    result.r = strtof(red_begin, &channel_end);

    char* g_begin = channel_end + 1; // Skip ,
    result.g = strtof(g_begin, &channel_end);

    char* b_begin = channel_end + 1; // Skip ,
    result.b = strtof(b_begin, &channel_end);

    return result;
}

void main(int argc, char** argv) {

    std::string command = g_scene = argc >= 2 ? std::string(argv[1]) : "";
    if (command.compare("-h") == 0 || command.compare("--help") == 0) {
        print_usage();
        return;
    }  

    // Parse command line arguments.
    int argument = 1;
    while (argument < argc) {
        if (strcmp(argv[argument], "--scene") == 0 || strcmp(argv[argument], "-s") == 0)
            g_scene = std::string(argv[++argument]);
        else if (strcmp(argv[argument], "--environment-map") == 0 || strcmp(argv[argument], "-e") == 0)
            g_environment = std::string(argv[++argument]);
        else if (strcmp(argv[argument], "--environment-color") == 0 || strcmp(argv[argument], "-c") == 0)
            g_environment_color = parse_RGB(std::string(argv[++argument]));
        else
            printf("Unknown argument: '%s'\n", argv[argument]);
        ++argument;
    }

    if (g_scene.empty())
        printf("SimpleViewer will display the Cornell Box scene.\n");

    GLFWDriver::run(initializer, initialize_window);
}
