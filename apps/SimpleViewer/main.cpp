#include <TestScene.h>

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/LightSource.h>

#include <GLFWDriver.h>

#include <ObjLoader/ObjLoader.h>

#include <OptiXRenderer/Renderer.h>

#include <cstdio>
#include <iostream>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Input;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

static std::string g_filepath;

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
                transform.translation += normalize(translation_offset) * velocity * engine.get_time().get_smooth_delta_time();
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

    static inline void navigate_callback(Cogwheel::Core::Engine& engine, void* state) {
        static_cast<Navigation*>(state)->navigate(engine);
    }

private:
    SceneNodes::UID m_node_ID;
    float m_vertical_rotation;
    float m_horizontal_rotation;
    float m_velocity;
};

static inline void scenenode_cleanup_callback(void* dummy) {
    LightSources::reset_change_notifications();
    Materials::reset_change_notifications();
    Meshes::reset_change_notifications();
    MeshModels::reset_change_notifications();
    SceneNodes::reset_change_notifications();
}

void initializer(Cogwheel::Core::Engine& engine) {
    engine.get_window().set_name("SimpleViewer");

    LightSources::allocate(8u);
    Materials::allocate(8u);
    Meshes::allocate(8u);
    MeshModels::allocate(8u);
    SceneNodes::allocate(8u);

    engine.add_tick_cleanup_callback(scenenode_cleanup_callback, nullptr);

    if (g_filepath.empty())
        engine.set_scene_root(create_test_scene(engine));
    else {
        engine.set_scene_root(ObjLoader::load(g_filepath));

        { // Add camera
            Cameras::allocate(1u);
            SceneNodes::UID cam_node_ID = SceneNodes::create("Cam");

            Transform cam_transform = SceneNodes::get_global_transform(cam_node_ID);
            cam_transform.translation = Vector3f(0, 1, -4);
            SceneNodes::set_global_transform(cam_node_ID, cam_transform);

            Matrix4x4f perspective_matrix, inverse_perspective_matrix;
            CameraUtils::compute_perspective_projection(0.1f, 100.0f, PI<float>() / 4.0f, 8.0f / 6.0f,
                perspective_matrix, inverse_perspective_matrix);

            Cameras::UID cam_ID = Cameras::create(cam_node_ID, perspective_matrix, inverse_perspective_matrix);
        }
    }

    // TODO Should be based on the transformed global mesh bounds and not the local bounds.
    //      These could be calculated as a helper function on the model.
    AABB scene_bounds = AABB::invalid();
    for (Meshes::UID mesh_ID : Meshes::get_iterable()) {
        AABB mesh_aabb = Meshes::get_bounds(mesh_ID);
        scene_bounds.grow_to_contain(mesh_aabb);
    }

    Cameras::UID cam_ID = *Cameras::begin();
    SceneNodes::UID cam_node_ID = Cameras::get_node_ID(cam_ID);
    float camera_velocity = magnitude(scene_bounds.size()) * 0.1f;
    Navigation* camera_navigation = new Navigation(cam_node_ID, camera_velocity);
    engine.add_mutating_callback(Navigation::navigate_callback, camera_navigation);

    Vector3f light_position = scene_bounds.center() + scene_bounds.size() * 10.0f;
    Transform light_transform = Transform(light_position);
    SceneNodes::UID light_node_ID = SceneNodes::create("Light", light_transform);
    LightSources::UID light_ID = LightSources::create_sphere_light(light_node_ID, RGB(1000000.0f), 0.0f);
}

void initialize_window(Cogwheel::Core::Window& window) {
    OptiXRenderer::Renderer* renderer = new OptiXRenderer::Renderer();
    Engine::get_instance()->add_non_mutating_callback(OptiXRenderer::render_callback, renderer);
}

void main(int argc, char** argv) {
    g_filepath = argc >= 2 ? std::string(argv[1]) : "";

    if (g_filepath.empty())
        printf("SimpleViewer will display the debug scene.\n");

    GLFWDriver::run(initializer, initialize_window);
}
