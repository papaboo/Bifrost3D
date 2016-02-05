#include <GLFWDriver.h>

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Core/IModule.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>
#include <Cogwheel/Scene/Camera.h>

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

class Navigation final : public IModule {
public:

    Navigation(SceneNodes::UID node_ID, float velocity) 
        : m_node_ID(node_ID)
        , m_vertical_rotation(0.0f) 
        , m_horizontal_rotation(0.0f)
        , m_velocity(velocity)
    { }

    void apply() override {
        Engine* engine = Engine::get_instance();
        const Keyboard* keyboard = engine->get_keyboard();
        const Mouse* mouse = engine->get_mouse();

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

            if (strafing != 0.0f || forward != 0.0f) {
                Vector3f translation_offset = transform.rotation * Vector3f(strafing, 0.0f, forward);
                transform.translation += normalize(translation_offset) * m_velocity * engine->get_time().get_smooth_delta_time();
            }
        }

        { // Rotation
            if (mouse->get_left_button().is_pressed) {
                m_vertical_rotation += degrees_to_radians(float(mouse->get_delta().x));

                // Clamp horizontal rotation to -89 and 89 degrees to avoid turning the camera on it's head and the singularities of cross products at the poles.
                m_horizontal_rotation += degrees_to_radians(float(mouse->get_delta().y));
                m_horizontal_rotation = clamp(m_horizontal_rotation, -PI<float>() * 0.49f, PI<float>() * 0.49f);

                transform.rotation = Quaternionf::from_angle_axis(m_vertical_rotation, Vector3f::up()) * Quaternionf::from_angle_axis(m_horizontal_rotation, Vector3f::right());
            }
        }

        node.set_global_transform(transform);
    }

    std::string get_name() override {
        return "Navigation";
    }

private:
    SceneNodes::UID m_node_ID;
    float m_vertical_rotation;
    float m_horizontal_rotation;
    float m_velocity;
};

void initializer(Cogwheel::Core::Engine& engine) {
    engine.get_window().set_name("SimpleViewer");

    SceneNodes::allocate(8u);
    Meshes::allocate(8u);
    MeshModels::allocate(8u);

    ObjLoader::load(g_filepath);

    AABB scene_bounds = AABB::invalid();
    for (Meshes::ConstUIDIterator uid_itr = Meshes::begin(); uid_itr != Meshes::end(); ++uid_itr) {
        AABB mesh_aabb = Meshes::get_bounds(*uid_itr);
        scene_bounds.grow_to_contain(mesh_aabb);
    }
    std::cout << "Scene bounds " << scene_bounds << std::endl;
    
    { // Add camera
        Cameras::allocate(1u);
        SceneNodes::UID cam_node_ID = SceneNodes::create("Cam");

        Matrix4x4f perspective_matrix, inverse_perspective_matrix;
        CameraUtils::compute_perspective_projection(0.1f, 100.0f, PI<float>() / 4.0f, 8.0f / 6.0f,
            perspective_matrix, inverse_perspective_matrix);

        Cameras::UID cam_ID = Cameras::create(cam_node_ID, perspective_matrix, inverse_perspective_matrix);

        float camera_velocity = magnitude(scene_bounds.size()) * 0.1f;
        engine.add_mutating_module(new Navigation(cam_node_ID, camera_velocity));
    }
}

void initialize_window(Cogwheel::Core::Window& window) {
    Engine::get_instance()->add_non_mutating_module(new OptiXRenderer::Renderer());
}

void main(int argc, char** argv) {
    g_filepath = argc >= 2 ? std::string(argv[1]) : "";
    // g_filepath = "../../data/models/teapot/teapot.obj";

    if (g_filepath.empty()) {
        printf("SimpleViewer requires path to model as first argument.\n");
        exit(1);
    }

    GLFWDriver::run(initializer, initialize_window);
}
