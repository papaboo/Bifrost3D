#include <GLFWDriver.h>

#include <Core/Engine.h>
#include <Core/IModule.h>
#include <Input/Keyboard.h>
#include <Input/Mouse.h>
#include <Scene/Camera.h>

#include <OptiXRenderer/Renderer.h>

#include <cstdio>
#include <iostream>

using namespace Cogwheel::Core;
using namespace Cogwheel::Input;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

class Navigation final : public IModule {
public:

    Navigation(SceneNodes::UID node_ID) 
        : m_node_ID(node_ID)
        , m_vertical_rotation(0.0f) 
        , m_horizontal_rotation(0.0f)
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
                strafing = engine->get_time().get_smooth_delta_time();
            if (keyboard->is_pressed(Keyboard::Key::A))
                strafing -= engine->get_time().get_smooth_delta_time();

            float forward = 0.0f;
            if (keyboard->is_pressed(Keyboard::Key::W))
                forward = engine->get_time().get_smooth_delta_time();
            if (keyboard->is_pressed(Keyboard::Key::S))
                forward -= engine->get_time().get_smooth_delta_time();

            Vector3f translation_offset = transform.rotation * Vector3f(strafing, 0.0f, forward);
            transform.translation += translation_offset;
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
};


void initializer(Cogwheel::Core::Engine& engine) {
    engine.get_window().set_name("SimpleViewer");

    SceneNodes::allocate(64u);

    { // Add camera
        Cameras::allocate(1u);
        SceneNodes::UID cam_node_ID = SceneNodes::create("Cam");

        Matrix4x4f perspective_matrix, inverse_perspective_matrix;
        CameraUtils::compute_perspective_projection(0.1f, 100.0f, PI<float>() / 4.0f, 8.0f / 6.0f,
            perspective_matrix, inverse_perspective_matrix);

        Cameras::UID cam_ID = Cameras::create(cam_node_ID, perspective_matrix, inverse_perspective_matrix);

        engine.add_mutating_module(new Navigation(cam_node_ID));
    }
}

void initialize_window(Cogwheel::Core::Window& window) {
    Engine::get_instance()->add_non_mutating_module(new OptiXRenderer::Renderer());
}

void main(int argc, char** argv) {
    
    GLFWDriver::run(initializer, initialize_window);
}
