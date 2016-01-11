#include <GLFWDriver.h>

#include <Core/Engine.h>
#include <Core/IModule.h>
#include <Input/Keyboard.h>
#include <Input/Mouse.h>

#include <OptiXRenderer/Renderer.h>

#include <cstdio>
#include <iostream>

using namespace Cogwheel;

class DebugInput final : public Core::IModule {
public:

    void apply() override {
        Core::Engine* engine = Core::Engine::get_instance();

        int keys_pressed = 0;
        int halftaps = 0;
        for (int k = 0; k < (int)Input::Keyboard::Key::KeyCount; ++k) {
            keys_pressed += engine->get_keyboard()->is_pressed(Input::Keyboard::Key(k));
            halftaps += engine->get_keyboard()->halftaps(Input::Keyboard::Key(k));
        }

        printf("Keys held down %u and total halftaps %u\n", keys_pressed, halftaps);
    }

    std::string get_name() override {
        return "Input Debugger";
    }
};

void initializer(Cogwheel::Core::Engine& engine) {
    std::cout << "Initialize baby!" << std::endl;
    engine.get_window().set_name("SimpleViewer");

    // engine.add_mutating_module(new DebugInput());
}

void initialize_window(Cogwheel::Core::Window& window) {
    Core::Engine::get_instance()->add_non_mutating_module(new OptiXRenderer::Renderer());
}

void main(int argc, char** argv) {
    GLFWDriver::run(initializer, initialize_window);
}
