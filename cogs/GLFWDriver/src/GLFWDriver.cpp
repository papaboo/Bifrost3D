// Cogwheel GLFW main.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#include <GLFWDriver.h>

#include <glfw/glfw3.h>

#include <Core/Engine.h>
#include <Input/Keyboard.h>
#include <Input/Mouse.h>

using Cogwheel::Core::Engine;
using Cogwheel::Input::Keyboard;
using Cogwheel::Input::Mouse;
using Cogwheel::Math::Vector2i;

static Keyboard* g_keyboard = NULL;
static Mouse* g_mouse = NULL;

namespace GLFWDriver {

void run(on_launch_callback on_launch, on_window_created_callback on_window_created) {
    if (!glfwInit())
        exit(EXIT_FAILURE);

    // TODO Splash screen. With possibility to print resource processing text.

    // Create engine.
    Engine engine;
    on_launch(engine);

    Cogwheel::Core::Window& engine_window = engine.get_window();
    GLFWwindow* window = glfwCreateWindow(engine_window.get_width(), engine_window.get_height(), engine_window.get_name().c_str(), NULL, NULL);

    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // glfwSetWindowSizeCallback(window, windowSizeCallback);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    on_window_created(engine_window);

    { // Setup keyboard
        g_keyboard = new Keyboard();
        engine.set_keyboard(g_keyboard);
        GLFWkeyfun keyboard_callback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
            if (action == GLFW_REPEAT)
                return;
            g_keyboard->key_tapped(Keyboard::Key(key), action == GLFW_PRESS);
        };
        glfwSetKeyCallback(window, keyboard_callback);
    }

    { // Setup mouse
        double mouse_pos_x, mouse_pos_y;
        glfwGetCursorPos(window, &mouse_pos_x, &mouse_pos_y);
        g_mouse = new Mouse(Vector2i(int(mouse_pos_x), int(mouse_pos_y)));
        engine.set_mouse(g_mouse);
        
        static GLFWcursorposfun mouse_position_callback = [](GLFWwindow* window, double x, double y) {
            g_mouse->set_position(Vector2i(int(x), int(y)));
        };
        glfwSetCursorPosCallback(window, mouse_position_callback);

        static GLFWmousebuttonfun mouse_button_callback = [](GLFWwindow* window, int button, int action, int mods) {
            if (action == GLFW_REPEAT || button > Mouse::BUTTON_COUNT)
                return;

            g_mouse->button_tapped(button, action == GLFW_PRESS);
        };
        glfwSetMouseButtonCallback(window, mouse_button_callback);

        static GLFWscrollfun mouse_scroll_callback = [](GLFWwindow* window, double horizontalScroll, double verticalScroll) {
            g_mouse->add_scroll_delta(float(verticalScroll));
        };
        glfwSetScrollCallback(window, mouse_scroll_callback);
    }

    double previous_time = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        g_keyboard->per_frame_reset();
        g_mouse->per_frame_reset();
        glfwPollEvents();

        // Poll and update time.
        double current_time = glfwGetTime();
        float delta_time = float(current_time - previous_time);
        previous_time = current_time;
            
        engine.do_loop(delta_time);

        glfwSwapBuffers(window);

        if (engine.requested_quit())
            glfwSetWindowShouldClose(window, GL_TRUE);
    }

    // Cleanup.
    delete g_keyboard;
    delete g_mouse;

    glfwDestroyWindow(window);
    glfwTerminate();
}

} // NS GLFWDriver
