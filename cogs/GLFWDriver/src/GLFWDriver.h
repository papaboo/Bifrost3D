// Cogwheel GLFW main.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_GLFW_DRIVER_H_
#define _COGWHEEL_GLFW_DRIVER_H_

#include <glfw/glfw3.h>

#include <Input/Keyboard.h>
#include <Input/Mouse.h>

#include <string>

// TODO Move into Cogwheel when done with prototyping.
namespace Cogwheel {
namespace Core {

class Window final {
public:
    Window(const std::string& name, int width, int height)
        : m_name(name), m_width(width), m_height(height) { }
    
    inline std::string get_name() const { return m_name; }
    inline void set_name(const std::string& name) { m_name = name; }

    inline int get_width() const { return m_width; }
    inline int get_height() const { return m_height; }

    inline int resize(int width, int height) {
        m_width = width; m_height = height;
    }

private:
    std::string m_name;
    int m_width, m_height;
};

// TODO Make it a singleton.
class Engine final {
public:
    Engine()
        : m_window(Window("Cogwheel", 640, 480))
        , m_quit(false)
        , m_iterations(0)
        , m_keyboard(nullptr)
        , m_mouse(nullptr) { }

    inline Window& get_window() { return m_window; }

    inline bool requested_quit() const { return m_quit; }

    void set_keyboard(const Input::Keyboard* const keyboard) { m_keyboard = keyboard; }
    const Input::Keyboard* const get_keyboard() const { return m_keyboard; } // So .... you're saying it's const?
    void set_mouse(const Input::Mouse* const mouse) { m_mouse = mouse; }
    const Input::Mouse* const get_mouse() const { return m_mouse; }

    void do_loop(double dt) {
        // TODO Time struct with smooth delta time as well. Smooth delta time is handled as smoothDt = lerp(dt, smoothDt, a), let a be 0.666 or setable by the user?
        // Or use the bitsquid approach. http://bitsquid.blogspot.dk/2010/10/time-step-smoothing.html.
        // Remember, all debt must be payed. Time, technical or loans.
        printf("dt: %f\n", dt);

        int keys_pressed = 0;
        int halftaps = 0;
        for (int k = 0; k < (int)Input::Keyboard::Key::KeyCount; ++k) {
            keys_pressed += m_keyboard->is_pressed(Input::Keyboard::Key(k));
            halftaps += m_keyboard->halftaps(Input::Keyboard::Key(k));
        }

        printf("Keys held down %u and total halftaps %u\n", keys_pressed, halftaps);

        // TODO Invoke modules.

        ++m_iterations;
    }

private:
    Window m_window;
    bool m_quit;
    unsigned int m_iterations;

    // Input should only be updated by whoever created it and not by access via the engine.
    const Input::Keyboard* m_keyboard;
    const Input::Mouse* m_mouse;
};

} // NS Core
} // NS Cogwheel

using Cogwheel::Core::Engine;
using Cogwheel::Input::Keyboard;
using Cogwheel::Input::Mouse;
using Cogwheel::Math::Vector2i;

static Keyboard* g_keyboard = NULL;
static Mouse* g_mouse = NULL;

namespace GLFWDriver {

typedef void(*on_launch_callback)(Cogwheel::Core::Engine& engine);
void run(on_launch_callback on_launch) {
    if (!glfwInit())
        exit(EXIT_FAILURE);

    // TODO Splash screen. With possibility to print resource processing text.

    // Create engine.
    Engine engine = Engine();
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

#endif // _COGWHEEL_GLFW_DRIVER_H_