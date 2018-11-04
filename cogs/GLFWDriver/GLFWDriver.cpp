// Cogwheel GLFW main.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef RGB
#include <string>

std::string get_data_path() {
    char exepath[512];
    GetModuleFileName(nullptr, exepath, 512);
    // Find the second last slash and terminate after by setting the next character to '0'.
    char* last_char = exepath + strlen(exepath);
    int slash_count = 0;
    while (slash_count != 2) {
        char c = *--last_char;
        if (c == '/' || c == '\\')
            ++slash_count;
    }
    *++last_char = 0;
    return std::string(exepath) + "Data\\";
}

#endif // _WIN32

#include <GLFWDriver.h>

#include <glfw/glfw3.h>

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>

using Cogwheel::Core::Engine;
using Cogwheel::Input::Keyboard;
using Cogwheel::Input::Mouse;
using Cogwheel::Math::Vector2i;

static Keyboard* g_keyboard = nullptr;
static Mouse* g_mouse = nullptr;

namespace GLFWDriver {

int run(OnLaunchCallback on_launch, OnWindowCreatedCallback on_window_created) {
    if (!glfwInit())
        return EXIT_FAILURE;

    // TODO Splash screen. With possibility to print resource processing text.

    // Create engine.
    std::string data_path = get_data_path();
    static Engine engine(data_path);
    if (on_launch != nullptr) {
        int error_code = on_launch(engine);
        if (error_code != 0) {
            glfwTerminate();
            return error_code;
        }
    }

    Cogwheel::Core::Window& engine_window = engine.get_window();
    GLFWwindow* window = glfwCreateWindow(engine_window.get_width(), engine_window.get_height(), engine_window.get_name().c_str(), nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        return EXIT_FAILURE;
    }

    GLFWwindowsizefun window_size_callback = [](GLFWwindow* window, int width, int height) {
        Cogwheel::Core::Window& engine_window = engine.get_window();
        engine_window.resize(width, height);
    };
    glfwSetWindowSizeCallback(window, window_size_callback);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    int error_code = 0;
    if (on_window_created != nullptr)
        error_code = on_window_created(engine, engine_window);

    if (error_code == 0) {

        { // Setup keyboard
            g_keyboard = new Keyboard();
            engine.set_keyboard(g_keyboard);
            static GLFWkeyfun keyboard_callback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
                if (action == GLFW_REPEAT)
                    return;
                g_keyboard->key_tapped(Keyboard::Key(key), action == GLFW_PRESS);

                if (action == GLFW_PRESS) {
                    if (key == GLFW_KEY_BACKSPACE)
                        g_keyboard->add_codepoint(8);
                    else if (key == GLFW_KEY_TAB)
                        g_keyboard->add_codepoint(9);
                    else if (key == GLFW_KEY_ENTER)
                        g_keyboard->add_codepoint(13);
                    else if (key == GLFW_KEY_DELETE)
                        g_keyboard->add_codepoint(127);
                }
            };
            glfwSetKeyCallback(window, keyboard_callback);

            static GLFWcharfun character_callback = [](GLFWwindow* window, unsigned int codepoint) {
                g_keyboard->add_codepoint(codepoint);
            };
            glfwSetCharCallback(window, character_callback);
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
                if (action == GLFW_REPEAT || button >= int(Mouse::Button::ButtonCount))
                    return;
                g_mouse->button_tapped(Mouse::Button(button), action == GLFW_PRESS);
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

            engine.do_tick(delta_time);

            glfwSwapBuffers(window);

            if (engine.is_quit_requested()) {
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            }

            if (engine_window.has_changes(Cogwheel::Core::Window::Changes::Resized))
                glfwSetWindowSize(window, engine_window.get_width(), engine_window.get_height());
            if (engine_window.has_changes(Cogwheel::Core::Window::Changes::Renamed))
                glfwSetWindowTitle(window, engine_window.get_name().c_str());
        }

        // Cleanup.
        delete g_keyboard;
        delete g_mouse;
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

} // NS GLFWDriver
