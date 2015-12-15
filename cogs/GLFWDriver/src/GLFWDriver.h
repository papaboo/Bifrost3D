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

class Window {
public:
    Window(const std::string& name, int width, int height)
        : mName(name), mWidth(width), mHeight(height) { }
    
    inline std::string getName() const { return mName; }
    inline void setName(const std::string& name) { mName = name; }

    inline int getWidth() const { return mWidth; }
    inline int getHeight() const { return mHeight; }

    inline int resize(int width, int height) {
        mWidth = width; mHeight = height;
    }

private:
    std::string mName;
    int mWidth, mHeight;
};

// TODO Make it a singleton.
class Engine {
public:
    Engine()
        : mWindow(Window("Cogwheel", 640, 480))
        , mQuit(false)
        , mIterations(0)
        , mKeyboard(nullptr)
        , mMouse(nullptr) { }

    inline Window& getWindow() { return mWindow; }

    inline bool quitRequested() const { return mQuit; }

    void setKeyboard(const Input::Keyboard* const keyboard) { mKeyboard = keyboard; }
    const Input::Keyboard* const getKeyboard() const { return mKeyboard; } // So .... you're saying it's const?
    void setMouse(const Input::Mouse* const mouse) { mMouse = mouse; }
    const Input::Mouse* const getMouse() const { return mMouse; }

    void doLoop(float dt) {
        // TODO Time struct with smooth delta time as well. Smooth delta time is handled as smoothDt = lerp(dt, smoothDt, a), let a be 0.666 or setable by the user?
        // Or use the bitsquid approach. http://bitsquid.blogspot.dk/2010/10/time-step-smoothing.html.
        // Remember, all debt must be payed. Time, technical or loans.
        printf("dt: %f\n", dt);

        int keysPressed = 0;
        int halfTaps = 0;
        for (int k = 0; k < (int)Input::Keyboard::Key::KeyCount; ++k) {
            keysPressed += mKeyboard->isPressed(Input::Keyboard::Key(k));
            halfTaps += mKeyboard->halftaps(Input::Keyboard::Key(k));
        }

        printf("Keys held down %u and total halftaps %u\n", keysPressed, halfTaps);

        // TODO Invoke modules.

        ++mIterations;
    }

private:
    Window mWindow;
    bool mQuit;
    unsigned int mIterations;

    // Input should only be updated by whoever created it and not by access via the engine.
    const Input::Keyboard* mKeyboard;
    const Input::Mouse* mMouse;
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

    GLFWwindow* window = glfwCreateWindow(engine.getWindow().getWidth(), engine.getWindow().getHeight(), engine.getWindow().getName().c_str(), NULL, NULL);

    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // glfwSetWindowSizeCallback(window, windowSizeCallback);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    { // Setup keyboard
        g_keyboard = new Keyboard();
        engine.setKeyboard(g_keyboard);
        GLFWkeyfun keyboard_callback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
            if (action == GLFW_REPEAT)
                return;
            g_keyboard->keyTapped(Keyboard::Key(key), action == GLFW_PRESS);
        };
        glfwSetKeyCallback(window, keyboard_callback);
    }

    { // Setup mouse
        g_mouse = new Mouse();
        engine.setMouse(g_mouse);
        double mouse_pos_x, mouse_pos_y;
        glfwGetCursorPos(window, &mouse_pos_x, &mouse_pos_y);
        g_mouse->position = Vector2i(int(mouse_pos_x), int(mouse_pos_y));
        
        static GLFWcursorposfun mouse_position_callback = [](GLFWwindow* window, double x, double y) {
            Vector2i new_pos = Vector2i(int(x), int(y));
            g_mouse->delta = new_pos - g_mouse->position;
            g_mouse->position = new_pos;
        };
        glfwSetCursorPosCallback(window, mouse_position_callback);

        static GLFWmousebuttonfun mouse_button_callback = [](GLFWwindow* window, int button, int action, int mods) {
            if (action == GLFW_REPEAT || button > Mouse::BUTTON_COUNT)
                return;

            g_mouse->buttonTapped(button, action == GLFW_PRESS);
        };
        glfwSetMouseButtonCallback(window, mouse_button_callback);

        static GLFWscrollfun mouse_scroll_callback = [](GLFWwindow* window, double horizontalScroll, double verticalScroll) {
            g_mouse->scrollDelta += float(verticalScroll);
        };
        glfwSetScrollCallback(window, mouse_scroll_callback);
    }

    double previous_time = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        g_keyboard->perFrameReset();
        g_mouse->perFrameReset();
        glfwPollEvents();

        // Poll and update time.
        double current_time = glfwGetTime();
        float delta_time = float(current_time - previous_time);
        previous_time = current_time;
            
        engine.doLoop(delta_time);

        glfwSwapBuffers(window);

        if (engine.quitRequested())
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