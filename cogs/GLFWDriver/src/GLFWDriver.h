// Cogwheel GLFW main.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_GLFW_DRIVER_H_
#define _COGWHEEL_GLFW_DRIVER_H_

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

namespace GLFWDriver {

typedef void(*on_launch_callback)(Cogwheel::Core::Engine& engine);

void run(on_launch_callback on_launch);

} // NS GLFWDriver

#endif // _COGWHEEL_GLFW_DRIVER_H_