// Cogwheel engine driver.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_ENGINE_H_
#define _COGWHEEL_CORE_ENGINE_H_

#include <Core/Window.h>

#include <Input/Keyboard.h>
#include <Input/Mouse.h>

namespace Cogwheel {
namespace Input {
class Keyboard;
class Mouse;
}
}

namespace Cogwheel {
namespace Core {

// ---------------------------------------------------------------------------
// Engine driver, responsible for invoking the modules and handling all engine
// 'tick' logic not related to the operating system.
// TODO 
//   * Make it a singleton.
//   * Time struct with smooth delta time as well. Smooth delta time is handled as smoothDt = lerp(dt, smoothDt, a), let a be 0.666 or setable by the user?
//     Or use the bitsquid approach. http://bitsquid.blogspot.dk/2010/10/time-step-smoothing.html.
//     Remember Lanister time deltas, all debts must be payed. Time, technical or loans.
//   * Modules.
// ---------------------------------------------------------------------------
class Engine final {
public:
    Engine();

    inline Window& get_window() { return m_window; }

    inline bool requested_quit() const { return m_quit; }

    void set_keyboard(const Input::Keyboard* const keyboard) { m_keyboard = keyboard; }
    const Input::Keyboard* const get_keyboard() const { return m_keyboard; } // So .... you're saying it's const?
    void set_mouse(const Input::Mouse* const mouse) { m_mouse = mouse; }
    const Input::Mouse* const get_mouse() const { return m_mouse; }

    void do_loop(double dt);

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

#endif // _COGWHEEL_CORE_ENGINE_H_