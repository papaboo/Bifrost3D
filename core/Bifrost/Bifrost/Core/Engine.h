// Bifrost engine driver.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_CORE_ENGINE_H_
#define _BIFROST_CORE_ENGINE_H_

#include <Bifrost/Core/Time.h>
#include <Bifrost/Core/Window.h>

#include <functional>
#include <vector>

namespace Bifrost {
namespace Input {
    class Keyboard;
    class Mouse;
}
}

namespace Bifrost {
namespace Core {

// ---------------------------------------------------------------------------
// Engine driver, responsible for invoking the modules and handling all engine
// 'tick' logic not related to the operating system.
// Future work
// * Add a 'mutation complete' (said in the Zerg voice) callback.
// * Add on_exit callback and deallocate the managers internal state.
// ---------------------------------------------------------------------------
class Engine final {
public:
    Engine(const std::string& data_path);

    inline Time& get_time() { return m_time; }
    inline Time get_time() const { return m_time; }
    inline Window& get_window() { return m_window; }
    inline const Window& get_window() const { return m_window; }

    inline void request_quit() { m_quit = true; }
    inline bool is_quit_requested() const { return m_quit; }

    // -----------------------------------------------------------------------
    // Input
    // -----------------------------------------------------------------------
    void set_keyboard(const Input::Keyboard* const keyboard) { m_keyboard = keyboard; }
    const Input::Keyboard* const get_keyboard() const { return m_keyboard; } // So .... you're saying it's const? TODO Return an immutable mouse and keyboard (Just wrap them in thin shells that only lets user query state.)
    void set_mouse(const Input::Mouse* const mouse) { m_mouse = mouse; }
    const Input::Mouse* const get_mouse() const { return m_mouse; }

    // -----------------------------------------------------------------------
    // Callbacks
    // -----------------------------------------------------------------------
    inline void add_mutating_callback(std::function<void()> callback) { m_mutating_callbacks.push_back(callback); }
    inline void add_non_mutating_callback(std::function<void()> callback) { m_non_mutating_callbacks.push_back(callback); }
    inline void add_tick_cleanup_callback(std::function<void()> callback) { m_tick_cleanup_callbacks.push_back(callback); }

    // -----------------------------------------------------------------------
    // Paths
    // -----------------------------------------------------------------------
    const std::string& data_path() { return m_data_path; }

    // -----------------------------------------------------------------------
    // Main loop
    // -----------------------------------------------------------------------
    void do_tick(double delta_time);

private:

    // Delete copy constructors.
    Engine(const Engine& rhs) = delete;
    Engine& operator=(Engine& rhs) = delete;

    Time m_time;
    Window m_window;
    bool m_quit;

    // All engine callbacks.
    std::vector<std::function<void()>> m_mutating_callbacks;
    std::vector<std::function<void()>> m_non_mutating_callbacks;
    std::vector<std::function<void()>> m_tick_cleanup_callbacks;

    // Input should only be updated by whoever created it and not by access via the engine.
    const Input::Keyboard* m_keyboard;
    const Input::Mouse* m_mouse;

    const std::string m_data_path;
};

} // NS Core
} // NS Bifrost

#endif // _BIFROST_CORE_ENGINE_H_
