// Cogwheel engine driver.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_ENGINE_H_
#define _COGWHEEL_CORE_ENGINE_H_

#include <Cogwheel/Core/Array.h>
#include <Cogwheel/Core/Time.h>
#include <Cogwheel/Core/Window.h>
#include <Cogwheel/Scene/SceneNode.h>

namespace Cogwheel {
namespace Core {
    class IModule;
}
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
// Future work
// * Add a 'mutation complete' (said in the Zerg voice) callback.
// * Add on_exit callback and deallocate the managers internal state.
// * Can I setup callbacks using lambdas?
// * Consider if anything is made easier by having the engine as a singleton.
// ---------------------------------------------------------------------------
class Engine final {
public:
    static inline Engine* get_instance() { return m_instance; }

    Engine(const std::string& data_path);
    ~Engine();

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
    typedef void(*mutating_callback)(Engine& engine, void* callback_state);
    void add_mutating_callback(mutating_callback callback, void* callback_state);

    typedef void(*non_mutating_callback)(const Engine& engine, void* callback_state);
    void add_non_mutating_callback(non_mutating_callback callback, void* callback_state);

    typedef void(*tick_cleanup_callback)(void* callback_state);
    void add_tick_cleanup_callback(tick_cleanup_callback callback, void* callback_state);

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

    static Engine* m_instance;

    Time m_time;
    Window m_window;
    bool m_quit;

    // A closure wrapping a callback function and its state.
    template <typename Function>
    struct Closure {
        Function callback;
        void* state;
    };

    // All engine callbacks.
    Core::Array<Closure<mutating_callback>> m_mutating_callbacks;
    Core::Array<Closure<non_mutating_callback>> m_non_mutating_callbacks;
    Core::Array<Closure<tick_cleanup_callback>> m_tick_cleanup_callbacks;

    // Input should only be updated by whoever created it and not by access via the engine.
    const Input::Keyboard* m_keyboard;
    const Input::Mouse* m_mouse;

    const std::string m_data_path;
};

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_ENGINE_H_