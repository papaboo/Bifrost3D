// Cogwheel engine driver.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Core/Engine.h>

#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>

namespace Cogwheel {
namespace Core {

Engine* Engine::m_instance = nullptr;

Engine::Engine()
    : m_window(Window("Cogwheel", 640, 480))
    , m_mutating_callbacks(0)
    , m_non_mutating_callbacks(0)
    , m_tick_cleanup_callbacks(0)
    , m_quit(false)
    , m_keyboard(nullptr)
    , m_mouse(nullptr) {
    
    m_instance = this;
}
    
Engine::~Engine() {
    if (m_instance == this)
        m_instance = nullptr;
}

void Engine::add_mutating_callback(mutating_callback callback, void* callback_state) {
    Closure<mutating_callback> callback_closure = { callback, callback_state };
    m_mutating_callbacks.push_back(callback_closure);
}

void Engine::add_non_mutating_callback(non_mutating_callback callback, void* callback_state) {
    Closure<non_mutating_callback> callback_closure = { callback, callback_state };
    m_non_mutating_callbacks.push_back(callback_closure);
}

void Engine::add_tick_cleanup_callback(tick_cleanup_callback callback, void* callback_state) {
    Closure<tick_cleanup_callback> callback_closure = { callback, callback_state };
    m_tick_cleanup_callbacks.push_back(callback_closure);
}

void Engine::do_tick(double delta_time) {
    m_time.tick(delta_time);
    
    for (Closure<mutating_callback> mutating_callback : m_mutating_callbacks)
        mutating_callback.callback(*this, mutating_callback.data);

    for (Closure<non_mutating_callback> non_mutating_callback : m_non_mutating_callbacks)
        non_mutating_callback.callback(*this, non_mutating_callback.data);

    for (Closure<tick_cleanup_callback> tick_cleanup_closure : m_tick_cleanup_callbacks)
        tick_cleanup_closure.callback(tick_cleanup_closure.data);
}

} // NS Core
} // NS Cogwheel