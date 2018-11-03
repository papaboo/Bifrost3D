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

Engine::Engine(const std::string& data_path)
    : m_window(Window("Cogwheel", 640, 480))
    , m_mutating_callbacks(0)
    , m_non_mutating_callbacks(0)
    , m_tick_cleanup_callbacks(0)
    , m_quit(false)
    , m_keyboard(nullptr)
    , m_mouse(nullptr) 
    , m_data_path(data_path) {
    m_instance = this;
}
    
Engine::~Engine() {
    if (m_instance == this)
        m_instance = nullptr;
}

void Engine::add_mutating_callback(mutating_callback callback, void* callback_state) {
    auto func = [=]() -> void { callback(*this, callback_state); };
    m_mutating_callbacks.push_back(func);
}

void Engine::add_non_mutating_callback(non_mutating_callback callback, void* callback_state) {
    auto func = [=]() -> void { callback(*this, callback_state); };
    m_non_mutating_callbacks.push_back(func);
}

void Engine::add_tick_cleanup_callback(tick_cleanup_callback callback, void* callback_state) {
    auto func = [=]() -> void { callback(callback_state); };
    m_tick_cleanup_callbacks.push_back(func);
}

void Engine::do_tick(double delta_time) {
    m_time.tick(delta_time);

    m_window.reset_change_notifications();

    for (auto callback : m_mutating_callbacks)
        callback();

    for (auto callback : m_non_mutating_callbacks)
        callback();

    for (auto callback : m_tick_cleanup_callbacks)
        callback();
}

} // NS Core
} // NS Cogwheel