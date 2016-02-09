// Cogwheel engine driver.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Cogwheel/Core/Engine.h>

#include <Cogwheel/Core/IModule.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>

namespace Cogwheel {
namespace Core {

Engine* Engine::m_instance = nullptr;

Engine::Engine()
    : m_window(Window("Cogwheel", 640, 480))
    , m_scene_root(Scene::SceneNodes::UID::invalid_UID())
    , m_mutating_callbacks(0)
    , m_non_mutating_callbacks(0)
    , m_on_tick_cleanup_callbacks(0)
    , m_quit(false)
    , m_keyboard(nullptr)
    , m_mouse(nullptr) {
    
    m_instance = this;
}
    
Engine::~Engine() {
    if (m_instance == this)
        m_instance = nullptr;
}

void Engine::add_mutating_callback(Core::IModule* callback) {
    m_mutating_callbacks.push_back(callback);
}

void Engine::add_non_mutating_callback(Core::IModule* callback) {
    m_non_mutating_callbacks.push_back(callback);
}

void Engine::add_tick_cleanup_callback(on_tick_cleanup_callback callback, void* callback_state) {
    Closure<on_tick_cleanup_callback> callback_closure = { callback, callback_state };
    m_on_tick_cleanup_callbacks.push_back(callback_closure);
}

void Engine::do_tick(double delta_time) {
    m_time.tick(delta_time);
    
    for (IModule* mutating_callback : m_mutating_callbacks)
        mutating_callback->apply();

    for (IModule* non_mutating_callback : m_non_mutating_callbacks)
        non_mutating_callback->apply();

    for (Closure<on_tick_cleanup_callback> on_tick_cleanup_closure : m_on_tick_cleanup_callbacks)
        on_tick_cleanup_closure.callback(on_tick_cleanup_closure.data);
}

} // NS Core
} // NS Cogwheel