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

Engine::Engine(const std::string& data_path)
    : m_window(Window("Cogwheel", 640, 480))
    , m_mutating_callbacks(0)
    , m_non_mutating_callbacks(0)
    , m_tick_cleanup_callbacks(0)
    , m_quit(false)
    , m_keyboard(nullptr)
    , m_mouse(nullptr) 
    , m_data_path(data_path) {
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