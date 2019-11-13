// Bifrost engine driver.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Core/Engine.h>

#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Input/Mouse.h>

namespace Bifrost {
namespace Core {

Engine::Engine(const std::filesystem::path& data_directory)
    : m_window(Window("Bifrost", 640, 480))
    , m_mutating_callbacks(0)
    , m_non_mutating_callbacks(0)
    , m_tick_cleanup_callbacks(0)
    , m_quit(false)
    , m_keyboard(nullptr)
    , m_mouse(nullptr) 
    , m_data_directory(data_directory) {
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
} // NS Bifrost
