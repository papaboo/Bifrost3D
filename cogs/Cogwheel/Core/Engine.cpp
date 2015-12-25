// Cogwheel engine driver.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Core/Engine.h>

#include <Input/Keyboard.h>
#include <Input/Mouse.h>

namespace Cogwheel {
namespace Core {

Engine::Engine()
    : m_window(Window("Cogwheel", 640, 480))
    , m_scene_root(Scene::SceneNodes::UID::invalid_UID())
    , m_iterations(0)
    , m_quit(false)
    , m_keyboard(nullptr)
    , m_mouse(nullptr) { }
    
void Engine::do_loop(double dt) {
    printf("dt: %f\n", dt);

    int keys_pressed = 0;
    int halftaps = 0;
    for (int k = 0; k < (int)Input::Keyboard::Key::KeyCount; ++k) {
        keys_pressed += m_keyboard->is_pressed(Input::Keyboard::Key(k));
        halftaps += m_keyboard->halftaps(Input::Keyboard::Key(k));
    }

    printf("Keys held down %u and total halftaps %u\n", keys_pressed, halftaps);

    ++m_iterations;
}

} // NS Core
} // NS Cogwheel