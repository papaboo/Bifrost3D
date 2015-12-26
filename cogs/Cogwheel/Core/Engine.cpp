// Cogwheel engine driver.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Core/Engine.h>
#include <Core/IModule.h>

#include <Input/Keyboard.h>
#include <Input/Mouse.h>

namespace Cogwheel {
namespace Core {

Engine* Engine::m_instance = nullptr;

Engine::Engine()
    : m_window(Window("Cogwheel", 640, 480))
    , m_scene_root(Scene::SceneNodes::UID::invalid_UID())
    , m_mutating_modules(0)
    , m_non_mutating_modules(0)
    , m_quit(false)
    , m_keyboard(nullptr)
    , m_mouse(nullptr) {
    
    m_instance = this;
}
    
Engine::~Engine() {
    if (m_instance == this)
        m_instance = nullptr;
}

void Engine::add_mutating_module(Core::IModule* module) {
    m_mutating_modules.resize(m_mutating_modules.size() + 1u);
    m_mutating_modules[m_mutating_modules.size() - 1] = module;
}

void Engine::add_mutating_modules(Core::IModule** begin, Core::IModule** end) {
    m_mutating_modules.push_back(begin, end);
}

void Engine::add_non_mutating_module(Core::IModule* module) {
    m_non_mutating_modules.resize(m_non_mutating_modules.size() + 1u);
    m_non_mutating_modules[m_non_mutating_modules.size() - 1] = module;
}

void Engine::add_non_mutating_modules(Core::IModule** begin, Core::IModule** end) {
    m_non_mutating_modules.push_back(begin, end);
}

void Engine::do_loop(double delta_time) {
    m_time.tick(delta_time);
    
    for (IModule* module : m_mutating_modules)
        module->apply();

    for (IModule* module : m_non_mutating_modules)
        module->apply();
}

} // NS Core
} // NS Cogwheel