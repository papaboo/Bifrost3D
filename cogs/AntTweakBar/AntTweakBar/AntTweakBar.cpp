// Ant tweak bar cogwheel wrapper.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <AntTweakBar/AntTweakBar.h>

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>
#include <Cogwheel/Math/Vector.h>

using namespace Cogwheel::Input;
using namespace Cogwheel::Math;

namespace AntTweakBar {

void handle_input(const Cogwheel::Core::Engine& engine) {

    // Handle window size.
    const Cogwheel::Core::Window& window = engine.get_window();
    Vector2i window_size = { window.get_width(), window.get_height() };
    TwWindowSize(window_size.x, window_size.y);

    { // Handle mouse input.
        const Mouse* const mouse = engine.get_mouse();

        Vector2i mouse_pos = mouse->get_position();
        TwMouseMotion(mouse_pos.x, mouse_pos.y);

        struct MouseKeyPair { Mouse::Button button; TwMouseButtonID tw_button; };
        MouseKeyPair key_pairs[] = { { Mouse::Button::Left , TW_MOUSE_LEFT },
                                     { Mouse::Button::Middle, TW_MOUSE_MIDDLE },
                                     { Mouse::Button::Right, TW_MOUSE_RIGHT } };
        for (auto key_pair : key_pairs) {
            if (mouse->was_pressed(key_pair.button))
                TwMouseButton(TW_MOUSE_PRESSED, key_pair.tw_button);
            if (mouse->was_released(key_pair.button))
                TwMouseButton(TW_MOUSE_RELEASED, key_pair.tw_button);
        }

        static float mouse_wheel_pos = 0;
        mouse_wheel_pos += mouse->get_scroll_delta();
        TwMouseWheel((int)mouse_wheel_pos);
    }

    { // Handle keyboard input.
        const Keyboard* const keyboard = engine.get_keyboard();

        // Modifiers.
        int modifiers = 0;
        if (keyboard->is_pressed(Keyboard::Key::LeftControl) || keyboard->is_pressed(Keyboard::Key::RightControl))
            modifiers |= TW_KMOD_CTRL;
        if (keyboard->is_pressed(Keyboard::Key::LeftAlt) || keyboard->is_pressed(Keyboard::Key::RightAlt))
            modifiers |= TW_KMOD_ALT;
        if (keyboard->is_pressed(Keyboard::Key::LeftShift) || keyboard->is_pressed(Keyboard::Key::RightShift))
            modifiers |= TW_KMOD_SHIFT;

        // Function keys.
        if (keyboard->was_released(Keyboard::Key::Pause))
            TwKeyPressed(TW_KEY_PAUSE, modifiers);
        else if (keyboard->was_released(Keyboard::Key::Escape))
            TwKeyPressed(TW_KEY_ESCAPE, modifiers);
        else if (keyboard->was_released(Keyboard::Key::Right))
            TwKeyPressed(TW_KEY_RIGHT, modifiers);
        else if (keyboard->was_released(Keyboard::Key::Left))
            TwKeyPressed(TW_KEY_LEFT, modifiers);
        else if (keyboard->was_released(Keyboard::Key::Down))
            TwKeyPressed(TW_KEY_DOWN, modifiers);
        else if (keyboard->was_released(Keyboard::Key::Up))
            TwKeyPressed(TW_KEY_UP, modifiers);
        else if (keyboard->was_released(Keyboard::Key::Home))
            TwKeyPressed(TW_KEY_HOME, modifiers);
        else if (keyboard->was_released(Keyboard::Key::End))
            TwKeyPressed(TW_KEY_END, modifiers);

        // Text.
        const std::wstring& text = keyboard->get_text();
        for (int i = 0; i < text.length(); ++i)
            TwKeyPressed(text[i], modifiers);
    }
}

} // NS AntTweakBar
