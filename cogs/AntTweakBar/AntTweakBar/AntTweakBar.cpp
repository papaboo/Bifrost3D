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

#include <AntTweakBar.h>

using namespace Cogwheel::Core;
using namespace Cogwheel::Input;
using namespace Cogwheel::Math;

namespace AntTweakBar {

int* create_keyboard_key_map() {
    int* key_map = new int[int(Keyboard::Key::KeyCount)];
    key_map[int(Keyboard::Key::Space)] = ' ';
    key_map[int(Keyboard::Key::Apostrophe)] = '\'';
    key_map[int(Keyboard::Key::Comma)] = ',';
    key_map[int(Keyboard::Key::Minus)] = '-';
    key_map[int(Keyboard::Key::Period)] = '.';
    key_map[int(Keyboard::Key::Slash)] = '/';

    for (int k = int(Keyboard::Key::Key0); k <= int(Keyboard::Key::Key9); ++k)
        key_map[k] = '0' + k - int(Keyboard::Key::Key0);

    key_map[int(Keyboard::Key::Semicolon)] = ';';
    key_map[int(Keyboard::Key::Equal)] = '=';

    for (int k = int(Keyboard::Key::A); k <= int(Keyboard::Key::Z); ++k)
        key_map[k] = 'a' + k - int(Keyboard::Key::A);

    key_map[int(Keyboard::Key::Semicolon)] = ';';
    key_map[int(Keyboard::Key::LeftBracket)] = '[';
    key_map[int(Keyboard::Key::Backslash)] = '\\';
    key_map[int(Keyboard::Key::RightBracket)] = ']';
    key_map[int(Keyboard::Key::GraveAccent)] = '`';

    key_map[int(Keyboard::Key::Escape)] = TW_KEY_ESCAPE;
    key_map[int(Keyboard::Key::Enter)] = TW_KEY_RETURN;
    key_map[int(Keyboard::Key::Tab)] = TW_KEY_TAB;
    key_map[int(Keyboard::Key::Backspace)] = TW_KEY_BACKSPACE;
    key_map[int(Keyboard::Key::Insert)] = TW_KEY_INSERT;
    key_map[int(Keyboard::Key::Delete)] = TW_KEY_DELETE;
    key_map[int(Keyboard::Key::Right)] = TW_KEY_RIGHT;
    key_map[int(Keyboard::Key::Left)] = TW_KEY_LEFT;
    key_map[int(Keyboard::Key::Down)] = TW_KEY_DOWN;
    key_map[int(Keyboard::Key::Up)] = TW_KEY_UP;
    key_map[int(Keyboard::Key::PageUp)] = TW_KEY_PAGE_UP;
    key_map[int(Keyboard::Key::PageDown)] = TW_KEY_PAGE_DOWN;
    key_map[int(Keyboard::Key::Home)] = TW_KEY_HOME;
    key_map[int(Keyboard::Key::End)] = TW_KEY_END;

    for (int k = int(Keyboard::Key::F1); k <= int(Keyboard::Key::F15); ++k)
        key_map[k] = TW_KEY_F1 + k - int(Keyboard::Key::F1);

    for (int k = int(Keyboard::Key::Keypad0); k <= int(Keyboard::Key::Keypad9); ++k)
        key_map[k] = '0' + k - int(Keyboard::Key::Keypad0);

    key_map[int(Keyboard::Key::KeypadDecimal)] = '.';
    key_map[int(Keyboard::Key::KeypadDivide)] = '/';
    key_map[int(Keyboard::Key::KeypadMultiply)] = '*';
    key_map[int(Keyboard::Key::KeypadSubtract)] = '-';
    key_map[int(Keyboard::Key::KeypadAdd)] = '+';
    key_map[int(Keyboard::Key::KeypadEnter)] = TW_KEY_RETURN;
    key_map[int(Keyboard::Key::KeypadEqual)] = '=';

    return key_map;
}

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
        static int* key_map = nullptr;
        if (key_map == nullptr)
            key_map = create_keyboard_key_map();

        const Keyboard* const keyboard = engine.get_keyboard();

        int modifiers = 0;
        if (keyboard->is_pressed(Keyboard::Key::LeftControl) || keyboard->is_pressed(Keyboard::Key::RightControl))
            modifiers |= TW_KMOD_CTRL;
        if (keyboard->is_pressed(Keyboard::Key::LeftAlt) || keyboard->is_pressed(Keyboard::Key::RightAlt))
            modifiers |= TW_KMOD_ALT;
        if (keyboard->is_pressed(Keyboard::Key::LeftShift) || keyboard->is_pressed(Keyboard::Key::RightShift))
            modifiers |= TW_KMOD_SHIFT;

        for (int k = 0; k < int(Keyboard::Key::KeyCount); ++k)
            if (keyboard->was_pressed(Keyboard::Key(k))) {
                int key = key_map[k];
                bool upper_case = int(Keyboard::Key::A) <= k && k <= int(Keyboard::Key::Z) && (modifiers & TW_KMOD_SHIFT) != 0;
                if (upper_case)
                    key += 'A' - 'a';
                TwKeyPressed(key, modifiers);
            }
    }
}

} // NS AntTweakBar
