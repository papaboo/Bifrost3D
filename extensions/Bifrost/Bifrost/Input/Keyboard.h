// Bifrost keyboard input.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_INPUT_KEYBOARD_H_
#define _BIFROST_INPUT_KEYBOARD_H_

#include <array>

namespace Bifrost {
namespace Input {

//----------------------------------------------------------------------------
// Implementats a keyboard abstraction.
// The keyboard supports up to MAX_HALFTAP_COUNT half taps, press or release, 
// pr frame. This can be used to implement such interactions as double tap 
// for dash without worrying (too much) about the framerate.
//
// Future work
// * 3 bits should be enough for halftaps. 
//   Compress Keystate to 5 bits and store 6 states pr. 32 bit uint.
// * Remap the key enumeration. GLFW's values are not tight and does in fact 
//   have a ton of holes between them, which causes us to use more memory than
//   necessary when allocating an array of length 'KeyCount'.
//   Instead try mapping to ASCII + additional keys.
//----------------------------------------------------------------------------
class Keyboard final {
public:

    static const int MAX_HALFTAP_COUNT = 7;

    // Maps to the same key values as GLFW to make the initial implementation faster.
    // ASCII http://www.asciitable.com/
    // GLFW  http://www.glfw.org/docs/latest/group__keys.html#gac556b360f7f6fca4b70ba0aecf313fd4
    // SDL   http://www.libsdl.org/release/SDL-1.2.15/include/SDL_keysym.h
    enum class Key : short {
        Invalid = 0,
        Space = 32,
        Apostrophe = 39,
        Comma = 44,
        Minus = 45,
        Period = 46,
        Slash = 47,

        // Numerical keys [48, 57]
        Key0 = 48, Key1, Key2, Key3, Key4, Key5, Key6, Key7, Key8, Key9 = 57,
        Semicolon = 59,
        Equal = 61,

        // Letters [65, 90]
        A = 65, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = 90,

        LeftBracket = 91,
        Backslash = 92,
        RightBracket = 93,
        GraveAccent = 96,

        // ??
        World1 = 161, World2 = 162,

        Escape = 256,
        Enter = 257,
        Tab = 258,
        Backspace = 259,
        Insert = 260,
        Delete = 261,
        Right = 262,
        Left = 263,
        Down = 264,
        Up = 265,
        PageUp = 266,
        PageDown = 267,
        Home = 268,
        End = 269,
        CapsLock = 280,
        ScrollLock = 281,
        NumLock = 282,
        PrintScreen = 283,
        Pause = 284,

        // F-keys, apparently up to F25 \o/ [290, 314]
        F1 = 290, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25 = 314,

        // Keypad
        Keypad0 = 320, Keypad1 = 321, Keypad2, Keypad3, Keypad4, Keypad5, Keypad6, Keypad7, Keypad8, Keypad9 = 329,
        KeypadDecimal = 330,
        KeypadDivide = 331,
        KeypadMultiply = 332,
        KeypadSubtract = 333,
        KeypadAdd = 334,
        KeypadEnter = 335,
        KeypadEqual = 336,

        // Modifiers [340, 347]
        LeftShift = 340,
        LeftControl = 341,
        LeftAlt = 342,
        LeftSuper = 343,
        RightShift = 344,
        RightControl = 345,
        RightAlt = 346,
        RightSuper = 347,

        Menu = 348, // ????
        KeyCount
    };

    Keyboard() {
        for (KeyState& state : m_key_states) {
            state.is_consumed = false;
            state.is_pressed = false;
            state.halftaps = 0u;
        }
        m_text.reserve(32);
    }

    inline bool is_pressed(Key key) const { return m_key_states[(short)key].is_pressed && !m_key_states[(short)key].is_consumed; }
    inline bool is_released(Key key) const { return !is_pressed(key); }
    inline unsigned int halftaps(Key key) const { return m_key_states[(short)key].halftaps; }

    inline bool was_pressed(Key key) const {
        const KeyState state = m_key_states[(short)key];
        return ((state.is_pressed && state.halftaps == 1) || state.halftaps > 1) && !state.is_consumed;
    }
    inline bool was_released(Key key) const {
        const KeyState state = m_key_states[(short)key];
        return ((!state.is_pressed && state.halftaps == 1) || state.halftaps > 1) && !state.is_consumed;
    }

    inline void key_tapped(Key key, bool pressed) {
        m_key_states[(unsigned short)key].is_pressed = pressed;
        unsigned int halftaps = m_key_states[(short)key].halftaps;
        m_key_states[(unsigned short)key].halftaps = (halftaps == MAX_HALFTAP_COUNT) ? (MAX_HALFTAP_COUNT-1) : (halftaps + 1); // Checking for overflow! In case of overflow the tap count is reduced by one to maintain proper even/odd tap count relationship.
    }

    inline bool is_modifiers_pressed() const {
        return is_pressed(Key::LeftShift) || is_pressed(Key::LeftControl) || is_pressed(Key::LeftAlt) || is_pressed(Key::LeftSuper)
            || is_pressed(Key::RightShift) || is_pressed(Key::RightControl) || is_pressed(Key::RightAlt) || is_pressed(Key::RightSuper);
    }

    inline void add_codepoint(wchar_t codepoint) {
        m_text.append(&codepoint, 1);
    }

    inline const std::wstring& get_text() const { return m_text; }

    inline bool is_consumed(Key key) const { return m_key_states[short(key)].is_consumed; }
    inline void consume_button_event(Key key) { m_key_states[short(key)].is_consumed = true; }

    inline void consume_all_button_events() {
        for (KeyState& state : m_key_states)
            state.is_consumed = true;
    }

    inline void per_frame_reset() {
        for (KeyState& state : m_key_states) {
            state.is_consumed = false;
            state.halftaps = 0u;
        }
        m_text.clear();
    }

private:

    // 8 bit struct containing state of a key; is it pressed or released and how many times was it pressed last frame.
    struct KeyState {
        bool is_consumed : 1;
        bool is_pressed : 1;
        unsigned char halftaps : 6;
    };

    std::array<KeyState, (unsigned int)Key::KeyCount> m_key_states;

    std::wstring m_text;
};

} // NS Input
} // NS Bifrost

#endif // _BIFROST_INPUT_KEYBOARD_H_
