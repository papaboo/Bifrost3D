// Cogwheel keyboard input.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_INPUT_KEYBOARD_H_
#define _COGWHEEL_INPUT_KEYBOARD_H_

#include <array>

namespace Cogwheel {
namespace Input {

class Keyboard {
public:

    static const int MAX_HALFTAP_COUNT = 7;

    // Maps to the same key values as GLFW to make the initial implementation faster.
    // TODO Remap the values. GLFW's values are not tight and does in fact have a ton of holes between them, which causes us to use more memory than we really should, when allocating an array of length 'KeyCount'. Instead try mapping to ASCII + additional keys.
    // ASCII http://www.asciitable.com/
    // GLFW  http://www.glfw.org/docs/latest/group__keys.html#gac556b360f7f6fca4b70ba0aecf313fd4
    // SDL   http://www.libsdl.org/release/SDL-1.2.15/include/SDL_keysym.h
    enum class Key {
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
        RrightBracket = 93,
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
        for (KeyState& state : mKeyStates) {
            state.isPressed = false;
            state.halftaps = 0u;
        }
    }

    bool isPressed(Key key) const { return mKeyStates[(int)key].isPressed; }
    bool isReleased(Key key) const { return !isPressed(key); }
    int halftaps(Key key) const { return mKeyStates[(int)key].halftaps; }

    void keyTapped(Key key, bool pressed) {
        mKeyStates[(int)key].isPressed = pressed;
        unsigned int halftaps = mKeyStates[(int)key].halftaps;
        mKeyStates[(int)key].halftaps = halftaps == MAX_HALFTAP_COUNT ? MAX_HALFTAP_COUNT-1 : (halftaps + 1); // Checking for overflow! In case of overflow the tap count is reduced by one to maintain proper even/odd tap count relationship.
    }

    void resetKeyState() {
        for (KeyState& state : mKeyStates)
            state.halftaps = 0u;
    }

private:

    // 8 bit struct containing state of a key; is it pressed or released and how many times has it been pressed per frame.
    // TODO
    //    3 bits for halftaps should be enough. Compress the keystate to 4 bits and store two states pr 8 bit.
    struct KeyState {
        bool isPressed : 1;
        unsigned int halftaps : 7;
    };

    std::array<KeyState, (int)Key::KeyCount> mKeyStates;
};

} // NS Input
} // NS Cogwheel

#endif // _COGWHEEL_INPUT_KEYBOARD_H_