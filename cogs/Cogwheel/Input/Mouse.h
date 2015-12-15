// Cogwheel mouses input.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_INPUT_MOUSE_H_
#define _COGWHEEL_INPUT_MOUSE_H_

#include <Math/Vector.h>

namespace Cogwheel {
namespace Input {

class Mouse {
public:

    Math::Vector2i position;
    Math::Vector2i delta;

    static const int BUTTON_COUNT = 4;
    static const int MAX_HALFTAP_COUNT = 127;

    // 8 bit struct containing state of a key; is it pressed or released and how many times was it pressed last frame.
    struct ButtonState {
        bool isPressed : 1;
        unsigned int halftaps : 7;
    };

    ButtonState leftButton;
    ButtonState rightButton;
    ButtonState middleButton;
    ButtonState button4;

    float scrollDelta;

    inline void buttonTapped(int buttonId, bool pressed) {
        ButtonState* buttons = &leftButton;
        buttons[buttonId].isPressed = pressed;
        unsigned int halftaps = buttons[buttonId].halftaps;
        buttons[buttonId].halftaps = halftaps == MAX_HALFTAP_COUNT ? MAX_HALFTAP_COUNT - 1 : (halftaps + 1); // Checking for overflow! In case of overflow the tap count is reduced by one to maintain proper even/odd tap count relationship.
    }

    inline void perFrameReset() {
        delta = Math::Vector2i::zero();

        leftButton.halftaps = 0u;
        rightButton.halftaps = 0u;
        middleButton.halftaps = 0u;
        button4.halftaps = 0u;

        scrollDelta = 0.0f;
    }

};

} // NS Input
} // NS Cogwheel

#endif // _COGWHEEL_INPUT_MOUSE_H_