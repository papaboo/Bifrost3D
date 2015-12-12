// Test Cogwheel Keyboard.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_INPUT_KEYBOARD_TEST_H_
#define _COGWHEEL_INPUT_KEYBOARD_TEST_H_

#include <Input/Keyboard.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Input {

GTEST_TEST(Input_Keyboard, Initial_state) {
    Keyboard keyboard = Keyboard();
    for (int k = 0; k < (int)Keyboard::Key::KeyCount; ++k) {
        EXPECT_EQ(keyboard.isPressed(Keyboard::Key(k)), false);
        EXPECT_EQ(keyboard.halftaps(Keyboard::Key(k)), 0u);
    }
}

GTEST_TEST(Input_Keyboard, key_taps) {
    Keyboard keyboard = Keyboard();
    
    keyboard.keyTapped(Keyboard::Key::C, true);
    EXPECT_TRUE(keyboard.isPressed(Keyboard::Key::C));
    EXPECT_EQ(keyboard.halftaps(Keyboard::Key::C), 1u);

    keyboard.keyTapped(Keyboard::Key::B, true);
    keyboard.keyTapped(Keyboard::Key::B, false);
    EXPECT_FALSE(keyboard.isPressed(Keyboard::Key::B));
    EXPECT_EQ(keyboard.halftaps(Keyboard::Key::B), 2u);

    keyboard.keyTapped(Keyboard::Key::A, true);
    keyboard.keyTapped(Keyboard::Key::A, false);
    keyboard.keyTapped(Keyboard::Key::A, true);
    EXPECT_TRUE(keyboard.isPressed(Keyboard::Key::A));
    EXPECT_EQ(keyboard.halftaps(Keyboard::Key::A), 3u);
}

GTEST_TEST(Input_Keyboard, resetting) {
    Keyboard keyboard = Keyboard();

    keyboard.keyTapped(Keyboard::Key::A, true);
    keyboard.keyTapped(Keyboard::Key::B, true);
    keyboard.keyTapped(Keyboard::Key::C, true);

    for (int k = 0; k < (int)Keyboard::Key::KeyCount; ++k) {
        Keyboard::Key key = Keyboard::Key(k);
        if (key == Keyboard::Key::A || key == Keyboard::Key::B || key == Keyboard::Key::C) {
            EXPECT_TRUE(keyboard.isPressed(key));
            EXPECT_EQ(keyboard.halftaps(key), 1u);
        } else {
            EXPECT_FALSE(keyboard.isPressed(key));
            EXPECT_EQ(keyboard.halftaps(key), 0u);
        }
    }

    keyboard.resetKeyState();
    for (int k = 0; k < (int)Keyboard::Key::KeyCount; ++k) {
        Keyboard::Key key = Keyboard::Key(k);
        if (key == Keyboard::Key::A || key == Keyboard::Key::B || key == Keyboard::Key::C)
            EXPECT_TRUE(keyboard.isPressed(key));
        else
            EXPECT_FALSE(keyboard.isPressed(key));
        EXPECT_EQ(keyboard.halftaps(key), 0u);
    }
}

GTEST_TEST(Input_Keyboard, tap_overflow_handling) {
    Keyboard keyboard = Keyboard();

    // Simulate pressing and relasing A more times than can be represented by the halftap precision.
    for (int i = 0; i < Keyboard::MAX_HALFTAP_COUNT+1; ++i) {
        keyboard.keyTapped(Keyboard::Key::A, true);
        keyboard.keyTapped(Keyboard::Key::A, false);
    }

    for (int i = 0; i < 3; ++i) {

        keyboard.keyTapped(Keyboard::Key::A, true);

        EXPECT_TRUE(keyboard.isPressed(Keyboard::Key::A));
        bool oddNumberOfHalftaps = (keyboard.halftaps(Keyboard::Key::A) % 2) == 1;
        EXPECT_TRUE(oddNumberOfHalftaps);
        EXPECT_GE(keyboard.halftaps(Keyboard::Key::A), Keyboard::MAX_HALFTAP_COUNT - 1);

        keyboard.keyTapped(Keyboard::Key::A, false);

        EXPECT_TRUE(keyboard.isReleased(Keyboard::Key::A));
        bool evenNumberOfHalftaps = (keyboard.halftaps(Keyboard::Key::A) % 2) == 0;
        EXPECT_TRUE(evenNumberOfHalftaps);
        EXPECT_GE(keyboard.halftaps(Keyboard::Key::A), Keyboard::MAX_HALFTAP_COUNT - 1);
    }
}

} // NS Input
} // NS Cogwheel

#endif // _COGWHEEL_INPUT_KEYBOARD_TEST_H_