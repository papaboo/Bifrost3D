// Test Cogwheel Keyboard.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_INPUT_KEYBOARD_TEST_H_
#define _COGWHEEL_INPUT_KEYBOARD_TEST_H_

#include <Cogwheel/Input/Keyboard.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Input {

GTEST_TEST(Input_Keyboard, initial_state) {
    Keyboard keyboard = Keyboard();
    for (int k = 0; k < (int)Keyboard::Key::KeyCount; ++k) {
        EXPECT_EQ(keyboard.is_pressed(Keyboard::Key(k)), false);
        EXPECT_EQ(keyboard.halftaps(Keyboard::Key(k)), 0u);
    }
}

GTEST_TEST(Input_Keyboard, key_taps) {
    Keyboard keyboard = Keyboard();
    
    keyboard.key_tapped(Keyboard::Key::C, true);
    EXPECT_TRUE(keyboard.is_pressed(Keyboard::Key::C));
    EXPECT_EQ(keyboard.halftaps(Keyboard::Key::C), 1u);

    keyboard.key_tapped(Keyboard::Key::B, true);
    keyboard.key_tapped(Keyboard::Key::B, false);
    EXPECT_FALSE(keyboard.is_pressed(Keyboard::Key::B));
    EXPECT_EQ(keyboard.halftaps(Keyboard::Key::B), 2u);

    keyboard.key_tapped(Keyboard::Key::A, true);
    keyboard.key_tapped(Keyboard::Key::A, false);
    keyboard.key_tapped(Keyboard::Key::A, true);
    EXPECT_TRUE(keyboard.is_pressed(Keyboard::Key::A));
    EXPECT_EQ(keyboard.halftaps(Keyboard::Key::A), 3u);
}

GTEST_TEST(Input_Keyboard, resetting) {
    Keyboard keyboard = Keyboard();

    keyboard.key_tapped(Keyboard::Key::A, true);
    keyboard.key_tapped(Keyboard::Key::B, true);
    keyboard.key_tapped(Keyboard::Key::C, true);

    for (int k = 0; k < (int)Keyboard::Key::KeyCount; ++k) {
        Keyboard::Key key = Keyboard::Key(k);
        if (key == Keyboard::Key::A || key == Keyboard::Key::B || key == Keyboard::Key::C) {
            EXPECT_TRUE(keyboard.is_pressed(key));
            EXPECT_EQ(keyboard.halftaps(key), 1u);
        } else {
            EXPECT_FALSE(keyboard.is_pressed(key));
            EXPECT_EQ(keyboard.halftaps(key), 0u);
        }
    }

    keyboard.per_frame_reset();
    for (int k = 0; k < (int)Keyboard::Key::KeyCount; ++k) {
        Keyboard::Key key = Keyboard::Key(k);
        if (key == Keyboard::Key::A || key == Keyboard::Key::B || key == Keyboard::Key::C)
            EXPECT_TRUE(keyboard.is_pressed(key));
        else
            EXPECT_FALSE(keyboard.is_pressed(key));
        EXPECT_EQ(keyboard.halftaps(key), 0u);
    }
}

GTEST_TEST(Input_Keyboard, tap_overflow_handling) {
    Keyboard keyboard = Keyboard();

    // Simulate pressing and relasing A more times than can be represented by the halftap precision.
    for (int i = 0; i < Keyboard::MAX_HALFTAP_COUNT+1; ++i) {
        keyboard.key_tapped(Keyboard::Key::A, true);
        keyboard.key_tapped(Keyboard::Key::A, false);
    }

    for (int i = 0; i < 3; ++i) {

        keyboard.key_tapped(Keyboard::Key::A, true);

        EXPECT_TRUE(keyboard.is_pressed(Keyboard::Key::A));
        bool odd_number_of_halftaps = (keyboard.halftaps(Keyboard::Key::A) % 2) == 1;
        EXPECT_TRUE(odd_number_of_halftaps);
        EXPECT_GE(keyboard.halftaps(Keyboard::Key::A), Keyboard::MAX_HALFTAP_COUNT - 1u);

        keyboard.key_tapped(Keyboard::Key::A, false);

        EXPECT_TRUE(keyboard.is_released(Keyboard::Key::A));
        bool even_number_of_halftaps = (keyboard.halftaps(Keyboard::Key::A) % 2) == 0;
        EXPECT_TRUE(even_number_of_halftaps);
        EXPECT_GE(keyboard.halftaps(Keyboard::Key::A), Keyboard::MAX_HALFTAP_COUNT - 1u);
    }
}

} // NS Input
} // NS Cogwheel

#endif // _COGWHEEL_INPUT_KEYBOARD_TEST_H_