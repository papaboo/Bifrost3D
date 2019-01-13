// Test Bifrost Keyboard.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_INPUT_KEYBOARD_TEST_H_
#define _BIFROST_INPUT_KEYBOARD_TEST_H_

#include <Bifrost/Input/Keyboard.h>

#include <gtest/gtest.h>

namespace Bifrost {
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

GTEST_TEST(Input_Keyboard, was_pressed_and_released) {
    Keyboard keyboard = Keyboard();

    EXPECT_FALSE(keyboard.is_pressed(Keyboard::Key::A));

    // Key held down.
    keyboard.key_tapped(Keyboard::Key::A, true);
    EXPECT_TRUE(keyboard.is_pressed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.is_released(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.was_pressed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_released(Keyboard::Key::A));

    // Key held down and released.
    keyboard.key_tapped(Keyboard::Key::A, false);
    EXPECT_FALSE(keyboard.is_pressed(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.is_released(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.was_pressed(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.was_released(Keyboard::Key::A));

    // Key held down, released and held down again.
    keyboard.key_tapped(Keyboard::Key::A, true);
    EXPECT_TRUE(keyboard.is_pressed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.is_released(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.was_pressed(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.was_released(Keyboard::Key::A));

    keyboard.per_frame_reset();

    // Key held down from before reset.
    EXPECT_TRUE(keyboard.is_pressed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.is_released(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_pressed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_released(Keyboard::Key::A));

    // Key held down and relased.
    keyboard.key_tapped(Keyboard::Key::A, false);
    EXPECT_FALSE(keyboard.is_pressed(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.is_released(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_pressed(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.was_released(Keyboard::Key::A));
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

GTEST_TEST(Input_Keyboard, consumed_keys) {
    Keyboard keyboard = Keyboard();

    EXPECT_FALSE(keyboard.is_consumed(Keyboard::Key::A));

    keyboard.key_tapped(Keyboard::Key::A, true);
    EXPECT_FALSE(keyboard.is_consumed(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.was_pressed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_released(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.is_pressed(Keyboard::Key::A));

    // Consuming a key clears its 'was' flags and shows it as being released, since logically either the pressed or the released state should be true.
    keyboard.consume_button_event(Keyboard::Key::A);
    EXPECT_TRUE(keyboard.is_consumed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_pressed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_released(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.is_pressed(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.is_released(Keyboard::Key::A));

    // Resetting per frame clears the consumed flag.
    keyboard.per_frame_reset();
    EXPECT_FALSE(keyboard.is_consumed(Keyboard::Key::A));

    keyboard.key_tapped(Keyboard::Key::A, false);
    EXPECT_FALSE(keyboard.is_consumed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_pressed(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.was_released(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.is_released(Keyboard::Key::A));

    keyboard.consume_button_event(Keyboard::Key::A);
    EXPECT_TRUE(keyboard.is_consumed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_pressed(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.was_released(Keyboard::Key::A));
    EXPECT_FALSE(keyboard.is_pressed(Keyboard::Key::A));
    EXPECT_TRUE(keyboard.is_released(Keyboard::Key::A));
}

} // NS Input
} // NS Bifrost

#endif // _BIFROST_INPUT_KEYBOARD_TEST_H_
