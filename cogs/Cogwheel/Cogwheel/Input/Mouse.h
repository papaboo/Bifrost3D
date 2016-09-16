// Cogwheel mouses input.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_INPUT_MOUSE_H_
#define _COGWHEEL_INPUT_MOUSE_H_

#include <Cogwheel/Math/Vector.h>

#include <array>

namespace Cogwheel {
namespace Input {

//----------------------------------------------------------------------------
// Implementats a mouse abstraction.
// The mouse supports up to MAX_HALFTAP_COUNT half taps, presses and releases, 
// pr frame. This can be used to implement such interactions as double tap 
// for dash without worrying (too much) about the framerate.
//----------------------------------------------------------------------------
class Mouse final {
public:
    static const int MAX_HALFTAP_COUNT = 127;

    enum class Button : unsigned char {
        Left,
        Right,
        Middle,
        Button4,
        ButtonCount
    };

    // 8 bit struct containing state of a key; is it pressed or released and how many times was it pressed last frame.
    struct ButtonState {
        bool is_pressed : 1;
        unsigned char halftaps : 7;
    };

    Mouse(Math::Vector2i initial_position)
        : m_position(initial_position)
        , m_delta(Math::Vector2i(0, 0))
        , m_scroll_delta(0.0f) {

        for (ButtonState& button : m_button_states) {
            button.halftaps = 0u;
            button.is_pressed = false;
        }
    }

    inline void set_position(Math::Vector2i new_position) {
        m_delta += new_position - m_position;
        m_position = new_position;
    }

    inline Math::Vector2i get_position() const { return m_position; }
    inline Math::Vector2i get_delta() const { return m_delta; }

    inline void button_tapped(int buttonId, bool pressed) {
        m_button_states[buttonId].is_pressed = pressed;
        unsigned int halftaps = m_button_states[buttonId].halftaps;
        m_button_states[buttonId].halftaps = (halftaps == MAX_HALFTAP_COUNT) ? (MAX_HALFTAP_COUNT - 1) : (halftaps + 1); // Checking for overflow! In case of overflow the tap count is reduced by one to maintain proper even/odd tap count relationship.
    }

    inline bool is_pressed(Button button) const { return m_button_states[(unsigned int)button].is_pressed; }
    inline bool is_released(Button button) const { return !is_pressed(button); }
    inline unsigned int halftaps(Button button) const { return m_button_states[(unsigned int)button].halftaps; }

    inline bool was_pressed(Button button) const {
        const ButtonState state = m_button_states[(unsigned int)button];
        return (state.is_pressed && state.halftaps == 1) || state.halftaps > 1;
    }
    inline bool was_released(Button button) const {
        const ButtonState state = m_button_states[(unsigned int)button];
        return (!state.is_pressed && state.halftaps == 1) || state.halftaps > 1;
    }

    inline void add_scroll_delta(float scroll_delta) { m_scroll_delta += scroll_delta; }
    inline float get_scroll_delta() const { return m_scroll_delta; }

    inline void per_frame_reset() {
        m_delta = Math::Vector2i::zero();

        for (ButtonState& button : m_button_states)
            button.halftaps = 0u;

        m_scroll_delta = 0.0f;
    }

private:
    Math::Vector2i m_position;
    Math::Vector2i m_delta;

    std::array<ButtonState, (unsigned int)Button::ButtonCount> m_button_states;

    float m_scroll_delta;
};

} // NS Input
} // NS Cogwheel

#endif // _COGWHEEL_INPUT_MOUSE_H_