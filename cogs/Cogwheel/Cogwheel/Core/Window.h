// Cogwheel window.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_WINDOW_H_
#define _COGWHEEL_CORE_WINDOW_H_

#include <string>

namespace Cogwheel {
namespace Core {

// ---------------------------------------------------------------------------
// Window class.
// Contains the size and name of windows.
// Future work
// * Is fullscreen bool and supported fullscreen resolutions.
//   The question is how the rendering backends should handle this.
//   Does OptiX need a new context if we go fullscreen? What about GL?
// ---------------------------------------------------------------------------
class Window final {
public:
    Window(const std::string& name, int width, int height)
        : m_name(name), m_width(width), m_height(height) { }

    inline std::string get_name() const { return m_name; }
    inline void set_name(const std::string& name) { m_name = name; }

    inline int get_width() const { return m_width; }
    inline int get_height() const { return m_height; }
    inline float get_aspect_ratio() const { return m_width / float(m_height); }

    inline void resize(int width, int height) {
        m_width = width; m_height = height;
    }

private:
    std::string m_name;
    int m_width, m_height;
};

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_WINDOW_H_