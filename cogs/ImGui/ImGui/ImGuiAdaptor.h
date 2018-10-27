// Cogwheel ImGui adaptor.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_IMGUI_ADAPTOR_H_
#define _COGWHEEL_IMGUI_ADAPTOR_H_

#include <ImGui/Src/imgui.h> // Convenience include of ImGui functionality

#include <memory>
#include <vector>

// ------------------------------------------------------------------------------------------------
// Forward declerations
// ------------------------------------------------------------------------------------------------
namespace Cogwheel {
namespace Core {
class Engine;
}
}

namespace ImGui {

class IImGuiFrame {
public:
    virtual ~IImGuiFrame() {}
    virtual void layout_frame() = 0;
};

// ------------------------------------------------------------------------------------------------
// Dear IMGUI adaptor for Cogwheel input.
// Future work:
// * Disable keyboard and mouse events based on io.WantCaptureMouse and io.WantCaptureKeyboard.
// * Load and set fonts.
// * Support setting different mouse cursor images.
// ------------------------------------------------------------------------------------------------
class ImGuiAdaptor {
public:
    ImGuiAdaptor();

    void new_frame(const Cogwheel::Core::Engine& engine);

    inline void add_frame(std::unique_ptr<IImGuiFrame> frame) {
        // Ridiculous workaround for VS2015 attempting to reference some deleted function if frame is pushed directly.
        IImGuiFrame* _frame = frame.release();
        m_frames.push_back(std::unique_ptr<IImGuiFrame>(_frame));
    }

    bool is_enabled() const { return m_enabled; }
    void set_enabled(bool enabled) { m_enabled = enabled; }

private:
    std::vector<std::unique_ptr<IImGuiFrame>> m_frames;
    bool m_enabled;
};

static inline void new_frame_callback(Cogwheel::Core::Engine& engine, void* adaptor) {
    static_cast<ImGuiAdaptor*>(adaptor)->new_frame(engine);
}

} // NS ImGui

#endif // _COGWHEEL_IMGUI_ADAPTOR_H_