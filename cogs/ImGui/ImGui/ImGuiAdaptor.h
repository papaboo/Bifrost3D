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
    virtual void layout_frame() = 0;
};

typedef IImGuiFrame*(*ImGuiFrameCreator)();
// ------------------------------------------------------------------------------------------------
// Dear IMGUI adaptor for Cogwheel input.
// Future work:
// * Disable keyboard and mouse events based on io.WantCaptureMouse and io.WantCaptureKeyboard.
// * Load and set fonts.
// ------------------------------------------------------------------------------------------------
class ImGuiAdaptor {
public:
    ImGuiAdaptor(const Cogwheel::Core::Engine& engine);

    void new_frame(const Cogwheel::Core::Engine& engine);

    void add_frame(ImGuiFrameCreator frame_creator);

private:
    std::vector<std::unique_ptr<IImGuiFrame>> m_frames;
};

static inline void new_frame_callback(Cogwheel::Core::Engine& engine, void* adaptor) {
    static_cast<ImGuiAdaptor*>(adaptor)->new_frame(engine);
}

} // NS ImGui

#endif // _COGWHEEL_IMGUI_ADAPTOR_H_