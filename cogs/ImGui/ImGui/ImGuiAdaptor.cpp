// Cogwheel Imgui adaptor.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <ImGui/ImGuiAdaptor.h>

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>
#include <Cogwheel/Math/Vector.h>

using namespace Cogwheel::Input;
using namespace Cogwheel::Math;

namespace ImGui {

ImGuiAdaptor::ImGuiAdaptor(const Cogwheel::Core::Engine& engine) {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io; // TODO Copied from examples. No idea what this is.

    ImGui::StyleColorsDark();

    // TODO Set io.keymap
}

void ImGuiAdaptor ::new_frame(const Cogwheel::Core::Engine& engine) {
    if (ImGui::GetCurrentContext() == NULL)
        return;

    ImGuiIO& io = ImGui::GetIO();

    // Handle window.
    const Cogwheel::Core::Window& window = engine.get_window();
    Vector2i window_size = { window.get_width(), window.get_height() };
    io.DisplaySize = ImVec2((float)window_size.x, (float)window_size.y);

    io.DeltaTime = engine.get_time().get_smooth_delta_time();

    { // Handle mouse input.
        const Mouse* const mouse = engine.get_mouse();

        Vector2i mouse_pos = mouse->get_position();
        io.MousePos = { float(mouse_pos.x), float(mouse_pos.y) };

        for (int k = 0; k < 5; ++k)
            io.MouseDown[k] = mouse->is_pressed((Mouse::Button)k);

        /*
        static float mouse_wheel_pos = 0;
        mouse_wheel_pos += mouse->get_scroll_delta();
        TwMouseWheel((int)mouse_wheel_pos);
        */
    }

    { // TODO Handle keyboard

    }

    ImGui::NewFrame();

    for (auto& frame : m_frames)
        frame->layout_frame();
}

void ImGuiAdaptor::add_frame(ImGuiFrameCreator frame_creator) {
    IImGuiFrame* frame = frame_creator();
    m_frames.push_back(std::unique_ptr<IImGuiFrame>(frame));
}

} // NS ImGui
