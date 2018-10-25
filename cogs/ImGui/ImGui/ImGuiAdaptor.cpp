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

ImGuiAdaptor::ImGuiAdaptor()
    : m_enabled(true) {

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    // io.ImeWindowHandle = hwnd;

    ImGui::StyleColorsDark();

    io.KeyMap[ImGuiKey_Tab] = int(Keyboard::Key::Tab);
    io.KeyMap[ImGuiKey_LeftArrow] = int(Keyboard::Key::Left);
    io.KeyMap[ImGuiKey_RightArrow] = int(Keyboard::Key::Right);
    io.KeyMap[ImGuiKey_UpArrow] = int(Keyboard::Key::Up);
    io.KeyMap[ImGuiKey_DownArrow] = int(Keyboard::Key::Down);
    io.KeyMap[ImGuiKey_PageUp] = int(Keyboard::Key::PageUp);
    io.KeyMap[ImGuiKey_PageDown] = int(Keyboard::Key::PageDown);
    io.KeyMap[ImGuiKey_Home] = int(Keyboard::Key::Home);
    io.KeyMap[ImGuiKey_End] = int(Keyboard::Key::End);
    io.KeyMap[ImGuiKey_Insert] = int(Keyboard::Key::Insert);
    io.KeyMap[ImGuiKey_Delete] = int(Keyboard::Key::Delete);
    io.KeyMap[ImGuiKey_Backspace] = int(Keyboard::Key::Backspace);
    io.KeyMap[ImGuiKey_Space] = int(Keyboard::Key::Space);
    io.KeyMap[ImGuiKey_Enter] = int(Keyboard::Key::Enter);
    io.KeyMap[ImGuiKey_Escape] = int(Keyboard::Key::Escape);
    io.KeyMap[ImGuiKey_A] = int(Keyboard::Key::A);
    io.KeyMap[ImGuiKey_C] = int(Keyboard::Key::C);
    io.KeyMap[ImGuiKey_V] = int(Keyboard::Key::V);
    io.KeyMap[ImGuiKey_X] = int(Keyboard::Key::X);
    io.KeyMap[ImGuiKey_Y] = int(Keyboard::Key::Y);
    io.KeyMap[ImGuiKey_Z] = int(Keyboard::Key::Z);
}

void ImGuiAdaptor ::new_frame(const Cogwheel::Core::Engine& engine) {
    ImGuiIO& io = ImGui::GetIO();

    // Handle window.
    const Cogwheel::Core::Window& window = engine.get_window();
    Vector2i window_size = { window.get_width(), window.get_height() };
    io.DisplaySize = ImVec2((float)window_size.x, (float)window_size.y);

    io.DeltaTime = engine.get_time().get_smooth_delta_time();

    Mouse* mouse = (Mouse*)engine.get_mouse();
    Keyboard* keyboard = (Keyboard*)engine.get_keyboard();

    { // Handle mouse input.
        Vector2i mouse_pos = mouse->get_position();
        io.MousePos = { float(mouse_pos.x), float(mouse_pos.y) };

        for (int k = 0; k < 5; ++k)
            io.MouseDown[k] = mouse->is_pressed((Mouse::Button)k);

        io.MouseWheel += mouse->get_scroll_delta();
    }

    { // Handle keyboard
        // Modifiers
        io.KeyCtrl = keyboard->is_pressed(Keyboard::Key::LeftControl) || keyboard->is_pressed(Keyboard::Key::RightControl);
        io.KeyShift = keyboard->is_pressed(Keyboard::Key::LeftShift) || keyboard->is_pressed(Keyboard::Key::RightShift);
        io.KeyAlt = keyboard->is_pressed(Keyboard::Key::LeftAlt) || keyboard->is_pressed(Keyboard::Key::RightAlt);
        io.KeySuper = keyboard->is_pressed(Keyboard::Key::LeftSuper) || keyboard->is_pressed(Keyboard::Key::RightSuper);

        // Keymapped keys
        for (int k = 0; k < ImGuiKey_COUNT; ++k)
            io.KeysDown[io.KeyMap[k]] = keyboard->is_pressed(Keyboard::Key(io.KeyMap[k]));

        // Text.
        const std::wstring& text = keyboard->get_text();
        for (int i = 0; i < text.length(); ++i)
            io.AddInputCharacter(text[i]);
    }

    ImGui::NewFrame();

    if (m_enabled)
        for (auto& frame : m_frames)
            frame->layout_frame();

    if (io.WantCaptureMouse) {
        mouse->consume_all_button_events();
        mouse->consume_scroll_delta();
    }
    if (io.WantCaptureKeyboard)
        keyboard->consume_all_button_events();
}

} // NS ImGui
