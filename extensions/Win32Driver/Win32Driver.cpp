// Bifrost Win32 main.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#include <Win32Driver.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#undef RGB

#include <Bifrost/Core/Engine.h>
#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Input/Mouse.h>

using Bifrost::Core::Engine;
using Bifrost::Input::Keyboard;
using Bifrost::Input::Mouse;
using Bifrost::Math::Vector2i;

namespace Win32Driver {

Engine* g_engine = NULL;
Keyboard* g_keyboard = NULL;
Mouse* g_mouse = NULL;

// Windows message handler callback.
LRESULT CALLBACK handle_messages(HWND window_handle, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_SIZE: {
        int width = GET_X_LPARAM(lParam); 
        int height = GET_Y_LPARAM(lParam);
        if (g_engine->get_window().get_width() != width || g_engine->get_window().get_height() != height)
            g_engine->get_window().resize(width, height);
        break;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(window_handle, message, wParam, lParam);
    }

    return 0;
}

// Incredibly heavily inspired by GLFW's keyboard handling. See win32_window.c, translateKey() and referencing code.
Keyboard::Key translate_key(WPARAM wParam, LPARAM lParam) {
    using Key = Keyboard::Key;

    static Key keymap[512] = {
        // [0, 15]
        Key::Invalid, Key::Escape, Key::Key1, Key::Key2, Key::Key3, Key::Key4, Key::Key5, Key::Key6, Key::Key7, Key::Key8, Key::Key9, Key::Key0, Key::Minus, Key::Equal, Key::Backspace, Key::Tab,
        // [16, 31]
        Key::Q, Key::W, Key::E, Key::R, Key::T, Key::Y, Key::U, Key::I, Key::O, Key::P, Key::LeftBracket, Key::RightBracket, Key::Enter, Key::LeftControl, Key::A, Key::S,
        // [32, 47]
        Key::D, Key::F, Key::G, Key::H, Key::J, Key::K, Key::L, Key::Semicolon, Key::Apostrophe, Key::GraveAccent, Key::LeftShift, Key::Backslash, Key::Z, Key::X, Key::C, Key::V,
        // [48, 63]
        Key::B, Key::N, Key::M, Key::Comma, Key::Period, Key::Slash, Key::RightShift, Key::KeypadMultiply, Key::LeftAlt, Key::Space, Key::CapsLock, Key::F1, Key::F2, Key::F3, Key::F4, Key::F5,
        // [64, 79]
        Key::F6, Key::F7, Key::F8, Key::F9, Key::F10, Key::Pause, Key::ScrollLock, Key::Keypad7, Key::Keypad8, Key::Keypad9, Key::KeypadSubtract, Key::Keypad4, Key::Keypad5, Key::Keypad6, Key::KeypadAdd, Key::Keypad1,
        // [80, 95]
        Key::Keypad2, Key::Keypad3, Key::Keypad0, Key::KeypadDecimal, Key::Invalid, Key::Invalid, Key::World2, Key::F11, Key::F12, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [96, 111]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::F13, Key::F14, Key::F15, Key::F16, Key::F17, Key::F18, Key::F19, Key::F20, Key::F21, Key::F22, Key::F23, Key::Invalid,
        // [112, 127]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::F24, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [128, 143]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [144, 159]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [160, 175]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [176, 191]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [192, 207]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [208, 223]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [224, 239]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [240, 255]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [256, 271]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [272, 287]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::KeypadEnter, Key::RightControl, Key::Invalid, Key::Invalid,
        // [288, 303]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [304, 319]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::KeypadDivide, Key::Invalid, Key::PrintScreen, Key::RightAlt, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [320, 335]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::NumLock, Key::Invalid, Key::Home, Key::Up, Key::PageUp, Key::Invalid, Key::Left, Key::Invalid, Key::Right, Key::Invalid, Key::End,
        // [336, 351]
        Key::Down, Key::PageDown, Key::Insert, Key::Delete, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::LeftSuper, Key::RightSuper, Key::Menu, Key::Invalid, Key::Invalid,
        // [352, 367]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [368, 383]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [384, 399]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [400, 415]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [416, 431]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [432, 447]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [448, 463]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [464, 479]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [480, 495]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid,
        // [496, 511]
        Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid, Key::Invalid
    };

    if (wParam == VK_CONTROL)
    {
        // The CTRL keys require special handling, since Alt Gr actually sends two messages, a LeftControl and then RightAlt.
        // We try to detect Alt Gr events by peeking ahead in the message queue.

        // Is this an extended key (i.e. right key)?
        if (lParam & 0x01000000)
            return Key::RightControl;

        DWORD time = GetMessageTime();
        MSG next;
        if (PeekMessageW(&next, nullptr, 0, 0, PM_NOREMOVE)) {
            if (next.message == WM_KEYDOWN || next.message == WM_SYSKEYDOWN ||
                next.message == WM_KEYUP || next.message == WM_SYSKEYUP) {
                if (next.wParam == VK_MENU && (next.lParam & 0x01000000) &&
                    next.time == time) {
                    // Next message is a RightAlt message, which
                    // means that this is not a proper LCTRL message
                    return Key::Invalid;
                }
            }
        }

        return Key::LeftControl;
    }

    return keymap[HIWORD(lParam) & 0x1FF];
}

LRESULT handle_input(UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_MOUSEMOVE:
        g_mouse->set_position(Vector2i(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)));
        break;
    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
        g_mouse->button_tapped(Mouse::Button::Left, message == WM_LBUTTONDOWN);
        break;
    case WM_RBUTTONDOWN:
    case WM_RBUTTONUP:
        g_mouse->button_tapped(Mouse::Button::Right, message == WM_RBUTTONDOWN);
        break;
    case WM_MBUTTONDOWN:
    case WM_MBUTTONUP:
        g_mouse->button_tapped(Mouse::Button::Middle, message == WM_MBUTTONDOWN);
        break;
    case WM_XBUTTONDOWN:
    case WM_XBUTTONUP: {
        Mouse::Button button = GET_XBUTTON_WPARAM(wParam) == XBUTTON1 ? Mouse::Button::Button4 : Mouse::Button::Button5;
        g_mouse->button_tapped(button, message == WM_XBUTTONDOWN);
        break;
    }
    case WM_MOUSEWHEEL:
        g_mouse->add_scroll_delta(GET_WHEEL_DELTA_WPARAM(wParam) / float(WHEEL_DELTA));
        break;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN: {
        Keyboard::Key key = translate_key(wParam, lParam);
        if (key != Keyboard::Key::Invalid)
            if (g_keyboard->is_pressed(key))
                g_keyboard->key_tapped(key, false);
        g_keyboard->key_tapped(key, true);
        break;
    }
    case WM_KEYUP:
    case WM_SYSKEYUP: {
        Keyboard::Key key = translate_key(wParam, lParam);
        if (key != Keyboard::Key::Invalid) {

            if (wParam == VK_SHIFT) {
                // Release both Shift keys on Shift up, as only one event
                // is sent even if both keys are released.
                if (g_keyboard->is_pressed(Keyboard::Key::LeftShift))
                    g_keyboard->key_tapped(Keyboard::Key::LeftShift, false);
                if (g_keyboard->is_pressed(Keyboard::Key::RightShift))
                    g_keyboard->key_tapped(Keyboard::Key::RightShift, false);
            } else
                // If a released key wasn't pressed, then quickly press is.
                // This also happens to be a wonderful fix to Print Screen not producing down events.
                if (g_keyboard->is_released(key))
                    g_keyboard->key_tapped(key, true);
            g_keyboard->key_tapped(key, false);
        }
        break;
    }
    case WM_CHAR:
    case WM_SYSCHAR:
    case WM_UNICHAR: {
        if (message == WM_UNICHAR && wParam == UNICODE_NOCHAR)
            // WM_UNICHAR is not sent by Windows, but is sent by some third-party input method engine
            // Returning TRUE here announces support for this message
            // See https://msdn.microsoft.com/en-us/library/windows/desktop/ms646288(v=vs.85).aspx
            return 1;

        unsigned int codepoint = unsigned int(wParam);

        // Ignore function keys
        const unsigned int BACKSPACE_CODE_POINT = 8;
        const unsigned int TAB_CODE_POINT = 9;
        const unsigned int RETURN_CODE_POINT = 13;
        if ((codepoint < 32 && codepoint != BACKSPACE_CODE_POINT && codepoint != TAB_CODE_POINT && codepoint != RETURN_CODE_POINT)
            || (codepoint > 127 && codepoint < 160))
            return 1;

        g_keyboard->add_codepoint(codepoint);
        break;
    }
    default:
        return 1;
    }

    return 0;
}

Vector2i compute_full_window_size(int backbuffer_width, int backbuffer_height) {
    DWORD window_ex_style = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
    
    DWORD window_style = 0;
    { // Window style
        window_style = WS_CLIPSIBLINGS | WS_CLIPCHILDREN;

        // Window is decorated.
        window_style |= WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;

        // Window is resizable.
        window_style |= WS_MAXIMIZEBOX | WS_SIZEBOX;
    }

    RECT rect = { 0, 0, backbuffer_width, backbuffer_height };
    AdjustWindowRectEx(&rect, window_style, false, window_ex_style);
    return Vector2i(rect.right - rect.left, rect.bottom - rect.top);
}

int run(OnLaunchCallback on_launch, OnWindowCreatedCallback on_window_created) {

    HINSTANCE instance_handle = GetModuleHandle(0);

    char exepath[512];
    GetModuleFileName(nullptr, exepath, 512);
    // Find the second last slash and terminate after by setting the next character to '0'.
    char* last_char = exepath + strlen(exepath);
    int slash_count = 0;
    while (slash_count != 2) {
        char c = *--last_char;
        if (c == '/' || c == '\\')
            ++slash_count;
    }
    *++last_char = 0;
    std::string data_path = std::string(exepath) + "Data\\";

    // Create engine.
    g_engine = new Engine(data_path);
    if (on_launch) {
        int error_code = on_launch(*g_engine);
        if (error_code != S_OK)
            return error_code;
    }

    // Create window.
    Bifrost::Core::Window& engine_window = g_engine->get_window();
    const char* window_title = engine_window.get_name().c_str();
    
    // Fill in window information and register it.
    WNDCLASSEX wc;
    ZeroMemory(&wc, sizeof(WNDCLASSEX));
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = handle_messages;
    wc.hInstance = instance_handle;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)COLOR_WINDOW;
    wc.lpszClassName = "main_window";
    RegisterClassEx(&wc);

    Vector2i full_window_size = compute_full_window_size(engine_window.get_width(), engine_window.get_height());

    // Create the window and its handle.
    HWND hwnd = CreateWindowEx(WS_EX_LEFT,
        "main_window", window_title,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, // (x, y) position of the window
        full_window_size.x, full_window_size.y,
        NULL, // Parent window
        NULL, // Menu handle
        instance_handle,
        NULL); // used with multiple windows, NULL

    if (hwnd == 0) {
        printf("ERROR: Could not create window.\n");
        return EXIT_FAILURE;
    }

    ShowWindow(hwnd, SW_SHOWDEFAULT);

    if (on_window_created) {
        int error_code = on_window_created(*g_engine, engine_window, hwnd);
        if (error_code != S_OK)
            return error_code;
    }

    // Setup keyboard.
    g_keyboard = new Keyboard();
    g_engine->set_keyboard(g_keyboard);

    // Setup mouse.
    POINT p;
    GetCursorPos(&p);
    ScreenToClient(hwnd, &p);
    g_mouse = new Mouse(Vector2i(p.x, p.y));
    g_engine->set_mouse(g_mouse);

    // Setup timer.
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    double counter_hertz = 1.0 / freq.QuadPart;
    LARGE_INTEGER performance_count;
    QueryPerformanceCounter(&performance_count);
    double previous_time = performance_count.QuadPart * counter_hertz;

    bool running = true;
    while (running && !g_engine->is_quit_requested()) {
        // Handle window updates. It's irrelevant whether this is at the end or the beginning of the loop, 
        // but if done first we get to handle all window updates from initialization calls as well.
        if (engine_window.has_changes(Bifrost::Core::Window::Changes::Resized)) {
            Vector2i full_window_size = compute_full_window_size(engine_window.get_width(), engine_window.get_height());
            SetWindowPos(hwnd, HWND_TOP,
                0, 0, full_window_size.x, full_window_size.y,
                SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOOWNERZORDER);
        }

        if (engine_window.has_changes(Bifrost::Core::Window::Changes::Renamed))
            SetWindowText(hwnd, engine_window.get_name().c_str());

        // Poll events.
        g_keyboard->per_frame_reset();
        g_mouse->per_frame_reset();
        MSG msg = {};
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT)
                running = false;

            handle_input(msg.message, msg.wParam, msg.lParam);
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // Poll and update time.
        QueryPerformanceCounter(&performance_count);
        double current_time = performance_count.QuadPart * counter_hertz;
        float delta_time = float(current_time - previous_time);
        previous_time = current_time;

        g_engine->do_tick(delta_time);
    }

    delete g_mouse;
    delete g_keyboard;

    return S_OK;
}

} // NS Win32Driver
