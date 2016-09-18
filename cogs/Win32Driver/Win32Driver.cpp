// Cogwheel Win32 main.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#include <Win32Driver.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#undef RGB

#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>

using Cogwheel::Core::Engine;
using Cogwheel::Input::Keyboard;
using Cogwheel::Input::Mouse;
using Cogwheel::Math::Vector2i;

namespace Win32Driver {

Engine* g_engine = NULL;
Keyboard* g_keyboard = NULL;
Mouse* g_mouse = NULL;

// Windows message handler callback.
LRESULT CALLBACK handle_messages(HWND window_handle, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_SIZE:
        g_engine->get_window().resize(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(window_handle, message, wParam, lParam);
    }

    return 0;
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
    case WM_KEYUP:
        // TODO
        break;
    default:
        return 1;
    }

    return 0;
}

int run(OnLaunchCallback on_launch, OnWindowCreatedCallback on_window_created) {

    HINSTANCE instance_handle = GetModuleHandle(0);

    // Create engine.
    g_engine = new Engine();
    on_launch(*g_engine);

    // Create window.
    Cogwheel::Core::Window& engine_window = g_engine->get_window();
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

    // Create the window and its handle.
    HWND hwnd = CreateWindowEx(WS_EX_LEFT,
        "main_window", window_title,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, // (x, y) position of the window
        engine_window.get_width(), engine_window.get_height(),
        NULL, // Parent window
        NULL, // Menu handle
        instance_handle,
        NULL); // used with multiple windows, NULL

    if (hwnd == 0) {
        printf("ERROR: Could not create window.\n");
        return EXIT_FAILURE;
    }

    ShowWindow(hwnd, SW_SHOWDEFAULT);

    on_window_created(*g_engine, engine_window);

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

    while (!g_engine->is_quit_requested()) {
        // Poll events.
        g_keyboard->per_frame_reset();
        g_mouse->per_frame_reset();
        MSG msg = {};
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            handle_input(msg.message, msg.wParam, msg.lParam);

            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (msg.message == WM_QUIT)
            break;

        // Poll and update time.
        QueryPerformanceCounter(&performance_count);
        double current_time = performance_count.QuadPart * counter_hertz;
        float delta_time = float(current_time - previous_time);
        previous_time = current_time;

        g_engine->do_tick(delta_time);

        /* TODO
        if (engine_window.has_changes(Cogwheel::Core::Window::Changes::Resized))
            glfwSetWindowSize(window, engine_window.get_width(), engine_window.get_height());
        if (engine_window.has_changes(Cogwheel::Core::Window::Changes::Renamed))
            glfwSetWindowTitle(window, engine_window.get_name().c_str());
            */
    }

    return 0;
}

} // NS Win32Driver
