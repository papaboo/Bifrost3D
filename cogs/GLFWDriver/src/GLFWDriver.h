// Cogwheel GLFW main.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_GLFW_DRIVER_H_
#define _COGWHEEL_GLFW_DRIVER_H_

//----------------------------------------------------------------------------
// Forward declerations
//----------------------------------------------------------------------------
namespace Cogwheel {
namespace Core {
class Engine;
class Window;
}
}

namespace GLFWDriver {

typedef void (*on_launch_callback)(Cogwheel::Core::Engine& engine);
typedef void (*on_window_created_callback)(Cogwheel::Core::Window& window);

void run(on_launch_callback on_launch, on_window_created_callback on_window_created);

} // NS GLFWDriver

#endif // _COGWHEEL_GLFW_DRIVER_H_