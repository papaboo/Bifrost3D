// Cogwheel GLFW main.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_GLFW_DRIVER_H_
#define _COGWHEEL_GLFW_DRIVER_H_

#include <Core/Engine.h>

namespace GLFWDriver {

typedef void (*on_launch_callback)(Cogwheel::Core::Engine& engine);

void run(on_launch_callback on_launch);

} // NS GLFWDriver

#endif // _COGWHEEL_GLFW_DRIVER_H_