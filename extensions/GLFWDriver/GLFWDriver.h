// Bifrost GLFW main.
// ----------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _BIFROST_GLFW_DRIVER_H_
#define _BIFROST_GLFW_DRIVER_H_

//----------------------------------------------------------------------------
// Forward declerations
//----------------------------------------------------------------------------
namespace Bifrost::Core {
class Engine;
class Window;
}

namespace GLFWDriver {

typedef int (*OnLaunchCallback)(Bifrost::Core::Engine&);
typedef int (*OnWindowCreatedCallback)(Bifrost::Core::Engine&, Bifrost::Core::Window&);

int run(OnLaunchCallback on_launch, OnWindowCreatedCallback on_window_created);

} // NS GLFWDriver

#endif // _BIFROST_GLFW_DRIVER_H_
