// Ant tweak bar cogwheel wrapper.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_ANT_TWEAK_BAR_H_
#define _COGWHEEL_ANT_TWEAK_BAR_H_

#include <AntTweakBar.h> // Convenience include of ant tweak bar fanctionality

// ------------------------------------------------------------------------------------------------
// Forward declerations
// ------------------------------------------------------------------------------------------------
namespace Cogwheel {
namespace Core {
class Engine;
}
}

namespace AntTweakBar {

void handle_input(const Cogwheel::Core::Engine& engine);

} // NS AntTweakBar

#endif // _COGWHEEL_ANT_TWEAK_BAR_H_