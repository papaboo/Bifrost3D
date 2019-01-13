// Ant tweak bar bifrost wrapper.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_ANT_TWEAK_BAR_H_
#define _BIFROST_ANT_TWEAK_BAR_H_

#include <AntTweakBar.h> // Convenience include of ant tweak bar functionality

// ------------------------------------------------------------------------------------------------
// Forward declerations
// ------------------------------------------------------------------------------------------------
namespace Bifrost {
namespace Core {
class Engine;
}
}

namespace AntTweakBar {

void handle_input(const Bifrost::Core::Engine& engine);

} // NS AntTweakBar

#endif // _BIFROST_ANT_TWEAK_BAR_H_
