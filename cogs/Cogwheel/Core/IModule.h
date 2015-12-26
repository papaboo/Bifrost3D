// Cogwheel module inerface.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_MODULE_INTERFACE_H_
#define _COGWHEEL_CORE_MODULE_INTERFACE_H_

#include <string>

namespace Cogwheel {
namespace Core {

// ---------------------------------------------------------------------------
// Module interface.
// Modules can be hooked up to the engine and will be called every engine tick.
// A module needs to implement the apply() method, which will be invoked by 
// the engine. Additionally there is a get_name() method, which is used to
// quickly determine which modules are used.
// TODO
// * Serialization? Do we really care about that here or should it only be in the 'heaps abstractions', such as SceneNodes, Cameras and Models.
// ---------------------------------------------------------------------------
class IModule {
public:
    virtual void apply() = 0;

    virtual std::string get_name() = 0;
};

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_MODULE_INTERFACE_H_