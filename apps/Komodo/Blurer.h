// Komodo image blurer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _KOMODO_BLURER_H_
#define _KOMODO_BLURER_H_

#include <vector>

namespace Bifrost {
namespace Core {
class Engine;
}
}

class Blurer final {
public:
    Blurer(std::vector<char*> args, Bifrost::Core::Engine& engine);

private:
    // Pimpl the state to avoid exposing OptiX headers.
    struct Implementation;
    Implementation* m_impl;
};

#endif // _KOMODO_BLURER_H_