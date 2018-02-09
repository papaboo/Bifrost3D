// Komodo image tone mapper.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _KOMODO_TONE_MAPPER_H_
#define _KOMODO_TONE_MAPPER_H_

#include <vector>

namespace Cogwheel {
namespace Core {
class Engine;
}
}

class ToneMapper final {
public:
    ToneMapper(std::vector<char*> args, Cogwheel::Core::Engine& engine);

private:
    // Pimpl the state to avoid exposing OptiX headers.
    struct Implementation;
    Implementation* m_impl;
};

#endif // _KOMODO_TONE_MAPPER_H_