// OptiX renderer manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_RENDERER_H_
#define _OPTIXRENDERER_RENDERER_H_

#include <Core/IModule.h>

namespace OptiXRenderer {

// TODO Create method that only returns a valid pointer in case a Renderer could be created.
class Renderer final : public Cogwheel::Core::IModule {
public:
    Renderer();

    inline bool could_initialize() const { return m_device_ids.optix>= 0;  }

    void apply() override;

    std::string get_name() override;

private:

    struct {
        int optix;
        int cuda;
    } m_device_ids;

    // Pimpl the state to avoid exposing OptiX headers.
    struct State;
    State* m_state;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_RENDERER_H_