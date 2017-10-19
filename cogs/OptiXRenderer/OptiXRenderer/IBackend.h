// OptiX renderer backend interface.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2017, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_IBACKEND_H_
#define _OPTIXRENDERER_IBACKEND_H_

#include <optixu/optixpp_namespace.h>

namespace OptiXRenderer {

// ------------------------------------------------------------------------------------------------
// OptiX renderer backend interface.
// ------------------------------------------------------------------------------------------------
class IBackend {
public:
    virtual void resize_backbuffers(int width, int height) = 0;
    virtual void render(optix::Context& context, int width, int height) = 0;
};

// ------------------------------------------------------------------------------------------------
// Simple OptiX renderer backend.
// Launches the ray generation program and outputs directly to the output buffer.
// ------------------------------------------------------------------------------------------------
class SimpleBackend : public IBackend {
public:
    SimpleBackend(int entry_point) : m_entry_point(entry_point) { }
    void resize_backbuffers(int width, int height) {}
    void render(optix::Context& context, int width, int height) {
        context->launch(m_entry_point, width, height);
    }
private:
    int m_entry_point;
};

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_IBACKEND_H_