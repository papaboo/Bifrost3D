// Bifrost renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_CORE_RENDERER_H_
#define _BIFROST_CORE_RENDERER_H_

#include <Bifrost/Core/Iterable.h>
#include <Bifrost/Core/UniqueIDGenerator.h>

#include <string>

namespace Bifrost::Core {

//----------------------------------------------------------------------------
// Renderer ID
//----------------------------------------------------------------------------
class Renderers;
typedef Core::TypedUIDGenerator<Renderers> RendererIDGenerator;
typedef RendererIDGenerator::UID RendererID;

// ------------------------------------------------------------------------------------------------
// Renderer representation in Bifrost.
// Essentially just an ID that represents a renderer and can attach that renderer to a camera.
// ------------------------------------------------------------------------------------------------
class Renderers final {
public:
    using Iterator = RendererIDGenerator::ConstIterator;

    static bool is_allocated() { return m_names != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(RendererID renderer_ID) { return m_UID_generator.has(renderer_ID); }

    static RendererID create(const std::string& name);
    static void destroy(RendererID renderer_ID);

    static Iterator begin() { return m_UID_generator.begin(); }
    static Iterator end() { return m_UID_generator.end(); }
    static Iterator get_iterator(RendererID renderer_ID) { return m_UID_generator.get_iterator(renderer_ID); }
    static Iterable<Iterator> get_iterable() { return Iterable<Iterator>(begin(), end()); }

    static const std::string& get_name(RendererID renderer_ID) { return m_names[renderer_ID]; }

private:

    static void reserve_data(unsigned int new_capacity, unsigned int old_capacity);

    static RendererIDGenerator m_UID_generator;
    static std::string* m_names;
};

} // NS Bifrost::Core

#endif _BIFROST_CORE_RENDERER_H_
