// Cogwheel renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_RENDERER_H_
#define _COGWHEEL_CORE_RENDERER_H_

#include <Cogwheel/Core/Iterable.h>
#include <Cogwheel/Core/UniqueIDGenerator.h>

#include <string>

namespace Cogwheel {
namespace Core {

class Renderers final {
public:
    typedef Core::TypedUIDGenerator<Renderers> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static bool is_allocated() { return m_names != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(Renderers::UID renderer_ID) { return m_UID_generator.has(renderer_ID); }

    static Renderers::UID create(const std::string& name);
    static void destroy(Renderers::UID renderer_ID);

    static ConstUIDIterator begin() { return m_UID_generator.begin(); }
    static ConstUIDIterator end() { return m_UID_generator.end(); }
    static UIDGenerator::ConstIterator get_iterator(Renderers::UID renderer_ID) { return m_UID_generator.get_iterator(renderer_ID); }
    static Iterable<ConstUIDIterator> get_iterable() { return Iterable<ConstUIDIterator>(begin(), end()); }

    static std::string get_name(Renderers::UID renderer_ID) { return m_names[renderer_ID]; }

private:

    static void reserve_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;
    static std::string* m_names;
};

} // NS Core
} // NS Cogwheel

#endif _COGWHEEL_CORE_RENDERER_H_