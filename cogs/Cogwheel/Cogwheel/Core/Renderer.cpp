// Cogwheel renderer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Cogwheel/Core/Renderer.h>

namespace Cogwheel {
namespace Core {

Renderers::UIDGenerator Renderers::m_UID_generator = UIDGenerator(0u);
std::string* Renderers::m_names = nullptr;

void Renderers::allocate(unsigned int capacity) {
    assert(!is_allocated());

    m_UID_generator = UIDGenerator(capacity);
    m_names = new std::string[m_UID_generator.capacity()];
}

void Renderers::deallocate() {
    assert(is_allocated());

    m_UID_generator = UIDGenerator(0u);
    delete[] m_names; m_names = nullptr;
}

void Renderers::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    new_capacity = m_UID_generator.capacity();
    reserve_data(new_capacity, old_capacity);
}

void Renderers::reserve_data(unsigned int new_capacity, unsigned int old_capacity) {
    std::string* new_names = new std::string[new_capacity];
    std::copy(m_names, m_names + old_capacity, new_names);
    delete[] m_names;
    m_names = new_names;
}

Renderers::UID Renderers::create(const std::string& name) {
    assert(m_names != nullptr);

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_data(m_UID_generator.capacity(), old_capacity);

    m_names[id] = name;
    return id;
}

} // NS Core
} // NS Cogwheel