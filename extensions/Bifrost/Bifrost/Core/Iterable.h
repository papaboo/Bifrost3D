// Bifrost iterable.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_CORE_ITERABLE_H_
#define _BIFROST_CORE_ITERABLE_H_

namespace Bifrost {
namespace Core {

// ---------------------------------------------------------------------------
// Bifrost iterable.
// An iterable is an object that allows iteration over elements.
// Convenient for returning begin and end from function calls and 
// allowing easy use of c++11 for-each when an object contains multiple arrays.
// See fx Scene::SceneNodes.
// See http://ericniebler.com/2014/10/06/counted-ranges-and-efficiency/
// ---------------------------------------------------------------------------
template <typename Iterator>
struct Iterable final {
public:

    Iterable() = default;
    Iterable(Iterator begin, Iterator end)
        : m_begin(begin), m_end(end) {}

    Iterable(Iterator begin, size_t element_count)
        : m_begin(begin), m_end(begin + element_count) {}

    Iterator begin() const { return m_begin; }
    Iterator end() const { return m_end; }

    bool is_empty() const { return m_begin == m_end; }

private:
    const Iterator m_begin;
    const Iterator m_end;
};

} // NS Core
} // NS Bifrost

#endif // _BIFROST_CORE_ITERABLE_H_
