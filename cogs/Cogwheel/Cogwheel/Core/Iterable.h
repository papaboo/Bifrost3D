// Cogwheel iterable.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_ITERABLE_H_
#define _COGWHEEL_CORE_ITERABLE_H_

namespace Cogwheel {
namespace Core {

// ---------------------------------------------------------------------------
// Cogwheel iterable.
// An iterable is an object that allows iteration over elements.
// Convenient for returning begin and end from function calls and 
// allowing easy use of c++11 for-each when an object contains multiple arrays.
// See fx Scene::SceneNodes.
// See http://ericniebler.com/2014/10/06/counted-ranges-and-efficiency/
// ---------------------------------------------------------------------------
template <typename Iterator>
struct Iterable final {
public:

    Iterable(Iterator begin, Iterator end)
        : m_begin(begin), m_end(end) {}

    Iterator begin() { return m_begin; }
    Iterator end() { return m_end; }

private:
    Iterator m_begin;
    Iterator m_end;
};

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_ITERABLE_H_