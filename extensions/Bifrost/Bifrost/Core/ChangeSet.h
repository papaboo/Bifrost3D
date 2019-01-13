// Bifrost change set for resources.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_CORE_CHANGE_SET_H_
#define _BIFROST_CORE_CHANGE_SET_H_

#include <Bifrost/Core/Iterable.h>

#include <vector>

namespace Bifrost {
namespace Core {

// ---------------------------------------------------------------------------
// List changes for bifrost resources.
// ---------------------------------------------------------------------------
template <typename Bitmask, typename UID>
struct ChangeSet final {
public:
    typedef typename std::vector<UID>::iterator AssetIterator;

private:
    Bitmask* m_changes;
    std::vector<UID> m_resources_changed;
    int m_size;

public:

    ChangeSet() {
        m_size = 0;
        m_changes = nullptr;
    }

    ChangeSet(unsigned int size) {
        m_size = size;
        m_changes = new Bitmask[m_size];
        std::memset(m_changes, 0, m_size * sizeof(Bitmask));
        m_resources_changed.reserve(m_size / 4);
    }

    void resize(int new_size) {
        Bitmask* new_changes = new Bitmask[new_size];
        int copyable_elements = min(new_size, m_size);
        std::copy(m_changes, m_changes + copyable_elements, new_changes);
        delete[] m_changes;
        m_changes = new_changes;
        if (copyable_elements < new_size)
            std::memset(m_changes + copyable_elements, 0, (new_size - copyable_elements) * sizeof(Bitmask));
        m_size = new_size;
    }

    inline void set_change(UID id, Bitmask change) { 
        if (m_changes[id].none_set())
            m_resources_changed.push_back(id);
        m_changes[id] = change;
    }

    inline void add_change(UID id, Bitmask change) { set_change(id, m_changes[id] | change); }

    inline Bitmask get_changes(UID id) { return m_changes[id]; }

    inline Core::Iterable<AssetIterator> get_changed_resources() {
        return Core::Iterable<AssetIterator>(m_resources_changed.begin(), m_resources_changed.end());
    }

    inline void reset_change_notifications() {
        // NOTE We could use some heuristic here to choose between looping over 
        // the notifications and only resetting the changed materials instead of resetting all.
        std::memset(m_changes, 0, m_size * sizeof(Bitmask));
        m_resources_changed.resize(0);
    }

};

} // NS Core
} // NS Bifrost

#endif // _BIFROST_CORE_CHANGE_SET_H_
