// Cogwheel Unique ID Generator.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_UNIQUE_ID_GENERATOR_H_
#define _COGWHEEL_CORE_UNIQUE_ID_GENERATOR_H_

namespace Cogwheel {
namespace Core {

//----------------------------------------------------------------------------
// Unique ID Generator.
// Used to generate unique ID's, which can be associated with different resources, 
// fx a unique ID is created pr resource, to distinguish between all resources, 
// but a unique ID is also created pr SceneNode to distinguish all nodes.
// See http://bitsquid.blogspot.de/2011/09/managing-decoupling-part-4-id-lookup.html.
// Future work
// * Enable assert in has(UID). It requires the last free element to always point inside the array, preferably at the sentinel element. If that happens, then I need to special case erase() to the case where next_element is 0, because then last_element is invalid.
// * We could generate something like OwnedUIDs, that can only be stored in one place and will erase themselves when they are no longer needed.
// ** In order to avoid the user accidentally deleting an OwnedUID, the UID member would of course have to be unaccessable. (or require an owned ID for erase, but then it needs to be used everywhere!)
//----------------------------------------------------------------------------
template <typename T>
class TypedUIDGenerator final {
public:

    //------------------------------------------------------------------------
    // Unique identifier.
    // The unique identifier contains a 24bit identifier, mID. 
    // Apart from that it contains an 8 bit incarnation count, 
    // which is used to avoid clashes when an ID is reused.
    //------------------------------------------------------------------------
    struct UID final {
    private:
        unsigned int m_ID_incarnation;

        // Make the TypedUIDGenerator a friend class to allow it to construct UIDs.
        friend class TypedUIDGenerator;

        UID(unsigned int id, unsigned int incarnation) : m_ID_incarnation((incarnation << 24u) | id) {}

        inline void set_ID(unsigned int id) { m_ID_incarnation = (m_ID_incarnation & 0xFF000000) | id; }
        inline unsigned int get_incarnation_count() const { return m_ID_incarnation >> 24u; }
        inline void increment_incarnation() { m_ID_incarnation += 0x01000000; }

    public:
        static const unsigned int MAX_IDS = 0xFFFFFF;

        // Creates a sentinel UID that will never be valid.
        UID() : m_ID_incarnation(0u) { } // NOTE AVH Should really be private or non-existent, but that requires not using the UID in any containers that needs default initialization
        static inline UID invalid_UID() { return UID(0u, 0u); }

        // The ID.
        inline unsigned int get_ID() const { return m_ID_incarnation & MAX_IDS; }

        // Implicit conversion to unsigned int is a shorthand way of accessing the ID.
        inline operator unsigned int() const { return get_ID(); }

        // Equality operator.
        inline bool operator==(const UID& rhs) const {
            return m_ID_incarnation == rhs.m_ID_incarnation;
        }
    };

    //------------------------------------------------------------------------
    // Constant iterator.
    // Searches all UIDs linearly and returns the ones that are valid.
    class ConstIterator {
    public:
        ConstIterator(UID* id, const TypedUIDGenerator& UID_generator)
            : m_id(id), m_UID_generator(UID_generator) { }
        inline ConstIterator& operator++() {
            ++m_id;
            while (m_id != m_UID_generator.end().m_id && !m_UID_generator.has(*m_id))
            // while (m_id != m_UID_generator.end().m_id && m_UID_generator.m_IDs[*m_id] != *m_id)
                ++m_id;
            return *this;
        }
        inline ConstIterator operator++(int) { ConstIterator tmp(*this); operator++(); return tmp; }
        inline bool operator==(const ConstIterator& rhs) { return m_id == rhs.m_id; }
        inline bool operator!=(const ConstIterator& rhs) { return m_id != rhs.m_id; }
        inline UID operator*() { return *m_id; }
        inline UID operator->() { return *m_id; }
    private:
        UID* m_id;
        const TypedUIDGenerator& m_UID_generator;
    };

    TypedUIDGenerator(unsigned int start_capacity = 256);
    TypedUIDGenerator(TypedUIDGenerator<T>&& other);
    ~TypedUIDGenerator();

    TypedUIDGenerator<T>& operator=(TypedUIDGenerator<T>&& rhs);

    UID generate();
    void erase(UID id);
    bool has(UID id) const;

    unsigned int capacity() const { return m_capacity; }
    void reserve(unsigned int capacity);
    unsigned int max_capacity() { return UID::MAX_IDS; }

    inline ConstIterator begin() const {
        UID* first_valid_ID = m_IDs + 1u; // Skip sentinel invalid id.
        ConstIterator itr = ConstIterator(first_valid_ID, *this);
        if (!has(*itr))
            // Advance iterator if the first element in the array isn't valid.
            ++itr;
        return itr;
    }

    inline ConstIterator end() const { return ConstIterator(m_IDs + m_capacity, *this); }

    // Debug! std::string to_string();

private:
    // Delete copy constructors to avoid having multiple versions of the same UID generator.
    TypedUIDGenerator(TypedUIDGenerator<T>& other) = delete;
    TypedUIDGenerator<T>& operator=(const TypedUIDGenerator<T>& rhs) = delete;

    unsigned int m_capacity;
    UID* m_IDs;
        
    unsigned int m_next_index;
    unsigned int m_last_index;
};

// Typedefs for 'untyped' UIDs.
typedef TypedUIDGenerator<void> UIDGenerator;
typedef UIDGenerator::UID UID;

} // NS Core
} // NS Cogwheel


#include "UniqueIDGenerator.impl"

#endif // _COGWHEEL_CORE_GLOBAL_UNIQUE_ID_GENERATOR_H_