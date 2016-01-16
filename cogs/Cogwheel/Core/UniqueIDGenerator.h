// Cogwheel Unique ID Generator.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_UNIQUE_ID_GENERATOR_H_
#define _COGWHEEL_CORE_UNIQUE_ID_GENERATOR_H_

#include <vector>

namespace Cogwheel {
namespace Core {

// Unique ID Generator.
// Used to generate unique ID's, which can be associated with different resources, 
// fx a unique ID is created pr resource, to distinguish between all resources, 
// but a unique ID is also created pr SceneNode to distinguish all nodes.
// See http://bitsquid.blogspot.de/2011/09/managing-decoupling-part-4-id-lookup.html.
// TODO
// * Possibility of looping over active IDs.
// * We could generate something like OwnedUIDs, that can only be stored in one place an will erase themselves when they are no longer needed.
// ** In order to avoid the user accidentally deleting an OwnedUID, the UID member would of course have to be unaccessable. (or require an owned ID for erase, but then it needs to be used everywhere!)
template <typename T>
class TypedUIDGenerator final {
public:
        
    // Unique identifier.
    // The unique identifier contains a 24bit identifier, mID. Apart from that it contains an 8 bit incarnation count, used to avoid clashes when an ID is reused.
    struct UID final {
    private:
        unsigned int m_incarnation : 8;
        unsigned int m_ID : 24;

        // Make the TypedUIDGenerator a friend class to allow it to construct UIDs.
        friend class TypedUIDGenerator;

        UID(unsigned int id, unsigned int incarnation) : m_incarnation(incarnation), m_ID(id) { }
    public:
        static const unsigned int MAX_IDS = 0xFFFFFF;

        // Creates a sentinel UID that will never be valid.
        UID() : m_incarnation(0), m_ID(0) { } // NOTE AVH Should really be private or non-existent, but that requires not using the UID in any containers that needs default initialization
        static inline UID invalid_UID() { return UID(0, 0u); }

        // The ID.
        inline unsigned int ID() const { return m_ID; }

        // Implicit conversion to unsigned int is a shorthand way of accessing the ID.
        inline operator unsigned int() const { return m_ID; }

        // Equality operator.
        inline bool operator==(const UID& rhs) const {
            return m_ID == rhs.m_ID && m_incarnation == rhs.m_incarnation;
        }
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

    // Debug! std::string UIDGenerator::to_string();

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