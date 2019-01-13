// Test Bifrost Unique ID Generator.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_CORE_UNIQUE_ID_GENERATOR_TEST_H_
#define _BIFROST_CORE_UNIQUE_ID_GENERATOR_TEST_H_

#include <Bifrost/Core/UniqueIDGenerator.h>

#include <gtest/gtest.h>

#include <set>

namespace Bifrost {
namespace Core {

GTEST_TEST(Core_UniqueIDGenerator, capacity) {
    UIDGenerator gen = UIDGenerator(8u);
    EXPECT_GE(gen.capacity(), 8u);

    // Test that capacity can be increased.
    unsigned int largerCapacity = gen.capacity() + 4u;
    gen.reserve(largerCapacity);
    EXPECT_GE(gen.capacity(), largerCapacity);

    // Test that capacity won't be decreased.
    gen.reserve(5u);
    EXPECT_EQ(gen.capacity(), largerCapacity);
}

GTEST_TEST(Core_UniqueIDGenerator, generate_and_erase) {
    UIDGenerator gen = UIDGenerator(2u);
    UID id = gen.generate();
    EXPECT_TRUE(gen.has(id));

    bool erased = gen.erase(id);
    EXPECT_TRUE(erased);
    EXPECT_FALSE(gen.has(id));

    bool not_erased = !gen.erase(id);
    EXPECT_TRUE(not_erased);
}

GTEST_TEST(Core_UniqueIDGenerator, expanding) {
    UIDGenerator gen = UIDGenerator(2u);
    UID id0 = gen.generate();
    UID id1 = gen.generate();
    UID id2 = gen.generate();

    EXPECT_GE(gen.capacity(), 3u);
    EXPECT_TRUE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_TRUE(gen.has(id2));
}

GTEST_TEST(Core_UniqueIDGenerator, reusable_entries) {
    UIDGenerator gen = UIDGenerator(2u);

    UID id0 = gen.generate();
    EXPECT_TRUE(gen.has(id0));
    gen.erase(id0);
    EXPECT_FALSE(gen.has(id0));

    // Get capacity after the first element has been created and insure that it is never increased.
    unsigned int startCapacity = gen.capacity();

    UID id1 = gen.generate();
    EXPECT_TRUE(gen.has(id1));
    EXPECT_FALSE(gen.has(id0));
    gen.erase(id1);
    EXPECT_FALSE(gen.has(id1));

    UID id2 = gen.generate();
    EXPECT_TRUE(gen.has(id2));
    EXPECT_FALSE(gen.has(id1));
    EXPECT_FALSE(gen.has(id0));
    gen.erase(id2);
    EXPECT_FALSE(gen.has(id2));

    UID id3 = gen.generate();
    EXPECT_TRUE(gen.has(id3));
    EXPECT_FALSE(gen.has(id2));
    EXPECT_FALSE(gen.has(id1));
    EXPECT_FALSE(gen.has(id0));
    gen.erase(id3);
    EXPECT_FALSE(gen.has(id3));

    UID id4 = gen.generate();
    EXPECT_TRUE(gen.has(id4));
    EXPECT_FALSE(gen.has(id3));
    EXPECT_FALSE(gen.has(id2));
    EXPECT_FALSE(gen.has(id1));
    EXPECT_FALSE(gen.has(id0));
    gen.erase(id4);
    EXPECT_FALSE(gen.has(id4));

    EXPECT_EQ(startCapacity, gen.capacity());
}

GTEST_TEST(Core_UniqueIDGenerator, multiple_generate_and_erase) {
    UIDGenerator gen = UIDGenerator(8u);

    UID id0 = gen.generate();
    EXPECT_TRUE(gen.has(id0));
    
    UID id1 = gen.generate();
    EXPECT_TRUE(gen.has(id1));

    gen.erase(id0);
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));

    UID id2 = gen.generate();
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_TRUE(gen.has(id2));

    UID id3 = gen.generate();
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_TRUE(gen.has(id2));
    EXPECT_TRUE(gen.has(id3));

    UID id4 = gen.generate();
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_TRUE(gen.has(id2));
    EXPECT_TRUE(gen.has(id3));
    EXPECT_TRUE(gen.has(id4));

    gen.erase(id4);
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_TRUE(gen.has(id2));
    EXPECT_TRUE(gen.has(id3));
    EXPECT_FALSE(gen.has(id4));

    gen.erase(id2);
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_FALSE(gen.has(id2));
    EXPECT_TRUE(gen.has(id3));
    EXPECT_FALSE(gen.has(id4));

    UID id5 = gen.generate();
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_FALSE(gen.has(id2));
    EXPECT_TRUE(gen.has(id3));
    EXPECT_FALSE(gen.has(id4));
    EXPECT_TRUE(gen.has(id5));

    UID id6 = gen.generate();
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_FALSE(gen.has(id2));
    EXPECT_TRUE(gen.has(id3));
    EXPECT_FALSE(gen.has(id4));
    EXPECT_TRUE(gen.has(id5));
    EXPECT_TRUE(gen.has(id6));

    gen.erase(id5);
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_FALSE(gen.has(id2));
    EXPECT_TRUE(gen.has(id3));
    EXPECT_FALSE(gen.has(id4));
    EXPECT_FALSE(gen.has(id5));
    EXPECT_TRUE(gen.has(id6));

    UID id7 = gen.generate();
    EXPECT_FALSE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_FALSE(gen.has(id2));
    EXPECT_TRUE(gen.has(id3));
    EXPECT_FALSE(gen.has(id4));
    EXPECT_FALSE(gen.has(id5));
    EXPECT_TRUE(gen.has(id6));
    EXPECT_TRUE(gen.has(id7));
}

GTEST_TEST(Core_UniqueIDGenerator, iterator) {
    UIDGenerator gen = UIDGenerator(8u);

    UID id0 = gen.generate();
    EXPECT_TRUE(gen.has(id0));

    UID id1 = gen.generate();
    EXPECT_TRUE(gen.has(id1));

    UID id2 = gen.generate();
    EXPECT_TRUE(gen.has(id2));
    
    UID id3 = gen.generate();
    EXPECT_TRUE(gen.has(id3));
    
    UID id4 = gen.generate();
    EXPECT_TRUE(gen.has(id4));
    
    UID id5 = gen.generate();
    EXPECT_TRUE(gen.has(id5));

    { // Test that all ID's are looped over.
        std::set<UID> id_set;
        for (UID id : gen)
            id_set.insert(id);

        EXPECT_TRUE(id_set.find(id0) != id_set.end());
        EXPECT_TRUE(id_set.find(id1) != id_set.end());
        EXPECT_TRUE(id_set.find(id2) != id_set.end());
        EXPECT_TRUE(id_set.find(id3) != id_set.end());
        EXPECT_TRUE(id_set.find(id4) != id_set.end());
        EXPECT_TRUE(id_set.find(id5) != id_set.end());
    }

    { // Remove the first id.
        gen.erase(id0);

        std::set<UID> id_set;
        for (UID id : gen)
            id_set.insert(id);

        EXPECT_TRUE(id_set.find(id0) == id_set.end());
        EXPECT_TRUE(id_set.find(id1) != id_set.end());
        EXPECT_TRUE(id_set.find(id2) != id_set.end());
        EXPECT_TRUE(id_set.find(id3) != id_set.end());
        EXPECT_TRUE(id_set.find(id4) != id_set.end());
        EXPECT_TRUE(id_set.find(id5) != id_set.end());
    }
    
    { // Remove an id in the middle of the list.
        gen.erase(id3);

        std::set<UID> id_set;
        for (UID id : gen)
            id_set.insert(id);

        EXPECT_TRUE(id_set.find(id0) == id_set.end());
        EXPECT_TRUE(id_set.find(id1) != id_set.end());
        EXPECT_TRUE(id_set.find(id2) != id_set.end());
        EXPECT_TRUE(id_set.find(id3) == id_set.end());
        EXPECT_TRUE(id_set.find(id4) != id_set.end());
        EXPECT_TRUE(id_set.find(id5) != id_set.end());
    }

    { // Remove the id at the end of the list.
        gen.erase(id5);

        std::set<UID> id_set;
        for (UID id : gen)
            id_set.insert(id);

        EXPECT_TRUE(id_set.find(id0) == id_set.end());
        EXPECT_TRUE(id_set.find(id1) != id_set.end());
        EXPECT_TRUE(id_set.find(id2) != id_set.end());
        EXPECT_TRUE(id_set.find(id3) == id_set.end());
        EXPECT_TRUE(id_set.find(id4) != id_set.end());
        EXPECT_TRUE(id_set.find(id5) == id_set.end());
    }
}

} // NS Core
} // NS Bifrost

#endif // _BIFROST_CORE_UNIQUE_ID_GENERATOR_TEST_H_
