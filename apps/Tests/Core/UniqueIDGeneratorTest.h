// Test Cogwheel Unique ID Generator.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_UNIQUE_ID_GENERATOR_TEST_H_
#define _COGWHEEL_CORE_UNIQUE_ID_GENERATOR_TEST_H_

#include <Core/UniqueIDGenerator.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Core {

GTEST_TEST(Core_UniqueIDGeneratorTest, TestCapacity) {
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

GTEST_TEST(Core_UniqueIDGeneratorTest, TestGenerateAndErase) {
    UIDGenerator gen = UIDGenerator(2u);
    UID id = gen.generate();
    EXPECT_TRUE(gen.has(id));

    gen.erase(id);
    EXPECT_FALSE(gen.has(id));
}

GTEST_TEST(Core_UniqueIDGeneratorTest, TestExpanding) {
    UIDGenerator gen = UIDGenerator(2u);
    UID id0 = gen.generate();
    UID id1 = gen.generate();
    UID id2 = gen.generate();

    EXPECT_GE(gen.capacity(), 3u);
    EXPECT_TRUE(gen.has(id0));
    EXPECT_TRUE(gen.has(id1));
    EXPECT_TRUE(gen.has(id2));
}

GTEST_TEST(Core_UniqueIDGeneratorTest, TestReusableEntries) {
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

GTEST_TEST(Core_UniqueIDGeneratorTest, TestMultipleGenerateAndErase) {
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

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_UNIQUE_ID_GENERATOR_TEST_H_