// Test Cogwheel Bitmask.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_BITMASK_TEST_H_
#define _COGWHEEL_CORE_BITMASK_TEST_H_

#include <Cogwheel/Core/Bitmask.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Core {

enum class Core_TestEnum : unsigned char {
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    All = 7
};

GTEST_TEST(Core_Bitmask, set) {
    Bitmask<Core_TestEnum> bitmask = Core_TestEnum::Two;
    EXPECT_TRUE(bitmask.is_set(Core_TestEnum::Two));
    EXPECT_TRUE(bitmask == Core_TestEnum::Two);
}

GTEST_TEST(Core_Bitmask, union) {
    Bitmask<Core_TestEnum> bitmask = Core_TestEnum::Two;
    bitmask |= Core_TestEnum::One;

    EXPECT_TRUE(bitmask.any_set(Core_TestEnum::One));
    EXPECT_TRUE(bitmask.any_set(Core_TestEnum::Two));
    EXPECT_EQ(Core_TestEnum::Three, bitmask);
}

GTEST_TEST(Core_Bitmask, intersection) {
    Bitmask<Core_TestEnum> bitmask = Core_TestEnum::Three;
    bitmask &= Core_TestEnum::One;

    EXPECT_EQ(Core_TestEnum::One, bitmask);
    EXPECT_NE(Core_TestEnum::Two, bitmask);
    EXPECT_NE(Core_TestEnum::Three, bitmask);
}

GTEST_TEST(Core_Bitmask, comparison) {
    Bitmask<Core_TestEnum> bitmask1 = Core_TestEnum::One;
    Bitmask<Core_TestEnum> bitmask3 = Core_TestEnum::Three;

    // Bitmask/ enum comparison.
    EXPECT_TRUE(bitmask1 == Core_TestEnum::One);
    EXPECT_TRUE(bitmask3 == Core_TestEnum::Three);
    EXPECT_TRUE(bitmask1 != Core_TestEnum::Three);
    EXPECT_TRUE(bitmask3 != Core_TestEnum::One);
    
    // Bitmask/ bitmask comparison.
    EXPECT_TRUE(bitmask1 == bitmask1);
    EXPECT_TRUE(bitmask3 == bitmask3);
    EXPECT_TRUE(bitmask1 != bitmask3);
    EXPECT_TRUE(bitmask3 != bitmask1);
}

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_BITMASK_TEST_H_