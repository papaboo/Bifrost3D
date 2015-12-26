// Test Cogwheel Array.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_ARRAY_TEST_H_
#define _COGWHEEL_CORE_ARRAY_TEST_H_

#include <Core/Array.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Core {

GTEST_TEST(Core_Array, resizing) {
    Array<unsigned int> array = Array<unsigned int>(8u);
    EXPECT_EQ(8u, array.size());

    array.resize(25u);
    EXPECT_EQ(25u, array.size());

    array.resize(5u);
    EXPECT_EQ(5u, array.size());
}

GTEST_TEST(Core_Array, copying) {
    Array<unsigned int> array0 = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    Array<unsigned int> array1 = array0;

    EXPECT_NE(array0.begin(), array1.begin());
    EXPECT_NE(array0.end(), array1.end());

    for (unsigned int i = 0; i != array0.size(); ++i)
        EXPECT_EQ(array0[i], array1[i]);
}

GTEST_TEST(Core_Array, indexing) {
    // Create array and test that index and data are equal.
    Array<unsigned int> array = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    for (unsigned int i = 0; i != array.size(); ++i) {
        EXPECT_EQ(i, array[i]);
        EXPECT_EQ(i, array.data()[i]);
        EXPECT_EQ(i, array.begin()[i]);
    }

    // Shrink array and test that index and data are equal.
    array.resize(5u);
    for (unsigned int i = 0; i != array.size(); ++i)
        EXPECT_EQ(i, array[i]);
}

GTEST_TEST(Core_Array, zero_sized) {
    Array<unsigned int> array1 = Array<unsigned int>(0);
    array1.resize(2);
    array1.resize(0);

    Array<unsigned int> array2 = {};
    array2.resize(2);
    array2.resize(0);
}

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_ARRAY_TEST_H_