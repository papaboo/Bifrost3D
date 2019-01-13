// Test Bifrost Array.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_CORE_ARRAY_TEST_H_
#define _BIFROST_CORE_ARRAY_TEST_H_

#include <Bifrost/Core/Array.h>

#include <gtest/gtest.h>

namespace Bifrost {
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
    Array<unsigned int> array0 = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u };
    Array<unsigned int> array1 = array0;

    EXPECT_NE(array0.begin(), array1.begin());
    EXPECT_NE(array0.end(), array1.end());

    for (unsigned int i = 0; i != array0.size(); ++i)
        EXPECT_EQ(array0[i], array1[i]);
}

GTEST_TEST(Core_Array, indexing) {
    // Create array and test that index and data are equal.
    Array<unsigned int> array = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u };
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

GTEST_TEST(Core_Array, push_back) {
    Array<unsigned int> array = { 0u, 1u, 2u };
    
    Array<unsigned int> tail = { 3u, 4u, 5u };
    array.push_back(tail.begin(), tail.end());
    EXPECT_EQ(array.size(), 6u);
    for (unsigned int i = 0; i != array.size(); ++i)
        EXPECT_EQ(i, array[i]);

    array.push_back( {6u, 7u, 8u} );
    EXPECT_EQ(array.size(), 9u);
    for (unsigned int i = 0; i != array.size(); ++i)
        EXPECT_EQ(i, array[i]);
}

} // NS Core
} // NS Bifrost

#endif // _BIFROST_CORE_ARRAY_TEST_H_
