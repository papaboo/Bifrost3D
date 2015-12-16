// Cogwheel resiable array.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_CORE_ARRAY_H_
#define _COGWHEEL_CORE_ARRAY_H_

#include <initializer_list>

namespace Cogwheel {
namespace Core {

// ---------------------------------------------------------------------------
// Array is similar to std::vector, except that size and capacity are the same 
// and it will not require objects to have a default constructor, and as such 
// elements not explicitly initialized will contain undefined data.
// TODO
// * Specialize for booleans and add 'clearAll()' and 'setAll()' methods.
// * Specialize for arbitrary bits pr integer element?
// * Constructor taking two iterators as argument
// * Add emplace.
// * Add swap.
// ---------------------------------------------------------------------------
template <typename T, typename SizeType = unsigned int>
struct Array final {
public:
    typedef T value_type;
    typedef SizeType size_type;
    typedef T* iterator;
    typedef const T* const_iterator;

private:
    size_type mSize;
    T* mData;

public:

    // -----------------------------------------------------------------------
    // Constructors and destrcutor
    // -----------------------------------------------------------------------
    Array(size_type size)
        : mSize(size), mData(new T[mSize]) {}
    Array(Array<T>&& other)
        : mSize(other.mSize), mData(other.mData) {
        other.mSize = 0; other.mData = nullptr;
    }
	Array(const Array<T>& other)
		: mSize(other.mSize), mData(new T[mSize]) {
		std::copy(other.mData, other.mData + mSize, mData);
	}
    Array(const std::initializer_list<T>& list)
        : mSize(static_cast<size_type>(list.size())), mData(new T[mSize]) {
        std::copy(list.begin(), list.end(), mData);
    }
    ~Array() {
        if (mData) 
            delete[] mData;
    }

    // -----------------------------------------------------------------------
    // Assignment
    // -----------------------------------------------------------------------
    Array& operator=(Array<T>&& rhs) {
        mSize = rhs.mSize; 
        mData = rhs.mData;
        rhs.mSize = 0;
        rhs.mData = nullptr;
        return *this;
    }
    Array& operator=(const Array<T>& rhs) {
        mSize = rhs.mSize;
        mData = new T[mSize]; // TODO malloc to avoid default initialization.
        std::copy(rhs.mData, rhs.mData + mSize, mData);
        return *this;
    }
    
    // -----------------------------------------------------------------------
    // Iterators
    // -----------------------------------------------------------------------
    inline iterator begin() { return mData; }
    inline const_iterator begin() const { return mData + mSize; }
    inline iterator end() { return mData; }
    inline const_iterator end() const { return mData + mSize; }

    // -----------------------------------------------------------------------
    // Size
    // -----------------------------------------------------------------------
    inline size_type size() const { return mSize; }
    inline void resize(size_type size) {
        T* newData = new T[size];
        size_type minSize = size < mSize ? size : mSize;
        memcpy(newData, mData, sizeof(T) * minSize);
        delete[] mData;
        mData = newData;
        mSize = size;
    }

    // -----------------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------------
    inline T& operator[](size_type i) { return mData[i]; }
    inline const T& operator[](size_type i) const { return mData[i]; }
    inline T* data() { return mData; }
    inline const T* data() const { return mData; }
};

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_ARRAY_H_