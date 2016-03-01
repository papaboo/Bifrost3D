// Cogwheel resizable array.
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
// Future work
// * Support function pointers. (possibly by using aligned_alloc? Check how std::vector does it.)
// * Constructor taking two iterators as argument.
// * Add emplace.
// * Add swap.
// * Specialize for booleans and add 'clearAll()' and 'setAll()' methods.
// * Specialize for arbitrary bits pr integer element?
// * Use malloc to allocate, but make sure that it supports memory aligned structs.
// ---------------------------------------------------------------------------
template <typename T, typename SizeType = unsigned int>
struct Array final {
public:
    typedef T value_type;
    typedef SizeType size_type;
    typedef T* iterator;
    typedef const T* const_iterator;

private:
    size_type m_size;
    T* m_data;

public:

    // -----------------------------------------------------------------------
    // Constructors and destrcutor
    // -----------------------------------------------------------------------
    Array()
        : m_size(0), m_data(nullptr) {
    }
    explicit Array(size_type size)
        : m_size(size), m_data(new T[m_size]) {}
    Array(Array<T>&& other)
        : m_size(other.m_size), m_data(other.m_data) {
        other.m_size = 0; other.m_data = nullptr;
    }
	Array(const Array<T>& other)
		: m_size(other.m_size), m_data(new T[m_size]) {
		std::copy(other.m_data, other.m_data + m_size, m_data);
	}
    Array(const std::initializer_list<T>& list)
        : m_size(static_cast<size_type>(list.size())), m_data(new T[m_size]) {
        std::copy(list.begin(), list.end(), m_data);
    }
    ~Array() {
        if (m_data) 
            delete[] m_data;
    }

    // -----------------------------------------------------------------------
    // Assignment
    // -----------------------------------------------------------------------
    Array& operator=(Array<T>&& rhs) {
        m_size = rhs.m_size; 
        m_data = rhs.m_data;
        rhs.m_size = 0;
        rhs.m_data = nullptr;
        return *this;
    }
    Array& operator=(const Array<T>& rhs) {
        m_size = rhs.m_size;
        m_data = new T[m_size];
        std::copy(rhs.m_data, rhs.m_data + m_size, m_data);
        return *this;
    }
    
    // -----------------------------------------------------------------------
    // Iterators
    // -----------------------------------------------------------------------
    inline iterator begin() { return m_data; }
    inline const_iterator begin() const { return m_data; }
    inline iterator end() { return m_data + m_size; }
    inline const_iterator end() const { return m_data + m_size; }

    // -----------------------------------------------------------------------
    // Size
    // -----------------------------------------------------------------------
    inline size_type size() const { return m_size; }
    inline void resize(size_type size) {
        T* new_data = new T[size];
        size_type minSize = size < m_size ? size : m_size;
        memcpy(new_data, m_data, sizeof(T) * minSize);
        delete[] m_data;
        m_data = new_data;
        m_size = size;
    }

    // -----------------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------------
    inline T& operator[](size_type i) { return m_data[i]; }
    inline const T& operator[](size_type i) const { return m_data[i]; }
    inline T* data() { return m_data; }
    inline const T* data() const { return m_data; }

    // -----------------------------------------------------------------------
    // Modifiers
    // -----------------------------------------------------------------------
    void push_back(T element) {
        size_type previous_size = m_size;
        resize(m_size + 1u);
        m_data[previous_size] = element;
    }
    void push_back(const T* begin, const T* end) {
        size_type previous_size = m_size;
        resize(m_size + size_type(end - begin));
        const T* from = begin;
        T* to = m_data + previous_size;
        while (from != end) {
            *to = *from;
            ++from;
            ++to;
        }
    }
    void push_back(const std::initializer_list<T>& list) {
        push_back(list.begin(), list.end());
    }
};

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_CORE_ARRAY_H_