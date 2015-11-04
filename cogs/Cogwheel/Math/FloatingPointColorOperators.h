//*****************************************************************************
// Header with operators for floating point colors.
// Used by RGB and RGBA.
// Requires that 
// * the class is typedef'ed as Color.
// * The number of elements in the Vector is given by N.
// * A pointer to the first element can be obtained by calling begin().
// * The type of each element is T.
//
// TODO
// * Can/should I use copy constructors when returning Colors?
//*****************************************************************************

//*****************************************************************************
// Indexing operators.
//*****************************************************************************
inline T& operator[](const int i) { return begin()[i]; }
inline const T& operator[](const int i) const { return begin()[i]; }

//*****************************************************************************
// Addition operators.
//*****************************************************************************
inline void operator+=(const Color& v) {
    for (int i = 0; i < N; ++i)
        this[i] += v(i);
}
inline Color operator+(const Color& rhs) const {
    Color ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] += rhs[i];
    return ret;
}

//*****************************************************************************
// Subtraction operators.
//*****************************************************************************
inline void operator-=(const T v) {
    for (int i = 0; i < N; ++i)
        this[i] -= v;
}
inline void operator-=(const Color& v) {
    for (int i = 0; i < N; ++i)
        this[i] -= v(i);
}
inline Color operator-(const T rhs) const {
    Color ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] -= rhs;
    return ret;
}
inline Vector<T> operator-(const Vector<T>& rhs) const {
    Vector<T> ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] -= rhs[i];
    return ret;
}

//*****************************************************************************
// Multiplication operators.
//*****************************************************************************
inline void operator*=(const T v) {
    for (int i = 0; i < N; ++i)
        this[i] *= v;
}
inline void operator*=(const Vector<T>& v) {
    for (int i = 0; i < N; ++i)
        this[i] *= v(i);
}
inline Vector<T> operator*(const T rhs) const {
    Vector<T> ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] *= rhs;
    return ret;
}
inline Vector<T> operator*(const Vector<T>& rhs) const {
    Vector<T> ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] *= rhs[i];
    return ret;
}

//*****************************************************************************
// Division operators.
//*****************************************************************************
inline void operator/=(const T v) {
    for (int i = 0; i < N; ++i)
        this[i] /= v;
}
inline void operator/=(const Vector<T>& v) {
    for (int i = 0; i < N; ++i)
        this[i] /= v(i);
}
inline Vector<T> operator/(const T rhs) const {
    Vector<T> ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] /= rhs;
    return ret;
}
inline Vector<T> operator/(const Vector<T>& rhs) const {
    Vector<T> ret(*this);
        for (int i = 0; i < N; ++i)
            ret[i] /= rhs[i];
    return ret;
}

//*****************************************************************************
// Comparison operators.
//*****************************************************************************
inline bool operator==(const Vector<T>& rhs) const {
    for (int i = 0; i < N; ++i)
        if (this[i] != rhs[i]) return false;
    return true;
}
inline bool operator!=(const Vector<T>& rhs) const {
    for (int i = 0; i < N; ++i)
        if (this[i] == rhs[i]) return false;
    return true;
}